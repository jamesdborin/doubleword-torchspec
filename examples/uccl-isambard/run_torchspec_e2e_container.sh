#!/usr/bin/env bash

# Runs inside the persistent podman-hpc container on one of the two nodes.

set -euo pipefail

: "${E2E_RUN_DIR:?}"
: "${E2E_CONFIG_FILE:?}"
: "${E2E_DATASET_FILE:?}"
: "${E2E_LOCAL_IP:?}"
: "${E2E_NODE_ID:?}"
: "${RAY_GCS_PORT:?}"

RUN_DIR="$E2E_RUN_DIR"
NODE_ID="$E2E_NODE_ID"
LOCAL_IP="$E2E_LOCAL_IP"
SGLANG_COPY="$RUN_DIR/sglang"
RAY_TEMP="/tmp/ray-torchspec-${SLURM_JOB_ID}-${NODE_ID}"
export HOME="/tmp/torchspec-home-${SLURM_JOB_ID}-${NODE_ID}"
export XDG_CACHE_HOME="$HOME/.cache"
export FLASHINFER_WORKSPACE_DIR="$XDG_CACHE_HOME/flashinfer"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
mkdir -p "$RAY_TEMP" "$FLASHINFER_WORKSPACE_DIR" "$TRITON_CACHE_DIR"

wait_for_file() {
    local path="${1:?wait_for_file requires a path}"
    local wait_seconds="${2:-300}"
    local deadline
    deadline=$((SECONDS + wait_seconds))
    while [[ ! -s "$path" ]]; do
        if (( SECONDS >= deadline )); then
            echo "timed out waiting for $path" >&2
            return 1
        fi
        sleep 1
    done
}

stop_ray() {
    ray stop --force >/dev/null 2>&1 || true
}
trap stop_ray EXIT INT TERM

if [[ "$NODE_ID" == 0 ]]; then
    # Stage and patch SGLang once on the shared filesystem.  A fresh run
    # directory makes the operation deterministic and avoids mutating the image.
    [[ ! -e "$SGLANG_COPY" ]] || {
        echo "refusing to reuse existing staged SGLang tree: $SGLANG_COPY" >&2
        exit 1
    }
    mkdir -p "$SGLANG_COPY"
    cp -a --no-preserve=ownership /sgl-workspace/sglang/python "$SGLANG_COPY/"
    patch --batch --forward --silent -d "$SGLANG_COPY" -p1 \
        < /workspace/TorchSpec/patches/sglang/v0.5.12/transfer_backend.patch
    patch --batch --forward --silent \
        "$SGLANG_COPY/python/sglang/srt/models/qwen3_vl.py" \
        < /workspace/TorchSpec/examples/qwen35-9b-isambard/patches/qwen3_vl-qwen35-aux-capture.patch
    printf 'ready\n' >"$RUN_DIR/sglang.ready"
else
    wait_for_file "$RUN_DIR/sglang.ready" 300
fi
export PYTHONPATH="/workspace/TorchSpec:$SGLANG_COPY/python:${PYTHONPATH:-}"

if [[ "$NODE_ID" == 0 ]]; then
    ray start --head \
        --node-ip-address="$LOCAL_IP" \
        --port="$RAY_GCS_PORT" \
        --ray-client-server-port="$RAY_CLIENT_PORT" \
        --dashboard-port="$RAY_DASHBOARD_PORT" \
        --node-manager-port="$RAY_NODE_MANAGER_PORT" \
        --object-manager-port="$RAY_OBJECT_MANAGER_PORT" \
        --min-worker-port="$RAY_MIN_WORKER_PORT" \
        --max-worker-port="$RAY_MAX_WORKER_PORT" \
        --num-gpus=4 --num-cpus="${TORCHSPEC_RAY_NUM_CPUS:-8}" \
        --temp-dir="$RAY_TEMP" --include-dashboard=false --disable-usage-stats
    printf '%s\n' "$LOCAL_IP" >"$RUN_DIR/ray-head.ip"
    printf 'ready\n' >"$RUN_DIR/ray-head.ready"
    wait_for_file "$RUN_DIR/ray-worker.ready" 300

    export RAY_ADDRESS="$LOCAL_IP:$RAY_GCS_PORT"
    python3 - <<'PY'
import os
import time

import ray

ray.init(address=os.environ["RAY_ADDRESS"])
deadline = time.monotonic() + 180
while int(ray.cluster_resources().get("GPU", 0)) < 8:
    if time.monotonic() >= deadline:
        raise RuntimeError(f"Ray cluster resources did not reach 8 GPUs: {ray.cluster_resources()}")
    time.sleep(1)
print("RAY_EIGHT_GPU_CLUSTER_READY", ray.cluster_resources(), flush=True)
ray.shutdown()
PY

    TRAIN_IP="$(<"$RUN_DIR/node-0.ip")"
    INFERENCE_IP="$(<"$RUN_DIR/node-1.ip")"
    status=0
    python3 -m torchspec.train_entry \
        --config "$E2E_CONFIG_FILE" \
        "dataset.train_data_path=$E2E_DATASET_FILE" \
        training.placement_strategy=custom \
        "training.training_node_ips=[$TRAIN_IP]" \
        training.training_num_nodes=1 \
        training.training_num_gpus_per_node=4 \
        training.fsdp_strategy=FULL_SHARD \
        training.micro_batch_size=1 \
        training.draft_accumulation_steps=1 \
        "training.num_train_steps=$E2E_NUM_STEPS" \
        training.save_per_epoch=false \
        training.save_interval=1000000 \
        inference.inference_num_gpus=4 \
        inference.inference_num_gpus_per_engine=4 \
        inference.inference_num_gpus_per_node=4 \
        inference.sglang.tp_size=4 \
        "inference.inference_node_ips=[$INFERENCE_IP]" \
        transfer.backend=uccl-p2p \
        uccl.transport=cxi \
        uccl.use_gpu_direct=false \
        logging.report_to=tensorboard \
        "output_dir=$RUN_DIR/output" \
        "cache_dir=$RUN_DIR/cache" \
        "model_download_dir=$HF_HOME" \
        2>&1 | tee "$RUN_DIR/train.log" || status=${PIPESTATUS[0]}
    if [[ "$status" == 0 ]]; then
        python3 /workspace/TorchSpec/examples/uccl-isambard/e2e/extract_tensorboard_metrics.py \
            "$RUN_DIR" --expected-steps "$E2E_NUM_STEPS" \
            --output "$RUN_DIR/metrics.json" || status=$?
    fi
    if [[ "$status" == 0 ]]; then
        printf 'complete\n' >"$RUN_DIR/complete.marker"
    fi
    printf '%s\n' "$status" >"$RUN_DIR/driver.status"
    printf 'done\n' >"$RUN_DIR/driver.done"
    exit "$status"
fi

wait_for_file "$RUN_DIR/ray-head.ready" 300
HEAD_IP="$(<"$RUN_DIR/ray-head.ip")"
ray start \
    --address="$HEAD_IP:$RAY_GCS_PORT" \
    --node-ip-address="$LOCAL_IP" \
    --node-manager-port="$RAY_NODE_MANAGER_PORT" \
    --object-manager-port="$RAY_OBJECT_MANAGER_PORT" \
    --min-worker-port="$RAY_MIN_WORKER_PORT" \
    --max-worker-port="$RAY_MAX_WORKER_PORT" \
    --num-gpus=4 --num-cpus="${TORCHSPEC_RAY_NUM_CPUS:-8}" --disable-usage-stats
printf 'ready\n' >"$RUN_DIR/ray-worker.ready"
wait_for_file "$RUN_DIR/driver.done" 14400
exit "$(<"$RUN_DIR/driver.status")"
