#!/usr/bin/env bash

set -euo pipefail

: "${E2E_RUN_DIR:?}"
: "${E2E_CONFIG_FILE:?}"
: "${E2E_DATASET_FILE:?}"

SGLANG_COPY="$E2E_RUN_DIR/sglang"
export TMPDIR="/tmp/ts-tmp-${SLURM_JOB_ID}"
export RAY_TMPDIR="/tmp/ray-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$RAY_TMPDIR"
export HOME="/tmp/torchspec-home-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$HOME/.cache"
export FLASHINFER_WORKSPACE_DIR="$XDG_CACHE_HOME/flashinfer"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
mkdir -p "$FLASHINFER_WORKSPACE_DIR" "$TRITON_CACHE_DIR"
[[ ! -e "$SGLANG_COPY" ]] || { echo "staged SGLang already exists: $SGLANG_COPY" >&2; exit 1; }
mkdir -p "$SGLANG_COPY"
cp -a --no-preserve=ownership /sgl-workspace/sglang/python "$SGLANG_COPY/"
patch --batch --forward --silent -d "$SGLANG_COPY" -p1 \
    < /workspace/TorchSpec/patches/sglang/v0.5.12/transfer_backend.patch
patch --batch --forward --silent \
    "$SGLANG_COPY/python/sglang/srt/models/qwen3_vl.py" \
    < /workspace/TorchSpec/examples/qwen35-9b-isambard/patches/qwen3_vl-qwen35-aux-capture.patch
echo "SGLANG_E2E_PATCHES_APPLIED path=$SGLANG_COPY"

export PYTHONPATH="/workspace/TorchSpec:$SGLANG_COPY/python:${PYTHONPATH:-}"

# Starting Ray implicitly via ray.init(address="local") can hang after GCS
# startup under rootless Podman.  Start it explicitly and keep its diagnostics
# on the shared run directory so failures are observable from the login node.
LOCAL_IP="${E2E_LOCAL_IP:?E2E_LOCAL_IP must be resolved on the host}"
[[ -n "$LOCAL_IP" ]] || { echo "could not resolve hsn0 address" >&2; exit 1; }
RAY_GCS_PORT="$((24000 + SLURM_JOB_ID % 1000))"
RAY_NODE_MANAGER_PORT="$((RAY_GCS_PORT + 1))"
RAY_OBJECT_MANAGER_PORT="$((RAY_GCS_PORT + 2))"
RAY_MIN_WORKER_PORT="$((RAY_GCS_PORT + 100))"
RAY_MAX_WORKER_PORT="$((RAY_GCS_PORT + 1099))"
RAY_LOCAL_TEMP="${E2E_RAY_LOCAL_TEMP:-/tmp/ray-live}"
mkdir -p "$RAY_LOCAL_TEMP" "$E2E_RUN_DIR/ray"
# Ray eagerly pre-starts roughly one Python worker per advertised CPU.  A
# 64-CPU advertisement created 66 idle workers before SGLang launched and the
# rootless container was SIGKILLed as the engine actor spawned.  The smoke test
# only needs a few controller/actor slots, so keep the Ray process tree small
# while retaining all Slurm CPUs for model loading and native kernels.
ray start --head \
    --node-ip-address="$LOCAL_IP" \
    --port="$RAY_GCS_PORT" \
    --node-manager-port="$RAY_NODE_MANAGER_PORT" \
    --object-manager-port="$RAY_OBJECT_MANAGER_PORT" \
    --min-worker-port="$RAY_MIN_WORKER_PORT" \
    --max-worker-port="$RAY_MAX_WORKER_PORT" \
    --num-gpus=2 --num-cpus="${TORCHSPEC_RAY_NUM_CPUS:-8}" \
    --temp-dir="$RAY_LOCAL_TEMP" --include-dashboard=false --disable-usage-stats
cleanup_ray() {
    cp -a --no-preserve=ownership "$RAY_LOCAL_TEMP/." "$E2E_RUN_DIR/ray/" \
        >/dev/null 2>&1 || true
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup_ray EXIT INT TERM
export RAY_ADDRESS="$LOCAL_IP:$RAY_GCS_PORT"
ray status --address="$RAY_ADDRESS"
echo "RAY_EXPLICIT_HEAD_READY address=$RAY_ADDRESS"

cd /workspace/TorchSpec
python3 -m torchspec.train_entry \
    --config "$E2E_CONFIG_FILE" \
    "dataset.train_data_path=$E2E_DATASET_FILE" \
    training.placement_strategy=inference_first \
    training.training_num_nodes=1 \
    training.training_num_gpus_per_node=1 \
    training.fsdp_strategy=REPLICATE \
    training.micro_batch_size=1 \
    training.draft_accumulation_steps=1 \
    training.num_train_steps=1 \
    training.save_interval=0 \
    training.save_per_epoch=false \
    inference.inference_num_gpus=1 \
    inference.inference_num_gpus_per_engine=1 \
    inference.inference_num_gpus_per_node=1 \
    inference.inference_batch_size=1 \
    inference.sglang.tp_size=1 \
    inference.sglang.mem_fraction_static=0.3 \
    inference.sglang.extra_args.disable_cuda_graph=true \
    transfer.backend=uccl-p2p \
    uccl.transport=cxi \
    uccl.use_gpu_direct=false \
    logging.report_to=tensorboard \
    "output_dir=$E2E_RUN_DIR/output" \
    "cache_dir=$E2E_RUN_DIR/cache" \
    "model_download_dir=$HF_HOME" \
    2>&1 | tee "$E2E_RUN_DIR/train.log"

python3 /workspace/TorchSpec/examples/uccl-isambard/e2e/extract_tensorboard_metrics.py \
    "$E2E_RUN_DIR" --expected-steps 1 --output "$E2E_RUN_DIR/metrics.json"
printf 'complete\n' >"$E2E_RUN_DIR/complete.marker"
echo "UCCL_SINGLE_NODE_ONE_STEP_PASSED run_dir=$E2E_RUN_DIR"
