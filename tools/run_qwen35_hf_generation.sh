#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-local/torchspec:gpt-oss-20b-sglang}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
PORT="${PORT:-30000}"
WORKSPACE="${WORKSPACE:-$(pwd)}"
WORK_DIR="${WORK_DIR:-/workspace/outputs/qwen-3.5-9B-hf-generation}"
HUB_OUTPUT_PATH="${HUB_OUTPUT_PATH:-data/qwen-3.5-9B}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
CONCURRENCY="${CONCURRENCY:-1024}"
SHARD_SIZE="${SHARD_SIZE:-1024}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.82}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1024}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-1048576}"
DATASETS="${DATASETS:-jamesdborin/Magpie-Llama-3.1-Pro-300K-Filtered-prompt-only jamesdborin/UltraChat-200K-prompt-only}"

if [[ -z "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ]]; then
  echo "HF_TOKEN or HUGGING_FACE_HUB_TOKEN must be set for Hugging Face uploads." >&2
  exit 1
fi

docker run --rm --gpus all --ipc=host --network host \
  -e HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" \
  -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}" \
  -e HF_HOME="${HF_HOME:-/root/.cache/huggingface}" \
  -v "${WORKSPACE}:/workspace" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace \
  --entrypoint bash \
  "${IMAGE}" \
  -lc "
set -euo pipefail
mkdir -p '${WORK_DIR}'
sglang serve \
  --model-path '${MODEL}' \
  --served-model-name '${MODEL}' \
  --host 0.0.0.0 \
  --port '${PORT}' \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --context-length '${CONTEXT_LENGTH}' \
  --mem-fraction-static '${MEM_FRACTION_STATIC}' \
  --max-running-requests '${MAX_RUNNING_REQUESTS}' \
  --max-prefill-tokens '${MAX_PREFILL_TOKENS}' \
  > '${WORK_DIR}/sglang.log' 2>&1 &
server_pid=\$!
trap 'kill \$server_pid 2>/dev/null || true' EXIT

python3 - <<'PY'
import time
import requests

url = 'http://127.0.0.1:${PORT}/health'
for attempt in range(360):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print('SGLang server is healthy')
            break
    except Exception:
        pass
    time.sleep(5)
else:
    raise SystemExit('SGLang server did not become healthy')
PY

python3 tools/generate_hf_sglang_dataset.py \
  --model '${MODEL}' \
  --server-address 'http://127.0.0.1:${PORT}/v1' \
  --hub-output-path '${HUB_OUTPUT_PATH}' \
  --work-dir '${WORK_DIR}' \
  --progress-file '${WORK_DIR}/progress.json' \
  --batch-size '${BATCH_SIZE}' \
  --concurrency '${CONCURRENCY}' \
  --shard-size '${SHARD_SIZE}' \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 1.5 \
  --repetition-penalty 1.0 \
  --max-new-tokens 8192 \
  --datasets ${DATASETS}
"
