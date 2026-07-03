#!/usr/bin/env bash
set -euo pipefail

exec "$(dirname "$0")/run_qwen35_hf_generation.sh" "$@"
