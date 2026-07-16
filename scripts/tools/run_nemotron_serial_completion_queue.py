#!/usr/bin/env python3
"""Run the current rate-limited Nemotron repos sequentially to completion."""

from __future__ import annotations

import subprocess
import time


def run(name: str, cmd: list[str]) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S %z')} START {name}", flush=True)
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S %z')} DONE {name}", flush=True)


def main() -> None:
    run(
        "rlhf-genrm",
        [
            "python",
            "scripts/tools/resume_nemotron_chunked_full.py",
            "--repo-id",
            "jamesdborin/Nemotron-RLHF-GenRM-v1-prompt-only",
            "--expected-count",
            "299517",
            "--chunk-size",
            "10000",
            "--chunk-dir",
            "data/nemotron-prompt-only/Nemotron-RLHF-GenRM-v1-prompt-only/dw_batch_requests_gpt-oss-20b_chunks_10000",
            "--start",
            "1",
            "--end",
            "30",
            "--output-id",
            "data/nemotron-prompt-only/Nemotron-RLHF-GenRM-v1-prompt-only/gpt-oss-20b-temperature-1-reasoning_effort-low-max_token-2048.jsonl.batch_ids",
            "--known-file-id",
            "020=77693af8-dc53-40f3-b232-ed415ac6393c",
            "--success-sleep",
            "60",
            "--create-rate-limit-sleep",
            "300",
        ],
    )
    run("arc-agi", ["python3", "scripts/tools/resume_arc_agi_full.py"])
    run(
        "competitive-programming-v2",
        [
            "python",
            "scripts/tools/resume_nemotron_chunked_full.py",
            "--repo-id",
            "jamesdborin/Nemotron-SFT-Competitive-Programming-v2-prompt-only",
            "--expected-count",
            "841555",
            "--chunk-size",
            "10000",
            "--chunk-dir",
            "data/nemotron-prompt-only/Nemotron-SFT-Competitive-Programming-v2-prompt-only/dw_batch_requests_gpt-oss-20b_chunks_10000",
            "--start",
            "1",
            "--end",
            "85",
            "--output-id",
            "data/nemotron-prompt-only/Nemotron-SFT-Competitive-Programming-v2-prompt-only/gpt-oss-20b-temperature-1-reasoning_effort-low-max_token-2048.jsonl.batch_ids",
            "--known-file-id",
            "014=c9594e0a-cd18-49db-814c-bbd6e7a36c10",
            "--success-sleep",
            "60",
            "--create-rate-limit-sleep",
            "300",
        ],
    )


if __name__ == "__main__":
    main()
