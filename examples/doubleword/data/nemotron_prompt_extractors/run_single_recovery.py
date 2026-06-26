#!/usr/bin/env python3
"""Run remaining Nemotron prompt exports one at a time with cache cleanup."""

from __future__ import annotations

import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
WORKER = Path(__file__).with_name("export_prompt_only_dataset.py")
OUTPUT_ROOT = Path("/tmp/nemotron_prompt_only_exports")
OWNER = "jamesdborin"
COLLECTION_TITLE = "Nemotron-Post-Training-v3 Prompt-Only"

DATASETS = [
    "nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2",
    "nvidia/Nemotron-RL-Instruction-Following-Citation-Formatting-v1",
    "nvidia/Nemotron-RL-litmus-bench-v0.1",
    "nvidia/Nemotron-RL-Super-Training-Blends",
    "nvidia/Nemotron-SFT-OpenCode-v1",
    "nvidia/Nemotron-Math-Proofs-v1",
    "nvidia/Nemotron-Agentic-v1",
    "nvidia/Nemotron-Competitive-Programming-v1",
    "nvidia/Nemotron-Math-v2",
    "nvidia/Nemotron-SWE-v1",
    "nvidia/Nemotron-SFT-SWE-v2",
    "nvidia/Nemotron-SFT-Instruction-Following-Chat-v2",
    "nvidia/Nemotron-RLHF-GenRM-v1",
    "nvidia/Nemotron-RL-ReasoningGym-v1",
    "nvidia/Nemotron-SFT-Multilingual-v1",
    "nvidia/Nemotron-RL-Safety-v1",
    "nvidia/Nemotron-RL-Identity-Following-v1",
    "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1",
    "nvidia/Nemotron-RL-Instruction-Following-Adversarial-v1",
    "nvidia/Nemotron-SFT-CUDA-v1",
    "nvidia/Nemotron-SFT-Instruction-Following-Chat-v3",
    "nvidia/Nemotron-SFT-Science-v2",
    "nvidia/Nemotron-RL-QA-Abstention-v1",
    "nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1",
    "nvidia/Nemotron-RL-ARC-AGI-v1",
    "nvidia/Nemotron-RL-InverseIFEval-v1",
    "nvidia/Nemotron-SFT-Math-v3",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def local_dataset_name(dataset_id: str) -> str:
    return dataset_id.replace("/", "__")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def dataset_dir(dataset_id: str) -> Path:
    return OUTPUT_ROOT / local_dataset_name(dataset_id)


def cache_dir(dataset_id: str) -> Path:
    return OUTPUT_ROOT / ".hf_cache" / local_dataset_name(dataset_id)


def summary_total(dataset_id: str) -> dict[str, str]:
    path = dataset_dir(dataset_id) / "summary.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("config") == "__total__":
                return row
    return {}


def has_reusable_export(dataset_id: str) -> bool:
    row = summary_total(dataset_id)
    if not row:
        return False
    return (
        (dataset_dir(dataset_id) / "prompts.jsonl").exists()
        and row.get("status") == "ok"
        and row.get("row_count_delta") in {"", "0"}
        and row.get("failed_prompt_rows") in {"", "0"}
    )


def backup_and_clear_for_rerun(dataset_id: str) -> None:
    root = dataset_dir(dataset_id)
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = root / "failed_runs" / stamp
    backup.mkdir(parents=True, exist_ok=True)
    for name in [
        "README.md",
        "summary.csv",
        "status.json",
        "null_or_empty_rows.csv",
        "prompts.jsonl",
        "prompts.jsonl.tmp",
        "prompts.jsonl.partial",
    ]:
        path = root / name
        if path.exists():
            shutil.move(str(path), backup / name)
    log(f"{dataset_id}: moved stale local artifacts to {backup}")


def clean_cache(dataset_id: str) -> None:
    shutil.rmtree(cache_dir(dataset_id), ignore_errors=True)


def run_worker(dataset_id: str, force: bool) -> int:
    command = [
        "uv",
        "run",
        "--no-project",
        "--isolated",
        "--with",
        "datasets==5.0.0",
        "--with",
        "huggingface_hub>=0.33.0",
        "--with",
        "requests",
        "python",
        str(WORKER),
        "--dataset",
        dataset_id,
        "--output-root",
        str(OUTPUT_ROOT),
        "--owner",
        OWNER,
        "--collection-title",
        COLLECTION_TITLE,
        "--semaphore-dir",
        str(OUTPUT_ROOT / "semaphore" / "extract-single"),
        "--max-concurrent",
        "1",
        "--upload-semaphore-dir",
        str(OUTPUT_ROOT / "semaphore" / "upload-single"),
        "--max-upload-concurrent",
        "1",
        "--wait-for-auth",
    ]
    if force:
        command.append("--force")

    log_path = OUTPUT_ROOT / "logs" / f"{local_dataset_name(dataset_id)}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"{dataset_id}: running {' '.join(shlex.quote(part) for part in command)}")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now()}] single-worker recovery start force={force}\n")
        handle.flush()
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        ).returncode


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / ".hf_cache").mkdir(parents=True, exist_ok=True)
    log(f"single recovery worker starting with {len(DATASETS)} dataset(s)")
    for dataset_id in DATASETS:
        log(f"{dataset_id}: preparing")
        clean_cache(dataset_id)
        reusable = has_reusable_export(dataset_id)
        if not reusable:
            backup_and_clear_for_rerun(dataset_id)
        rc = run_worker(dataset_id, force=not reusable)
        clean_cache(dataset_id)
        log(f"{dataset_id}: finished rc={rc}; cache cleaned")
        if rc != 0:
            log(f"{dataset_id}: stopping queue after failure")
            return rc
    log("single recovery worker complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
