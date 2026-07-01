#!/usr/bin/env python3
"""Run remaining Nemotron prompt exports one at a time with cache cleanup."""

from __future__ import annotations

import argparse
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
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        "NEMOTRON_PROMPT_OUTPUT_ROOT",
        (
            "/workspace/nemotron_prompt_only_exports"
            if Path("/workspace").is_dir()
            else "/tmp/nemotron_prompt_only_exports"
        ),
    )
)
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
OWNER = "jamesdborin"
COLLECTION_TITLE = "Nemotron-Post-Training-v3 Prompt-Only"
SUMMARY_JSON_MARKER = "```json\n"
STALE_ARTIFACT_NAMES = [
    "README.md",
    "summary.md",
    "status.json",
    "status.json.tmp",
    "null_or_empty_rows.md",
    "prompts.csv",
    "prompts.csv.tmp",
    "prompts.csv.partial",
    "summary.csv",
    "summary.md.tmp",
    "summary.csv.tmp",
    "null_or_empty_rows.csv",
    "prompts.jsonl",
    "prompts.jsonl.tmp",
    "prompts.jsonl.partial",
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


def recovery_complete_path() -> Path:
    return OUTPUT_ROOT / "recovery_completed.jsonl"


def load_recovery_completed() -> set[str]:
    path = recovery_complete_path()
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            dataset_id = payload.get("dataset")
            if isinstance(dataset_id, str):
                completed.add(dataset_id)
    return completed


def mark_recovery_complete(dataset_id: str) -> None:
    payload = {"dataset": dataset_id, "completed_at": utc_now()}
    with recovery_complete_path().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def status_phase(dataset_id: str) -> str:
    path = dataset_dir(dataset_id) / "status.json"
    if not path.exists():
        return ""
    try:
        return json.loads(path.read_text(encoding="utf-8") or "{}").get("phase", "")
    except json.JSONDecodeError:
        return "invalid"


def load_manifest_datasets() -> list[str]:
    manifest_path = OUTPUT_ROOT / "dataset_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        return [row["dataset_id"] for row in csv.DictReader(handle)]


def summary_total(dataset_id: str) -> dict[str, str]:
    path = dataset_dir(dataset_id) / "summary.md"
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    start = text.find(SUMMARY_JSON_MARKER)
    if start < 0:
        return {}
    start += len(SUMMARY_JSON_MARKER)
    end = text.find("\n```", start)
    if end < 0:
        return {}
    try:
        rows = json.loads(text[start:end])
    except json.JSONDecodeError:
        return {}
    for row in rows:
        if isinstance(row, dict) and row.get("config") == "__total__":
            return row
    return {}


def has_reusable_export(dataset_id: str) -> bool:
    row = summary_total(dataset_id)
    if not row:
        return False
    return (
        (dataset_dir(dataset_id) / "prompts.csv").exists()
        and row.get("status") == "ok"
        and row.get("row_count_delta") in {"", "0"}
        and row.get("failed_prompt_rows") in {"", "0"}
    )


def clear_for_rerun(dataset_id: str) -> None:
    root = dataset_dir(dataset_id)
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        return
    removed = 0
    for name in STALE_ARTIFACT_NAMES:
        path = root / name
        if path.exists():
            path.unlink()
            removed += 1
    failed_runs = root / "failed_runs"
    if failed_runs.exists():
        shutil.rmtree(failed_runs)
        removed += 1
    log(f"{dataset_id}: removed {removed} stale local artifact(s)")


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
        "--cleanup-local-artifacts",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Nemotron prompt exports one at a time with cache cleanup."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--owner", default=OWNER)
    parser.add_argument("--collection-title", default=COLLECTION_TITLE)
    parser.add_argument(
        "--dataset",
        action="append",
        help="Run only this dataset; may be passed more than once. Defaults to incomplete manifest entries.",
    )
    return parser.parse_args()


def main() -> int:
    global OUTPUT_ROOT, OWNER, COLLECTION_TITLE
    args = parse_args()
    OUTPUT_ROOT = args.output_root.expanduser().resolve()
    OWNER = args.owner
    COLLECTION_TITLE = args.collection_title
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / ".hf_cache").mkdir(parents=True, exist_ok=True)
    completed = load_recovery_completed()
    datasets = args.dataset or []
    if not datasets:
        datasets = [
            dataset_id
            for dataset_id in load_manifest_datasets()
            if dataset_id not in completed
        ]
    log(f"single recovery worker starting with {len(datasets)} dataset(s)")
    for dataset_id in datasets:
        log(f"{dataset_id}: preparing")
        clean_cache(dataset_id)
        reusable = has_reusable_export(dataset_id)
        if not reusable:
            clear_for_rerun(dataset_id)
        rc = run_worker(dataset_id, force=not reusable)
        clean_cache(dataset_id)
        log(f"{dataset_id}: finished rc={rc}; cache cleaned")
        if rc != 0:
            log(f"{dataset_id}: stopping queue after failure")
            return rc
        mark_recovery_complete(dataset_id)
    log("single recovery worker complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
