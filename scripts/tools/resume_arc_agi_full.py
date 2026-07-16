#!/usr/bin/env python3
"""Resume the ARC-AGI Nemotron Doubleword batch run end to end."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


REPO_ID = "jamesdborin/Nemotron-SFT-ARC-AGI-v1-prompt-only"
EXPECTED_COUNT = 252_069
ROOT = Path("data/nemotron-prompt-only/Nemotron-SFT-ARC-AGI-v1-prompt-only")
RESULTS_NAME = "gpt-oss-20b-temperature-1-reasoning_effort-low-max_token-2048.jsonl"
BATCH_IDS = ROOT / f"{RESULTS_NAME}.batch_ids"
CHUNK_DIR = ROOT / "dw_batch_requests_gpt-oss-20b_chunks_10000"
KNOWN_008 = "9657e304-3083-4908-ac07-72424f19b390"
KNOWN_008_BYTES = 170_578_006


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S %z')} {message}", flush=True)


def run(cmd: list[str]) -> None:
    log("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def capture_json(cmd: list[str]) -> dict[str, object]:
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return json.loads(proc.stdout)


def ids() -> list[str]:
    return [line.strip() for line in BATCH_IDS.read_text(encoding="utf-8").splitlines() if line.strip()]


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def verify_local_state() -> None:
    if count_lines(ROOT / "dw_batch_requests_gpt-oss-20b.jsonl") != EXPECTED_COUNT:
        raise RuntimeError("prepared request count mismatch")
    current = ids()
    if len(current) != len(set(current)):
        raise RuntimeError("duplicate batch ids in ledger")
    if len(current) < 7:
        raise RuntimeError(f"expected at least seven seed ids, found {len(current)}")
    chunks = sorted(CHUNK_DIR.glob("*.jsonl"))
    if len(chunks) != 26:
        raise RuntimeError(f"expected 26 chunks, found {len(chunks)}")
    meta = capture_json(["dw", "files", "get", KNOWN_008, "--output", "json"])
    if meta.get("filename") != "dw_batch_requests_gpt-oss-20b-008.jsonl":
        raise RuntimeError(f"known chunk 008 filename mismatch: {meta}")
    if int(meta.get("bytes") or 0) != KNOWN_008_BYTES:
        raise RuntimeError(f"known chunk 008 byte mismatch: {meta}")
    if Path(CHUNK_DIR / "dw_batch_requests_gpt-oss-20b-008.jsonl").stat().st_size != KNOWN_008_BYTES:
        raise RuntimeError("local chunk 008 byte mismatch")


def create_remaining() -> None:
    run(
        [
            "python3",
            "scripts/tools/resume_dw_chunk_batches.py",
            "--chunk-dir",
            str(CHUNK_DIR),
            "--start",
            "1",
            "--end",
            "26",
            "--output-id",
            str(BATCH_IDS),
            "--known-file-id",
            f"008={KNOWN_008}",
            "--success-sleep",
            "60",
            "--create-rate-limit-sleep",
            "300",
        ]
    )
    current = ids()
    if len(current) != 26 or len(current) != len(set(current)):
        raise RuntimeError(f"batch ledger is not exactly 26 unique ids: {len(current)} total")


def download_error_sample(batch_id: str, error_file_id: str) -> None:
    error_dir = ROOT / "batch_errors"
    error_dir.mkdir(exist_ok=True)
    error_path = error_dir / f"{batch_id}.errors.jsonl"
    run(["dw", "files", "content", error_file_id, "--output-file", str(error_path)])
    log(f"error_file_id={error_file_id} downloaded_to={error_path}")
    with error_path.open("r", encoding="utf-8") as handle:
        for index, line in zip(range(5), handle):
            row = json.loads(line)
            log(f"failed sample {index + 1}: custom_id={row.get('custom_id')} schema={json.dumps(row)[:1000]}")


def wait_clean_batches() -> None:
    while True:
        current = ids()
        if len(current) != 26:
            raise RuntimeError(f"expected 26 ids before watching, found {len(current)}")
        completed = 0
        total_failed = 0
        for batch_id in current:
            batch = capture_json(["dw", "batches", "get", batch_id, "--output", "json"])
            counts = batch.get("request_counts") or {}
            failed = int(counts.get("failed") or 0)
            status = str(batch.get("status"))
            total_failed += failed
            if failed or status in {"failed", "cancelled", "expired"}:
                log(f"stopping on batch_id={batch_id} status={status} request_counts={counts}")
                error_file_id = str(batch.get("error_file_id") or "")
                if error_file_id:
                    download_error_sample(batch_id, error_file_id)
                raise RuntimeError("request-level or batch-level failure detected")
            if status == "completed":
                completed += 1
        log(f"batch watch: completed={completed}/26 failed={total_failed}")
        if completed == 26:
            return
        time.sleep(300)


def finalize() -> None:
    run(
        [
            "python3",
            "scripts/tools/run_nemotron_dw_repo.py",
            "--repo-id",
            REPO_ID,
            "--expected-count",
            str(EXPECTED_COUNT),
            "--chunk-size",
            "10000",
            "--reuse-prepared",
            "--skip-generic-upload",
        ]
    )


def main() -> None:
    verify_local_state()
    create_remaining()
    wait_clean_batches()
    finalize()


if __name__ == "__main__":
    main()
