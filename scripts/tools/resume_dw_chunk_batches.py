#!/usr/bin/env python3
"""Submit batch chunk files one at a time with rate-limit retries."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


RATE_LIMIT_RE = re.compile(r"Retry after (\d+)s")


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S %z')} {message}", flush=True)


def run_json(cmd: list[str], *, min_rate_limit_wait: int = 60, transient_wait: int = 60) -> dict[str, object]:
    transient_attempts = 0
    while True:
        proc = subprocess.run(cmd, text=True, capture_output=True)
        if proc.returncode == 0:
            return json.loads(proc.stdout)
        combined = f"{proc.stdout}\n{proc.stderr}"
        if "Rate limited" in combined:
            match = RATE_LIMIT_RE.search(combined)
            wait = int(match.group(1)) if match else min_rate_limit_wait
            wait = max(wait, min_rate_limit_wait)
            log(f"rate limited; sleeping {wait}s before retrying")
            time.sleep(wait)
            continue
        if "500 Internal Server Error" in combined or "Internal server error" in combined:
            transient_attempts += 1
            wait = min(transient_wait * transient_attempts, 300)
            log(f"server error from {' '.join(cmd[:3])}; sleeping {wait}s before retrying")
            time.sleep(wait)
            continue
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)


def active_batch_count() -> int:
    proc = subprocess.run(
        ["dw", "batches", "list", "--active-first", "--limit", "100"],
        text=True,
        capture_output=True,
        check=True,
    )
    active_statuses = {"validating", "in_progress", "finalizing", "cancelling"}
    count = 0
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        batch = json.loads(line)
        if batch.get("status") in active_statuses:
            count += 1
    return count


def wait_for_capacity(limit: int, sleep_seconds: int) -> None:
    if limit <= 0:
        return
    while True:
        count = active_batch_count()
        if count < limit:
            return
        log(f"{count} active batches >= cap {limit}; sleeping {sleep_seconds}s before create")
        time.sleep(sleep_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-dir", required=True, type=Path)
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--output-id", required=True, type=Path)
    parser.add_argument("--known-file-id", action="append", default=[], help="Map a chunk to an existing upload as CHUNK=FILE_ID, e.g. 006=file-...")
    parser.add_argument("--success-sleep", type=int, default=0)
    parser.add_argument("--create-rate-limit-sleep", type=int, default=60)
    parser.add_argument("--active-limit", type=int, default=0, help="Wait before creating if active batches are at or above this count; 0 disables")
    parser.add_argument("--active-sleep", type=int, default=300)
    args = parser.parse_args()

    known_file_ids: dict[str, str] = {}
    for item in args.known_file_id:
        chunk_num, sep, file_id = item.partition("=")
        if sep != "=" or not chunk_num or not file_id:
            raise ValueError(f"invalid --known-file-id value: {item!r}")
        known_file_ids[chunk_num.zfill(3)] = file_id

    args.output_id.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if args.output_id.exists():
        existing = [line.strip() for line in args.output_id.read_text(encoding="utf-8").splitlines() if line.strip()]

    chunks = sorted(args.chunk_dir.glob("*.jsonl"))
    selected = chunks[args.start - 1 : args.end]
    if len(selected) != args.end - args.start + 1:
        raise RuntimeError(f"expected {args.end - args.start + 1} chunks, found {len(selected)}")

    submitted_in_selected = max(0, len(existing) - (args.start - 1))
    remaining = selected[submitted_in_selected:]
    log(
        f"{len(existing)} existing ids; {submitted_in_selected} already cover selected range; "
        f"submitting {len(remaining)} chunks"
    )
    with args.output_id.open("a", encoding="utf-8") as handle:
        for chunk in remaining:
            chunk_match = re.search(r"-(\d+)\.jsonl$", chunk.name)
            chunk_num = chunk_match.group(1) if chunk_match else ""
            file_id = known_file_ids.get(chunk_num)
            if file_id is None:
                upload = run_json(["dw", "files", "upload", str(chunk), "--output", "json"])
                file_id = str(upload["id"])
                log(f"{chunk.name}: uploaded file={file_id}")
            wait_for_capacity(args.active_limit, args.active_sleep)
            batch = run_json(
                ["dw", "batches", "create", "--file", file_id, "--output", "json"],
                min_rate_limit_wait=args.create_rate_limit_sleep,
            )
            batch_id = str(batch["id"])
            handle.write(batch_id + "\n")
            handle.flush()
            log(f"{chunk.name}: file={file_id} batch={batch_id}")
            if args.success_sleep > 0:
                time.sleep(args.success_sleep)


if __name__ == "__main__":
    main()
