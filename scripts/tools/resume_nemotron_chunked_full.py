#!/usr/bin/env python3
"""Resume a chunked Nemotron Doubleword run, then finalize it end to end."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--chunk-size", required=True, type=int)
    parser.add_argument("--output-id", required=True, type=Path)
    parser.add_argument("--chunk-dir", required=True, type=Path)
    parser.add_argument("--start", default=1, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--known-file-id", action="append", default=[])
    parser.add_argument("--success-sleep", default=60, type=int)
    parser.add_argument("--create-rate-limit-sleep", default=300, type=int)
    args = parser.parse_args()

    resume = [
        "python",
        "scripts/tools/resume_dw_chunk_batches.py",
        "--chunk-dir",
        str(args.chunk_dir),
        "--start",
        str(args.start),
        "--end",
        str(args.end),
        "--output-id",
        str(args.output_id),
        "--success-sleep",
        str(args.success_sleep),
        "--create-rate-limit-sleep",
        str(args.create_rate_limit_sleep),
    ]
    for known in args.known_file_id:
        resume.extend(["--known-file-id", known])
    run(resume)

    ids = count_lines(args.output_id)
    if ids != args.end:
        raise RuntimeError(f"expected {args.end} batch ids, found {ids}")

    run(
        [
            "python",
            "scripts/tools/run_nemotron_dw_repo.py",
            "--repo-id",
            args.repo_id,
            "--expected-count",
            str(args.expected_count),
            "--chunk-size",
            str(args.chunk_size),
            "--skip-generic-upload",
            "--reuse-prepared",
        ]
    )


if __name__ == "__main__":
    main()
