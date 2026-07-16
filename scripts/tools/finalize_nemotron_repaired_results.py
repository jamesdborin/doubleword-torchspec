#!/usr/bin/env python3
"""Merge repaired Doubleword results for a Nemotron repo and upload artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from run_nemotron_dw_repo import (
    MODEL,
    MODEL_SLUG,
    RESULTS_NAME,
    convert_torchspec,
    count_lines,
    result_is_bad,
    upload,
    verify_if_practical,
)


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_results(paths: list[Path]) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                row = json.loads(line)
                custom_id = row.get("custom_id")
                if not isinstance(custom_id, str) or not custom_id:
                    raise RuntimeError(f"{path}:{line_no}: missing custom_id")
                if result_is_bad(row):
                    raise RuntimeError(f"{path}:{line_no}: malformed result for {custom_id}")
                rows[custom_id] = row
    return rows


def ordered_request_ids(prepared: Path) -> list[str]:
    ids: list[str] = []
    with prepared.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            custom_id = row.get("custom_id")
            if not isinstance(custom_id, str) or not custom_id:
                raise RuntimeError(f"{prepared}:{line_no}: missing custom_id")
            ids.append(custom_id)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--original-results", required=True, type=Path)
    parser.add_argument("--repair-results", required=True, type=Path, action="append")
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--verify-max-bytes", type=int, default=250_000_000)
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    request_ids = ordered_request_ids(args.prepared)
    if len(request_ids) != args.expected_count:
        raise RuntimeError(f"request count {len(request_ids)} != expected {args.expected_count}")

    rows = load_results([args.original_results, *args.repair_results])
    missing = [custom_id for custom_id in request_ids if custom_id not in rows]
    if missing:
        raise RuntimeError(f"missing {len(missing)} result rows; first missing custom_id={missing[0]}")

    final_results = args.local_dir / RESULTS_NAME
    tmp = final_results.with_suffix(final_results.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        for custom_id in request_ids:
            out.write(json.dumps(rows[custom_id], ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(final_results)
    if count_lines(final_results) != args.expected_count:
        raise RuntimeError(f"final results count {count_lines(final_results)} != expected {args.expected_count}")

    torchspec = args.local_dir / "data" / f"{MODEL_SLUG}.jsonl"
    if torchspec.exists():
        backup = torchspec.with_suffix(torchspec.suffix + ".bak")
        shutil.copy2(torchspec, backup)
    convert_torchspec(args.local_dir, args.prepared, final_results, torchspec)
    if count_lines(torchspec) != args.expected_count:
        raise RuntimeError(f"torchspec count {count_lines(torchspec)} != expected {args.expected_count}")

    verify: dict[str, str] = {}
    if not args.skip_upload:
        upload(args.repo_id, final_results, RESULTS_NAME, f"Add {MODEL} batch results")
        verify[RESULTS_NAME] = verify_if_practical(args.repo_id, final_results, RESULTS_NAME, args.verify_max_bytes)
        upload(args.repo_id, torchspec, f"data/{MODEL_SLUG}.jsonl", f"Add TorchSpec split for {MODEL}")
        verify[f"data/{MODEL_SLUG}.jsonl"] = verify_if_practical(
            args.repo_id, torchspec, f"data/{MODEL_SLUG}.jsonl", args.verify_max_bytes
        )

    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "results_count": count_lines(final_results),
                "torchspec_count": count_lines(torchspec),
                "verify": verify,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
