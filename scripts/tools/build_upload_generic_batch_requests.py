#!/usr/bin/env python3
"""Build and upload generic Doubleword batch request JSONL files for HF repos."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


DEFAULT_CONVERTER = Path.home() / ".codex/skills/hf-dw-batch-inference/scripts/csv_to_batch_jsonl.py"


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S %z')} {message}", flush=True)


def safe_name(repo_id: str) -> str:
    return repo_id.rsplit("/", 1)[-1]


def run(cmd: list[str]) -> None:
    log("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_converter(converter: Path, prompt_csv: Path, generic_jsonl: Path) -> None:
    log(f"converting {prompt_csv} -> {generic_jsonl}")
    csv.field_size_limit(sys.maxsize)
    spec = importlib.util.spec_from_file_location("csv_to_batch_jsonl", converter)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load converter: {converter}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    generic_jsonl.parent.mkdir(parents=True, exist_ok=True)
    used_ids: set[str] = set()
    count = 0
    with prompt_csv.open(newline="", encoding="utf-8") as src, generic_jsonl.open("w", encoding="utf-8") as dst:
        rows = (line.replace("\0", "") for line in src)
        reader = csv.DictReader(rows)
        if not reader.fieldnames:
            raise RuntimeError("CSV has no header")
        if "prompt" not in reader.fieldnames:
            raise RuntimeError("CSV is missing prompt column: prompt")
        for row in reader:
            request = module.row_to_request(
                row=row,
                generated_id=module.short_id(used_ids, 12),
                prompt_column="prompt",
                model_placeholder="[MODEL]",
            )
            dst.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    log(f"Wrote {count} requests to {generic_jsonl}")


def line_count(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for count, _ in enumerate(handle, 1):
            pass
    return count


def remote_has_generic(api: HfApi, repo_id: str) -> bool:
    return "dw_batch_requests.jsonl" in api.list_repo_files(repo_id, repo_type="dataset")


def process_repo(
    api: HfApi,
    repo_id: str,
    output_root: Path,
    converter: Path,
    cleanup_new: bool,
    skip_remote_existing: bool,
    force_convert: bool,
) -> dict[str, object]:
    if skip_remote_existing and remote_has_generic(api, repo_id):
        log(f"{repo_id}: remote dw_batch_requests.jsonl already exists; skipping")
        return {"repo_id": repo_id, "status": "skipped_remote_exists"}

    repo_dir = output_root / safe_name(repo_id)
    existed_before = repo_dir.exists()
    repo_dir.mkdir(parents=True, exist_ok=True)
    prompt_csv = repo_dir / "prompts.csv"
    generic_jsonl = repo_dir / "dw_batch_requests.jsonl"

    if not prompt_csv.exists():
        log(f"{repo_id}: downloading prompts.csv")
        hf_hub_download(
            repo_id=repo_id,
            filename="prompts.csv",
            repo_type="dataset",
            local_dir=repo_dir,
        )
    else:
        log(f"{repo_id}: reusing local prompts.csv ({prompt_csv.stat().st_size} bytes)")

    if force_convert and generic_jsonl.exists():
        log(f"{repo_id}: removing existing dw_batch_requests.jsonl before forced conversion")
        generic_jsonl.unlink()

    if not generic_jsonl.exists():
        run_converter(converter, prompt_csv, generic_jsonl)
    else:
        log(f"{repo_id}: reusing local dw_batch_requests.jsonl ({generic_jsonl.stat().st_size} bytes)")

    run(["dw", "files", "validate", str(generic_jsonl)])
    requests = line_count(generic_jsonl)

    run(
        [
            "huggingface-cli",
            "upload",
            repo_id,
            str(generic_jsonl),
            "dw_batch_requests.jsonl",
            "--repo-type",
            "dataset",
            "--commit-message",
            "Add generic Doubleword batch requests",
        ]
    )

    if not remote_has_generic(api, repo_id):
        raise RuntimeError(f"{repo_id}: upload completed but remote dw_batch_requests.jsonl is not visible")

    result = {
        "repo_id": repo_id,
        "status": "uploaded",
        "requests": requests,
        "local_jsonl": str(generic_jsonl),
        "bytes": generic_jsonl.stat().st_size,
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)

    if cleanup_new and not existed_before:
        log(f"{repo_id}: removing newly-created local staging directory {repo_dir}")
        shutil.rmtree(repo_dir)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", action="append", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/nemotron-prompt-only"))
    parser.add_argument("--converter", type=Path, default=DEFAULT_CONVERTER)
    parser.add_argument("--cleanup-new", action="store_true")
    parser.add_argument("--skip-remote-existing", action="store_true")
    parser.add_argument("--force-convert", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    results = []
    for repo_id in args.repo_id:
        results.append(
            process_repo(
                api=api,
                repo_id=repo_id,
                output_root=args.output_root,
                converter=args.converter,
                cleanup_new=args.cleanup_new,
                skip_remote_existing=args.skip_remote_existing,
                force_convert=args.force_convert,
            )
        )
    log("summary")
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
