#!/usr/bin/env python3
"""Extract prompt-only CSV and generic Doubleword JSONL from external HF datasets."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


PROMPT_COLUMNS = [
    "dataset",
    "config",
    "split",
    "row_index",
    "prompt",
    "prompt_source",
    "prompt_source_detail",
    "system_prompt",
    "system_source",
    "tools",
    "tools_source",
    "schema_str",
    "extraction_error",
]


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S %z')} {message}", flush=True)


def run(cmd: list[str]) -> None:
    log("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def short_id(used: set[str]) -> str:
    while True:
        value = uuid.uuid4().hex[:12]
        if value not in used:
            used.add(value)
            return value


def batch_request(custom_id: str, prompt: str) -> dict[str, object]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "[MODEL]",
            "messages": [{"role": "user", "content": prompt}],
        },
    }


def source_files(api: HfApi, repo_id: str, prefix: str = "data/", include_file_prefix: str | None = None) -> list[str]:
    files = [
        sibling.rfilename
        for sibling in api.dataset_info(repo_id, files_metadata=True).siblings
        if sibling.rfilename.startswith(prefix) and sibling.rfilename.endswith(".parquet")
    ]
    if include_file_prefix:
        files = [filename for filename in files if filename.startswith(include_file_prefix)]
    return sorted(files)


def split_from_filename(path: str) -> str:
    name = Path(path).name
    if "-" not in name:
        return Path(path).stem
    return name.split("-", 1)[0]


def iter_prompt_rows(
    repo_id: str,
    files: Iterable[str],
    prompt_column: str,
    id_column: str | None,
    local_dir: Path,
) -> Iterable[dict[str, str]]:
    global_index = 0
    columns = [prompt_column]
    if id_column:
        columns.append(id_column)
    for filename in files:
        split = split_from_filename(filename)
        log(f"{repo_id}: downloading {filename}")
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=local_dir)
        parquet = pq.ParquetFile(path)
        if prompt_column not in parquet.schema_arrow.names:
            raise RuntimeError(f"{repo_id} {filename}: missing prompt column {prompt_column!r}")
        read_columns = [column for column in columns if column in parquet.schema_arrow.names]
        for batch in parquet.iter_batches(batch_size=10_000, columns=read_columns):
            data = batch.to_pydict()
            prompts = data[prompt_column]
            ids = data.get(id_column or "", [None] * len(prompts))
            for local_index, prompt in enumerate(prompts):
                prompt = (prompt or "").strip()
                detail = {
                    "source_file": filename,
                    "source_row_index": local_index,
                }
                if id_column and ids[local_index] is not None:
                    detail[id_column] = str(ids[local_index])
                yield {
                    "dataset": repo_id,
                    "config": "default",
                    "split": split,
                    "row_index": str(global_index),
                    "prompt": prompt,
                    "prompt_source": prompt_column,
                    "prompt_source_detail": json.dumps(detail, ensure_ascii=False, separators=(",", ":")),
                    "system_prompt": "",
                    "system_source": "",
                    "tools": "",
                    "tools_source": "",
                    "schema_str": "",
                    "extraction_error": "" if prompt else "empty prompt",
                }
                global_index += 1


def write_artifacts(
    source_repo: str,
    prompt_column: str,
    id_column: str | None,
    repo_dir: Path,
    files: list[str],
) -> tuple[int, Path, Path]:
    prompts_csv = repo_dir / "prompts.csv"
    requests_jsonl = repo_dir / "dw_batch_requests.jsonl"
    used_ids: set[str] = set()
    count = 0
    with prompts_csv.open("w", newline="", encoding="utf-8") as csv_handle, requests_jsonl.open(
        "w", encoding="utf-8"
    ) as jsonl_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=PROMPT_COLUMNS, quoting=csv.QUOTE_ALL, escapechar="\\")
        writer.writeheader()
        for row in iter_prompt_rows(source_repo, files, prompt_column, id_column, repo_dir / "source"):
            writer.writerow(row)
            if row["prompt"]:
                custom_id = short_id(used_ids)
                jsonl_handle.write(
                    json.dumps(batch_request(custom_id, row["prompt"]), ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                count += 1
    return count, prompts_csv, requests_jsonl


def write_readme(path: Path, source_repo: str, prompt_column: str, count: int, files: list[str]) -> None:
    path.write_text(
        "\n".join(
            [
                "---",
                "license: other",
                "task_categories:",
                "- text-generation",
                "---",
                "",
                f"# {path.parent.name}",
                "",
                f"Prompt-only extraction from `{source_repo}`.",
                "",
                f"- Source prompt column: `{prompt_column}`",
                f"- Generic Doubleword requests: `{count}`",
                "- Request model placeholder: `[MODEL]`",
                "- Source files:",
                *(f"  - `{filename}`" for filename in files),
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--target-repo", required=True)
    parser.add_argument("--prompt-column", required=True)
    parser.add_argument("--id-column")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/external-prompt-only"))
    parser.add_argument("--include-file-prefix")
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    files = source_files(api, args.source_repo, include_file_prefix=args.include_file_prefix)
    if not files:
        raise RuntimeError(f"{args.source_repo}: no parquet files found")

    repo_dir = args.output_root / args.target_repo.rsplit("/", 1)[-1]
    repo_dir.mkdir(parents=True, exist_ok=True)
    log(f"{args.source_repo}: found {len(files)} parquet files")
    count, prompts_csv, requests_jsonl = write_artifacts(
        source_repo=args.source_repo,
        prompt_column=args.prompt_column,
        id_column=args.id_column,
        repo_dir=repo_dir,
        files=files,
    )
    log(f"{args.source_repo}: wrote {count} generic requests")

    run(["dw", "files", "validate", str(requests_jsonl)])
    write_readme(repo_dir / "README.md", args.source_repo, args.prompt_column, count, files)

    api.create_repo(args.target_repo, repo_type="dataset", exist_ok=True)
    run(
        [
            "huggingface-cli",
            "upload",
            args.target_repo,
            str(prompts_csv),
            "prompts.csv",
            "--repo-type",
            "dataset",
            "--commit-message",
            "Add prompt-only source CSV",
        ]
    )
    run(
        [
            "huggingface-cli",
            "upload",
            args.target_repo,
            str(requests_jsonl),
            "dw_batch_requests.jsonl",
            "--repo-type",
            "dataset",
            "--commit-message",
            "Add generic Doubleword batch requests",
        ]
    )
    run(
        [
            "huggingface-cli",
            "upload",
            args.target_repo,
            str(repo_dir / "README.md"),
            "README.md",
            "--repo-type",
            "dataset",
            "--commit-message",
            "Add dataset card",
        ]
    )
    api.add_collection_item(args.collection, args.target_repo, "dataset", exists_ok=True)

    remote_files = set(api.list_repo_files(args.target_repo, repo_type="dataset"))
    required = {"prompts.csv", "dw_batch_requests.jsonl", "README.md"}
    missing = sorted(required - remote_files)
    if missing:
        raise RuntimeError(f"{args.target_repo}: missing uploaded files: {missing}")
    print(
        json.dumps(
            {
                "source_repo": args.source_repo,
                "target_repo": args.target_repo,
                "requests": count,
                "files": files,
                "status": "uploaded",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    if args.cleanup:
        import shutil

        log(f"removing local staging directory {repo_dir}")
        shutil.rmtree(repo_dir)


if __name__ == "__main__":
    main()
