#!/usr/bin/env python3
"""Create prompt-only eval dataset repos with generic Doubleword JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

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


def request(custom_id: str, prompt: str) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": "[MODEL]", "messages": [{"role": "user", "content": prompt}]},
    }


def prompt_row(
    source_repo: str,
    config: str,
    split: str,
    index: int,
    prompt: str,
    prompt_source: str,
    detail: dict[str, Any],
) -> dict[str, str]:
    return {
        "dataset": source_repo,
        "config": config,
        "split": split,
        "row_index": str(index),
        "prompt": (prompt or "").strip(),
        "prompt_source": prompt_source,
        "prompt_source_detail": json.dumps(detail, ensure_ascii=False, separators=(",", ":")),
        "system_prompt": "",
        "system_source": "",
        "tools": "",
        "tools_source": "",
        "schema_str": "",
        "extraction_error": "" if (prompt or "").strip() else "empty prompt",
    }


def humaneval_rows(repo_dir: Path) -> Iterable[dict[str, str]]:
    repo = "openai/openai_humaneval"
    filename = "openai_humaneval/test-00000-of-00001.parquet"
    path = hf_hub_download(repo, filename, repo_type="dataset", local_dir=repo_dir / "source")
    table = pq.read_table(path, columns=["task_id", "prompt", "entry_point"])
    for index, row in enumerate(table.to_pylist()):
        yield prompt_row(
            repo,
            "default",
            "test",
            index,
            row["prompt"],
            "prompt",
            {"source_file": filename, "task_id": row["task_id"], "entry_point": row["entry_point"]},
        )


def ifeval_rows(repo_dir: Path) -> Iterable[dict[str, str]]:
    repo = "google/IFEval"
    filename = "ifeval_input_data.jsonl"
    path = hf_hub_download(repo, filename, repo_type="dataset", local_dir=repo_dir / "source")
    with open(path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            yield prompt_row(
                repo,
                "default",
                "test",
                index,
                row["prompt"],
                "prompt",
                {
                    "source_file": filename,
                    "key": row.get("key"),
                    "instruction_id_list": row.get("instruction_id_list"),
                    "kwargs": row.get("kwargs"),
                },
            )


def gpqa_prompt(row: dict[str, str]) -> str:
    choices = [
        ("A", row["Correct Answer"].strip()),
        ("B", row["Incorrect Answer 1"].strip()),
        ("C", row["Incorrect Answer 2"].strip()),
        ("D", row["Incorrect Answer 3"].strip()),
    ]
    lines = [
        row["Question"].strip(),
        "",
        "Choose the correct answer from the following options.",
    ]
    lines.extend(f"{label}. {text}" for label, text in choices)
    return "\n".join(lines).strip()


def gpqa_rows(repo_dir: Path) -> Iterable[dict[str, str]]:
    repo = "Idavidrein/gpqa"
    files = ["gpqa_diamond.csv", "gpqa_main.csv", "gpqa_extended.csv"]
    index = 0
    for filename in files:
        path = hf_hub_download(repo, filename, repo_type="dataset", local_dir=repo_dir / "source")
        config = filename.removesuffix(".csv")
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for source_index, row in enumerate(reader):
                yield prompt_row(
                    repo,
                    config,
                    config,
                    index,
                    gpqa_prompt(row),
                    "Question+choices",
                    {
                        "source_file": filename,
                        "source_row_index": source_index,
                        "record_id": row.get("Record ID"),
                        "high_level_domain": row.get("High-level domain"),
                        "subdomain": row.get("Subdomain"),
                    },
                )
                index += 1


def aime_rows(repo_dir: Path) -> Iterable[dict[str, str]]:
    repo = "MathArena/aime_2025"
    filename = "data/train-00000-of-00001.parquet"
    path = hf_hub_download(repo, filename, repo_type="dataset", local_dir=repo_dir / "source")
    table = pq.read_table(path, columns=["problem_idx", "problem", "problem_type"])
    for index, row in enumerate(table.to_pylist()):
        yield prompt_row(
            repo,
            "default",
            "train",
            index,
            row["problem"],
            "problem",
            {"source_file": filename, "problem_idx": row["problem_idx"], "problem_type": row["problem_type"]},
        )


EXTRACTORS = {
    "humaneval": humaneval_rows,
    "ifeval": ifeval_rows,
    "gpqa": gpqa_rows,
    "aime_2025": aime_rows,
}


def write_artifacts(repo_dir: Path, rows: Iterable[dict[str, str]]) -> int:
    prompts_csv = repo_dir / "prompts.csv"
    requests_jsonl = repo_dir / "dw_batch_requests.jsonl"
    used_ids: set[str] = set()
    count = 0
    with prompts_csv.open("w", newline="", encoding="utf-8") as csv_handle, requests_jsonl.open(
        "w", encoding="utf-8"
    ) as jsonl_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=PROMPT_COLUMNS, quoting=csv.QUOTE_ALL, escapechar="\\")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            if row["prompt"]:
                jsonl_handle.write(
                    json.dumps(request(short_id(used_ids), row["prompt"]), ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                count += 1
    return count


def write_readme(repo_dir: Path, source_repo: str, count: int) -> None:
    (repo_dir / "README.md").write_text(
        "\n".join(
            [
                "---",
                "task_categories:",
                "- text-generation",
                "---",
                "",
                f"# {repo_dir.name}",
                "",
                f"Prompt-only eval extraction from `{source_repo}`.",
                "",
                f"- Generic Doubleword requests: `{count}`",
                "- Request model placeholder: `[MODEL]`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def upload_repo(api: HfApi, repo_id: str, repo_dir: Path, collection_slug: str) -> None:
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    for filename, message in [
        ("prompts.csv", "Add prompt-only source CSV"),
        ("dw_batch_requests.jsonl", "Add generic Doubleword batch requests"),
        ("README.md", "Add dataset card"),
    ]:
        run(
            [
                "huggingface-cli",
                "upload",
                repo_id,
                str(repo_dir / filename),
                filename,
                "--repo-type",
                "dataset",
                "--commit-message",
                message,
            ]
        )
    api.add_collection_item(collection_slug, repo_id, "dataset", exists_ok=True)
    remote = set(api.list_repo_files(repo_id, repo_type="dataset"))
    required = {"prompts.csv", "dw_batch_requests.jsonl", "README.md"}
    missing = required - remote
    if missing:
        raise RuntimeError(f"{repo_id}: missing remote files: {sorted(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-title", default="evals-prompt-only")
    parser.add_argument("--namespace", default="jamesdborin")
    parser.add_argument("--output-root", type=Path, default=Path("data/evals-prompt-only"))
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    collection = api.create_collection(
        args.collection_title,
        namespace=args.namespace,
        description="Prompt-only evaluation datasets with generic Doubleword batch request JSONL.",
        exists_ok=True,
    )
    collection_slug = collection.slug
    log(f"collection {collection_slug}")

    specs = [
        ("humaneval", "openai/openai_humaneval", "jamesdborin/openai-humaneval-prompt-only"),
        ("ifeval", "google/IFEval", "jamesdborin/IFEval-prompt-only"),
        ("gpqa", "Idavidrein/gpqa", "jamesdborin/GPQA-prompt-only"),
        ("aime_2025", "MathArena/aime_2025", "jamesdborin/AIME-2025-prompt-only"),
    ]
    results = []
    for key, source_repo, target_repo in specs:
        repo_dir = args.output_root / target_repo.rsplit("/", 1)[-1]
        repo_dir.mkdir(parents=True, exist_ok=True)
        log(f"{source_repo}: extracting to {target_repo}")
        count = write_artifacts(repo_dir, EXTRACTORS[key](repo_dir))
        log(f"{source_repo}: wrote {count} generic requests")
        run(["dw", "files", "validate", str(repo_dir / "dw_batch_requests.jsonl")])
        write_readme(repo_dir, source_repo, count)
        upload_repo(api, target_repo, repo_dir, collection_slug)
        results.append({"source_repo": source_repo, "target_repo": target_repo, "requests": count})
        if args.cleanup:
            import shutil

            log(f"removing local staging directory {repo_dir}")
            shutil.rmtree(repo_dir)

    print(json.dumps({"collection": collection_slug, "results": results}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
