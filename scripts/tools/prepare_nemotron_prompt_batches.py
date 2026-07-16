#!/usr/bin/env python3
"""Download Nemotron prompt CSVs and create generic Doubleword batch JSONL files.

This is a workspace helper for collection planning. It mirrors the bundled
hf-dw-batch-inference converter but skips rows whose prompt field is empty,
because some extracted prompt-only repos contain non-request rows.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

SKILL_DIR = Path.home() / ".codex/skills/hf-dw-batch-inference"
sys.path.insert(0, str(SKILL_DIR / "scripts"))

from csv_to_batch_jsonl import row_to_request, short_id  # noqa: E402


OUTPUT_ROOT = Path("data/nemotron-prompt-only")


def normalize_openai_tools(request: dict[str, object]) -> None:
    body = request.get("body")
    if not isinstance(body, dict):
        return
    response_format = body.get("response_format")
    if isinstance(response_format, dict):
        json_schema = response_format.get("json_schema")
        if isinstance(json_schema, dict):
            normalize_schema(json_schema.get("schema"))
    tools = body.get("tools")
    if not isinstance(tools, list):
        return
    normalized = []
    for tool in tools:
        if (
            isinstance(tool, dict)
            and tool.get("type") == "function"
            and "function" not in tool
            and "name" in tool
        ):
            function = {key: value for key, value in tool.items() if key != "type"}
            normalize_function_parameters(function)
            normalized.append({"type": "function", "function": function})
        else:
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict):
                normalize_function_parameters(tool["function"])
            normalized.append(tool)
    body["tools"] = normalized


def normalize_schema(schema: object) -> None:
    if not isinstance(schema, dict):
        return
    for key in ("oneOf", "anyOf", "allOf"):
        schema.pop(key, None)
    if "properties" in schema and "type" not in schema:
        schema["type"] = "object"
    properties = schema.get("properties")
    required = schema.get("required")
    if isinstance(properties, dict) and isinstance(required, list):
        schema["required"] = [item for item in required if item in properties]
    if schema.get("type") == "array" and "items" not in schema:
        schema["items"] = {}
    for value in list(schema.values()):
        if isinstance(value, dict):
            normalize_schema(value)
        elif isinstance(value, list):
            for item in value:
                normalize_schema(item)


def normalize_function_parameters(function: dict[str, object]) -> None:
    params = function.get("parameters")
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            return
        function["parameters"] = params
    if isinstance(params, dict):
        if "type" not in params:
            params["type"] = "object"
        normalize_schema(params)


def list_repos() -> list[dict[str, object]]:
    proc = subprocess.run(
        [
            "python",
            str(SKILL_DIR / "scripts/nemotron_collection.py"),
            "list",
            "--check-files",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def download_prompts(repo_id: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    local = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="prompts.csv")
    target = out_dir / "prompts.csv"
    if not target.exists() or Path(local).read_bytes() != target.read_bytes():
        target.write_bytes(Path(local).read_bytes())
    return target


def convert_skip_empty(csv_path: Path, output_path: Path, meta_path: Path) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(sys.maxsize)
    used_ids: set[str] = set()
    count = 0
    skipped_empty = 0
    total = 0
    skipped_examples: list[dict[str, str]] = []

    with csv_path.open(newline="", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        reader = csv.DictReader(line.replace("\0", "") for line in src)
        if not reader.fieldnames:
            raise SystemExit(f"{csv_path}: CSV has no header")
        if "prompt" not in reader.fieldnames:
            raise SystemExit(f"{csv_path}: CSV is missing prompt column")

        for row in reader:
            total += 1
            if not (row.get("prompt") or "").strip():
                skipped_empty += 1
                if len(skipped_examples) < 10:
                    skipped_examples.append(
                        {
                            "dataset": row.get("dataset", ""),
                            "config": row.get("config", ""),
                            "split": row.get("split", ""),
                            "row_index": row.get("row_index", ""),
                        }
                    )
                continue
            request = row_to_request(
                row=row,
                generated_id=short_id(used_ids, 12),
                prompt_column="prompt",
                model_placeholder="[MODEL]",
            )
            normalize_openai_tools(request)
            dst.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1

    metadata = {
        "csv": str(csv_path),
        "output": str(output_path),
        "total_rows": total,
        "requests": count,
        "skipped_empty_prompt_rows": skipped_empty,
        "skipped_empty_examples": skipped_examples,
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    csv.field_size_limit(sys.maxsize)

    for repo in list_repos():
        repo_id = str(repo["repo_id"])
        slug = repo_id.split("/", 1)[1]
        out_dir = OUTPUT_ROOT / slug
        csv_path = out_dir / "prompts.csv"
        output_path = out_dir / "dw_batch_requests.jsonl"
        meta_path = out_dir / "dw_batch_requests.meta.json"

        if not csv_path.exists():
            csv_path = download_prompts(repo_id, out_dir)

        can_reuse_output = output_path.exists() and (
            meta_path.exists() or bool(repo.get("has_dw_batch_requests"))
        )

        if can_reuse_output:
            requests = sum(1 for _ in output_path.open("r", encoding="utf-8"))
            metadata = {
                "csv": str(csv_path),
                "output": str(output_path),
                "requests": requests,
                "existing_output_reused": True,
            }
        else:
            metadata = convert_skip_empty(csv_path, output_path, meta_path)

        print(json.dumps({"repo_id": repo_id, **metadata}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
