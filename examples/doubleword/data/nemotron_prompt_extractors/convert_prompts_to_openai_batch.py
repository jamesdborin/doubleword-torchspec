#!/usr/bin/env python3
"""Convert prompt-only Nemotron JSONL rows to OpenAI Batch chat requests."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MAX_REQUESTS_PER_SHARD = 50_000
DEFAULT_MAX_BYTES_PER_SHARD = 190 * 1024 * 1024
ENDPOINT = "/v1/chat/completions"


@dataclass
class ShardStats:
    path: str
    requests: int = 0
    bytes: int = 0


@dataclass
class ConversionStats:
    input_path: str
    output_dir: str
    model: str
    total_rows: int = 0
    written_requests: int = 0
    skipped_rows: int = 0
    shards: list[ShardStats] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "output_dir": self.output_dir,
            "model": self.model,
            "total_rows": self.total_rows,
            "written_requests": self.written_requests,
            "skipped_rows": self.skipped_rows,
            "shard_count": len(self.shards),
            "shards": [
                {"path": shard.path, "requests": shard.requests, "bytes": shard.bytes}
                for shard in self.shards
            ],
        }


class ShardWriter:
    def __init__(self, output_dir: Path, max_requests: int, max_bytes: int) -> None:
        if max_requests <= 0:
            raise ValueError("--max-requests-per-shard must be positive")
        if max_bytes <= 0:
            raise ValueError("--max-bytes-per-shard must be positive")

        self.output_dir = output_dir
        self.max_requests = max_requests
        self.max_bytes = max_bytes
        self.index = -1
        self.handle: Any = None
        self.current: ShardStats | None = None
        self.shards: list[ShardStats] = []

    def __enter__(self) -> ShardWriter:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None
            self.current = None

    def _open_next(self) -> None:
        self.close()
        self.index += 1
        path = self.output_dir / f"batch_{self.index:03d}.jsonl"
        self.handle = path.open("w", encoding="utf-8")
        self.current = ShardStats(path=str(path))
        self.shards.append(self.current)

    def write_line(self, line: str) -> None:
        line_bytes = len(line.encode("utf-8"))
        if line_bytes > self.max_bytes:
            raise ValueError(
                f"single request is {line_bytes} bytes, larger than shard byte limit "
                f"{self.max_bytes}"
            )

        if self.current is None:
            self._open_next()
        elif (
            self.current.requests >= self.max_requests
            or self.current.bytes + line_bytes > self.max_bytes
        ):
            self._open_next()

        assert self.handle is not None
        assert self.current is not None
        self.handle.write(line)
        self.current.requests += 1
        self.current.bytes += line_bytes


def load_extra_body(extra_body_json: str | None) -> dict[str, Any]:
    if extra_body_json is None:
        return {}
    value = json.loads(extra_body_json)
    if not isinstance(value, dict):
        raise ValueError("--extra-body-json must decode to a JSON object")
    return value


def compact_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def custom_id_for_record(record: dict[str, Any]) -> str:
    dataset_name = str(record.get("dataset") or "dataset").rsplit("/", 1)[-1]
    slug = dataset_name.lower().replace("_", "-")
    for prefix in ("nemotron-3-", "nemotron-"):
        if slug.startswith(prefix):
            slug = slug[len(prefix) :]
            break
    config = str(record.get("config") or "default").replace("/", "-")
    split = str(record.get("split") or "split").replace("/", "-")
    row_index = int(record["row_index"])
    return f"{slug}_{config}_{split}_{row_index:06d}"


def convert_tool(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") != "function":
        return tool

    if isinstance(tool.get("function"), dict):
        return tool

    function: dict[str, Any] = {}
    for key in ("name", "description", "parameters", "strict"):
        if key in tool:
            function[key] = tool[key]
    return {"type": "function", "function": function}


def convert_tools(tools: Any) -> list[dict[str, Any]] | None:
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise ValueError("tools must be a list when present")
    converted = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError("each tool must be a JSON object")
        converted.append(convert_tool(tool))
    return converted or None


def build_request(
    record: dict[str, Any],
    model: str,
    extra_body: dict[str, Any],
    max_completion_tokens: int | None = None,
    temperature: float | None = None,
) -> dict[str, Any] | None:
    prompt = record.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        return None

    messages: list[dict[str, str]] = []
    system_prompt = record.get("system_prompt")
    if isinstance(system_prompt, str) and system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    body: dict[str, Any] = {"model": model, "messages": messages}
    body.update(extra_body)
    if max_completion_tokens is not None:
        body["max_completion_tokens"] = max_completion_tokens
    if temperature is not None:
        body["temperature"] = temperature

    tools = convert_tools(record.get("tools"))
    if tools is not None:
        body["tools"] = tools

    return {
        "custom_id": custom_id_for_record(record),
        "method": "POST",
        "url": ENDPOINT,
        "body": body,
    }


def manifest_record(record: dict[str, Any], custom_id: str) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "dataset": record.get("dataset"),
        "config": record.get("config"),
        "split": record.get("split"),
        "row_index": record.get("row_index"),
    }


def skipped_record(record: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "dataset": record.get("dataset"),
        "config": record.get("config"),
        "split": record.get("split"),
        "row_index": record.get("row_index"),
        "reason": reason,
        "prompt_source": record.get("prompt_source"),
        "prompt_source_detail": record.get("prompt_source_detail"),
        "extraction_error": record.get("extraction_error"),
    }


def convert_file(
    input_path: Path,
    output_dir: Path,
    model: str,
    max_requests_per_shard: int = DEFAULT_MAX_REQUESTS_PER_SHARD,
    max_bytes_per_shard: int = DEFAULT_MAX_BYTES_PER_SHARD,
    max_completion_tokens: int | None = None,
    temperature: float | None = None,
    extra_body: dict[str, Any] | None = None,
) -> ConversionStats:
    extra_body = extra_body or {}
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = ConversionStats(
        input_path=str(input_path),
        output_dir=str(output_dir),
        model=model,
    )
    manifest_path = output_dir / "manifest.jsonl"
    skipped_path = output_dir / "skipped_rows.jsonl"
    summary_path = output_dir / "summary.json"

    with (
        input_path.open(encoding="utf-8") as input_handle,
        manifest_path.open("w", encoding="utf-8") as manifest_handle,
        skipped_path.open("w", encoding="utf-8") as skipped_handle,
        ShardWriter(output_dir, max_requests_per_shard, max_bytes_per_shard) as shards,
    ):
        for line_number, line in enumerate(input_handle, start=1):
            if not line.strip():
                continue
            stats.total_rows += 1
            record = json.loads(line)

            request = build_request(
                record,
                model=model,
                extra_body=extra_body,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
            )
            if request is None:
                stats.skipped_rows += 1
                skipped_handle.write(compact_json(skipped_record(record, "null_or_empty_prompt")))
                continue

            custom_id = request["custom_id"]
            shards.write_line(compact_json(request))
            manifest_handle.write(compact_json(manifest_record(record, custom_id)))
            stats.written_requests += 1

        stats.shards = shards.shards

    summary_path.write_text(
        json.dumps(stats.to_json(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert prompt-only Nemotron JSONL to OpenAI Batch chat JSONL."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input prompts.jsonl path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for batch files.")
    parser.add_argument("--model", required=True, help="Chat Completions model to use.")
    parser.add_argument("--max-completion-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument(
        "--extra-body-json",
        help="JSON object merged into each Chat Completions request body.",
    )
    parser.add_argument(
        "--max-requests-per-shard",
        type=int,
        default=DEFAULT_MAX_REQUESTS_PER_SHARD,
        help="Maximum requests per output JSONL shard.",
    )
    parser.add_argument(
        "--max-bytes-per-shard",
        type=int,
        default=DEFAULT_MAX_BYTES_PER_SHARD,
        help="Maximum UTF-8 bytes per output JSONL shard.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = convert_file(
        input_path=args.input,
        output_dir=args.output_dir,
        model=args.model,
        max_requests_per_shard=args.max_requests_per_shard,
        max_bytes_per_shard=args.max_bytes_per_shard,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        extra_body=load_extra_body(args.extra_body_json),
    )
    print(json.dumps(stats.to_json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
