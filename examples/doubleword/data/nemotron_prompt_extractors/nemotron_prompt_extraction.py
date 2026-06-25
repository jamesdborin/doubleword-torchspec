#!/usr/bin/env python3
"""Shared prompt extraction utilities for Nemotron Post-Training v3 datasets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = "default"
IFBENCH_MARKERS = ("ifbench", "if_bench", "instruction_following")


DATASET_SPECS: dict[str, dict[str, Any]] = {
    "nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2": {
        "priority": ["responses_input"],
    },
    "nvidia/Nemotron-RL-Instruction-Following-Citation-Formatting-v1": {
        "priority": ["responses_input"],
    },
    "nvidia/Nemotron-RL-Instruction-Following-Free-Form-Formatting-v1": {
        "priority": ["responses_input"],
    },
    "nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1": {
        "priority": ["responses_input"],
    },
    "nvidia/Nemotron-RL-Instruction-Following-Calendar-v2": {
        "priority": ["responses_input"],
    },
    "nvidia/Nemotron-SFT-Agentic-v2": {
        "priority": ["messages"],
        "split_aliases": {"train": "interactive_agent"},
        "data_files": [
            {"config": DEFAULT_CONFIG, "split": "interactive_agent", "path": "data/interactive_agent.jsonl"},
            {"config": DEFAULT_CONFIG, "split": "search", "path": "data/search.jsonl"},
            {"config": DEFAULT_CONFIG, "split": "tool_calling", "path": "data/tool_calling.jsonl"},
        ],
    },
    "nvidia/Nemotron-RL-litmus-bench-v0.1": {
        "priority": ["responses_input"],
    },
    "nvidia/Nemotron-RL-Super-Training-Blends": {
        "priority": ["responses_input", "prompt", "messages", "question", "problem"],
        "data_files": [
            {"config": DEFAULT_CONFIG, "split": "rlvr1", "path": "rlvr1.jsonl"},
            {"config": DEFAULT_CONFIG, "split": "rlvr2", "path": "rlvr2.jsonl"},
            {"config": DEFAULT_CONFIG, "split": "rlvr3", "path": "rlvr3.jsonl"},
            {"config": DEFAULT_CONFIG, "split": "swe1", "path": "swe1.jsonl"},
            {"config": DEFAULT_CONFIG, "split": "swe2", "path": "swe2.jsonl"},
            {"config": DEFAULT_CONFIG, "split": "rlhf", "path": "rlhf.jsonl"},
        ],
    },
    "nvidia/Nemotron-SFT-OpenCode-v1": {
        "priority": ["messages", "question", "agent_prompt"],
    },
    "nvidia/Nemotron-3-Nano-RL-Training-Blend": {
        "priority": ["responses_input"],
        "data_files": [
            {"config": DEFAULT_CONFIG, "split": "train", "path": "train.jsonl"},
        ],
    },
    "nvidia/Nemotron-Math-Proofs-v1": {
        "priority": ["messages", "problem"],
    },
    "nvidia/Nemotron-Agentic-v1": {
        "priority": ["messages"],
        "data_files": [
            {"config": DEFAULT_CONFIG, "split": "interactive_agent", "path": "data/interactive_agent.jsonl"},
            {"config": DEFAULT_CONFIG, "split": "tool_calling", "path": "data/tool_calling.jsonl"},
        ],
    },
    "nvidia/Nemotron-Competitive-Programming-v1": {
        "priority": ["messages"],
        "data_files": [
            {"config": DEFAULT_CONFIG, "split": "competitive_coding_cpp_part00", "path": "data/competitive_coding_cpp.part_00.jsonl"},
            {"config": DEFAULT_CONFIG, "split": "competitive_coding_cpp_part01", "path": "data/competitive_coding_cpp.part_01.jsonl"},
            {"config": DEFAULT_CONFIG, "split": "competitive_coding_python_part00", "path": "data/competitive_coding_python.part_00.jsonl"},
            {"config": DEFAULT_CONFIG, "split": "competitive_coding_python_part01", "path": "data/competitive_coding_python.part_01.jsonl"},
            {"config": DEFAULT_CONFIG, "split": "infinibyte_part00", "path": "data/infinibyte.part_00.jsonl"},
            {"config": DEFAULT_CONFIG, "split": "infinibyte_part01", "path": "data/infinibyte.part_01.jsonl"},
        ],
    },
    "nvidia/Nemotron-Math-v2": {
        "priority": ["messages", "problem"],
    },
    "nvidia/Nemotron-SWE-v1": {
        "priority": ["messages"],
    },
    "nvidia/Nemotron-SFT-SWE-v2": {
        "priority": ["messages"],
        "data_files": [
            {"config": DEFAULT_CONFIG, "split": "agentless", "path": "data/agentless.jsonl"},
            {"config": DEFAULT_CONFIG, "split": "openhands_swe", "path": "data/swe.jsonl"},
        ],
    },
    "nvidia/Nemotron-SFT-Safety-v1": {
        "priority": ["messages"],
    },
    "nvidia/Nemotron-SFT-Competitive-Programming-v2": {
        "priority": ["messages"],
        "data_files": [
            {"config": DEFAULT_CONFIG, "split": "exercism", "path": "data/exercism.jsonl"},
            {"config": DEFAULT_CONFIG, "split": "text_to_sql", "path": "data/text_to_sql.jsonl"},
            {"config": DEFAULT_CONFIG, "split": "competitive_coding_cpp_00", "path": "data/competitive_programming_cpp_00.jsonl"},
            {"config": DEFAULT_CONFIG, "split": "competitive_coding_cpp_01", "path": "data/competitive_programming_cpp_01.jsonl"},
            {"config": DEFAULT_CONFIG, "split": "competitive_coding_python_00", "path": "data/competitive_programming_python_00.jsonl"},
            {"config": DEFAULT_CONFIG, "split": "competitive_coding_python_01", "path": "data/competitive_programming_python_01.jsonl"},
        ],
    },
    "nvidia/Nemotron-SpecializedDomains-Finance-v1": {
        "priority": ["messages"],
    },
    "nvidia/Nemotron-SFT-Instruction-Following-Chat-v2": {
        "priority": ["messages"],
        "data_files": [
            {"config": DEFAULT_CONFIG, "split": "reasoning_off", "path": "data/reasoning_off.jsonl"},
            {"config": DEFAULT_CONFIG, "split": "reasoning_on", "path": "data/reasoning_on.jsonl"},
        ],
    },
    "nvidia/Nemotron-RLHF-GenRM-v1": {
        "priority": ["nested_messages", "messages"],
        "data_files": [
            {"config": DEFAULT_CONFIG, "split": "train", "path": "data/train.jsonl"},
        ],
    },
    "nvidia/Nemotron-RL-ReasoningGym-v1": {
        "priority": ["responses_input", "question"],
    },
    "nvidia/Nemotron-SFT-Multilingual-v1": {
        "priority": ["messages"],
    },
    "nvidia/Nemotron-RL-Safety-v1": {
        "priority": ["prompt"],
    },
    "nvidia/Nemotron-RL-Identity-Following-v1": {
        "priority": ["responses_input"],
    },
    "nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1": {
        "priority": ["responses_input"],
        "data_files": [
            {"config": DEFAULT_CONFIG, "split": "train", "path": "train.jsonl"},
        ],
    },
    "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1": {
        "priority": ["responses_input"],
        "data_files": [
            {"config": DEFAULT_CONFIG, "split": "train", "path": "train.jsonl"},
        ],
    },
    "nvidia/Nemotron-RL-Instruction-Following-Adversarial-v1": {
        "priority": ["prompt", "responses_input"],
    },
    "nvidia/Nemotron-RL-Instruction-Following-MultiTurnChat-v1": {
        "priority": ["responses_input"],
    },
    "nvidia/Nemotron-SFT-ARC-AGI-v1": {
        "priority": ["messages"],
    },
    "nvidia/Nemotron-SFT-CUDA-v1": {
        "priority": ["messages"],
    },
    "nvidia/Nemotron-SFT-Instruction-Following-Chat-v3": {
        "priority": ["messages"],
    },
    "nvidia/Nemotron-SFT-Math-v4": {
        "priority": ["messages", "problem"],
    },
    "nvidia/Nemotron-Math-Proofs-v2": {
        "priority": ["messages", "problem"],
    },
    "nvidia/Nemotron-SFT-Multilingual-v2": {
        "priority": ["messages"],
    },
    "nvidia/Nemotron-SFT-Safety-v2": {
        "priority": ["messages"],
        "data_files": [
            {"config": DEFAULT_CONFIG, "split": "train", "path": "data/train.jsonl"},
        ],
    },
    "nvidia/Nemotron-SFT-Science-v2": {
        "priority": ["messages"],
    },
    "nvidia/Nemotron-RL-Science-v1": {
        "priority": ["responses_input", "problem"],
        "data_files": [
            {"config": DEFAULT_CONFIG, "split": "so_openq", "path": "so_openq.jsonl"},
        ],
    },
    "nvidia/Nemotron-RL-Math-v2": {
        "priority": ["responses_input", "question"],
    },
    "nvidia/Nemotron-RL-QA-Abstention-v1": {
        "priority": ["responses_input", "messages", "question"],
    },
    "nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1": {
        "priority": ["responses_input"],
    },
    "nvidia/Nemotron-RL-ARC-AGI-v1": {
        "priority": ["responses_input"],
    },
    "nvidia/Nemotron-RL-SysBench-v1": {
        "priority": ["responses_input", "messages"],
    },
    "nvidia/Nemotron-RL-CFBench-v1": {
        "priority": ["responses_input"],
    },
    "nvidia/Nemotron-RL-Multichallenge-v1": {
        "priority": ["responses_input", "messages"],
    },
    "nvidia/Nemotron-RL-InverseIFEval-v1": {
        "priority": ["responses_input", "messages"],
    },
    "nvidia/Nemotron-RL-Ultra-Training-Blends": {
        "priority": ["ifbench_prompt", "responses_input", "messages"],
        "data_files": [
            {"config": "rlvr1", "split": "train", "path": "rlvr1.jsonl"},
            {"config": "rlvr2", "split": "train", "path": "rlvr2.jsonl"},
            {"config": "ifbench", "split": "train", "path": "ifbench.jsonl"},
            {"config": "rlhf", "split": "train", "path": "rlhf.jsonl"},
            {"config": "reasoning", "split": "train", "path": "reasoning.jsonl"},
            {"config": "swe", "split": "train", "path": "swe.jsonl"},
            {"config": "mopd", "split": "train", "path": "mopd.jsonl"},
        ],
    },
    "nvidia/Nemotron-SFT-SWE-v3": {
        "priority": ["messages"],
    },
    "nvidia/Nemotron-SFT-Math-v3": {
        "priority": ["messages", "problem"],
    },
}


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in "[{":
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def content_to_text(content: Any) -> str | None:
    content = parse_jsonish(content)
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            item = parse_jsonish(item)
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text is not None:
                    converted = content_to_text(text)
                    if converted is not None:
                        parts.append(converted)
        return "\n".join(parts) if parts else None
    if isinstance(content, dict):
        for key in ("text", "content", "value", "prompt"):
            if key in content:
                return content_to_text(content[key])
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    return str(content)


def is_message(value: Any) -> bool:
    return isinstance(value, dict) and ("role" in value or "content" in value)


def first_message_prompt(messages: Any) -> tuple[str | None, str | None]:
    messages = parse_jsonish(messages)
    if not isinstance(messages, list):
        return content_to_text(messages), "scalar"

    flattened = flatten_messages(messages)
    for message in flattened:
        role = str(message.get("role", "")).lower()
        if role == "user":
            return content_to_text(message.get("content")), "first_user_message"

    for message in flattened:
        role = str(message.get("role", "")).lower()
        if role not in {"assistant", "tool", "function"}:
            text = content_to_text(message.get("content"))
            if text:
                return text, "first_non_assistant_message"

    if flattened:
        return content_to_text(flattened[0].get("content")), "first_message"
    return None, None


def flatten_messages(value: Any) -> list[dict[str, Any]]:
    value = parse_jsonish(value)
    if is_message(value):
        return [value]
    if not isinstance(value, list):
        return []

    flattened: list[dict[str, Any]] = []
    for item in value:
        item = parse_jsonish(item)
        if is_message(item):
            flattened.append(item)
        elif isinstance(item, list):
            flattened.extend(flatten_messages(item))
    return flattened


def responses_input(row: dict[str, Any]) -> Any:
    params = parse_jsonish(row.get("responses_create_params"))
    if not isinstance(params, dict):
        return None
    return params.get("input")


def is_ifbench_row(row: dict[str, Any], config: str | None = None) -> bool:
    haystack: list[str] = [config] if config else []
    for key in ("dataset", "source", "subset", "task", "benchmark", "category", "used_in"):
        value = row.get(key)
        if isinstance(value, str):
            haystack.append(value)
        elif isinstance(value, list):
            haystack.extend(item for item in value if isinstance(item, str))
    text = " ".join(haystack).lower()
    return any(marker in text for marker in IFBENCH_MARKERS)


def extract_prompt(
    row: dict[str, Any], dataset_id: str, config: str | None = None
) -> tuple[str | None, str | None, str | None]:
    spec = DATASET_SPECS[dataset_id]
    for source in spec["priority"]:
        if source == "responses_input":
            prompt, detail = first_message_prompt(responses_input(row))
        elif source == "ifbench_prompt":
            prompt = content_to_text(row.get("prompt")) if is_ifbench_row(row, config) else None
            detail = "prompt" if prompt else "not_ifbench"
        elif source == "nested_messages":
            prompt, detail = first_message_prompt(row.get("messages"))
        elif source == "messages":
            prompt, detail = first_message_prompt(row.get("messages"))
        else:
            prompt = content_to_text(row.get(source))
            detail = source

        if prompt:
            return prompt, source, detail

    return None, None, "no_prompt_found"


def dataset_configs(dataset_id: str, requested_config: str | None) -> list[str | None]:
    if requested_config:
        return [None if requested_config == DEFAULT_CONFIG else requested_config]
    try:
        from datasets import get_dataset_config_names

        names = get_dataset_config_names(dataset_id)
    except Exception:
        return [None]
    if not names:
        return [None]
    return [None if name == DEFAULT_CONFIG else name for name in names]


def dataset_splits(dataset_id: str, config: str | None, requested_split: str | None) -> list[str]:
    if requested_split:
        aliases = DATASET_SPECS[dataset_id].get("split_aliases", {})
        return [aliases.get(requested_split, requested_split)]
    try:
        from datasets import get_dataset_split_names

        if config is None:
            return list(get_dataset_split_names(dataset_id))
        return list(get_dataset_split_names(dataset_id, config))
    except Exception:
        return ["train"]


def load_stream(dataset_id: str, config: str | None, split: str) -> Iterable[dict[str, Any]]:
    if dataset_id == "nvidia/Nemotron-RL-ReasoningGym-v1" and config is None and split == "train":
        return load_reasoninggym_raw_jsonl(dataset_id)

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The Hugging Face 'datasets' package is required to extract prompts. "
            "Install the project dependencies before running full extraction."
        ) from exc

    def load(streaming: bool) -> Iterable[dict[str, Any]]:
        kwargs: dict[str, Any] = {"split": split, "streaming": streaming}
        if config is not None:
            return load_dataset(dataset_id, config, **kwargs)
        return load_dataset(dataset_id, **kwargs)

    def rows() -> Iterable[dict[str, Any]]:
        yielded = 0
        try:
            for row in load(streaming=True):
                yielded += 1
                yield row
        except Exception:
            if yielded:
                raise
            for row in load(streaming=False):
                yield row

    return rows()


def load_reasoninggym_raw_jsonl(dataset_id: str) -> Iterable[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=dataset_id, repo_type="dataset", filename="data/train.jsonl")
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def dataset_data_files(
    dataset_id: str, requested_config: str | None, requested_split: str | None
) -> list[dict[str, str]]:
    data_files = DATASET_SPECS[dataset_id].get("data_files")
    if not data_files:
        return []

    selected: list[dict[str, str]] = []
    for data_file in data_files:
        config = data_file["config"]
        if requested_config and requested_config != config:
            continue
        if requested_split and requested_split != data_file["split"]:
            continue
        selected.append(data_file)
    return selected


def load_jsonl_file(dataset_id: str, path: str) -> Iterable[dict[str, Any]]:
    try:
        import requests
        from huggingface_hub import hf_hub_url
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The Hugging Face Hub and requests packages are required to download JSONL files."
        ) from exc

    local_path = download_jsonl_file(dataset_id, path, requests, hf_hub_url)
    decoder = json.JSONDecoder(strict=False)
    with local_path.open(encoding="utf-8") as handle:
        buffer = ""
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() and not buffer:
                continue
            if "\x00" in line:
                if buffer.strip():
                    yield {
                        "_raw_json_error": "NUL byte found while buffering JSON object",
                        "_raw_json_path": path,
                        "_raw_json_line": line_number,
                    }
                    buffer = ""
                yield {
                    "_raw_json_error": "NUL byte found in JSONL record",
                    "_raw_json_path": path,
                    "_raw_json_line": line_number,
                }
                continue
            buffer += line
            while buffer:
                stripped = buffer.lstrip()
                if not stripped:
                    buffer = ""
                    break
                try:
                    row, end = decoder.raw_decode(stripped)
                except json.JSONDecodeError as exc:
                    if exc.pos >= len(stripped) - 1 or "Unterminated string" in exc.msg:
                        break
                    raise
                yield row
                buffer = stripped[end:]
        if buffer.strip():
            yield decoder.decode(buffer)


def download_jsonl_file(
    dataset_id: str, path: str, requests: Any, hf_hub_url: Any
) -> Path:
    cache_root = Path("/tmp/nemotron_prompt_full_extraction/hf_jsonl_cache")
    local_path = cache_root / dataset_id.replace("/", "__") / path
    parts_dir = local_path.with_suffix(local_path.suffix + ".parts")
    local_path.parent.mkdir(parents=True, exist_ok=True)

    url = hf_hub_url(dataset_id, path, repo_type="dataset")
    head = requests.head(url, allow_redirects=True, timeout=60)
    head.raise_for_status()
    total_size = int(head.headers["content-length"])
    url = head.url

    if local_path.exists() and local_path.stat().st_size == total_size:
        return local_path

    chunk_size = 64 * 1024 * 1024
    chunks = [
        (index, start, min(start + chunk_size - 1, total_size - 1))
        for index, start in enumerate(range(0, total_size, chunk_size))
    ]
    parts_dir.mkdir(parents=True, exist_ok=True)

    def download_part(item: tuple[int, int, int]) -> Path:
        index, start, end = item
        expected_size = end - start + 1
        part_path = parts_dir / f"{index:06d}.part"
        if part_path.exists() and part_path.stat().st_size == expected_size:
            return part_path

        headers = {"Range": f"bytes={start}-{end}"}
        tmp_path = part_path.with_suffix(".tmp")
        for attempt in range(10):
            with requests.get(
                url, headers=headers, stream=True, timeout=(30, 120)
            ) as response:
                if response.status_code != 206:
                    response.raise_for_status()
                    raise RuntimeError(
                        f"Expected HTTP 206 for range {start}-{end}, got {response.status_code}"
                    )
                response.raise_for_status()
                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if tmp_path.stat().st_size == expected_size:
                tmp_path.replace(part_path)
                return part_path
            tmp_path.unlink(missing_ok=True)
            time.sleep(min(60, 2 ** (attempt + 1)))
        raise RuntimeError(f"Failed to download range {start}-{end} for {path}")

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(download_part, chunks))

    tmp_local = local_path.with_suffix(local_path.suffix + ".tmp")
    with tmp_local.open("wb") as output:
        for index, _, _ in chunks:
            with (parts_dir / f"{index:06d}.part").open("rb") as part:
                shutil.copyfileobj(part, output, length=8 * 1024 * 1024)
    if tmp_local.stat().st_size != total_size:
        raise RuntimeError(f"Downloaded size mismatch for {path}")
    tmp_local.replace(local_path)
    return local_path


def write_prompts(
    dataset_id: str,
    output_path: Path,
    requested_config: str | None = None,
    requested_split: str | None = None,
    limit: int | None = None,
) -> int:
    total = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        data_files = dataset_data_files(dataset_id, requested_config, requested_split)
        if data_files:
            sources = (
                (
                    data_file["config"],
                    data_file["split"],
                    load_jsonl_file(dataset_id, data_file["path"]),
                )
                for data_file in data_files
            )
        else:
            sources = (
                (config or DEFAULT_CONFIG, split, load_stream(dataset_id, config, split))
                for config in dataset_configs(dataset_id, requested_config)
                for split in dataset_splits(dataset_id, config, requested_split)
            )

        for config, split, rows in sources:
            for row_index, row in enumerate(rows):
                    error = None
                    try:
                        prompt, source, source_detail = extract_prompt(row, dataset_id, config)
                    except Exception as exc:
                        prompt, source, source_detail = None, None, None
                        error = f"{type(exc).__name__}: {exc}"

                    record = {
                        "dataset": dataset_id,
                        "config": config,
                        "split": split,
                        "row_index": row_index,
                        "prompt": prompt,
                        "prompt_source": source,
                        "prompt_source_detail": source_detail,
                    }
                    if error is not None:
                        record["extraction_error"] = error

                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total += 1
                    if limit is not None and total >= limit:
                        return total
    return total


def build_parser(dataset_id: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract first prompts from Nemotron datasets.")
    if dataset_id is None:
        parser.add_argument("dataset", choices=sorted(DATASET_SPECS))
    parser.add_argument("--output", required=True, type=Path, help="Destination JSONL path.")
    parser.add_argument("--config", help="Optional Hugging Face dataset config to extract.")
    parser.add_argument("--split", help="Optional Hugging Face split to extract.")
    parser.add_argument("--limit", type=int, help="Optional maximum rows for smoke tests.")
    return parser


def main(dataset_id: str | None = None) -> None:
    parser = build_parser(dataset_id)
    args = parser.parse_args()
    selected_dataset = dataset_id or args.dataset
    count = write_prompts(
        dataset_id=selected_dataset,
        output_path=args.output,
        requested_config=args.config,
        requested_split=args.split,
        limit=args.limit,
    )
    print(f"Wrote {count} prompts to {args.output}", file=sys.stderr)
    sys.stderr.flush()
    sys.stdout.flush()
    # Some transient `uv --with datasets` environments abort in PyArrow/Python
    # finalizers after successful streaming. Exit directly once output is flushed.
    os._exit(0)


if __name__ == "__main__":
    main()
