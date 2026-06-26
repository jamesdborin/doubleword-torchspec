#!/usr/bin/env python3
"""Tests for OpenAI Batch conversion from prompt-only JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from convert_prompts_to_openai_batch import convert_file


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def assert_valid_batch_request(request: dict[str, object]) -> None:
    assert request["method"] == "POST"
    assert request["url"] == "/v1/chat/completions"
    body = request["body"]
    assert isinstance(body, dict)
    assert body["model"] == "gpt-test"
    assert body["messages"]


def test_convert_prompt_system_tools_and_skips_null_rows() -> None:
    rows = [
        {
            "dataset": "nvidia/Nemotron-3-Nano-RL-Training-Blend",
            "config": "default",
            "split": "train",
            "row_index": 1,
            "prompt": "Plain prompt",
            "prompt_source": "responses_input",
            "prompt_source_detail": "first_user_message",
            "system_prompt": None,
            "tools": None,
        },
        {
            "dataset": "nvidia/Nemotron-3-Nano-RL-Training-Blend",
            "config": "default",
            "split": "train",
            "row_index": 2,
            "prompt": "Use a tool",
            "prompt_source": "responses_input",
            "prompt_source_detail": "first_user_message",
            "system_prompt": "System rules",
            "tools": [
                {
                    "type": "function",
                    "name": "search",
                    "description": "Search docs.",
                    "parameters": {"type": "object"},
                    "strict": False,
                }
            ],
        },
        {
            "dataset": "nvidia/Nemotron-3-Nano-RL-Training-Blend",
            "config": "default",
            "split": "train",
            "row_index": 3,
            "prompt": None,
            "prompt_source": None,
            "prompt_source_detail": "no_prompt_found",
            "system_prompt": None,
            "tools": None,
        },
    ]

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        input_path = root / "prompts.jsonl"
        output_dir = root / "batch"
        write_jsonl(input_path, rows)

        stats = convert_file(
            input_path=input_path,
            output_dir=output_dir,
            model="gpt-test",
            max_completion_tokens=100,
            temperature=0.2,
            extra_body={"seed": 123},
        )

        requests = read_jsonl(output_dir / "batch_000.jsonl")
        manifest = read_jsonl(output_dir / "manifest.jsonl")
        skipped = read_jsonl(output_dir / "skipped_rows.jsonl")
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert stats.total_rows == 3
    assert stats.written_requests == 2
    assert stats.skipped_rows == 1
    assert summary["written_requests"] == 2
    assert summary["skipped_rows"] == 1

    assert len(requests) == 2
    for request in requests:
        assert_valid_batch_request(request)

    first_body = requests[0]["body"]
    assert first_body["messages"] == [{"role": "user", "content": "Plain prompt"}]
    assert first_body["max_completion_tokens"] == 100
    assert first_body["temperature"] == 0.2
    assert first_body["seed"] == 123
    assert "tools" not in first_body

    second_body = requests[1]["body"]
    assert second_body["messages"] == [
        {"role": "system", "content": "System rules"},
        {"role": "user", "content": "Use a tool"},
    ]
    assert second_body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search docs.",
                "parameters": {"type": "object"},
                "strict": False,
            },
        }
    ]

    assert [item["custom_id"] for item in manifest] == [
        "nano-rl-training-blend_default_train_000001",
        "nano-rl-training-blend_default_train_000002",
    ]
    assert skipped == [
        {
            "dataset": "nvidia/Nemotron-3-Nano-RL-Training-Blend",
            "config": "default",
            "split": "train",
            "row_index": 3,
            "reason": "null_or_empty_prompt",
            "prompt_source": None,
            "prompt_source_detail": "no_prompt_found",
            "extraction_error": None,
        }
    ]


def test_shards_by_request_count_and_byte_limit() -> None:
    rows = [
        {
            "dataset": "nvidia/Nemotron-3-Nano-RL-Training-Blend",
            "config": "default",
            "split": "train",
            "row_index": index,
            "prompt": f"Prompt {index}",
        }
        for index in range(5)
    ]

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        input_path = root / "prompts.jsonl"
        output_dir = root / "batch"
        write_jsonl(input_path, rows)

        stats = convert_file(
            input_path=input_path,
            output_dir=output_dir,
            model="gpt-test",
            max_requests_per_shard=2,
            max_bytes_per_shard=400,
        )

        shard_paths = sorted(output_dir.glob("batch_*.jsonl"))
        seen_ids: set[str] = set()
        for shard in shard_paths:
            assert shard.stat().st_size <= 400
            requests = read_jsonl(shard)
            assert len(requests) <= 2
            for request in requests:
                assert_valid_batch_request(request)
                custom_id = str(request["custom_id"])
                assert custom_id not in seen_ids
                seen_ids.add(custom_id)

    assert stats.written_requests == 5
    assert len(stats.shards) == 3
    assert [shard.requests for shard in stats.shards] == [2, 2, 1]
    assert len(seen_ids) == 5
