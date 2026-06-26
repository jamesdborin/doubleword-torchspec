#!/usr/bin/env python3
"""Dependency-light smoke tests for prompt normalization."""

import csv
from tempfile import TemporaryDirectory
from pathlib import Path

from nemotron_prompt_extraction import (
    extract_prompt,
    extract_system_for_source,
    extract_tools_for_source,
    write_prompts,
)


def test_responses_input_first_user() -> None:
    row = {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Extract this prompt."},
                {"role": "assistant", "content": "No."},
                {"role": "user", "content": "Ignore later turns."},
            ]
        }
    }

    prompt, source, detail = extract_prompt(
        row, "nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2"
    )
    system_prompt, system_source = extract_system_for_source(row, source)

    assert prompt == "Extract this prompt."
    assert source == "responses_input"
    assert detail == "first_user_message"
    assert system_prompt == "Be concise."
    assert system_source == "responses_create_params.input"


def test_messages_first_user() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "Rules"},
            {"role": "user", "content": [{"type": "text", "text": "First user"}]},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Second user"},
        ]
    }

    prompt, source, detail = extract_prompt(row, "nvidia/Nemotron-SFT-Agentic-v2")
    system_prompt, system_source = extract_system_for_source(row, source)

    assert prompt == "First user"
    assert source == "messages"
    assert detail == "first_user_message"
    assert system_prompt == "Rules"
    assert system_source == "messages"


def test_nested_genrm_messages() -> None:
    row = {"messages": [[{"role": "user", "content": "Judge prompt"}]]}

    prompt, source, detail = extract_prompt(row, "nvidia/Nemotron-RLHF-GenRM-v1")

    assert prompt == "Judge prompt"
    assert source == "nested_messages"
    assert detail == "first_user_message"


def test_stringified_messages() -> None:
    row = {
        "messages": '[{"role": "system", "content": "Rules"}, '
        '{"role": "user", "content": "Fix the issue."}]'
    }

    prompt, source, detail = extract_prompt(row, "nvidia/Nemotron-SWE-v1")
    system_prompt, system_source = extract_system_for_source(row, source)

    assert prompt == "Fix the issue."
    assert source == "messages"
    assert detail == "first_user_message"
    assert system_prompt == "Rules"
    assert system_source == "messages"


def test_messages_without_initial_context_stay_user_only() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Solve this."},
            {"role": "assistant", "content": "Answer"},
        ]
    }

    prompt, source, detail = extract_prompt(row, "nvidia/Nemotron-SFT-Math-v4")

    assert prompt == "Solve this."
    assert source == "messages"
    assert detail == "first_user_message"


def test_empty_first_user_uses_first_non_empty_user() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "Rules"},
            {"role": "user", "content": None},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "Real prompt"},
        ]
    }

    prompt, source, detail = extract_prompt(
        row, "nvidia/Nemotron-SFT-Instruction-Following-Chat-v3"
    )
    system_prompt, system_source = extract_system_for_source(row, source)

    assert prompt == "Real prompt"
    assert source == "messages"
    assert detail == "first_user_message"
    assert system_prompt == "Rules"
    assert system_source == "messages"


def test_multiple_initial_system_messages_are_separate() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "System rules"},
            {"role": "developer", "content": "Developer rules"},
            {"role": "user", "content": "Do the task."},
        ]
    }

    prompt, source, detail = extract_prompt(row, "nvidia/Nemotron-SFT-Agentic-v2")
    system_prompt, system_source = extract_system_for_source(row, source)

    assert prompt == "Do the task."
    assert detail == "first_user_message"
    assert system_prompt == "System rules\n\nDeveloper rules"
    assert system_source == "messages"


def test_messages_include_top_level_tools() -> None:
    row = {
        "messages": [{"role": "user", "content": "Edit the file."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "write",
                    "description": "Write a file.",
                    "parameters": {"type": "object"},
                },
            }
        ],
    }

    prompt, source, detail = extract_prompt(row, "nvidia/Nemotron-SFT-CUDA-v1")
    tools, tools_source = extract_tools_for_source(row, source)

    assert prompt == "Edit the file."
    assert source == "messages"
    assert detail == "first_user_message"
    assert tools[0]["function"]["name"] == "write"
    assert tools_source == "tools"


def test_responses_input_include_response_tools() -> None:
    row = {
        "responses_create_params": {
            "input": [{"role": "user", "content": "Use the search tool."}],
            "tools": [{"type": "function", "name": "search"}],
        }
    }

    prompt, source, detail = extract_prompt(
        row, "nvidia/Nemotron-RL-Science-v1"
    )
    tools, tools_source = extract_tools_for_source(row, source)

    assert prompt == "Use the search tool."
    assert source == "responses_input"
    assert detail == "first_user_message"
    assert tools == [{"type": "function", "name": "search"}]
    assert tools_source == "responses_create_params.tools"


def test_structured_outputs_writer_includes_schema_str() -> None:
    import nemotron_prompt_extraction

    rows = [
        {
            "responses_create_params": {
                "input": [{"role": "user", "content": "Return JSON."}],
            },
            "schema_str": '{"type": "object", "properties": {"answer": {"type": "string"}}}',
        }
    ]
    original_load_parquet_file = nemotron_prompt_extraction.load_parquet_file
    nemotron_prompt_extraction.load_parquet_file = lambda _dataset_id, _path: iter(rows)
    try:
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "prompts.csv"
            count = write_prompts(
                "nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2",
                output_path,
                requested_split="direct_generation",
            )
            with output_path.open(encoding="utf-8", newline="") as handle:
                record = next(csv.DictReader(handle))
    finally:
        nemotron_prompt_extraction.load_parquet_file = original_load_parquet_file

    assert count == 1
    assert record["prompt"] == "Return JSON."
    assert record["schema_str"] == rows[0]["schema_str"]


if __name__ == "__main__":
    test_responses_input_first_user()
    test_messages_first_user()
    test_nested_genrm_messages()
    test_stringified_messages()
    test_messages_without_initial_context_stay_user_only()
    test_empty_first_user_uses_first_non_empty_user()
    test_multiple_initial_system_messages_are_separate()
    test_messages_include_top_level_tools()
    test_responses_input_include_response_tools()
    test_structured_outputs_writer_includes_schema_str()
