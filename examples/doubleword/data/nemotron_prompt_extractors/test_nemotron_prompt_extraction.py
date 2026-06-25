#!/usr/bin/env python3
"""Dependency-light smoke tests for prompt normalization."""

from nemotron_prompt_extraction import extract_prompt


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

    assert prompt == "Extract this prompt."
    assert source == "responses_input"
    assert detail == "first_user_message"


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

    assert prompt == "First user"
    assert source == "messages"
    assert detail == "first_user_message"


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

    assert prompt == "Fix the issue."
    assert source == "messages"
    assert detail == "first_user_message"


if __name__ == "__main__":
    test_responses_input_first_user()
    test_messages_first_user()
    test_nested_genrm_messages()
    test_stringified_messages()
