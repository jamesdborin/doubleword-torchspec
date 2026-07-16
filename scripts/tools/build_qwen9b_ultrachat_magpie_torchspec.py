#!/usr/bin/env python3
"""Join prompt JSONL and Qwen output JSONL into a TorchSpec-compatible dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_inputs(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            custom_id = row.get("custom_id")
            if not custom_id:
                raise ValueError(f"{path}:{line_no}: missing custom_id")
            body = row.get("body") or {}
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{path}:{line_no}: missing body.messages")
            if custom_id in rows:
                raise ValueError(f"{path}:{line_no}: duplicate custom_id {custom_id}")
            rows[custom_id] = row
    return rows


def assistant_content(row: dict[str, Any]) -> str | None:
    response = row.get("response") or {}
    if response.get("status_code") != 200:
        return None
    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return None
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    return content


def first_user_turn(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


def build_source(
    source_name: str,
    input_path: Path,
    output_path: Path,
    out_handle,
    start_index: int,
) -> dict[str, int]:
    inputs = read_inputs(input_path)
    seen_outputs: set[str] = set()
    written = 0
    bad_outputs = 0
    missing_inputs = 0
    duplicate_outputs = 0
    with output_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            custom_id = row.get("custom_id") or row.get("id")
            if not custom_id:
                bad_outputs += 1
                continue
            if custom_id in seen_outputs:
                duplicate_outputs += 1
                continue
            seen_outputs.add(custom_id)
            request = inputs.get(custom_id)
            if request is None:
                missing_inputs += 1
                continue
            content = assistant_content(row)
            if content is None:
                bad_outputs += 1
                continue
            messages = []
            for message in request["body"]["messages"]:
                role = message.get("role")
                msg_content = message.get("content")
                if isinstance(role, str) and isinstance(msg_content, str):
                    messages.append({"role": role, "content": msg_content})
            if not messages:
                bad_outputs += 1
                continue
            messages.append({"role": "assistant", "content": content})
            row_id = f"{source_name}_{start_index + written:06d}"
            out_handle.write(
                json.dumps(
                    {
                        "id": row_id,
                        "question_id": row_id,
                        "category": source_name,
                        "sub_category": source_name,
                        "source": source_name,
                        "custom_id": custom_id,
                        "conversations": messages,
                        "turns": [first_user_turn(messages)],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            written += 1
    return {
        "input_rows": len(inputs),
        "output_rows": len(seen_outputs) + duplicate_outputs,
        "written": written,
        "bad_outputs": bad_outputs,
        "missing_inputs": missing_inputs,
        "duplicate_outputs": duplicate_outputs,
        "unused_inputs": len(set(inputs) - seen_outputs),
    }


def validate(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            conversations = row.get("conversations")
            turns = row.get("turns")
            if not isinstance(conversations, list) or len(conversations) < 2:
                raise ValueError(f"{path}:{line_no}: invalid conversations")
            if conversations[-1].get("role") != "assistant":
                raise ValueError(f"{path}:{line_no}: final message is not assistant")
            if not isinstance(turns, list) or not turns or not isinstance(turns[0], str):
                raise ValueError(f"{path}:{line_no}: invalid turns")
            if not row.get("id") or not row.get("question_id") or not row.get("sub_category"):
                raise ValueError(f"{path}:{line_no}: missing id/question_id/sub_category")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/qwen9B-500k-ultrachat-magpie"))
    args = parser.parse_args()

    root = args.root
    raw = root / "raw"
    out_dir = root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "train.jsonl"
    summary_path = root / "summary.json"

    summary: dict[str, dict[str, int]] = {}
    with output.open("w", encoding="utf-8") as out:
        offset = 0
        for source_name, input_name, output_name in [
            ("magpie", "magpie_input.jsonl", "magpie_output.jsonl"),
            ("ultrachat", "ultrachat_input.jsonl", "ultrachat_output.jsonl"),
        ]:
            stats = build_source(source_name, raw / input_name, raw / output_name, out, offset)
            summary[source_name] = stats
            offset += stats["written"]

    total = validate(output)
    payload = {"output": str(output), "total_rows": total, "sources": summary}
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
