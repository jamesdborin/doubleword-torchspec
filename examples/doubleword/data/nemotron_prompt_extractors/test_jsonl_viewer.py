#!/usr/bin/env python3
"""Smoke tests for the local prompt JSONL viewer."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from jsonl_viewer import build_index, read_page


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_viewer_discovers_prompt_jsonl_variants() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        write_jsonl(root / "nvidia__Dataset-A" / "prompts.jsonl", [{"prompt": "a"}])
        write_jsonl(root / "nvidia__Dataset-B" / "prompt.jsonl", [{"prompt": "b"}])

        index = build_index(root, include_counts=True)

    assert [item["label"] for item in index] == ["nvidia/Dataset-A", "nvidia/Dataset-B"]
    assert [item["rows"] for item in index] == [1, 1]


def test_read_page_returns_headings_and_rows() -> None:
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "category" / "prompts.jsonl"
        write_jsonl(
            path,
            [
                {"dataset": "a", "prompt": "one"},
                {"dataset": "a", "prompt": "two", "tools": [{"name": "search"}]},
                {"dataset": "a", "prompt": "three"},
            ],
        )

        page = read_page(path, page=1, page_size=2)

    assert page["total_rows"] == 3
    assert page["total_pages"] == 2
    assert page["headings"] == ["dataset", "prompt", "tools"]
    assert [row["line"] for row in page["records"]] == [1, 2]
    assert page["records"][1]["values"]["tools"] == [{"name": "search"}]
