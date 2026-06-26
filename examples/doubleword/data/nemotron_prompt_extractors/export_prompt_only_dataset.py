#!/usr/bin/env python3
"""Export one Nemotron dataset as prompt-only CSV and upload it to HF."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from nemotron_prompt_extraction import (
    DATASET_SPECS,
    DEFAULT_CONFIG,
    dataset_configs,
    dataset_data_files,
    dataset_splits,
    extract_dataset_metadata,
    extract_prompt,
    extract_system_for_source,
    extract_tools_for_source,
    load_jsonl_file,
    load_parquet_file,
    load_stream,
    PROMPT_COLUMNS,
    prompt_record_to_csv_row,
)


DEFAULT_OWNER = "jamesdborin"
DEFAULT_COLLECTION_TITLE = "Nemotron-Post-Training-v3 Prompt-Only"
DEFAULT_OUTPUT_ROOT = Path("/tmp/nemotron_prompt_only_exports")

SUMMARY_COLUMNS = [
    "dataset_id",
    "dataset_title",
    "prompt_only_repo_id",
    "config",
    "split",
    "source_path",
    "expected_original_rows",
    "observed_original_rows",
    "original_rows_for_delta",
    "original_rows_basis",
    "extracted_rows",
    "row_count_delta",
    "row_count_mismatch",
    "null_prompt_rows",
    "empty_prompt_rows",
    "failed_prompt_rows",
    "extraction_error_rows",
    "status",
    "upload_status",
    "started_at",
    "finished_at",
    "duration_seconds",
    "output_csv",
    "error",
]
SUMMARY_JSON_MARKER = "```json\n"

BAD_ROW_COLUMNS = [
    "dataset_id",
    "config",
    "split",
    "row_index",
    "reason",
    "prompt_source",
    "prompt_source_detail",
    "extraction_error",
]


@dataclass
class SourceSpec:
    config: str
    split: str
    path: str | None
    loader: Callable[[], Iterable[dict[str, Any]]]


@dataclass
class SourceStats:
    dataset_id: str
    dataset_title: str
    prompt_only_repo_id: str
    config: str
    split: str
    source_path: str
    expected_original_rows: int | None
    observed_original_rows: int
    extracted_rows: int
    null_prompt_rows: int
    empty_prompt_rows: int
    extraction_error_rows: int
    status: str
    started_at: str
    finished_at: str
    duration_seconds: float
    output_csv: str
    error: str = ""

    def to_row(self, upload_status: str) -> dict[str, Any]:
        original_rows = (
            self.expected_original_rows
            if self.expected_original_rows is not None
            else self.observed_original_rows
        )
        basis = "expected" if self.expected_original_rows is not None else "observed"
        delta = self.extracted_rows - original_rows
        failed_prompt_rows = self.null_prompt_rows + self.empty_prompt_rows
        return {
            "dataset_id": self.dataset_id,
            "dataset_title": self.dataset_title,
            "prompt_only_repo_id": self.prompt_only_repo_id,
            "config": self.config,
            "split": self.split,
            "source_path": self.source_path,
            "expected_original_rows": self.expected_original_rows,
            "observed_original_rows": self.observed_original_rows,
            "original_rows_for_delta": original_rows,
            "original_rows_basis": basis,
            "extracted_rows": self.extracted_rows,
            "row_count_delta": delta,
            "row_count_mismatch": bool(delta),
            "null_prompt_rows": self.null_prompt_rows,
            "empty_prompt_rows": self.empty_prompt_rows,
            "failed_prompt_rows": failed_prompt_rows,
            "extraction_error_rows": self.extraction_error_rows,
            "status": self.status,
            "upload_status": upload_status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": f"{self.duration_seconds:.3f}",
            "output_csv": self.output_csv,
            "error": self.error,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def local_dataset_name(dataset_id: str) -> str:
    return dataset_id.replace("/", "__")


def dataset_title(dataset_id: str) -> str:
    return dataset_id.rsplit("/", 1)[1]


def prompt_only_repo_id(dataset_id: str, owner: str) -> str:
    return f"{owner}/{dataset_title(dataset_id)}-prompt-only"


def configure_isolated_cache(output_root: Path, dataset_id: str) -> Path:
    cache_root = output_root / ".hf_cache" / local_dataset_name(dataset_id)
    os.environ["HF_DATASETS_CACHE"] = str(cache_root / "datasets")
    os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
    os.environ["NEMOTRON_PROMPT_JSONL_CACHE"] = str(cache_root / "jsonl")
    return cache_root


def pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def maybe_clear_stale_slot(slot_path: Path) -> None:
    owner_path = slot_path / "owner.json"
    try:
        payload = json.loads(owner_path.read_text())
        pid = int(payload["pid"])
    except Exception:
        return
    if not pid_is_alive(pid):
        shutil.rmtree(slot_path, ignore_errors=True)


@contextmanager
def concurrency_slot(
    semaphore_dir: Path | None,
    max_concurrent: int,
    label: str,
) -> Iterable[None]:
    if semaphore_dir is None or max_concurrent <= 0:
        yield
        return

    semaphore_dir.mkdir(parents=True, exist_ok=True)
    slot_path: Path | None = None
    last_wait_log = 0.0
    while slot_path is None:
        for index in range(max_concurrent):
            candidate = semaphore_dir / f"slot-{index:02d}"
            maybe_clear_stale_slot(candidate)
            try:
                candidate.mkdir()
            except FileExistsError:
                continue
            owner_payload = {
                "pid": os.getpid(),
                "label": label,
                "started_at": utc_now(),
            }
            (candidate / "owner.json").write_text(json.dumps(owner_payload) + "\n")
            slot_path = candidate
            log(f"acquired concurrency slot {candidate.name} for {label}")
            break
        if slot_path is None:
            now = time.time()
            if now - last_wait_log >= 60:
                log(f"waiting for a concurrency slot for {label}")
                last_wait_log = now
            time.sleep(5)

    try:
        yield
    finally:
        shutil.rmtree(slot_path, ignore_errors=True)
        log(f"released concurrency slot {slot_path.name} for {label}")


def expected_rows_by_source(dataset_id: str) -> dict[tuple[str, str], int]:
    try:
        from datasets import get_dataset_infos

        infos = get_dataset_infos(dataset_id)
    except Exception as exc:
        log(f"could not fetch expected row metadata for {dataset_id}: {exc}")
        return {}

    expected: dict[tuple[str, str], int] = {}
    for config_name, info in infos.items():
        config = DEFAULT_CONFIG if config_name == DEFAULT_CONFIG else config_name
        splits = getattr(info, "splits", None)
        if not splits:
            continue
        for split_name, split_info in splits.items():
            num_examples = getattr(split_info, "num_examples", None)
            if num_examples is not None:
                expected[(config, split_name)] = int(num_examples)
    return expected


def build_sources(
    dataset_id: str,
    requested_config: str | None,
    requested_split: str | None,
) -> list[SourceSpec]:
    data_files = dataset_data_files(dataset_id, requested_config, requested_split)
    if data_files:
        return [
            SourceSpec(
                config=data_file["config"],
                split=data_file["split"],
                path=data_file["path"],
                loader=lambda data_file=data_file: (
                    load_parquet_file(dataset_id, data_file["path"])
                    if data_file.get("format") == "parquet"
                    else load_jsonl_file(dataset_id, data_file["path"])
                ),
            )
            for data_file in data_files
        ]

    sources: list[SourceSpec] = []
    for config in dataset_configs(dataset_id, requested_config):
        config_label = config or DEFAULT_CONFIG
        for split in dataset_splits(dataset_id, config, requested_split):
            sources.append(
                SourceSpec(
                    config=config_label,
                    split=split,
                    path=None,
                    loader=lambda config=config, split=split: load_stream(
                        dataset_id, config, split
                    ),
                )
            )
    return sources


def write_status(dataset_dir: Path, **payload: Any) -> None:
    payload["updated_at"] = utc_now()
    tmp_path = dataset_dir / "status.json.tmp"
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(dataset_dir / "status.json")


def empty_to_csv(value: Any) -> Any:
    return "" if value is None else value


def markdown_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def write_markdown_table_header(handle: Any, columns: list[str]) -> None:
    handle.write("| " + " | ".join(columns) + " |\n")
    handle.write("| " + " | ".join("---" for _ in columns) + " |\n")


def write_markdown_table_row(
    handle: Any, columns: list[str], row: dict[str, Any]
) -> None:
    handle.write("| " + " | ".join(markdown_cell(row.get(column)) for column in columns) + " |\n")


def read_summary_rows(summary_path: Path) -> list[dict[str, Any]]:
    if not summary_path.exists():
        return []
    text = summary_path.read_text(encoding="utf-8")
    start = text.find(SUMMARY_JSON_MARKER)
    if start < 0:
        return []
    start += len(SUMMARY_JSON_MARKER)
    end = text.find("\n```", start)
    if end < 0:
        return []
    try:
        rows = json.loads(text[start:end])
    except json.JSONDecodeError:
        return []
    return rows if isinstance(rows, list) else []


def total_summary_row(rows: list[SourceStats], upload_status: str) -> dict[str, Any]:
    expected_values = [row.expected_original_rows for row in rows]
    expected_total = (
        sum(value for value in expected_values if value is not None)
        if all(value is not None for value in expected_values)
        else None
    )
    observed_total = sum(row.observed_original_rows for row in rows)
    extracted_total = sum(row.extracted_rows for row in rows)
    null_total = sum(row.null_prompt_rows for row in rows)
    empty_total = sum(row.empty_prompt_rows for row in rows)
    error_total = sum(row.extraction_error_rows for row in rows)
    original_total = expected_total if expected_total is not None else observed_total
    status = "ok" if all(row.status == "ok" for row in rows) else "needs_review"
    if any(row.status == "failed" for row in rows):
        status = "failed"
    elif any(row.status == "limited" for row in rows):
        status = "limited"

    first = rows[0]
    started = min(row.started_at for row in rows)
    finished = max(row.finished_at for row in rows)
    duration = sum(row.duration_seconds for row in rows)
    delta = extracted_total - original_total
    return {
        "dataset_id": first.dataset_id,
        "dataset_title": first.dataset_title,
        "prompt_only_repo_id": first.prompt_only_repo_id,
        "config": "__total__",
        "split": "__total__",
        "source_path": "",
        "expected_original_rows": expected_total,
        "observed_original_rows": observed_total,
        "original_rows_for_delta": original_total,
        "original_rows_basis": "expected" if expected_total is not None else "observed",
        "extracted_rows": extracted_total,
        "row_count_delta": delta,
        "row_count_mismatch": bool(delta),
        "null_prompt_rows": null_total,
        "empty_prompt_rows": empty_total,
        "failed_prompt_rows": null_total + empty_total,
        "extraction_error_rows": error_total,
        "status": status,
        "upload_status": upload_status,
        "started_at": started,
        "finished_at": finished,
        "duration_seconds": f"{duration:.3f}",
        "output_csv": first.output_csv,
        "error": "; ".join(row.error for row in rows if row.error),
    }


def write_summary(
    summary_path: Path,
    rows: list[SourceStats],
    upload_status: str,
) -> None:
    summary_rows = [row.to_row(upload_status) for row in rows]
    if rows:
        summary_rows.append(total_summary_row(rows, upload_status))
    normalized_rows = [
        {key: empty_to_csv(row.get(key)) for key in SUMMARY_COLUMNS}
        for row in summary_rows
    ]
    total = next(
        (row for row in normalized_rows if row.get("config") == "__total__"),
        {},
    )
    tmp_path = summary_path.with_suffix(".md.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write("# Extraction Summary\n\n")
        if total:
            handle.write("## Totals\n\n")
            write_markdown_table_header(handle, SUMMARY_COLUMNS)
            write_markdown_table_row(handle, SUMMARY_COLUMNS, total)
            handle.write("\n")
        handle.write("## Sources\n\n")
        write_markdown_table_header(handle, SUMMARY_COLUMNS)
        for row in normalized_rows:
            if row.get("config") != "__total__":
                write_markdown_table_row(handle, SUMMARY_COLUMNS, row)
        handle.write("\n## Machine-readable rows\n\n")
        handle.write(SUMMARY_JSON_MARKER)
        handle.write(json.dumps(normalized_rows, ensure_ascii=False, indent=2))
        handle.write("\n```\n")
    tmp_path.replace(summary_path)


def set_summary_upload_status(summary_path: Path, upload_status: str) -> None:
    rows = read_summary_rows(summary_path)
    if not rows:
        return
    for row in rows:
        row["upload_status"] = upload_status
    source_rows = [row for row in rows if row.get("config") != "__total__"]
    tmp_path = summary_path.with_suffix(".md.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write("# Extraction Summary\n\n")
        total = next((row for row in rows if row.get("config") == "__total__"), {})
        if total:
            handle.write("## Totals\n\n")
            write_markdown_table_header(handle, SUMMARY_COLUMNS)
            write_markdown_table_row(handle, SUMMARY_COLUMNS, total)
            handle.write("\n")
        handle.write("## Sources\n\n")
        write_markdown_table_header(handle, SUMMARY_COLUMNS)
        for row in source_rows:
            write_markdown_table_row(handle, SUMMARY_COLUMNS, row)
        handle.write("\n## Machine-readable rows\n\n")
        handle.write(SUMMARY_JSON_MARKER)
        handle.write(json.dumps(rows, ensure_ascii=False, indent=2))
        handle.write("\n```\n")
    tmp_path.replace(summary_path)


def refresh_aggregate_summary(output_root: Path) -> None:
    import fcntl

    lock_path = output_root / ".summary.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        rows: list[dict[str, str]] = []
        for summary_path in sorted(output_root.glob("*/summary.md")):
            for row in read_summary_rows(summary_path):
                if row.get("config") == "__total__":
                    rows.append(row)
        tmp_path = output_root / "summary.md.tmp"
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.write("# Aggregate Extraction Summary\n\n")
            write_markdown_table_header(handle, SUMMARY_COLUMNS)
            for row in rows:
                write_markdown_table_row(handle, SUMMARY_COLUMNS, row)
            handle.write("\n## Machine-readable rows\n\n")
            handle.write(SUMMARY_JSON_MARKER)
            handle.write(json.dumps(rows, ensure_ascii=False, indent=2))
            handle.write("\n```\n")
        tmp_path.replace(output_root / "summary.md")


def extract_dataset(
    dataset_id: str,
    owner: str,
    output_root: Path,
    requested_config: str | None,
    requested_split: str | None,
    limit: int | None,
    force: bool,
) -> tuple[list[SourceStats], Path, Path]:
    title = dataset_title(dataset_id)
    repo_id = prompt_only_repo_id(dataset_id, owner)
    dataset_dir = output_root / local_dataset_name(dataset_id)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_path = dataset_dir / "prompts.csv"
    summary_path = dataset_dir / "summary.md"
    bad_rows_path = dataset_dir / "null_or_empty_rows.md"

    if output_path.exists() and summary_path.exists() and not force and limit is None:
        log(f"using existing export for {dataset_id}: {output_path}")
        return [], output_path, summary_path

    expected_rows = expected_rows_by_source(dataset_id)
    sources = build_sources(dataset_id, requested_config, requested_split)
    if not sources:
        raise RuntimeError(f"No sources selected for {dataset_id}")

    log(f"selected {len(sources)} source(s) for {dataset_id}")
    tmp_output_path = output_path.with_suffix(".csv.tmp")
    partial_output_path = output_path.with_suffix(".csv.partial")
    rows: list[SourceStats] = []
    total_written = 0
    limit_reached = False

    with tmp_output_path.open("w", encoding="utf-8", newline="") as output_handle, (
        bad_rows_path.open("w", encoding="utf-8")
    ) as bad_handle:
        output_writer = csv.DictWriter(output_handle, fieldnames=PROMPT_COLUMNS)
        output_writer.writeheader()
        bad_handle.write("# Null Or Empty Prompt Rows\n\n")
        write_markdown_table_header(bad_handle, BAD_ROW_COLUMNS)

        for source in sources:
            started_time = time.time()
            started_at = utc_now()
            observed_rows = 0
            extracted_rows = 0
            null_rows = 0
            empty_rows = 0
            extraction_errors = 0
            status = "ok"
            error = ""
            log(
                f"extracting {dataset_id} config={source.config} "
                f"split={source.split} path={source.path or '-'}"
            )
            try:
                for row_index, row in enumerate(source.loader()):
                    if limit is not None and total_written >= limit:
                        limit_reached = True
                        status = "limited"
                        break

                    observed_rows += 1
                    extraction_error = None
                    try:
                        prompt, prompt_source, source_detail = extract_prompt(
                            row, dataset_id, source.config
                        )
                        system_prompt, system_source = extract_system_for_source(
                            row, prompt_source
                        )
                        tools, tools_source = extract_tools_for_source(row, prompt_source)
                    except Exception as exc:
                        prompt, prompt_source, source_detail = None, None, None
                        system_prompt, system_source = None, None
                        tools, tools_source = None, None
                        extraction_error = f"{type(exc).__name__}: {exc}"
                        extraction_errors += 1

                    record = {
                        "dataset": dataset_id,
                        "config": source.config,
                        "split": source.split,
                        "row_index": row_index,
                        "prompt": prompt,
                        "prompt_source": prompt_source,
                        "prompt_source_detail": source_detail,
                        "system_prompt": system_prompt,
                        "system_source": system_source,
                        "tools": tools,
                        "tools_source": tools_source,
                    }
                    record.update(extract_dataset_metadata(row, dataset_id))
                    if extraction_error is not None:
                        record["extraction_error"] = extraction_error
                    output_writer.writerow(prompt_record_to_csv_row(record))

                    reason = None
                    if prompt is None:
                        null_rows += 1
                        reason = "null_prompt"
                    elif isinstance(prompt, str) and not prompt.strip():
                        empty_rows += 1
                        reason = "empty_prompt"
                    if extraction_error is not None:
                        reason = "extraction_error"
                    if reason is not None:
                        write_markdown_table_row(
                            bad_handle,
                            BAD_ROW_COLUMNS,
                            {
                                "dataset_id": dataset_id,
                                "config": source.config,
                                "split": source.split,
                                "row_index": row_index,
                                "reason": reason,
                                "prompt_source": prompt_source,
                                "prompt_source_detail": source_detail,
                                "extraction_error": extraction_error or "",
                            },
                        )

                    extracted_rows += 1
                    total_written += 1
                    if extracted_rows == 1 or extracted_rows % 10000 == 0:
                        log(
                            f"{dataset_id} {source.config}/{source.split}: "
                            f"{extracted_rows} rows, null={null_rows}, "
                            f"empty={empty_rows}, errors={extraction_errors}"
                        )
            except Exception as exc:
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                log(f"FAILED {dataset_id} {source.config}/{source.split}: {error}")

            finished_at = utc_now()
            rows.append(
                SourceStats(
                    dataset_id=dataset_id,
                    dataset_title=title,
                    prompt_only_repo_id=repo_id,
                    config=source.config,
                    split=source.split,
                    source_path=source.path or "",
                    expected_original_rows=expected_rows.get(
                        (source.config, source.split)
                    ),
                    observed_original_rows=observed_rows,
                    extracted_rows=extracted_rows,
                    null_prompt_rows=null_rows,
                    empty_prompt_rows=empty_rows,
                    extraction_error_rows=extraction_errors,
                    status=status,
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_seconds=time.time() - started_time,
                    output_csv=str(output_path),
                    error=error,
                )
            )
            write_summary(summary_path, rows, upload_status="not_started")
            refresh_aggregate_summary(output_root)

            if status == "failed":
                output_handle.flush()
                tmp_output_path.replace(partial_output_path)
                raise RuntimeError(error)
            if limit_reached:
                log(f"limit reached for {dataset_id}: {limit}")
                break

    tmp_output_path.replace(output_path)
    write_summary(summary_path, rows, upload_status="pending")
    refresh_aggregate_summary(output_root)
    log(f"wrote {total_written} prompt rows to {output_path}")
    return rows, output_path, summary_path


def cleanup_cache(cache_root: Path) -> None:
    if cache_root.exists():
        shutil.rmtree(cache_root, ignore_errors=True)
        log(f"deleted isolated Hugging Face cache {cache_root}")


def total_row_from_summary(summary_path: Path) -> dict[str, str]:
    for row in read_summary_rows(summary_path):
        if row.get("config") == "__total__":
            return row
    return {}


def write_readme(
    dataset_dir: Path,
    dataset_id: str,
    owner: str,
    summary_path: Path,
) -> Path:
    title = dataset_title(dataset_id)
    repo_title = f"{title}-prompt-only"
    total = total_row_from_summary(summary_path)
    extracted_rows = total.get("extracted_rows", "")
    failed_prompt_rows = total.get("failed_prompt_rows", "")
    row_count_delta = total.get("row_count_delta", "")
    readme_path = dataset_dir / "README.md"
    readme = f"""---
pretty_name: "{repo_title}"
tags:
- nemotron
- prompt-only
- post-training
source_datasets:
- "{dataset_id}"
configs:
- config_name: default
  data_files:
  - split: train
    path: prompts.csv
---

# {repo_title}

Prompt-only extraction from `{dataset_id}`.

Files:

- `prompts.csv`: one prompt extraction record per source row. Records include
  `prompt`, separated `system_prompt`, and structured `tools` when the source row
  defines available tools. Nested values are JSON-encoded inside CSV cells.
- `summary.md`: source row counts, extracted row counts, count deltas, and failed prompt counts.
- `null_or_empty_rows.md`: row indexes where prompt extraction produced a null or empty prompt.

Summary:

- extracted rows: {extracted_rows}
- failed prompt rows: {failed_prompt_rows}
- row count delta: {row_count_delta}

Uploaded under `{owner}` from the Nemotron Post-Training v3 prompt extractor workflow.
"""
    readme_path.write_text(readme)
    return readme_path


def wait_for_owner_auth(owner: str, wait: bool):
    from huggingface_hub import HfApi

    api = HfApi()
    while True:
        try:
            whoami = api.whoami(token=True)
            name = whoami.get("name")
            if name == owner:
                log(f"authenticated to Hugging Face as {name}")
                return api
            message = f"authenticated as {name}; waiting for {owner}"
        except Exception as exc:
            message = f"not authenticated to Hugging Face: {exc}"

        if not wait:
            raise RuntimeError(message)
        log(f"{message}; rechecking in 60 seconds")
        time.sleep(60)


def upload_dataset(
    dataset_dir: Path,
    dataset_id: str,
    owner: str,
    collection_title: str,
    wait_for_auth: bool,
) -> None:
    api = wait_for_owner_auth(owner, wait_for_auth)
    repo_id = prompt_only_repo_id(dataset_id, owner)
    log(f"creating dataset repo {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    log(f"uploading {dataset_dir} to {repo_id}")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=dataset_dir,
        allow_patterns=[
            "README.md",
            "prompts.csv",
            "summary.md",
            "null_or_empty_rows.md",
        ],
        commit_message=f"Upload prompt-only extract for {dataset_title(dataset_id)}",
    )
    log(f"ensuring collection {owner}/{collection_title}")
    collection = api.create_collection(
        title=collection_title,
        namespace=owner,
        exists_ok=True,
    )
    api.add_collection_item(
        collection_slug=collection.slug,
        item_id=repo_id,
        item_type="dataset",
        exists_ok=True,
    )
    log(f"added {repo_id} to collection {collection.slug}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one Nemotron dataset to prompt-only CSV."
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_SPECS))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--collection-title", default=DEFAULT_COLLECTION_TITLE)
    parser.add_argument("--config")
    parser.add_argument("--split")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--wait-for-auth", action="store_true")
    parser.add_argument("--cleanup-cache", action="store_true", default=True)
    parser.add_argument("--no-cleanup-cache", dest="cleanup_cache", action="store_false")
    parser.add_argument("--semaphore-dir", type=Path)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--upload-semaphore-dir", type=Path)
    parser.add_argument("--max-upload-concurrent", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_root / local_dataset_name(args.dataset)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    cache_root = configure_isolated_cache(output_root, args.dataset)
    summary_path = dataset_dir / "summary.md"

    write_status(
        dataset_dir,
        dataset=args.dataset,
        phase="starting",
        output_root=str(output_root),
        repo_id=prompt_only_repo_id(args.dataset, args.owner),
    )

    try:
        with concurrency_slot(
            args.semaphore_dir,
            args.max_concurrent,
            f"extract {args.dataset}",
        ):
            write_status(dataset_dir, dataset=args.dataset, phase="extracting")
            rows, output_path, summary_path = extract_dataset(
                dataset_id=args.dataset,
                owner=args.owner,
                output_root=output_root,
                requested_config=args.config,
                requested_split=args.split,
                limit=args.limit,
                force=args.force,
            )
            if not rows and summary_path.exists():
                refresh_aggregate_summary(output_root)

        if args.cleanup_cache:
            cleanup_cache(cache_root)

        write_readme(dataset_dir, args.dataset, args.owner, summary_path)

        if args.skip_upload:
            set_summary_upload_status(summary_path, "skipped")
            refresh_aggregate_summary(output_root)
            write_status(dataset_dir, dataset=args.dataset, phase="upload_skipped")
            log("upload skipped")
            return 0

        set_summary_upload_status(summary_path, "waiting_for_auth")
        refresh_aggregate_summary(output_root)
        with concurrency_slot(
            args.upload_semaphore_dir,
            args.max_upload_concurrent,
            f"upload {args.dataset}",
        ):
            write_status(dataset_dir, dataset=args.dataset, phase="uploading")
            upload_dataset(
                dataset_dir=dataset_dir,
                dataset_id=args.dataset,
                owner=args.owner,
                collection_title=args.collection_title,
                wait_for_auth=args.wait_for_auth,
            )
        set_summary_upload_status(summary_path, "complete")
        refresh_aggregate_summary(output_root)
        if args.cleanup_cache:
            cleanup_cache(cache_root)
        write_status(dataset_dir, dataset=args.dataset, phase="complete")
        return 0
    except Exception as exc:
        if summary_path.exists():
            set_summary_upload_status(summary_path, "failed")
            refresh_aggregate_summary(output_root)
        write_status(
            dataset_dir,
            dataset=args.dataset,
            phase="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        log(f"ERROR: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
