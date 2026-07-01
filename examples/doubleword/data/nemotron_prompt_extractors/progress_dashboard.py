#!/usr/bin/env python3
"""Live dashboard for Nemotron prompt-only export progress."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        "NEMOTRON_PROMPT_OUTPUT_ROOT",
        (
            "/workspace/nemotron_prompt_only_exports"
            if Path("/workspace").is_dir()
            else "/tmp/nemotron_prompt_only_exports"
        ),
    )
)
DEFAULT_SURVEY = Path(__file__).resolve().parent.parent / "nemotron_post_training_v3_dataset_survey.md"
SUMMARY_JSON_MARKER = "```json\n"
TRACKER_FILENAME = ".dashboard_progress.json"

LOG_ROWS_RE = re.compile(r":\s+([0-9]+)\s+rows,\s+null=([0-9]+),\s+empty=([0-9]+),\s+errors=([0-9]+)")
LOG_PROGRESS_RE = re.compile(
    r"\]\s+(\S+)\s+([^/\s]+)/([^:]+):\s+"
    r"observed=([0-9]+),\s+written=([0-9]+),\s+skipped=([0-9]+),\s+"
    r"null=([0-9]+),\s+empty=([0-9]+),\s+errors=([0-9]+)"
)
LOG_EXTRACT_RE = re.compile(r"extracting\s+(\S+)\s+config=(\S+)\s+split=(\S+)\s+path=(.*)$")
LOG_WAIT_RE = re.compile(r"waiting for a concurrency slot for (extract|upload)\s+(.+)$")
LOG_UPLOAD_RE = re.compile(r"(uploading|creating dataset repo|added .+ to collection|released concurrency slot .* upload)")
SURVEY_ROW_RE = re.compile(r"^\|\s*\d+\s*\|\s*\[([^\]]+)\]\(https://huggingface\.co/datasets/([^)]+)\)\s*\|\s*([^|]+)\|")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


def read_summary_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
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


def parse_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    match = re.search(r"\d+", text)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def plausible_row_count(value: Any, expected: int | None) -> int | None:
    parsed = parse_int(value)
    if parsed is None:
        return None
    if expected is not None and parsed > expected:
        return None
    return parsed


def parse_survey_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        match = SURVEY_ROW_RE.match(line)
        if not match:
            continue
        dataset_id = match.group(2)
        samples = match.group(3)
        value = parse_int(samples)
        if value is not None:
            counts[dataset_id] = value
    return counts


def local_dataset_name(dataset_id: str) -> str:
    return dataset_id.replace("/", "__")


def dataset_title(dataset_id: str) -> str:
    return dataset_id.rsplit("/", 1)[1]


def tail_lines(path: Path, max_bytes: int = 128 * 1024) -> list[str]:
    if not path.exists():
        return []
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > max_bytes:
            handle.seek(size - max_bytes)
            handle.readline()
        data = handle.read()
    return data.decode("utf-8", errors="replace").splitlines()


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def line_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        output = subprocess.check_output(["wc", "-l", str(path)], text=True, timeout=5)
        lines = int(output.strip().split()[0])
        if path.suffix == ".csv":
            return max(0, lines - 1)
        return lines
    except Exception:
        return None


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def human_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024
    return f"{size}B"


def read_active_slots(root: Path) -> dict[str, dict[str, Any]]:
    active: dict[str, dict[str, Any]] = {}
    for owner_path in sorted((root / "semaphore").glob("*/*/owner.json")):
        payload = read_json(owner_path)
        label = str(payload.get("label", ""))
        match = re.match(r"(extract|upload)\s+(.+)$", label)
        if not match:
            continue
        phase, dataset_id = match.groups()
        active[dataset_id] = {
            "phase": phase,
            "slot": owner_path.parent.name,
            "pid": payload.get("pid"),
            "started_at": payload.get("started_at"),
        }
    return active


def read_log_state(root: Path, dataset_id: str) -> dict[str, Any]:
    log_path = root / "logs" / f"{local_dataset_name(dataset_id)}.log"
    lines = tail_lines(log_path)
    state: dict[str, Any] = {"last_line": lines[-1] if lines else "", "last_rows": None}
    for line in lines:
        if "[requeue " in line or "[reschedule " in line:
            state.pop("last_rows", None)
            state.pop("last_null", None)
            state.pop("last_empty", None)
            state.pop("last_errors", None)
            state.pop("current_config", None)
            state.pop("current_split", None)
            state.pop("current_path", None)
            state.pop("waiting_phase", None)
            state.pop("upload_activity", None)
        rows_match = LOG_ROWS_RE.search(line)
        if rows_match:
            state["last_rows"] = int(rows_match.group(1))
            state["last_null"] = int(rows_match.group(2))
            state["last_empty"] = int(rows_match.group(3))
            state["last_errors"] = int(rows_match.group(4))
        progress_match = LOG_PROGRESS_RE.search(line)
        if progress_match:
            state["last_rows"] = int(progress_match.group(5))
            state["last_observed"] = int(progress_match.group(4))
            state["last_skipped"] = int(progress_match.group(6))
            state["last_null"] = int(progress_match.group(7))
            state["last_empty"] = int(progress_match.group(8))
            state["last_errors"] = int(progress_match.group(9))
        extracting = LOG_EXTRACT_RE.search(line)
        if extracting:
            state["current_config"] = extracting.group(2)
            state["current_split"] = extracting.group(3)
            state["current_path"] = extracting.group(4)
        waiting = LOG_WAIT_RE.search(line)
        if waiting:
            state["waiting_phase"] = waiting.group(1)
        if LOG_UPLOAD_RE.search(line):
            state["upload_activity"] = line
    return state


def source_key(config: str, split: str, path: str = "") -> str:
    return "\x1f".join([config, split, path])


def merge_source(
    sources: dict[str, dict[str, Any]],
    *,
    config: str,
    split: str,
    path: str = "",
    observed_rows: int | None = None,
    extracted_rows: int | None = None,
    expected_rows: int | None = None,
    null_rows: int | None = None,
    empty_rows: int | None = None,
    error_rows: int | None = None,
    status: str = "",
) -> None:
    key = source_key(config, split, path)
    row = sources.setdefault(
        key,
        {
            "config": config,
            "split": split,
            "path": path,
            "observed_rows": 0,
            "extracted_rows": 0,
            "expected_rows": None,
            "null_rows": 0,
            "empty_rows": 0,
            "error_rows": 0,
            "status": "",
        },
    )
    for field, value in [
        ("observed_rows", observed_rows),
        ("extracted_rows", extracted_rows),
        ("null_rows", null_rows),
        ("empty_rows", empty_rows),
        ("error_rows", error_rows),
    ]:
        if value is not None:
            row[field] = max(parse_int(row.get(field)) or 0, value)
    if expected_rows is not None:
        current_expected = parse_int(row.get("expected_rows"))
        row["expected_rows"] = max(current_expected or 0, expected_rows)
    if status:
        row["status"] = status


def source_totals(sources: dict[str, dict[str, Any]]) -> dict[str, int | None]:
    extracted = sum(parse_int(row.get("extracted_rows")) or 0 for row in sources.values())
    observed = sum(parse_int(row.get("observed_rows")) or 0 for row in sources.values())
    expected_values = [parse_int(row.get("expected_rows")) for row in sources.values()]
    expected = (
        sum(value for value in expected_values if value is not None)
        if expected_values and all(value is not None for value in expected_values)
        else None
    )
    null_rows = sum(parse_int(row.get("null_rows")) or 0 for row in sources.values())
    empty_rows = sum(parse_int(row.get("empty_rows")) or 0 for row in sources.values())
    error_rows = sum(parse_int(row.get("error_rows")) or 0 for row in sources.values())
    return {
        "extracted_rows": extracted,
        "observed_rows": observed,
        "expected_rows": expected,
        "failed_prompt_rows": null_rows + empty_rows,
        "extraction_error_rows": error_rows,
    }


def read_log_progress(root: Path, dataset_id: str) -> dict[str, Any]:
    log_path = root / "logs" / f"{local_dataset_name(dataset_id)}.log"
    sources: dict[str, dict[str, Any]] = {}
    current_config = ""
    current_split = ""
    current_path = ""
    last_line = ""
    for line in read_lines(log_path):
        last_line = line
        extracting = LOG_EXTRACT_RE.search(line)
        if extracting:
            current_config = extracting.group(2)
            current_split = extracting.group(3)
            current_path = "" if extracting.group(4) == "-" else extracting.group(4)
            merge_source(
                sources,
                config=current_config,
                split=current_split,
                path=current_path,
            )
            continue
        progress = LOG_PROGRESS_RE.search(line)
        if not progress:
            continue
        config = progress.group(2)
        split = progress.group(3)
        path = current_path if config == current_config and split == current_split else ""
        merge_source(
            sources,
            config=config,
            split=split,
            path=path,
            observed_rows=int(progress.group(4)),
            extracted_rows=int(progress.group(5)),
            null_rows=int(progress.group(7)),
            empty_rows=int(progress.group(8)),
            error_rows=int(progress.group(9)),
        )
    totals = source_totals(sources)
    return {
        "sources": sources,
        "totals": totals,
        "current_config": current_config,
        "current_split": current_split,
        "current_path": current_path,
        "last_line": last_line,
    }


def merge_summary_progress(summary_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        if row.get("config") == "__total__":
            continue
        merge_source(
            sources,
            config=str(row.get("config") or ""),
            split=str(row.get("split") or ""),
            path=str(row.get("source_path") or ""),
            observed_rows=parse_int(row.get("observed_original_rows")),
            extracted_rows=parse_int(row.get("extracted_rows")),
            expected_rows=parse_int(row.get("original_rows_for_delta")),
            null_rows=parse_int(row.get("null_prompt_rows")),
            empty_rows=parse_int(row.get("empty_prompt_rows")),
            error_rows=parse_int(row.get("extraction_error_rows")),
            status=str(row.get("status") or ""),
        )
    return sources


def read_tracker(root: Path) -> dict[str, Any]:
    payload = read_json(root / TRACKER_FILENAME)
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict):
        return {"version": 1, "datasets": {}}
    return {"version": 1, "datasets": datasets}


def update_tracker(root: Path, datasets: list[dict[str, Any]]) -> None:
    payload = read_tracker(root)
    tracked = payload["datasets"]
    for dataset in datasets:
        dataset_id = dataset["dataset_id"]
        current = tracked.setdefault(dataset_id, {"sources": {}})
        current["updated_at"] = utc_now()
        current["state"] = dataset.get("state") or current.get("state") or ""
        current["phase"] = dataset.get("phase") or current.get("phase") or ""
        current["upload_status"] = dataset.get("upload_status") or current.get("upload_status") or ""
        current["extract_status"] = dataset.get("extract_status") or current.get("extract_status") or ""
        for field in ["expected_rows", "extracted_rows", "failed_prompt_rows", "row_delta"]:
            value = parse_int(dataset.get(field))
            if value is not None:
                existing = parse_int(current.get(field)) or 0
                expected = parse_int(dataset.get("expected_rows")) or parse_int(current.get("expected_rows"))
                if field == "extracted_rows" and expected is not None and existing > expected >= value:
                    current[field] = value
                else:
                    current[field] = max(existing, value)
        sources = current.setdefault("sources", {})
        for key, row in (dataset.get("_tracker_sources") or {}).items():
            existing = sources.setdefault(key, {})
            existing.update({k: v for k, v in row.items() if k in {"config", "split", "path", "status"}})
            for field in [
                "observed_rows",
                "extracted_rows",
                "expected_rows",
                "null_rows",
                "empty_rows",
                "error_rows",
            ]:
                value = parse_int(row.get(field))
                if value is not None:
                    existing[field] = max(parse_int(existing.get(field)) or 0, value)
    payload["updated_at"] = utc_now()
    write_json_atomic(root / TRACKER_FILENAME, payload)


def build_snapshot(output_root: Path, survey_path: Path) -> dict[str, Any]:
    manifest_rows = read_csv(output_root / "dataset_manifest.csv")
    summary_rows = {
        row["dataset_id"]: row
        for row in read_summary_rows(output_root / "summary.md")
        if row.get("config") == "__total__"
    }
    survey_counts = parse_survey_counts(survey_path)
    active = read_active_slots(output_root)
    tracker = read_tracker(output_root)
    tracked_datasets = tracker["datasets"]

    datasets: list[dict[str, Any]] = []
    for manifest in manifest_rows:
        dataset_id = manifest["dataset_id"]
        local_dir = Path(manifest["local_output_dir"])
        local_summary_rows = read_summary_rows(local_dir / "summary.md")
        local_total = next(
            (row for row in local_summary_rows if row.get("config") == "__total__"),
            {},
        )
        summary = summary_rows.get(dataset_id, {}) or local_total
        status = read_json(local_dir / "status.json")
        log_state = read_log_state(output_root, dataset_id)
        log_progress = read_log_progress(output_root, dataset_id)
        tracked = tracked_datasets.get(dataset_id, {})
        prompt_path = local_dir / "prompts.csv"
        tmp_prompt_path = local_dir / "prompts.csv.tmp"
        bad_rows_path = local_dir / "null_or_empty_rows.md"

        tracker_sources: dict[str, dict[str, Any]] = {}
        for key, row in (tracked.get("sources") or {}).items():
            if isinstance(row, dict):
                tracker_sources[key] = dict(row)
        for row in merge_summary_progress(local_summary_rows).values():
            merge_source(
                tracker_sources,
                config=str(row.get("config") or ""),
                split=str(row.get("split") or ""),
                path=str(row.get("path") or ""),
                observed_rows=parse_int(row.get("observed_rows")),
                extracted_rows=parse_int(row.get("extracted_rows")),
                expected_rows=parse_int(row.get("expected_rows")),
                null_rows=parse_int(row.get("null_rows")),
                empty_rows=parse_int(row.get("empty_rows")),
                error_rows=parse_int(row.get("error_rows")),
                status=str(row.get("status") or ""),
            )
        for row in log_progress["sources"].values():
            merge_source(
                tracker_sources,
                config=str(row.get("config") or ""),
                split=str(row.get("split") or ""),
                path=str(row.get("path") or ""),
                observed_rows=parse_int(row.get("observed_rows")),
                extracted_rows=parse_int(row.get("extracted_rows")),
                expected_rows=parse_int(row.get("expected_rows")),
                null_rows=parse_int(row.get("null_rows")),
                empty_rows=parse_int(row.get("empty_rows")),
                error_rows=parse_int(row.get("error_rows")),
                status=str(row.get("status") or ""),
            )
        totals = source_totals(tracker_sources)

        expected_candidates = [
            parse_int(summary.get("original_rows_for_delta")),
            parse_int(summary.get("expected_original_rows")),
            parse_int(tracked.get("expected_rows")),
            parse_int(totals.get("expected_rows")),
            survey_counts.get(dataset_id),
        ]
        expected = max((value for value in expected_candidates if value is not None), default=None)
        raw_csv_rows = line_count(tmp_prompt_path) or line_count(prompt_path)
        csv_rows = plausible_row_count(raw_csv_rows, expected)
        extracted_candidates = [
            plausible_row_count(summary.get("extracted_rows"), expected),
            plausible_row_count(tracked.get("extracted_rows"), expected),
            plausible_row_count(totals.get("extracted_rows"), expected),
            log_state.get("last_rows"),
            csv_rows,
        ]
        extracted = max((value for value in extracted_candidates if value is not None), default=0)

        upload_status = summary.get("upload_status") or tracked.get("upload_status") or "not_started"
        status_phase = str(status.get("phase") or tracked.get("phase") or "")
        extract_status = summary.get("status") or tracked.get("extract_status") or status_phase or "not_started"
        active_info = active.get(dataset_id)
        is_waiting = "waiting for a concurrency slot" in log_state.get("last_line", "")
        is_rescheduled = status_phase in {
            "starting",
            "extracting",
            "uploading",
            "waiting_for_auth",
        }
        is_failed = summary.get("status") == "failed" or upload_status == "failed"
        if is_failed:
            state = "failed"
        elif active_info or is_rescheduled or is_waiting:
            state = "running"
        elif upload_status == "complete" or tracked.get("state") == "completed":
            state = "completed"
        elif summary or extracted:
            state = "running"
        else:
            state = "not_started"

        if state == "running" and (is_rescheduled or is_waiting):
            extracted = max(log_state.get("last_rows") or 0, extracted or 0, csv_rows or 0)
            upload_status = "pending"

        if state == "completed":
            progress = 100.0
        elif expected and expected > 0:
            progress = min(99.0, max(0.0, extracted / expected * 100.0))
        elif active_info or is_rescheduled or is_waiting or summary:
            progress = 5.0
        else:
            progress = 0.0

        phase = "done"
        if state == "failed":
            phase = "failed"
        elif state == "not_started":
            phase = "queued"
        elif active_info:
            phase = active_info["phase"]
        elif is_waiting:
            phase = log_state.get("waiting_phase") or "queued"
        elif is_rescheduled:
            phase = status_phase
        elif summary and upload_status != "complete":
            phase = "upload_pending"

        null_rows = max(
            parse_int(summary.get("failed_prompt_rows")) or 0,
            parse_int(tracked.get("failed_prompt_rows")) or 0,
            parse_int(totals.get("failed_prompt_rows")) or 0,
        )
        row_delta = max(
            parse_int(summary.get("row_count_delta")) or 0,
            parse_int(tracked.get("row_delta")) or 0,
        )
        has_issue = bool(row_delta or null_rows or summary.get("status") == "failed")
        current_split = log_state.get("current_split") or log_progress.get("current_split", "")
        current_path = log_state.get("current_path") or log_progress.get("current_path", "")

        datasets.append(
            {
                "index": int(manifest["agent_index"]),
                "dataset_id": dataset_id,
                "title": dataset_title(dataset_id),
                "repo_id": manifest["prompt_only_repo_id"],
                "window": manifest["tmux_window"],
                "state": state,
                "phase": phase,
                "extract_status": extract_status,
                "upload_status": upload_status,
                "progress": round(progress, 1),
                "expected_rows": expected,
                "extracted_rows": extracted,
                "row_delta": row_delta,
                "failed_prompt_rows": null_rows,
                "has_issue": has_issue,
                "current_split": current_split,
                "current_path": current_path,
                "last_line": log_state.get("last_line", ""),
                "active": active_info,
                "started_at": summary.get("started_at") or status.get("updated_at") or tracked.get("updated_at") or "",
                "finished_at": summary.get("finished_at") or "",
                "duration_seconds": summary.get("duration_seconds") or "",
                "output_size": human_bytes(file_size(prompt_path) or file_size(tmp_prompt_path)),
                "bad_rows_file": str(bad_rows_path) if bad_rows_path.exists() else "",
                "_tracker_sources": tracker_sources,
            }
        )

    update_tracker(output_root, datasets)
    for dataset in datasets:
        dataset.pop("_tracker_sources", None)

    counts = {
        "completed": sum(1 for item in datasets if item["state"] == "completed"),
        "running": sum(1 for item in datasets if item["state"] == "running"),
        "not_started": sum(1 for item in datasets if item["state"] == "not_started"),
        "failed": sum(1 for item in datasets if item["state"] == "failed"),
        "issues": sum(1 for item in datasets if item["has_issue"]),
    }
    return {
        "updated_at": utc_now(),
        "output_root": str(output_root),
        "counts": counts,
        "datasets": datasets,
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Nemotron Prompt Export Progress</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --text: #18202f;
      --muted: #687386;
      --line: #dfe4ec;
      --green: #1d8f5f;
      --blue: #2d6cdf;
      --amber: #b56a00;
      --red: #b42318;
      --bar: #dbe4f0;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font: 14px/1.4 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: var(--bg);
    }
    header {
      position: sticky;
      top: 0;
      z-index: 5;
      background: rgba(247, 248, 251, 0.96);
      border-bottom: 1px solid var(--line);
      padding: 14px 18px;
    }
    h1 { margin: 0 0 10px; font-size: 20px; font-weight: 700; letter-spacing: 0; }
    .stats { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    .stat {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      min-width: 118px;
    }
    .stat strong { display: block; font-size: 18px; }
    .stat span { color: var(--muted); font-size: 12px; }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 14px 18px 10px;
      align-items: center;
    }
    input, select {
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      background: white;
      color: var(--text);
    }
    input { min-width: 280px; flex: 1; }
    main { padding: 0 18px 24px; }
    table {
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    th, td {
      text-align: left;
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      vertical-align: middle;
    }
    th {
      font-size: 12px;
      color: var(--muted);
      background: #f1f4f8;
      position: sticky;
      top: 88px;
      z-index: 3;
    }
    tr:last-child td { border-bottom: 0; }
    .title { font-weight: 650; }
    .sub { color: var(--muted); font-size: 12px; margin-top: 2px; overflow-wrap: anywhere; }
    .pill {
      display: inline-flex;
      align-items: center;
      height: 24px;
      padding: 0 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 650;
      border: 1px solid transparent;
      white-space: nowrap;
    }
    .completed { color: var(--green); background: #e9f7ef; border-color: #bee8d0; }
    .running { color: var(--blue); background: #eaf1ff; border-color: #c7d7ff; }
    .not_started { color: var(--muted); background: #f2f4f7; border-color: #d8dde6; }
    .issue { color: var(--amber); background: #fff5df; border-color: #f0d398; }
    .failed { color: var(--red); background: #fff0ee; border-color: #ffd0ca; }
    .progress-wrap { min-width: 180px; }
    .progress {
      width: 100%;
      height: 10px;
      background: var(--bar);
      border-radius: 999px;
      overflow: hidden;
      margin-bottom: 5px;
    }
    .progress > div { height: 100%; background: var(--blue); border-radius: 999px; transition: width .25s; }
    tr[data-state="completed"] .progress > div { background: var(--green); }
    .mono { font-variant-numeric: tabular-nums; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .right { text-align: right; }
    .last-line {
      max-width: 340px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      color: var(--muted);
      font-size: 12px;
    }
    @media (max-width: 900px) {
      th:nth-child(5), td:nth-child(5), th:nth-child(7), td:nth-child(7) { display: none; }
      input { min-width: 180px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Nemotron Prompt Export Progress</h1>
    <div class="stats">
      <div class="stat"><strong id="completed">0</strong><span>Completed</span></div>
      <div class="stat"><strong id="running">0</strong><span>Running</span></div>
      <div class="stat"><strong id="notStarted">0</strong><span>Not started</span></div>
      <div class="stat"><strong id="failed">0</strong><span>Failed</span></div>
      <div class="stat"><strong id="issues">0</strong><span>Flagged</span></div>
      <div class="stat"><strong id="updated">-</strong><span>Updated</span></div>
    </div>
  </header>
  <div class="controls">
    <input id="search" placeholder="Filter datasets" />
    <select id="stateFilter">
      <option value="all">All states</option>
      <option value="completed">Completed</option>
      <option value="running">Running</option>
      <option value="not_started">Not started</option>
      <option value="failed">Failed</option>
      <option value="issues">Flagged</option>
    </select>
  </div>
  <main>
    <table>
      <thead>
        <tr>
          <th>Dataset</th>
          <th>State</th>
          <th>Progress</th>
          <th class="right">Rows</th>
          <th class="right">Null/Empty</th>
          <th>Phase</th>
          <th>Last activity</th>
          <th class="right">Size</th>
        </tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
  </main>
  <script>
    const rowsEl = document.getElementById('rows');
    const searchEl = document.getElementById('search');
    const stateEl = document.getElementById('stateFilter');
    let snapshot = null;

    function fmtInt(value) {
      if (value === null || value === undefined || value === '') return '-';
      return Number(value).toLocaleString();
    }

    function esc(value) {
      return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }

    function render() {
      if (!snapshot) return;
      const q = searchEl.value.trim().toLowerCase();
      const filter = stateEl.value;
      document.getElementById('completed').textContent = snapshot.counts.completed;
      document.getElementById('running').textContent = snapshot.counts.running;
      document.getElementById('notStarted').textContent = snapshot.counts.not_started;
      document.getElementById('failed').textContent = snapshot.counts.failed;
      document.getElementById('issues').textContent = snapshot.counts.issues;
      document.getElementById('updated').textContent = new Date(snapshot.updated_at).toLocaleTimeString();

      const rows = snapshot.datasets.filter(d => {
        if (q && !(d.title.toLowerCase().includes(q) || d.dataset_id.toLowerCase().includes(q))) return false;
        if (filter === 'issues') return d.has_issue;
        if (filter !== 'all' && d.state !== filter) return false;
        return true;
      });

      rowsEl.innerHTML = rows.map(d => {
        const issue = d.has_issue ? `<span class="pill issue">flagged</span>` : '';
        const stateClass = d.extract_status === 'failed' ? 'failed' : d.state;
        const rowsText = `${fmtInt(d.extracted_rows)} / ${fmtInt(d.expected_rows)}`;
        const detail = d.current_split ? `${d.current_split}${d.current_path ? ' · ' + d.current_path : ''}` : d.repo_id;
        return `<tr data-state="${esc(d.state)}">
          <td><div class="title">${esc(d.title)}</div><div class="sub">${esc(detail)}</div></td>
          <td><span class="pill ${stateClass}">${esc(d.state.replace('_', ' '))}</span> ${issue}</td>
          <td class="progress-wrap"><div class="progress"><div style="width:${d.progress}%"></div></div><div class="mono">${d.progress.toFixed(1)}%</div></td>
          <td class="right mono">${rowsText}</td>
          <td class="right mono">${fmtInt(d.failed_prompt_rows)}</td>
          <td>${esc(d.phase)}<div class="sub">${esc(d.upload_status)}</div></td>
          <td><div class="last-line" title="${esc(d.last_line)}">${esc(d.last_line || '-')}</div></td>
          <td class="right mono">${esc(d.output_size)}</td>
        </tr>`;
      }).join('');
    }

    async function refresh() {
      try {
        const response = await fetch('/api/progress', {cache: 'no-store'});
        snapshot = await response.json();
        render();
      } catch (error) {
        console.error(error);
      }
    }

    searchEl.addEventListener('input', render);
    stateEl.addEventListener('change', render);
    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    output_root: Path
    survey_path: Path

    def log_message(self, format: str, *args: Any) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {self.address_string()} {format % args}")

    def send_json(self, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self) -> None:
        body = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self.send_html()
            return
        if self.path == "/api/progress":
            self.send_json(build_snapshot(self.output_root, self.survey_path))
            return
        self.send_response(404)
        self.end_headers()

    def do_HEAD(self) -> None:
        if self.path in {"/", "/index.html", "/api/progress"}:
            content_type = (
                "application/json; charset=utf-8"
                if self.path == "/api/progress"
                else "text/html; charset=utf-8"
            )
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a live Nemotron export dashboard.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--survey", type=Path, default=DEFAULT_SURVEY)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    DashboardHandler.output_root = args.output_root.expanduser().resolve()
    DashboardHandler.survey_path = args.survey.expanduser().resolve()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Serving dashboard at http://{args.host}:{args.port}")
    print(f"Reading run data from {DashboardHandler.output_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
