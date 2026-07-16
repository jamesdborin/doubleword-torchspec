#!/usr/bin/env python3
"""Monitor current Nemotron repair batches and finalize repos when complete."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path("data/nemotron-prompt-only")


@dataclass(frozen=True)
class RepairJob:
    name: str
    repo_id: str
    local_dir: Path
    prepared: Path
    original_results: Path
    repair_ids: Path
    repair_results: Path
    expected_count: int


JOBS = [
    RepairJob(
        name="structured",
        repo_id="jamesdborin/Nemotron-RL-Instruction-Following-Structured-Outputs-v2-prompt-only",
        local_dir=ROOT / "Nemotron-RL-Instruction-Following-Structured-Outputs-v2-prompt-only",
        prepared=ROOT
        / "Nemotron-RL-Instruction-Following-Structured-Outputs-v2-prompt-only"
        / "dw_batch_requests_gpt-oss-20b.jsonl",
        original_results=ROOT
        / "Nemotron-RL-Instruction-Following-Structured-Outputs-v2-prompt-only"
        / "gpt-oss-20b-temperature-1-reasoning_effort-low-max_token-2048.original-successes.jsonl",
        repair_ids=ROOT
        / "Nemotron-RL-Instruction-Following-Structured-Outputs-v2-prompt-only"
        / "repairs"
        / "structured_missing_no_schema.batch_ids",
        repair_results=ROOT
        / "Nemotron-RL-Instruction-Following-Structured-Outputs-v2-prompt-only"
        / "repairs"
        / "structured_missing_no_schema.results.jsonl",
        expected_count=62696,
    ),
    RepairJob(
        name="conversational",
        repo_id="jamesdborin/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1-prompt-only",
        local_dir=ROOT / "Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1-prompt-only",
        prepared=ROOT
        / "Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1-prompt-only"
        / "dw_batch_requests_gpt-oss-20b.jsonl",
        original_results=ROOT
        / "Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1-prompt-only"
        / "gpt-oss-20b-temperature-1-reasoning_effort-low-max_token-2048.partial-originals.jsonl",
        repair_ids=ROOT
        / "Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1-prompt-only"
        / "repairs"
        / "conversational_missing_no_tools.batch_ids",
        repair_results=ROOT
        / "Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1-prompt-only"
        / "repairs"
        / "conversational_missing_no_tools.results.jsonl",
        expected_count=96968,
    ),
]


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S %z')} {message}", flush=True)


def run(cmd: list[str]) -> None:
    log("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def capture_json(cmd: list[str]) -> dict[str, object]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=True)
    return json.loads(proc.stdout)


def ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def batch_summary(job: RepairJob) -> tuple[bool, int, int, int, dict[str, int]]:
    terminal = {"completed", "failed", "cancelled", "expired"}
    all_terminal = True
    completed = failed = total = 0
    states: dict[str, int] = {}
    for batch_id in ids(job.repair_ids):
        batch = capture_json(["dw", "batches", "get", batch_id, "--output", "json"])
        state = str(batch.get("status"))
        states[state] = states.get(state, 0) + 1
        counts = batch.get("request_counts") or {}
        completed += int(counts.get("completed") or 0)
        failed += int(counts.get("failed") or 0)
        total += int(counts.get("total") or 0)
        if state not in terminal:
            all_terminal = False
    return all_terminal, completed, failed, total, states


def finalize(job: RepairJob) -> None:
    run(["dw", "batches", "results", "--from-file", str(job.repair_ids), "--output-file", str(job.repair_results)])
    run(
        [
            "python",
            "scripts/tools/finalize_nemotron_repaired_results.py",
            "--repo-id",
            job.repo_id,
            "--local-dir",
            str(job.local_dir),
            "--prepared",
            str(job.prepared),
            "--original-results",
            str(job.original_results),
            "--repair-results",
            str(job.repair_results),
            "--expected-count",
            str(job.expected_count),
        ]
    )


def main() -> None:
    done: set[str] = set()
    while len(done) < len(JOBS):
        for job in JOBS:
            if job.name in done:
                continue
            all_terminal, completed, failed, total, states = batch_summary(job)
            log(f"{job.name}: states={states} completed={completed} failed={failed} total={total}")
            if all_terminal:
                if failed:
                    raise RuntimeError(f"{job.name}: repair batches finished with {failed} failed rows")
                if completed != total:
                    raise RuntimeError(f"{job.name}: completed {completed} != total {total}")
                finalize(job)
                done.add(job.name)
        if len(done) < len(JOBS):
            time.sleep(300)
    log("all repair jobs finalized")


if __name__ == "__main__":
    main()
