#!/usr/bin/env python3
"""Run one Nemotron prompt-only repo through Doubleword and TorchSpec upload."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path

from huggingface_hub import hf_hub_download

from prepare_nemotron_prompt_batches import convert_skip_empty, download_prompts, normalize_openai_tools


ROOT = Path("data/nemotron-prompt-only")
MODEL = "openai/gpt-oss-20b"
MODEL_SLUG = "gpt-oss-20b"
RESULTS_NAME = "gpt-oss-20b-temperature-1-reasoning_effort-low-max_token-2048.jsonl"
CONVERTER_SRC = (
    ROOT
    / "Nemotron-RL-litmus-bench-v0.1-prompt-only"
    / "scripts"
    / "convert_torchspec_gpt_oss_20b_harmony.py"
)


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def capture_json(cmd: list[str]) -> dict[str, object]:
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return json.loads(proc.stdout)


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def jsonl_rows(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            yield row


def result_is_bad(row: dict[str, object]) -> bool:
    if row.get("error"):
        return True
    response = row.get("response")
    if not isinstance(response, dict):
        return True
    status_code = response.get("status_code")
    if isinstance(status_code, int) and status_code >= 400:
        return True
    body = response.get("body")
    if not isinstance(body, dict):
        return True
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return True
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    return not isinstance(message, dict)


def scan_bad_result_ids(results: Path) -> list[str]:
    bad: list[str] = []
    seen: set[str] = set()
    for row in jsonl_rows(results):
        custom_id = row.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id:
            raise RuntimeError(f"{results}: result row missing custom_id")
        if custom_id in seen:
            raise RuntimeError(f"{results}: duplicate result custom_id {custom_id}")
        seen.add(custom_id)
        if result_is_bad(row):
            bad.append(custom_id)
    return bad


def write_retry_requests(prepared: Path, bad_ids: set[str], retry_requests: Path) -> int:
    found = 0
    retry_requests.parent.mkdir(parents=True, exist_ok=True)
    with retry_requests.open("w", encoding="utf-8") as out:
        for row in jsonl_rows(prepared):
            custom_id = row.get("custom_id")
            if isinstance(custom_id, str) and custom_id in bad_ids:
                out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                found += 1
    if found != len(bad_ids):
        missing = len(bad_ids) - found
        raise RuntimeError(f"{prepared}: failed to find {missing} retry request rows")
    return found


def replace_result_rows(results: Path, retry_results: Path) -> None:
    replacements: dict[str, dict[str, object]] = {}
    for row in jsonl_rows(retry_results):
        custom_id = row.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id:
            raise RuntimeError(f"{retry_results}: retry result row missing custom_id")
        if result_is_bad(row):
            raise RuntimeError(f"{retry_results}: retry row still bad for custom_id {custom_id}")
        replacements[custom_id] = row

    tmp = results.with_suffix(results.suffix + ".tmp")
    replaced = 0
    with results.open("r", encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            custom_id = row.get("custom_id")
            if isinstance(custom_id, str) and custom_id in replacements:
                row = replacements[custom_id]
                replaced += 1
            dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(results)
    if replaced != len(replacements):
        raise RuntimeError(f"{results}: replaced {replaced} rows, expected {len(replacements)}")


def download_results(batch_ids: Path, results: Path, attempts: int = 5) -> None:
    last_error = ""
    for attempt in range(1, attempts + 1):
        proc = subprocess.run(
            ["dw", "batches", "results", "--from-file", str(batch_ids), "--output-file", str(results)],
            text=True,
            capture_output=True,
        )
        if proc.returncode == 0:
            print(proc.stdout, end="", flush=True)
            return
        last_error = f"{proc.stdout}\n{proc.stderr}".strip()
        wait = min(300, 15 * attempt)
        print(f"results download failed on attempt {attempt}; sleeping {wait}s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"failed to download batch results: {last_error[-2000:]}")


def repair_bad_results(prepared: Path, results: Path, local_dir: Path, request_count: int, max_rounds: int = 3) -> None:
    for round_idx in range(1, max_rounds + 1):
        bad_ids = scan_bad_result_ids(results)
        if not bad_ids:
            return
        print(f"found {len(bad_ids)} malformed result rows; retry round {round_idx}", flush=True)
        retry_dir = local_dir / "retries"
        retry_requests = retry_dir / f"{RESULTS_NAME}.retry-{round_idx}.jsonl"
        retry_results = retry_dir / f"{RESULTS_NAME}.retry-{round_idx}.results.jsonl"
        retry_ids = retry_dir / f"{RESULTS_NAME}.retry-{round_idx}.batch_ids"
        write_retry_requests(prepared, set(bad_ids), retry_requests)
        run(["dw", "files", "validate", str(retry_requests)])
        if retry_ids.exists():
            ids = [line.strip() for line in retry_ids.read_text(encoding="utf-8").splitlines() if line.strip()]
            wait_for_batches(ids)
        else:
            run(["dw", "batches", "run", str(retry_requests), "--watch", "--output-id", str(retry_ids)])
        download_results(retry_ids, retry_results)
        if count_lines(retry_results) != len(bad_ids):
            raise RuntimeError(f"retry results count {count_lines(retry_results)} != retry request count {len(bad_ids)}")
        replace_result_rows(results, retry_results)
        if count_lines(results) != request_count:
            raise RuntimeError(f"results count {count_lines(results)} != request count {request_count} after retry")
    remaining = scan_bad_result_ids(results)
    if remaining:
        raise RuntimeError(f"{len(remaining)} malformed result rows remain after {max_rounds} retry rounds")


def rewrite_tools(path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    changed = False
    with path.open("r", encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            before = json.dumps(row.get("body", {}).get("tools", None), sort_keys=True)
            normalize_openai_tools(row)
            after = json.dumps(row.get("body", {}).get("tools", None), sort_keys=True)
            changed = changed or before != after
            dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)
    if changed:
        print(f"normalized tools in {path}", flush=True)


def upload(repo_id: str, local: Path, remote: str, message: str) -> None:
    run(
        [
            "huggingface-cli",
            "upload",
            repo_id,
            str(local),
            remote,
            "--repo-type",
            "dataset",
            "--commit-message",
            message,
        ]
    )


def verify_if_practical(repo_id: str, local: Path, remote: str, max_bytes: int) -> str:
    size = local.stat().st_size
    if size > max_bytes:
        return f"skipped-byte-compare:size={size}"
    downloaded = Path(
        hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=remote, local_dir="/tmp/hf_dw_verify")
    )
    if downloaded.read_bytes() != local.read_bytes():
        raise RuntimeError(f"uploaded file mismatch: {repo_id}:{remote}")
    return "byte-verified"


def ensure_generic(repo_id: str, local_dir: Path, force_convert: bool) -> Path:
    prompts = local_dir / "prompts.csv"
    generic = local_dir / "dw_batch_requests.jsonl"
    meta = local_dir / "dw_batch_requests.meta.json"
    if not prompts.exists():
        download_prompts(repo_id, local_dir)
    if force_convert or not generic.exists():
        convert_skip_empty(prompts, generic, meta)
    rewrite_tools(generic)
    run(["dw", "files", "validate", str(generic)])
    return generic


def prepare(generic: Path, prepared: Path) -> None:
    run(
        [
            "dw",
            "files",
            "prepare",
            str(generic),
            "--model",
            MODEL,
            "--temperature",
            "1",
            "--max-tokens",
            "2048",
            "--set",
            "body.reasoning_effort=low",
            "--output-file",
            str(prepared),
        ]
    )
    rewrite_tools(prepared)
    run(["dw", "files", "validate", str(prepared)])


def split_if_needed(prepared: Path, chunk_size: int) -> Path:
    total = count_lines(prepared)
    if total <= chunk_size:
        return prepared
    chunk_dir = prepared.parent / f"{prepared.stem}_chunks_{chunk_size}"
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True)
    run(["dw", "files", "split", str(prepared), "--chunk-size", str(chunk_size), "--output-dir", str(chunk_dir)])
    return chunk_dir


def submit_watch(target: Path, batch_ids: Path) -> list[str]:
    if batch_ids.exists():
        ids = [line.strip() for line in batch_ids.read_text(encoding="utf-8").splitlines() if line.strip()]
        if ids:
            wait_for_batches(ids)
            return ids
    run(["dw", "batches", "run", str(target), "--watch", "--output-id", str(batch_ids)])
    return [line.strip() for line in batch_ids.read_text(encoding="utf-8").splitlines() if line.strip()]


def get_batch_with_retry(batch_id: str, attempts: int = 5) -> dict[str, object]:
    last_error = ""
    for attempt in range(1, attempts + 1):
        proc = subprocess.run(["dw", "batches", "get", batch_id, "--output", "json"], text=True, capture_output=True)
        if proc.returncode == 0:
            return json.loads(proc.stdout)
        last_error = f"{proc.stdout}\n{proc.stderr}".strip()
        time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"failed to fetch batch {batch_id}: {last_error[-1000:]}")


def wait_for_batches(ids: list[str]) -> None:
    terminal = {"completed", "failed", "cancelled", "expired"}
    while True:
        terminal_count = 0
        states: dict[str, int] = {}
        failed_requests = 0
        for batch_id in ids:
            batch = get_batch_with_retry(batch_id)
            counts = batch.get("request_counts") or {}
            status = str(batch.get("status"))
            states[status] = states.get(status, 0) + 1
            failed_requests += int(counts.get("failed") or 0)
            if status in {"cancelled", "expired"}:
                raise RuntimeError(f"{batch_id}: status={status} counts={counts}")
            if status in terminal:
                terminal_count += 1
        print(f"batch poll: terminal={terminal_count}/{len(ids)} states={states} failed_requests={failed_requests}", flush=True)
        if terminal_count == len(ids):
            return
        time.sleep(300)


def retry_failed_batches(ids: list[str], request_count: int, max_rounds: int = 3) -> None:
    for round_idx in range(1, max_rounds + 1):
        wait_for_batches(ids)
        failed_ids: list[tuple[str, int, str]] = []
        completed_total = 0
        for batch_id in ids:
            batch = get_batch_with_retry(batch_id)
            counts = batch.get("request_counts") or {}
            completed_total += int(counts.get("completed") or 0)
            failed = int(counts.get("failed") or 0)
            status = str(batch.get("status"))
            if failed or status == "failed":
                failed_ids.append((batch_id, failed, status))
        if not failed_ids:
            return
        if completed_total == request_count:
            return
        print(
            f"retry round {round_idx}: completed_total={completed_total}/{request_count}; "
            f"retrying {len(failed_ids)} failed batches",
            flush=True,
        )
        for batch_id, failed, status in failed_ids:
            print(f"+ dw batches retry {batch_id} # status={status} failed={failed}", flush=True)
            proc = subprocess.run(["dw", "batches", "retry", batch_id, "--output", "json"], text=True, capture_output=True)
            if proc.returncode != 0:
                combined = f"{proc.stdout}\n{proc.stderr}".strip()
                if "Rate limited" in combined:
                    print("rate limited while retrying failed batch; sleeping 300s", flush=True)
                    time.sleep(300)
                    proc = subprocess.run(
                        ["dw", "batches", "retry", batch_id, "--output", "json"],
                        text=True,
                        capture_output=True,
                    )
                if proc.returncode != 0:
                    raise RuntimeError(f"failed to retry batch {batch_id}: {combined[-1000:]}")
        time.sleep(30)
    wait_for_batches(ids)


def validate_batches(ids: list[str], batch_jsonl: Path) -> int:
    total = 0
    completed = 0
    failed = 0
    with batch_jsonl.open("w", encoding="utf-8") as out:
        for batch_id in ids:
            batch = get_batch_with_retry(batch_id)
            out.write(json.dumps(batch, sort_keys=True) + "\n")
            counts = batch.get("request_counts") or {}
            total += int(counts.get("total") or 0)
            completed += int(counts.get("completed") or 0)
            failed += int(counts.get("failed") or 0)
            if batch.get("status") not in {"completed", "failed"}:
                raise RuntimeError(f"{batch_id}: status={batch.get('status')}")
    if failed:
        print(f"batch validation still has failed request counts: total={total} completed={completed} failed={failed}", flush=True)
    return completed


def convert_torchspec(local_dir: Path, prepared: Path, results: Path, torchspec: Path) -> None:
    script = local_dir / "scripts" / "convert_torchspec_gpt_oss_20b_harmony.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    if not script.exists():
        shutil.copy2(CONVERTER_SRC, script)
    run(
        [
            "python",
            str(script),
            "--requests",
            str(prepared),
            "--results",
            str(results),
            "--output",
            str(torchspec),
            "--model",
            MODEL,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--skip-generic-upload", action="store_true")
    parser.add_argument("--reuse-prepared", action="store_true")
    parser.add_argument("--verify-max-bytes", type=int, default=250_000_000)
    args = parser.parse_args()

    slug = args.repo_id.split("/", 1)[1]
    local_dir = ROOT / slug
    local_dir.mkdir(parents=True, exist_ok=True)
    generic = ensure_generic(args.repo_id, local_dir, args.force_convert)
    request_count = count_lines(generic)
    if args.expected_count is not None and request_count != args.expected_count:
        raise RuntimeError(f"request count {request_count} != expected {args.expected_count}")

    if args.skip_generic_upload:
        generic_verify = "skipped"
    else:
        upload(args.repo_id, generic, "dw_batch_requests.jsonl", "Add generic Doubleword batch requests")
        generic_verify = verify_if_practical(args.repo_id, generic, "dw_batch_requests.jsonl", args.verify_max_bytes)

    prepared = local_dir / f"dw_batch_requests_{MODEL_SLUG}.jsonl"
    if args.reuse_prepared and prepared.exists():
        rewrite_tools(prepared)
        run(["dw", "files", "validate", str(prepared)])
    else:
        prepare(generic, prepared)
    target = split_if_needed(prepared, args.chunk_size)
    batch_ids = local_dir / f"{RESULTS_NAME}.batch_ids"
    ids = submit_watch(target, batch_ids)
    retry_failed_batches(ids, request_count)
    batch_total = validate_batches(ids, local_dir / f"{RESULTS_NAME}.batches.jsonl")
    if batch_total != request_count:
        raise RuntimeError(f"batch completed count {batch_total} != request count {request_count}")

    results = local_dir / RESULTS_NAME
    download_results(batch_ids, results)
    if count_lines(results) != request_count:
        raise RuntimeError(f"results count {count_lines(results)} != request count {request_count}")
    repair_bad_results(prepared, results, local_dir, request_count)
    upload(args.repo_id, results, RESULTS_NAME, f"Add {MODEL} batch results")
    results_verify = verify_if_practical(args.repo_id, results, RESULTS_NAME, args.verify_max_bytes)

    torchspec = local_dir / "data" / f"{MODEL_SLUG}.jsonl"
    convert_torchspec(local_dir, prepared, results, torchspec)
    if count_lines(torchspec) != request_count:
        raise RuntimeError(f"torchspec count {count_lines(torchspec)} != request count {request_count}")
    upload(args.repo_id, torchspec, f"data/{MODEL_SLUG}.jsonl", f"Add TorchSpec split for {MODEL}")
    torchspec_verify = verify_if_practical(args.repo_id, torchspec, f"data/{MODEL_SLUG}.jsonl", args.verify_max_bytes)

    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "request_count": request_count,
                "batch_ids": ids,
                "results_count": count_lines(results),
                "torchspec_count": count_lines(torchspec),
                "verify": {
                    "dw_batch_requests.jsonl": generic_verify,
                    RESULTS_NAME: results_verify,
                    f"data/{MODEL_SLUG}.jsonl": torchspec_verify,
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
