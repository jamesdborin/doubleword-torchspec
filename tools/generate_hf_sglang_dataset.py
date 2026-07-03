#!/usr/bin/env python3
"""Generate torchspec JSONL datasets from HF Hub datasets with an SGLang server."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset, load_dataset_builder
from huggingface_hub import HfApi, hf_hub_download
from openai import OpenAI
from tqdm import tqdm


DEFAULT_DATASETS = (
    "jamesdborin/Magpie-Llama-3.1-Pro-300K-Filtered-prompt-only",
    "jamesdborin/UltraChat-200K-prompt-only",
)


@dataclass
class DatasetProgress:
    repo_id: str
    total: int | None
    completed: int
    errors: int
    elapsed_seconds: float
    records_per_second: float
    eta_seconds: float | None
    eta_human: str | None
    current_shard: int
    output_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--server-address", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--hub-output-path", default="data/qwen-3.5-9B")
    parser.add_argument("--work-dir", default="outputs/qwen-3.5-9B-hf-generation")
    parser.add_argument("--progress-file", default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=1024)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    return parser.parse_args()


def seconds_to_human(seconds: float | None) -> str | None:
    if seconds is None or not math.isfinite(seconds):
        return None
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def repo_slug(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def get_total_rows(repo_id: str) -> int | None:
    try:
        builder = load_dataset_builder(repo_id)
        if builder.info.splits and "train" in builder.info.splits:
            return builder.info.splits["train"].num_examples
    except Exception:
        pass
    return None


def iter_dataset(repo_id: str):
    return load_dataset(repo_id, split="train", streaming=True)


def load_uploaded_or_local_shards(
    api: HfApi,
    repo_id: str,
    local_dir: Path,
    hub_output_path: str,
    token: str | None,
) -> tuple[set[str], set[str]]:
    local_dir.mkdir(parents=True, exist_ok=True)
    processed_ids = set()
    remote_names = set()

    for path in sorted(local_dir.glob("part-*.jsonl")):
        processed_ids.update(read_record_ids(path))

    try:
        files = api.list_repo_files(repo_id, repo_type="dataset", token=token)
    except Exception:
        files = []

    prefix = hub_output_path.rstrip("/") + "/"
    for filename in files:
        if not filename.startswith(prefix) or not filename.endswith(".jsonl"):
            continue
        remote_names.add(Path(filename).name)
        local_path = local_dir / Path(filename).name
        if not local_path.exists():
            downloaded = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                token=token,
            )
            local_path.write_bytes(Path(downloaded).read_bytes())
        processed_ids.update(read_record_ids(local_path))

    return processed_ids, remote_names


def read_record_ids(path: Path) -> set[str]:
    ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_id = row.get("source_custom_id") or row.get("id")
            if record_id is not None:
                ids.add(str(record_id))
    return ids


def get_source_id(row: dict[str, Any], row_index: int) -> str:
    source_id = row.get("id") or row.get("source_custom_id") or row.get("custom_id") or row_index
    return str(source_id)


def extract_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("conversations") or row.get("messages")
    body = row.get("body")
    if not isinstance(messages, list) and isinstance(body, dict):
        messages = body.get("messages")
    if not isinstance(messages, list):
        prompt = row.get("prompt") or row.get("text")
        if not isinstance(prompt, str) and isinstance(body, dict):
            prompt = body.get("prompt") or body.get("text")
        if not isinstance(prompt, str):
            raise ValueError("row has no conversations/messages/prompt/text field")
        return [{"role": "user", "content": prompt}]

    prompt_messages = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role in {"system", "user", "assistant", "tool"}:
            prompt_messages.append(
                {
                    "role": role,
                    "content": message.get("content", ""),
                }
            )
        elif role in {"assistant_analysis", "assistant_final"}:
            break
    while prompt_messages and prompt_messages[-1]["role"] == "assistant":
        prompt_messages.pop()
    if not prompt_messages:
        raise ValueError("no usable prompt messages found")
    return prompt_messages


def build_request(args: argparse.Namespace, messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "presence_penalty": args.presence_penalty,
        "extra_body": {
            "top_k": args.top_k,
            "min_p": args.min_p,
            "repetition_penalty": args.repetition_penalty,
        },
        "stream": False,
    }


def call_model(args: argparse.Namespace, client: OpenAI, row: dict[str, Any], row_index: int) -> dict[str, Any]:
    messages = extract_messages(row)
    last_error = None
    for attempt in range(1, args.max_retries + 1):
        try:
            response = client.chat.completions.create(**build_request(args, messages))
            response_message = response.choices[0].message
            assistant_content = response_message.content or ""
            output = dict(row)
            source_id = get_source_id(row, row_index)
            output["id"] = source_id
            output["conversations"] = messages + [{"role": "assistant", "content": assistant_content}]
            output["model"] = args.model
            output["source_custom_id"] = source_id
            output["generation_parameters"] = {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "min_p": args.min_p,
                "presence_penalty": args.presence_penalty,
                "repetition_penalty": args.repetition_penalty,
                "max_new_tokens": args.max_new_tokens,
            }
            if response.usage is not None:
                output["input_tokens"] = response.usage.prompt_tokens
                output["output_tokens"] = response.usage.completion_tokens
                output["context_length"] = response.usage.total_tokens
            return output
        except Exception as exc:
            last_error = exc
            if attempt < args.max_retries:
                time.sleep(args.retry_sleep * attempt)
    raise RuntimeError(str(last_error))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def upload_shard(
    api: HfApi,
    repo_id: str,
    shard_path: Path,
    hub_output_path: str,
    token: str | None,
) -> None:
    api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=str(shard_path),
        path_in_repo=f"{hub_output_path.rstrip('/')}/{shard_path.name}",
        token=token,
        commit_message=f"Add {shard_path.name} Qwen3.5-9B completions",
    )


def write_progress(path: Path, progress: list[DatasetProgress]) -> None:
    payload = {
        "updated_at_unix": time.time(),
        "overall": summarize_progress(progress),
        "datasets": [asdict(item) for item in progress],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize_progress(progress: list[DatasetProgress]) -> dict[str, Any]:
    completed = sum(item.completed for item in progress)
    errors = sum(item.errors for item in progress)
    known_total = all(item.total is not None for item in progress)
    total = sum(item.total or 0 for item in progress) if known_total else None
    elapsed = max((item.elapsed_seconds for item in progress), default=0.0)
    rate = completed / elapsed if elapsed > 0 else 0.0
    eta = ((total - completed) / rate) if total is not None and rate > 0 else None
    return {
        "total": total,
        "completed": completed,
        "errors": errors,
        "records_per_second": rate,
        "eta_seconds": eta,
        "eta_human": seconds_to_human(eta),
    }


def process_dataset(args: argparse.Namespace, api: HfApi, repo_id: str, progress_path: Path, all_progress: list[DatasetProgress]) -> DatasetProgress:
    client = OpenAI(base_url=args.server_address.rstrip("/"), api_key="None", timeout=3600.0)
    dataset_dir = Path(args.work_dir) / repo_slug(repo_id)
    data_dir = dataset_dir / args.hub_output_path
    error_path = dataset_dir / "errors.jsonl"
    token = None if args.no_upload else args.token
    processed_ids, remote_names = load_uploaded_or_local_shards(
        api, repo_id, data_dir, args.hub_output_path, token
    )
    if not args.no_upload:
        for shard_path in sorted(data_dir.glob("part-*.jsonl")):
            if shard_path.name not in remote_names:
                upload_shard(api, repo_id, shard_path, args.hub_output_path, args.token)
    total = get_total_rows(repo_id)
    completed = len(processed_ids)
    errors = 0
    started_at = time.time()
    shard_index = completed // args.shard_size
    shard_rows: list[dict[str, Any]] = []

    progress = DatasetProgress(
        repo_id=repo_id,
        total=total,
        completed=completed,
        errors=errors,
        elapsed_seconds=0.0,
        records_per_second=0.0,
        eta_seconds=None,
        eta_human=None,
        current_shard=shard_index,
        output_dir=str(data_dir),
    )
    all_progress.append(progress)
    write_progress(progress_path, all_progress)

    pending_batch: list[tuple[int, dict[str, Any]]] = []

    def flush_batch(batch: list[tuple[int, dict[str, Any]]]) -> None:
        nonlocal completed, errors, shard_index, shard_rows
        if not batch:
            return

        def refresh_progress() -> None:
            elapsed = time.time() - started_at
            processed_this_run = completed + errors - len(processed_ids)
            rate = processed_this_run / elapsed if elapsed > 0 else 0.0
            remaining = (total - completed - errors) if total is not None else None
            eta = remaining / rate if remaining is not None and rate > 0 else None
            progress.completed = completed
            progress.errors = errors
            progress.elapsed_seconds = elapsed
            progress.records_per_second = rate
            progress.eta_seconds = eta
            progress.eta_human = seconds_to_human(eta)
            progress.current_shard = shard_index
            write_progress(progress_path, all_progress)

        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_map = {
                executor.submit(call_model, args, client, row, row_index): (row_index, row)
                for row_index, row in batch
            }
            for future in tqdm(as_completed(future_map), total=len(future_map), desc=repo_id):
                row_index, row = future_map[future]
                try:
                    generated = future.result()
                except Exception as exc:
                    errors += 1
                    error_record = {"row_index": row_index, "error": str(exc), "source": row}
                    write_jsonl(error_path, [error_record])
                    refresh_progress()
                    continue

                shard_rows.append(generated)
                completed += 1
                if len(shard_rows) >= args.shard_size:
                    shard_path = data_dir / f"part-{shard_index:06d}.jsonl"
                    write_jsonl(shard_path, shard_rows)
                    shard_rows = []
                    shard_index += 1
                    if not args.no_upload:
                        upload_shard(api, repo_id, shard_path, args.hub_output_path, args.token)

                refresh_progress()

    for row_index, row in enumerate(iter_dataset(repo_id)):
        source_id = get_source_id(row, row_index)
        if source_id in processed_ids:
            continue
        pending_batch.append((row_index, row))
        if len(pending_batch) >= args.batch_size:
            flush_batch(pending_batch)
            pending_batch = []

    flush_batch(pending_batch)
    if shard_rows:
        shard_path = data_dir / f"part-{shard_index:06d}.jsonl"
        write_jsonl(shard_path, shard_rows)
        if not args.no_upload:
            upload_shard(api, repo_id, shard_path, args.hub_output_path, args.token)
    return progress


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.concurrency <= 0 or args.shard_size <= 0:
        raise ValueError("batch-size, concurrency, and shard-size must be positive")
    if not args.no_upload and not args.token:
        raise RuntimeError("HF_TOKEN or HUGGING_FACE_HUB_TOKEN is required unless --no-upload is set")

    progress_path = Path(args.progress_file or Path(args.work_dir) / "progress.json")
    api = HfApi()
    all_progress: list[DatasetProgress] = []
    for repo_id in args.datasets:
        process_dataset(args, api, repo_id, progress_path, all_progress)


if __name__ == "__main__":
    main()
