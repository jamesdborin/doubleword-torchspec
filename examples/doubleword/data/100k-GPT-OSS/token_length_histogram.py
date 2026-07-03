#!/usr/bin/env python3
"""Build token-length histograms for the GPT-OSS Harmony JSONL dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "gpt-oss-20b.harmony.jsonl"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "token_histograms"
DEFAULT_TOKENIZER = "openai/gpt-oss-20b"
DEFAULT_REASONING_LEVEL = "low"
DEFAULT_BATCH_SIZE = 1024
DEFAULT_BUCKET_SIZE = 128
DEFAULT_MAX_DISPLAY_BUCKETS = 60

DEFAULT_REASONING_MESSAGE = (
    "You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\n"
    "Current date: 2025-06-28\n\n"
    "Reasoning: {reasoning_level}\n\n"
    "# Valid channels: analysis, commentary, final. Channel must be included for every message."
)

ASSISTANT_OUTPUT_ROLES = {"assistant_analysis", "assistant_final"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize GPT-OSS Harmony examples and write length histograms."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input Harmony JSONL path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=f"Hugging Face tokenizer name/path (default: {DEFAULT_TOKENIZER})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of strings to tokenize per batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for histogram CSVs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--bucket-size",
        type=int,
        default=DEFAULT_BUCKET_SIZE,
        help=f"Terminal histogram bucket width in tokens (default: {DEFAULT_BUCKET_SIZE})",
    )
    parser.add_argument(
        "--max-display-buckets",
        type=int,
        default=DEFAULT_MAX_DISPLAY_BUCKETS,
        help=(
            "Maximum number of terminal histogram buckets to print per counter "
            f"(default: {DEFAULT_MAX_DISPLAY_BUCKETS})"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of JSONL rows to process for smoke tests.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code to AutoTokenizer.from_pretrained (default: true).",
    )
    return parser.parse_args()


def render_message(role: str, content: str) -> str:
    if role == "system":
        return f"<|start|>system<|message|>{content}<|end|>"
    if role == "assistant_reasoning_effort":
        reasoning_message = DEFAULT_REASONING_MESSAGE.format(reasoning_level=content.lower())
        return f"<|start|>system<|message|>{reasoning_message}<|end|>"
    if role == "user":
        return f"<|start|>user<|message|>{content}<|end|>"
    if role == "assistant_analysis":
        return f"<|start|>assistant<|channel|>analysis<|message|>{content}<|end|>"
    if role == "assistant_commentary":
        return f"<|start|>assistant<|channel|>commentary<|message|>{content}<|end|>"
    if role == "assistant_final":
        return f"<|start|>assistant<|channel|>final<|message|>{content}<|end|>"
    raise ValueError(f"Unknown role: {role}")


def render_conversation_parts(conversation: list[dict]) -> tuple[str, str, str]:
    """Return prompt-only, full-example, and output-only Harmony renderings."""
    if not conversation:
        return "", "", ""

    rendered_messages = []
    prompt_parts = []
    if conversation[0].get("role") not in {"system", "assistant_reasoning_effort"}:
        rendered_default = render_message("assistant_reasoning_effort", DEFAULT_REASONING_LEVEL)
        rendered_messages.append(rendered_default)
        prompt_parts.append(rendered_default)

    output_parts = []
    for message in conversation:
        role = message["role"]
        content = message.get("content") or ""
        rendered = render_message(role, content)
        rendered_messages.append(rendered)
        if role in ASSISTANT_OUTPUT_ROLES:
            output_parts.append(rendered)
        else:
            prompt_parts.append(rendered)

    return "".join(prompt_parts), "".join(rendered_messages), "".join(output_parts)


def iter_examples(path: Path, limit: int | None) -> Iterable[tuple[int, list[dict]]]:
    with path.open() as f:
        for index, line in enumerate(f, start=1):
            if limit is not None and index > limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            conversation = row.get("conversations")
            if not isinstance(conversation, list):
                raise ValueError(f"Row {index} does not contain a conversations list")
            yield index, conversation


def tokenize_lengths(tokenizer, texts: list[str]) -> list[int]:
    encoding = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )
    return [len(input_ids) for input_ids in encoding["input_ids"]]


def update_counter(counter: Counter[int], lengths: list[int]) -> None:
    counter.update(lengths)


def flush_batch(
    tokenizer,
    prompt_texts: list[str],
    full_texts: list[str],
    output_texts: list[str],
    prompt_lengths: Counter[int],
    full_example_lengths: Counter[int],
    output_lengths: Counter[int],
) -> None:
    if not prompt_texts:
        return
    update_counter(prompt_lengths, tokenize_lengths(tokenizer, prompt_texts))
    update_counter(full_example_lengths, tokenize_lengths(tokenizer, full_texts))
    update_counter(output_lengths, tokenize_lengths(tokenizer, output_texts))
    prompt_texts.clear()
    full_texts.clear()
    output_texts.clear()


def write_counter_csv(path: Path, counter: Counter[int]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token_length", "count"])
        for token_length in sorted(counter):
            writer.writerow([token_length, counter[token_length]])


def counter_total(counter: Counter[int]) -> int:
    return sum(counter.values())


def weighted_percentile(counter: Counter[int], percentile: float) -> int:
    total = counter_total(counter)
    if total == 0:
        return 0
    threshold = max(1, math.ceil(total * percentile))
    running = 0
    for token_length in sorted(counter):
        running += counter[token_length]
        if running >= threshold:
            return token_length
    return max(counter)


def summary_stats(counter: Counter[int]) -> dict[str, float | int]:
    total = counter_total(counter)
    if total == 0:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
        }
    weighted_sum = sum(token_length * count for token_length, count in counter.items())
    return {
        "count": total,
        "min": min(counter),
        "max": max(counter),
        "mean": weighted_sum / total,
        "p50": weighted_percentile(counter, 0.50),
        "p90": weighted_percentile(counter, 0.90),
        "p95": weighted_percentile(counter, 0.95),
        "p99": weighted_percentile(counter, 0.99),
    }


def bucket_counter(counter: Counter[int], bucket_size: int) -> Counter[tuple[int, int]]:
    buckets: Counter[tuple[int, int]] = Counter()
    for token_length, count in counter.items():
        bucket_start = (token_length // bucket_size) * bucket_size
        bucket_end = bucket_start + bucket_size - 1
        buckets[(bucket_start, bucket_end)] += count
    return buckets


def display_buckets(
    buckets: Counter[tuple[int, int]], max_display_buckets: int
) -> list[tuple[tuple[int, int] | None, int | None]]:
    ordered = sorted(buckets.items())
    if len(ordered) <= max_display_buckets:
        return ordered

    keep_each_side = max(1, (max_display_buckets - 1) // 2)
    omitted = len(ordered) - (keep_each_side * 2)
    return ordered[:keep_each_side] + [(None, omitted)] + ordered[-keep_each_side:]


def print_counter_report(
    name: str,
    counter: Counter[int],
    bucket_size: int,
    max_display_buckets: int,
) -> None:
    stats = summary_stats(counter)
    print(f"\n{name}")
    print("-" * len(name))
    print(
        "count={count} min={min} max={max} mean={mean:.2f} "
        "p50={p50} p90={p90} p95={p95} p99={p99}".format(**stats)
    )

    buckets = bucket_counter(counter, bucket_size)
    max_count = max(buckets.values(), default=0)
    if max_count == 0:
        return

    for bucket, count in display_buckets(buckets, max_display_buckets):
        if bucket is None:
            print(f"... omitted {count} buckets ...")
            continue
        start, end = bucket
        bar_width = max(1, round((count / max_count) * 40))
        print(f"{start:>7}-{end:<7} {count:>8} {'#' * bar_width}")


def load_tokenizer(tokenizer_name: str, trust_remote_code: bool):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is required to run tokenization. Install the project "
            "dependencies, then rerun this script."
        ) from exc

    return AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.bucket_size <= 0:
        raise ValueError("--bucket-size must be positive")
    if args.max_display_buckets <= 0:
        raise ValueError("--max-display-buckets must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided")

    tokenizer = load_tokenizer(args.tokenizer, args.trust_remote_code)

    prompt_lengths: Counter[int] = Counter()
    full_example_lengths: Counter[int] = Counter()
    output_lengths: Counter[int] = Counter()
    prompt_texts: list[str] = []
    full_texts: list[str] = []
    output_texts: list[str] = []

    processed = 0
    for _, conversation in iter_examples(args.input, args.limit):
        prompt_text, full_text, output_text = render_conversation_parts(conversation)
        prompt_texts.append(prompt_text)
        full_texts.append(full_text)
        output_texts.append(output_text)
        processed += 1

        if len(prompt_texts) >= args.batch_size:
            flush_batch(
                tokenizer,
                prompt_texts,
                full_texts,
                output_texts,
                prompt_lengths,
                full_example_lengths,
                output_lengths,
            )
            print(f"Processed {processed} rows", flush=True)

    flush_batch(
        tokenizer,
        prompt_texts,
        full_texts,
        output_texts,
        prompt_lengths,
        full_example_lengths,
        output_lengths,
    )

    totals = {
        "prompt_lengths": counter_total(prompt_lengths),
        "full_example_lengths": counter_total(full_example_lengths),
        "output_lengths": counter_total(output_lengths),
    }
    for name, total in totals.items():
        if total != processed:
            raise RuntimeError(f"{name} counted {total} rows, expected {processed}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_counter_csv(args.output_dir / "prompt_lengths.csv", prompt_lengths)
    write_counter_csv(args.output_dir / "full_example_lengths.csv", full_example_lengths)
    write_counter_csv(args.output_dir / "output_lengths.csv", output_lengths)

    print(f"\nProcessed {processed} total rows")
    print(f"Wrote histogram CSVs to {args.output_dir}")
    print_counter_report(
        "Prompt token lengths", prompt_lengths, args.bucket_size, args.max_display_buckets
    )
    print_counter_report(
        "Full example token lengths",
        full_example_lengths,
        args.bucket_size,
        args.max_display_buckets,
    )
    print_counter_report(
        "Output token lengths", output_lengths, args.bucket_size, args.max_display_buckets
    )


if __name__ == "__main__":
    main()
