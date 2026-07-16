#!/usr/bin/env python3
"""Run nvidia/SPEED-Bench with an SGLang server and aggregate DFlash acceptance."""

import argparse
import asyncio
import json
import time
import urllib.request
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.load(response)


def wait_until_ready(base_url: str, timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health_generate", timeout=10) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(5)
    raise TimeoutError(f"SGLang did not become ready within {timeout}s")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--num-draft-tokens", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--hf-cache")
    parser.add_argument("--ready-timeout", type=int, default=1800)
    args = parser.parse_args()

    await asyncio.to_thread(wait_until_ready, args.base_url, args.ready_timeout)
    dataset = load_dataset(
        "nvidia/SPEED-Bench", "qualitative", split="test", cache_dir=args.hf_cache
    )
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=args.hf_cache, trust_remote_code=True
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    async def generate(row: dict) -> dict:
        messages = []
        responses = []
        total_histogram = Counter()
        total_verify = total_correct = total_proposed = total_completion = 0
        for turn in row["turns"]:
            messages.append({"role": "user", "content": turn})
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            async with semaphore:
                result = await asyncio.to_thread(
                    post_json,
                    f"{args.base_url}/generate",
                    {
                        "text": prompt,
                        "sampling_params": {
                            "temperature": 0.0,
                            "max_new_tokens": args.max_new_tokens,
                        },
                    },
                )
            text = result["text"]
            meta = result.get("meta_info", {})
            responses.append({"text": text, "meta_info": meta})
            messages.append({"role": "assistant", "content": text})
            histogram = meta.get("spec_correct_drafts_histogram", [])
            total_histogram.update({i: count for i, count in enumerate(histogram)})
            total_verify += int(meta.get("spec_verify_ct", 0))
            total_correct += int(meta.get("spec_num_correct_drafts", 0))
            total_proposed += int(meta.get("spec_num_proposed_drafts", 0))
            total_completion += int(meta.get("completion_tokens", 0))
        return {
            "question_id": row["question_id"],
            "category": row["category"],
            "responses": responses,
            "histogram": dict(total_histogram),
            "verify_steps": total_verify,
            "correct_drafts": total_correct,
            "proposed_drafts": total_proposed,
            "completion_tokens": total_completion,
        }

    results = await asyncio.gather(*(generate(row) for row in dataset))
    histogram = Counter()
    for result in results:
        histogram.update({int(k): v for k, v in result["histogram"].items()})
    verify_steps = sum(x["verify_steps"] for x in results)
    correct = sum(x["correct_drafts"] for x in results)
    proposed = sum(x["proposed_drafts"] for x in results)
    completion = sum(x["completion_tokens"] for x in results)
    max_drafts = args.num_draft_tokens - 1
    if max_drafts <= 0:
        raise ValueError("--num-draft-tokens must be at least 2")
    accepted_by_position = [
        sum(v for k, v in histogram.items() if k >= position)
        for position in range(1, max_drafts + 1)
    ]
    summary = {
        "dataset": "nvidia/SPEED-Bench",
        "config": "qualitative",
        "split": "test",
        "requests": len(results),
        "conversation_turns": sum(len(x["responses"]) for x in results),
        "verify_steps": verify_steps,
        "average_acceptance_length_including_bonus": completion / verify_steps if verify_steps else None,
        "average_accepted_draft_tokens": correct / verify_steps if verify_steps else None,
        "overall_draft_token_acceptance_percentage": 100 * correct / proposed if proposed else None,
        "accepted_draft_count_histogram": dict(sorted(histogram.items())),
        "draft_token_acceptance_percentage_by_position": {
            str(i): 100 * count / verify_steps
            for i, count in enumerate(accepted_by_position, 1)
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "results": results}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
