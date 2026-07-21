#!/usr/bin/env python3
"""Extract comparable TorchSpec training scalars from TensorBoard event files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = (
    "train/avg_loss",
    "train/avg_acc",
    "train/grad_norm",
    "train/lr",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-steps", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    event_files = sorted(args.run_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise SystemExit(f"no TensorBoard event files found below {args.run_dir}")

    # Keep the latest wall-time value if a restarted run emitted a duplicate
    # step. This also makes extraction deterministic across multiple event files.
    collected: dict[str, dict[int, tuple[float, float]]] = {tag: {} for tag in TAGS}
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        scalar_tags = set(accumulator.Tags().get("scalars", ()))
        for tag in TAGS:
            if tag not in scalar_tags:
                continue
            for event in accumulator.Scalars(tag):
                previous = collected[tag].get(event.step)
                if previous is None or event.wall_time >= previous[0]:
                    collected[tag][event.step] = (event.wall_time, float(event.value))

    loss_steps = sorted(collected["train/avg_loss"])
    expected = list(range(1, args.expected_steps + 1))
    if loss_steps != expected:
        raise SystemExit(f"expected loss steps {expected}, found {loss_steps}")

    output: dict[str, list[dict[str, float | int]]] = {}
    for tag, by_step in collected.items():
        values = []
        for step in sorted(by_step):
            value = by_step[step][1]
            if not math.isfinite(value):
                raise SystemExit(f"non-finite {tag} at step {step}: {value}")
            values.append({"step": step, "value": value})
        output[tag] = values

    payload = {
        "run_dir": str(args.run_dir.resolve()),
        "event_files": [str(path.resolve()) for path in event_files],
        "expected_steps": args.expected_steps,
        "metrics": output,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
