#!/usr/bin/env python3
"""Aggregate DFlash SPEED-Bench evaluations into CSV and a nine-series plot."""

import argparse
import csv
import json
from pathlib import Path


SERIES = [f"draft_token_{i}_acceptance_pct" for i in range(1, 8)] + [
    "overall_draft_acceptance_pct",
    "average_acceptance_length",
]


def parse_spec(spec: str) -> tuple[str, str, Path]:
    try:
        label, step, result = spec.split("=", 2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected LABEL=STEP=RESULTS_JSON") from exc
    return label, step, Path(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, type=parse_spec)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--plot", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for label, step, path in args.run:
        summary = json.loads(path.read_text())["summary"]
        positions = summary["draft_token_acceptance_percentage_by_position"]
        row = {
            "label": label,
            "training_step": step,
            "source_results": str(path),
            "requests": summary["requests"],
            "conversation_turns": summary["conversation_turns"],
            "verify_steps": summary["verify_steps"],
            "num_draft_token_slots": len(positions),
            "average_accepted_draft_tokens": summary["average_accepted_draft_tokens"],
            "average_acceptance_length": summary[
                "average_acceptance_length_including_bonus"
            ],
            "overall_draft_acceptance_pct": summary[
                "overall_draft_token_acceptance_percentage"
            ],
        }
        for position in range(1, 8):
            row[f"draft_token_{position}_acceptance_pct"] = positions[str(position)]
        rows.append(row)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label", "training_step", "requests", "conversation_turns", "verify_steps",
        "num_draft_token_slots", *SERIES, "average_accepted_draft_tokens", "source_results",
    ]
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["label"] for row in rows]
    import matplotlib.pyplot as plt

    x = list(range(len(rows)))
    fig, acceptance_axis = plt.subplots(figsize=(13, 7.5))
    colors = plt.cm.viridis([i / 8 for i in range(8)])
    for position in range(1, 8):
        key = f"draft_token_{position}_acceptance_pct"
        acceptance_axis.plot(
            x, [row[key] for row in rows], marker="o", color=colors[position - 1],
            label=f"Draft token {position} accepted (%)",
        )
    acceptance_axis.plot(
        x, [row["overall_draft_acceptance_pct"] for row in rows], marker="s",
        linewidth=2.5, color=colors[7], label="Overall draft acceptance (%)",
    )
    acceptance_axis.set_ylabel("Acceptance probability (%)")
    acceptance_axis.set_ylim(0, 100)
    acceptance_axis.grid(True, alpha=0.25)

    length_axis = acceptance_axis.twinx()
    length_axis.plot(
        x, [row["average_acceptance_length"] for row in rows], marker="D",
        linewidth=3, linestyle="--", color="black", label="Average acceptance length",
    )
    length_axis.set_ylabel("Average acceptance length (tokens, including bonus)")
    length_axis.set_ylim(bottom=0)

    acceptance_axis.set_xticks(x, labels, rotation=15, ha="right")
    acceptance_axis.set_xlabel("Training checkpoint / reference model")
    acceptance_axis.set_title(
        "Qwen3.5-9B DFlash acceptance on NVIDIA SPEED-Bench qualitative"
    )
    lines = acceptance_axis.lines + length_axis.lines
    acceptance_axis.legend(lines, [line.get_label() for line in lines], loc="upper right", ncol=2)
    fig.tight_layout()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=200)
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
