#!/usr/bin/env python3
"""Plot GPT-OSS token-length histogram CSVs with matplotlib."""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "token_histograms"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "plots"
DEFAULT_BUCKET_SIZE = 128

HISTOGRAMS = {
    "prompt_lengths": "Prompt token lengths",
    "full_example_lengths": "Full example token lengths",
    "output_lengths": "Output token lengths",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot token-length histogram CSVs.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing histogram CSVs (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for PNG plots (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--bucket-size",
        type=int,
        default=DEFAULT_BUCKET_SIZE,
        help=f"Bucket width in tokens for plotted bars (default: {DEFAULT_BUCKET_SIZE})",
    )
    parser.add_argument(
        "--log-y",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a logarithmic y-axis (default: true).",
    )
    return parser.parse_args()


def read_counter(path: Path) -> Counter[int]:
    counter: Counter[int] = Counter()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            counter[int(row["token_length"])] = int(row["count"])
    return counter


def bucket_counter(counter: Counter[int], bucket_size: int) -> tuple[list[int], list[int]]:
    buckets: Counter[int] = Counter()
    for token_length, count in counter.items():
        buckets[(token_length // bucket_size) * bucket_size] += count
    starts = sorted(buckets)
    counts = [buckets[start] for start in starts]
    return starts, counts


def plot_counter(
    name: str,
    title: str,
    counter: Counter[int],
    output_dir: Path,
    bucket_size: int,
    log_y: bool,
) -> Path:
    import matplotlib.pyplot as plt

    starts, counts = bucket_counter(counter, bucket_size)
    output_path = output_dir / f"{name}.png"

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(starts, counts, width=bucket_size, align="edge", color="#2f6f9f", edgecolor="#1d405c")
    ax.set_title(title)
    ax.set_xlabel("Token length")
    ax.set_ylabel("Examples")
    if log_y:
        ax.set_yscale("log")
        ax.set_ylabel("Examples (log scale)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    if args.bucket_size <= 0:
        raise ValueError("--bucket-size must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = args.output_dir / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    for name, title in HISTOGRAMS.items():
        counter = read_counter(args.input_dir / f"{name}.csv")
        output_path = plot_counter(
            name,
            title,
            counter,
            args.output_dir,
            args.bucket_size,
            args.log_y,
        )
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
