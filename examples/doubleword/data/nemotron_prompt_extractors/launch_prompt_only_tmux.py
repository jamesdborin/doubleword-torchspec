#!/usr/bin/env python3
"""Launch one tmux pane per Nemotron prompt-only export worker."""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from pathlib import Path

from nemotron_prompt_extraction import DATASET_SPECS


DEFAULT_ORIGINAL_COLLECTION = "nvidia/nemotron-post-training-v3"
DEFAULT_COLLECTION_TITLE = "Nemotron-Post-Training-v3 Prompt-Only"
DEFAULT_OUTPUT_ROOT = Path("/tmp/nemotron_prompt_only_exports")
DEFAULT_OWNER = "jamesdborin"
DEFAULT_SESSION = "nemotron-prompts"


def run(command: list[str], capture: bool = False) -> str:
    if capture:
        return subprocess.check_output(command, text=True).strip()
    subprocess.run(command, check=True)
    return ""


def session_exists(session_name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def dataset_title(dataset_id: str) -> str:
    return dataset_id.rsplit("/", 1)[1]


def local_dataset_name(dataset_id: str) -> str:
    return dataset_id.replace("/", "__")


def prompt_only_repo_id(dataset_id: str, owner: str) -> str:
    return f"{owner}/{dataset_title(dataset_id)}-prompt-only"


def load_collection_datasets(collection_slug: str) -> list[str]:
    try:
        from huggingface_hub import HfApi

        collection = HfApi().get_collection(collection_slug)
        dataset_ids = [
            item.item_id
            for item in collection.items
            if item.item_type == "dataset" and item.item_id in DATASET_SPECS
        ]
    except Exception as exc:
        print(
            f"Could not read collection {collection_slug}; falling back to DATASET_SPECS: {exc}",
            file=sys.stderr,
        )
        dataset_ids = []

    seen = set(dataset_ids)
    missing = [dataset_id for dataset_id in DATASET_SPECS if dataset_id not in seen]
    return dataset_ids + missing


def write_manifest(
    output_root: Path,
    datasets: list[str],
    owner: str,
    panes_per_window: int,
) -> Path:
    manifest_path = output_root / "dataset_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "agent_index",
                "dataset_id",
                "prompt_only_repo_id",
                "local_output_dir",
                "tmux_window",
            ],
        )
        writer.writeheader()
        for index, dataset_id in enumerate(datasets):
            writer.writerow(
                {
                    "agent_index": index,
                    "dataset_id": dataset_id,
                    "prompt_only_repo_id": prompt_only_repo_id(dataset_id, owner),
                    "local_output_dir": str(output_root / local_dataset_name(dataset_id)),
                    "tmux_window": f"batch-{index // panes_per_window:02d}",
                }
            )
    return manifest_path


def build_worker_command(
    *,
    repo_root: Path,
    worker_script: Path,
    output_root: Path,
    dataset_id: str,
    owner: str,
    collection_title: str,
    max_concurrent: int,
    max_upload_concurrent: int,
    wait_for_auth: bool,
    skip_upload: bool,
    force: bool,
    limit: int | None,
    use_uv: bool,
) -> str:
    log_dir = output_root / "logs"
    log_path = log_dir / f"{local_dataset_name(dataset_id)}.log"
    if use_uv:
        command = [
            "uv",
            "run",
            "--no-project",
            "--isolated",
            "--with",
            "datasets==5.0.0",
            "--with",
            "huggingface_hub>=0.33.0",
            "--with",
            "requests",
            "python",
            str(worker_script),
        ]
    else:
        command = [sys.executable, str(worker_script)]
    command.extend(
        [
            "--dataset",
            dataset_id,
            "--output-root",
            str(output_root),
            "--owner",
            owner,
            "--collection-title",
            collection_title,
            "--semaphore-dir",
            str(output_root / "semaphore" / "extract"),
            "--max-concurrent",
            str(max_concurrent),
            "--upload-semaphore-dir",
            str(output_root / "semaphore" / "upload"),
            "--max-upload-concurrent",
            str(max_upload_concurrent),
        ]
    )
    if wait_for_auth:
        command.append("--wait-for-auth")
    if skip_upload:
        command.append("--skip-upload")
    if force:
        command.append("--force")
    if limit is not None:
        command.extend(["--limit", str(limit)])

    command_text = shlex.join(command)
    return (
        f"cd {shlex.quote(str(repo_root))} && "
        f"mkdir -p {shlex.quote(str(log_dir))} && "
        "set -o pipefail && "
        f"{command_text} 2>&1 | tee -a {shlex.quote(str(log_path))}; "
        "rc=${PIPESTATUS[0]}; "
        f"echo; echo '[exit '$rc'] {shlex.quote(dataset_id)}'; "
        "exec bash"
    )


def start_command(target: str, shell_command: str) -> None:
    run(["tmux", "respawn-pane", "-k", "-t", target, f"bash -lc {shlex.quote(shell_command)}"])


def set_pane_title(target: str, title: str) -> None:
    run(["tmux", "select-pane", "-t", target, "-T", title[:80]])


def first_pane_id(window_target: str) -> str:
    output = run(
        ["tmux", "list-panes", "-t", window_target, "-F", "#{pane_id}"],
        capture=True,
    )
    return output.splitlines()[0]


def launch_tmux(
    *,
    session_name: str,
    datasets: list[str],
    panes_per_window: int,
    repo_root: Path,
    worker_script: Path,
    output_root: Path,
    owner: str,
    collection_title: str,
    max_concurrent: int,
    max_upload_concurrent: int,
    wait_for_auth: bool,
    skip_upload: bool,
    force: bool,
    limit: int | None,
    use_uv: bool,
) -> None:
    run(["tmux", "new-session", "-d", "-s", session_name, "-n", "batch-00"])
    run(["tmux", "set-option", "-t", session_name, "pane-border-status", "top"])
    run(
        [
            "tmux",
            "set-option",
            "-t",
            session_name,
            "pane-border-format",
            "#{pane_index}: #{pane_title}",
        ]
    )

    for index, dataset_id in enumerate(datasets):
        window_index = index // panes_per_window
        pane_index = index % panes_per_window
        window_name = f"batch-{window_index:02d}"
        window_target = f"{session_name}:{window_name}"

        if pane_index == 0 and window_index > 0:
            run(["tmux", "new-window", "-t", session_name, "-n", window_name])
            pane_target = first_pane_id(window_target)
        elif pane_index == 0:
            pane_target = first_pane_id(window_target)
        else:
            pane_target = run(
                [
                    "tmux",
                    "split-window",
                    "-t",
                    window_target,
                    "-P",
                    "-F",
                    "#{pane_id}",
                ],
                capture=True,
            )

        title = f"agent-{index:02d} {dataset_title(dataset_id)}"
        set_pane_title(pane_target, title)
        shell_command = build_worker_command(
            repo_root=repo_root,
            worker_script=worker_script,
            output_root=output_root,
            dataset_id=dataset_id,
            owner=owner,
            collection_title=collection_title,
            max_concurrent=max_concurrent,
            max_upload_concurrent=max_upload_concurrent,
            wait_for_auth=wait_for_auth,
            skip_upload=skip_upload,
            force=force,
            limit=limit,
            use_uv=use_uv,
        )
        start_command(pane_target, shell_command)
        set_pane_title(pane_target, title)
        run(["tmux", "select-layout", "-t", window_target, "tiled"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch tmux workers for all Nemotron prompt-only exports."
    )
    parser.add_argument("--session-name", default=DEFAULT_SESSION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--collection-title", default=DEFAULT_COLLECTION_TITLE)
    parser.add_argument("--original-collection", default=DEFAULT_ORIGINAL_COLLECTION)
    parser.add_argument("--max-concurrent", type=int, default=3)
    parser.add_argument("--max-upload-concurrent", type=int, default=2)
    parser.add_argument("--panes-per-window", type=int, default=7)
    parser.add_argument("--no-wait-for-auth", dest="wait_for_auth", action="store_false")
    parser.set_defaults(wait_for_auth=True)
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--no-uv", dest="use_uv", action="store_false")
    parser.set_defaults(use_uv=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--kill-existing", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[4]
    worker_script = Path(__file__).with_name("export_prompt_only_dataset.py").resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    datasets = load_collection_datasets(args.original_collection)
    if len(datasets) != len(DATASET_SPECS):
        print(
            f"Warning: launching {len(datasets)} datasets, expected {len(DATASET_SPECS)}",
            file=sys.stderr,
        )
    manifest_path = write_manifest(
        output_root=output_root,
        datasets=datasets,
        owner=args.owner,
        panes_per_window=args.panes_per_window,
    )

    if args.dry_run:
        print(f"Would launch {len(datasets)} panes in session {args.session_name}")
        print(f"Manifest: {manifest_path}")
        return 0

    if session_exists(args.session_name):
        if args.kill_existing:
            run(["tmux", "kill-session", "-t", args.session_name])
        else:
            print(
                f"tmux session {args.session_name!r} already exists. "
                "Use --kill-existing to replace it.",
                file=sys.stderr,
            )
            return 1

    launch_tmux(
        session_name=args.session_name,
        datasets=datasets,
        panes_per_window=args.panes_per_window,
        repo_root=repo_root,
        worker_script=worker_script,
        output_root=output_root,
        owner=args.owner,
        collection_title=args.collection_title,
        max_concurrent=args.max_concurrent,
        max_upload_concurrent=args.max_upload_concurrent,
        wait_for_auth=args.wait_for_auth,
        skip_upload=args.skip_upload,
        force=args.force,
        limit=args.limit,
        use_uv=args.use_uv,
    )

    print(f"Launched {len(datasets)} panes in tmux session {args.session_name}")
    print(f"Attach: tmux attach -t {args.session_name}")
    print(f"Output root: {output_root}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
