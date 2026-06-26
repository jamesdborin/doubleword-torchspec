#!/usr/bin/env python3
"""Launch one tmux worker for sequential Nemotron prompt-only exports."""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path

from nemotron_prompt_extraction import DATASET_SPECS


DEFAULT_ORIGINAL_COLLECTION = "nvidia/nemotron-post-training-v3"
DEFAULT_COLLECTION_TITLE = "Nemotron-Post-Training-v3 Prompt-Only"
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
                    "tmux_window": "single-worker",
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
    cleanup_local_artifacts: bool,
    hold_open: bool = True,
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
    if cleanup_local_artifacts and not skip_upload:
        command.append("--cleanup-local-artifacts")

    command_text = shlex.join(command)
    shell_command = (
        f"cd {shlex.quote(str(repo_root))} && "
        f"mkdir -p {shlex.quote(str(log_dir))} && "
        "set -o pipefail && "
        f"{command_text} 2>&1 | tee -a {shlex.quote(str(log_path))}; "
        "rc=${PIPESTATUS[0]}; "
        f"echo; echo '[exit '$rc'] {shlex.quote(dataset_id)}'; "
    )
    if hold_open:
        shell_command += "exec bash"
    else:
        shell_command += "exit $rc"
    return shell_command


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


def write_single_worker_script(
    *,
    datasets: list[str],
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
    cleanup_local_artifacts: bool,
) -> Path:
    script_path = output_root / "run_single_worker.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -u",
        f"cd {shlex.quote(str(repo_root))}",
        f"mkdir -p {shlex.quote(str(output_root / 'logs'))}",
        f"echo '[single-worker start]' $(date -Iseconds) | tee -a {shlex.quote(str(output_root / 'logs' / 'single-worker.log'))}",
    ]

    for index, dataset_id in enumerate(datasets):
        command = build_worker_command(
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
            cleanup_local_artifacts=cleanup_local_artifacts,
            hold_open=False,
        )
        lines.extend(
            [
                "",
                f"echo '[dataset {index + 1}/{len(datasets)}] {dataset_id}' | tee -a {shlex.quote(str(output_root / 'logs' / 'single-worker.log'))}",
                f"bash -lc {shlex.quote(command)}",
                "rc=$?",
                "if [ \"$rc\" -ne 0 ]; then",
                f"  echo '[single-worker failed rc='$rc'] {dataset_id}' | tee -a {shlex.quote(str(output_root / 'logs' / 'single-worker.log'))}",
                "  exit \"$rc\"",
                "fi",
            ]
        )
    lines.extend(
        [
            "",
            f"rm -rf {shlex.quote(str(output_root / '.hf_cache'))}",
            f"find {shlex.quote(str(output_root))} -name '*.tmp' -o -name '*.partial' | xargs -r rm -f",
            f"echo '[single-worker complete]' $(date -Iseconds) | tee -a {shlex.quote(str(output_root / 'logs' / 'single-worker.log'))}",
            "exec bash",
        ]
    )
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def launch_tmux(
    *,
    session_name: str,
    script_path: Path,
) -> None:
    run(["tmux", "new-session", "-d", "-s", session_name, "-n", "single-worker"])
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
    pane_target = first_pane_id(f"{session_name}:single-worker")
    set_pane_title(pane_target, "single sequential worker")
    start_command(pane_target, str(script_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch one tmux worker for all Nemotron prompt-only exports."
    )
    parser.add_argument("--session-name", default=DEFAULT_SESSION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--collection-title", default=DEFAULT_COLLECTION_TITLE)
    parser.add_argument("--original-collection", default=DEFAULT_ORIGINAL_COLLECTION)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--max-upload-concurrent", type=int, default=1)
    parser.add_argument(
        "--panes-per-window",
        type=int,
        default=1,
        help="Deprecated; kept for CLI compatibility. The launcher uses one pane.",
    )
    parser.add_argument("--no-wait-for-auth", dest="wait_for_auth", action="store_false")
    parser.set_defaults(wait_for_auth=True)
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--no-uv", dest="use_uv", action="store_false")
    parser.set_defaults(use_uv=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--kill-existing", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--cleanup-local-artifacts",
        action="store_true",
        help="Delete each dataset's local artifact directory after successful upload.",
    )
    parser.add_argument(
        "--keep-local-artifacts",
        dest="cleanup_local_artifacts",
        action="store_false",
    )
    parser.set_defaults(cleanup_local_artifacts=True)
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
    )
    script_path = write_single_worker_script(
        datasets=datasets,
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
        cleanup_local_artifacts=args.cleanup_local_artifacts,
    )

    if args.dry_run:
        print(f"Would launch one sequential worker for {len(datasets)} datasets in session {args.session_name}")
        print(f"Manifest: {manifest_path}")
        print(f"Worker script: {script_path}")
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
        script_path=script_path,
    )

    print(f"Launched one sequential worker for {len(datasets)} datasets in tmux session {args.session_name}")
    print(f"Attach: tmux attach -t {args.session_name}")
    print(f"Output root: {output_root}")
    print(f"Manifest: {manifest_path}")
    print(f"Worker script: {script_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
