#!/usr/bin/env python3
"""Two-process UCCL low-level READ smoke using a shared rendezvous directory."""

import argparse
import base64
import json
import os
import time
from pathlib import Path

import torch
from uccl import p2p


def require(result, operation):
    if not isinstance(result, tuple) or len(result) != 2 or not result[0]:
        raise RuntimeError(f"{operation} failed: {result!r}")
    return result[1]


def wait_for(path: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.1)


def expected_bytes(nbytes: int, *, pinned: bool = False) -> torch.Tensor:
    pattern = torch.arange(251, dtype=torch.uint8)
    value = pattern.repeat((nbytes + pattern.numel() - 1) // pattern.numel())[:nbytes]
    if pinned:
        destination = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        destination.copy_(value)
        return destination
    return value


def producer(endpoint, rendezvous: Path, nbytes: int, timeout: float) -> None:
    if not endpoint.start_passive_accept():
        raise RuntimeError("start_passive_accept failed")
    source = expected_bytes(nbytes, pinned=True)
    ptrs, sizes = [source.data_ptr()], [source.numel()]
    mrids = list(require(endpoint.regv(ptrs, sizes), "producer regv"))
    adverts = list(require(endpoint.advertisev(mrids, ptrs, sizes, 1), "advertisev"))
    payload = {
        "endpoint": base64.b64encode(bytes(endpoint.get_metadata())).decode("ascii"),
        "advertisements": [base64.b64encode(bytes(value)).decode("ascii") for value in adverts],
        "nbytes": nbytes,
    }
    temp = rendezvous / "metadata.json.tmp"
    temp.write_text(json.dumps(payload))
    temp.replace(rendezvous / "metadata.json")
    wait_for(rendezvous / "complete", timeout)
    for mrid in mrids:
        endpoint.dereg(mrid)
    print(f"producer retained and released {nbytes} bytes", flush=True)


def consumer(endpoint, rendezvous: Path, timeout: float) -> None:
    metadata_path = rendezvous / "metadata.json"
    wait_for(metadata_path, timeout)
    payload = json.loads(metadata_path.read_text())
    metadata = base64.b64decode(payload["endpoint"], validate=True)
    adverts = [base64.b64decode(value, validate=True) for value in payload["advertisements"]]
    nbytes = int(payload["nbytes"])
    destination = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    ip, port, bdf = p2p.Endpoint.parse_metadata(metadata)
    conn = require(endpoint.connect(ip, bdf, port), "connect")
    ptrs, sizes = [destination.data_ptr()], [destination.numel()]
    mrids = list(require(endpoint.regv(ptrs, sizes), "consumer regv"))
    transfer = require(endpoint.readv_async(conn, mrids, ptrs, sizes, adverts, 1), "readv_async")
    deadline = time.monotonic() + timeout
    while not require(endpoint.poll_async(transfer), "poll_async"):
        if time.monotonic() >= deadline:
            raise TimeoutError("UCCL transfer timed out")
        time.sleep(0.001)
    expected = expected_bytes(nbytes)
    torch.testing.assert_close(destination, expected, rtol=0, atol=0)
    for mrid in mrids:
        endpoint.dereg(mrid)
    endpoint.remove_remote_endpoint(conn)
    (rendezvous / "complete").touch()
    print(f"consumer verified {nbytes} bytes", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("producer", "consumer"), required=True)
    parser.add_argument("--rendezvous", type=Path, required=True)
    parser.add_argument("--payload-mib", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    if args.payload_mib <= 0:
        parser.error("--payload-mib must be positive")
    args.rendezvous.mkdir(parents=True, exist_ok=True)
    # Slurm exposes one GPU per task, so its process-local CUDA index is zero.
    gpu = int(os.environ.get("UCCL_P2P_LOCAL_GPU_IDX", "0"))
    endpoint = p2p.Endpoint(gpu)
    print(
        f"role={args.role} host={os.uname().nodename} gpu={gpu} "
        f"transport={os.environ.get('UCCL_P2P_TRANSPORT')}",
        flush=True,
    )
    if args.role == "producer":
        producer(endpoint, args.rendezvous, args.payload_mib * 1024 * 1024, args.timeout)
    else:
        consumer(endpoint, args.rendezvous, args.timeout)


if __name__ == "__main__":
    main()
