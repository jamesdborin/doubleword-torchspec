#!/usr/bin/env python3
"""Two-process end-to-end smoke for TorchSpec's UCCL transfer backend."""

import argparse
import json
import time
from pathlib import Path

import torch

from torchspec.transfer import TransferRef, TransferRole
from torchspec.transfer.uccl import UcclConfig, UcclP2PBackend


def wait_for(path: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.1)


def expected_tensors() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.arange(4096, dtype=torch.int64).reshape(1, -1),
        "hidden_states": torch.arange(4096 * 8, dtype=torch.bfloat16).reshape(1, 4096, 8),
    }


def producer(rendezvous: Path, timeout: float) -> None:
    backend = UcclP2PBackend(UcclConfig.from_env())
    backend.setup(TransferRole.PRODUCER, "cuda:0")
    ref = backend.put("torchspec-cxi-smoke", expected_tensors())
    temp = rendezvous / "transfer-ref.json.tmp"
    temp.write_text(json.dumps(ref.to_dict()))
    temp.replace(rendezvous / "transfer-ref.json")
    wait_for(rendezvous / "released", timeout)
    backend.health_check()
    if len(backend._registry) != 0:
        raise RuntimeError("consumer ACK did not release the producer registration")
    backend.close()
    print("TORCHSPEC_UCCL_PRODUCER_PASS", flush=True)


def consumer(rendezvous: Path, timeout: float) -> None:
    ref_path = rendezvous / "transfer-ref.json"
    wait_for(ref_path, timeout)
    ref = TransferRef.from_dict(json.loads(ref_path.read_text()))
    backend = UcclP2PBackend(UcclConfig.from_env())
    backend.setup(TransferRole.CONSUMER, "cuda:0")
    received = backend.get(ref, device="cpu")
    for name, expected in expected_tensors().items():
        torch.testing.assert_close(received[name], expected, rtol=0, atol=0)
    backend.release(ref)
    backend.close()
    (rendezvous / "released").touch()
    print("TORCHSPEC_UCCL_CONSUMER_PASS", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("producer", "consumer"), required=True)
    parser.add_argument("--rendezvous", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    args.rendezvous.mkdir(parents=True, exist_ok=True)
    if args.role == "producer":
        producer(args.rendezvous, args.timeout)
    else:
        consumer(args.rendezvous, args.timeout)


if __name__ == "__main__":
    main()
