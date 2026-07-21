#!/usr/bin/env python3
"""Integration test for the sglang spec training patch.

Launches a mooncake master subprocess, starts a sglang engine with
spec training enabled, sends a prefill-only request, retrieves the
hidden states from mooncake, and verifies shapes and non-zero content.

Usage:
    python tools/test_sglang_engine_patch.py --model-path <path_to_small_model>

Requires:
    - sglang with the spec training patch applied
    - mooncake_master binary available (via PATH or MOONCAKE_BUILD_DIR)
    - A small model for fast testing (e.g., Qwen2.5-0.5B)
"""

import argparse
import atexit
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback

import torch


def resolve_stored_tensor_metadata(meta, store, hidden_size, num_aux_layers):
    """Resolve the tensors the producer says it actually stored.

    New patched servers provide an authoritative serialized ``TransferRef``.
    The legacy protocol only provides a base key, so optional tensors must be
    discovered instead of being unconditionally assumed by the smoke test.
    """
    transfer_refs = meta.get("spec_training_transfer_refs", [])
    store_keys = meta.get("spec_training_mooncake_store_keys", [])
    if transfer_refs:
        try:
            from torchspec.transfer import TransferRef
        except ImportError:
            # A clean baseline checkout predates the transfer interface. Its
            # response only has legacy Mooncake keys, so continue below.
            print(
                "  NOTE: TransferRef is unavailable in this TorchSpec version; "
                "using legacy Mooncake metadata discovery"
            )
        else:
            ref = TransferRef.from_dict(transfer_refs[0])
            if ref.backend != "mooncake":
                raise AssertionError(f"Expected Mooncake transfer ref, got {ref.backend!r}")
            shapes = {name: spec.shape for name, spec in ref.tensors.items()}
            dtypes = {name: getattr(torch, spec.dtype) for name, spec in ref.tensors.items()}
            return ref.object_id, shapes, dtypes

    if not store_keys:
        raise AssertionError("No Mooncake store keys or transfer references returned")

    key = store_keys[0]
    seq_len = meta.get("prompt_tokens", 10)
    shapes = {
        "hidden_states": (seq_len, num_aux_layers * hidden_size),
        "input_ids": (seq_len,),
    }
    dtypes = {
        "hidden_states": torch.bfloat16,
        "input_ids": torch.int64,
    }
    if store.exists(f"{key}_lhs"):
        shapes["last_hidden_states"] = (seq_len, hidden_size)
        dtypes["last_hidden_states"] = torch.bfloat16
    else:
        print(
            "  NOTE: legacy SGLang response did not store last_hidden_states; "
            "validating the tensors that are present"
        )
    return key, shapes, dtypes


def find_free_port(start=10000, end=60000):
    for port in range(start, end):
        try:
            # Mooncake master and SGLang listen on TCP.  Probing UDP allowed
            # concurrent jobs on the same Slurm node to choose the same port.
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("", port))
            return port
        except OSError:
            continue
    raise RuntimeError(f"No free port found in range {start}-{end}")


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def find_mooncake_master_bin():
    if "MOONCAKE_BUILD_DIR" in os.environ:
        path = os.path.join(os.environ["MOONCAKE_BUILD_DIR"], "mooncake-store/src/mooncake_master")
        if os.path.exists(path):
            return path

    path = shutil.which("mooncake_master")
    if path:
        return path

    try:
        import mooncake

        pkg_dir = os.path.dirname(mooncake.__file__)
        path = os.path.join(pkg_dir, "mooncake_master")
        if os.path.exists(path):
            return path
    except ImportError:
        pass

    return None


def launch_mooncake_master(host, grpc_port, http_port):
    binary = find_mooncake_master_bin()
    if binary is None:
        print("ERROR: mooncake_master binary not found")
        sys.exit(1)

    cmd = [
        binary,
        f"--port={grpc_port}",
        f"--http_metadata_server_port={http_port}",
        "--http_metadata_server_host=0.0.0.0",
        "--enable_http_metadata_server=true",
    ]
    print(f"Launching mooncake master: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _cleanup():
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    atexit.register(_cleanup)
    time.sleep(2)

    if proc.poll() is not None:
        stderr = proc.stderr.read().decode()
        print(f"ERROR: mooncake master failed to start:\n{stderr}")
        sys.exit(1)

    print(f"Mooncake master running (PID={proc.pid}) on {host}:{grpc_port}, HTTP={http_port}")
    return proc


def set_mooncake_env(host, grpc_port, http_port):
    os.environ["MOONCAKE_LOCAL_HOSTNAME"] = host
    os.environ["MOONCAKE_MASTER_SERVER"] = f"{host}:{grpc_port}"
    os.environ["MOONCAKE_METADATA_SERVER"] = f"http://{host}:{http_port}/metadata"
    os.environ["MOONCAKE_PROTOCOL"] = "tcp"
    os.environ["MOONCAKE_GLOBAL_SEGMENT_SIZE"] = str(512 * 1024 * 1024)
    os.environ["MOONCAKE_LOCAL_BUFFER_SIZE"] = str(128 * 1024 * 1024)
    os.environ["MOONCAKE_HOST_BUFFER_SIZE"] = str(512 * 1024 * 1024)


def test_spec_training(model_path, aux_layer_ids=None):
    import sglang as sgl

    from torchspec.config.mooncake_config import MooncakeConfig
    from torchspec.transfer.mooncake import EagleMooncakeStore

    if aux_layer_ids is None:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        num_layers = config.num_hidden_layers
        aux_layer_ids = [2, num_layers // 2, num_layers - 3]
        if any(layer_id < 0 for layer_id in aux_layer_ids):
            aux_layer_ids = [0]
    num_aux_layers = len(aux_layer_ids)

    print("\n=== Launching sglang engine ===")
    print(f"  model_path: {model_path}")
    print(f"  aux_layer_ids: {aux_layer_ids}")

    engine = sgl.Engine(
        model_path=model_path,
        disable_cuda_graph=True,
        disable_radix_cache=True,
        enable_return_hidden_states=True,
        enable_aux_hidden_states=True,
        aux_hidden_state_layer_ids=aux_layer_ids,
        enable_spec_training_mooncake=True,
        trust_remote_code=True,
        mem_fraction_static=0.7,
        log_level="info",
    )

    print("Engine started successfully")

    # --- Send a spec training request ---
    test_prompt = "The quick brown fox jumps over the lazy dog"
    data_id = "test_sample_001"

    print("\n=== Sending spec training request ===")
    print(f"  prompt: {test_prompt!r}")
    print(f"  data_id: {data_id}")

    results = engine.generate(
        prompt=test_prompt,
        sampling_params={"max_new_tokens": 0},
        spec_training_data_id=data_id,
        return_hidden_states=True,
    )

    assert isinstance(results, dict), f"Expected dict, got {type(results)}"
    meta = results["meta_info"]
    store_keys = meta.get("spec_training_mooncake_store_keys", [])

    print(f"  meta_info keys: {list(meta.keys())}")
    print(f"  mooncake store keys: {store_keys}")
    print(f"  prompt_tokens: {meta.get('prompt_tokens')}")

    transfer_refs = meta.get("spec_training_transfer_refs", [])
    assert store_keys or transfer_refs, "No Mooncake store keys or transfer references returned"

    # --- Retrieve from mooncake and verify ---
    print("\n=== Retrieving hidden states from mooncake ===")

    mc_config = MooncakeConfig.from_env()
    mc_store = EagleMooncakeStore(mc_config)
    mc_store.setup(device=torch.device("cuda:0"))

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    hidden_size = config.hidden_size

    key, shapes, dtypes = resolve_stored_tensor_metadata(
        meta, mc_store, hidden_size, num_aux_layers
    )
    print(f"  Retrieving key: {key}")
    print(f"  Expected shapes: {shapes}")

    output = mc_store.get(key, shapes, dtypes, device=torch.device("cuda:0"))

    # --- Verify shapes ---
    print("\n=== Verifying tensors ===")

    hs = output.hidden_states
    ids = output.input_ids
    lhs = output.last_hidden_states

    print(f"  hidden_states: shape={tuple(hs.shape)}, dtype={hs.dtype}")
    print(f"  input_ids: shape={tuple(ids.shape)}, dtype={ids.dtype}")
    print(
        f"  last_hidden_states: shape={tuple(lhs.shape) if lhs is not None else None}, dtype={lhs.dtype if lhs is not None else None}"
    )

    assert tuple(hs.shape) == shapes["hidden_states"], (
        f"hidden_states shape mismatch: {tuple(hs.shape)} != {shapes['hidden_states']}"
    )
    assert tuple(ids.shape) == shapes["input_ids"], (
        f"input_ids shape mismatch: {tuple(ids.shape)} != {shapes['input_ids']}"
    )
    if "last_hidden_states" in shapes:
        assert lhs is not None, "last_hidden_states metadata was present but tensor is missing"
        assert tuple(lhs.shape) == shapes["last_hidden_states"], (
            f"last_hidden_states shape mismatch: {tuple(lhs.shape)} "
            f"!= {shapes['last_hidden_states']}"
        )
    else:
        assert lhs is None

    # --- Verify non-zero ---
    assert not torch.all(hs == 0), "hidden_states is all zeros"
    if lhs is not None:
        assert not torch.all(lhs == 0), "last_hidden_states is all zeros"
    assert not torch.all(ids == 0), "input_ids is all zeros"

    # --- Verify input_ids are valid token IDs ---
    assert ids.dtype == torch.int64, f"input_ids dtype should be int64, got {ids.dtype}"
    assert ids.min() >= 0, f"input_ids contains negative values: min={ids.min()}"

    # --- Verify hidden states have reasonable values (not NaN/Inf) ---
    assert not torch.any(torch.isnan(hs)), "hidden_states contains NaN"
    assert not torch.any(torch.isinf(hs)), "hidden_states contains Inf"
    if lhs is not None:
        assert not torch.any(torch.isnan(lhs)), "last_hidden_states contains NaN"
        assert not torch.any(torch.isinf(lhs)), "last_hidden_states contains Inf"

    print(f"\n  hidden_states norm: {hs.float().norm():.4f}")
    if lhs is not None:
        print(f"  last_hidden_states norm: {lhs.float().norm():.4f}")
    print(f"  input_ids sample: {ids[:10].tolist()}")

    # --- Cleanup ---
    mc_store.remove_eagle3_tensors(
        key, has_last_hidden_states="last_hidden_states" in shapes
    )
    mc_store.close()
    engine.shutdown()

    print(f"\n{'=' * 50}")
    print("ALL TESTS PASSED")
    print(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(description="Test sglang spec training patch")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to a small model for testing (e.g., Qwen/Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--aux-layer-ids",
        type=int,
        nargs="+",
        default=None,
        help="Layer IDs for aux hidden state capture. Auto-detected if not set.",
    )
    args = parser.parse_args()

    host = get_local_ip()
    grpc_port = find_free_port(51000)
    http_port = find_free_port(8100)

    master_proc = launch_mooncake_master(host, grpc_port, http_port)
    set_mooncake_env(host, grpc_port, http_port)

    try:
        test_spec_training(args.model_path, args.aux_layer_ids)
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        if master_proc.poll() is None:
            master_proc.terminate()
            master_proc.wait(timeout=5)


if __name__ == "__main__":
    main()
