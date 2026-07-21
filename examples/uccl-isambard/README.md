# UCCL-P2P on Isambard Slingshot

These scripts validate TorchSpec's UCCL-P2P transfer dependency over Isambard's
Slingshot/CXI fabric. Mooncake remains TorchSpec's default backend.

## Install the pinned build

The tested source pin is `4312141c93c4b2f3ff7b9f274ad58c355c14fab6`.
The installer builds the low-level P2P extension inside the cached TorchSpec
SIF and installs it into a persistent user-writable directory. Run it on a
compute node, not the login node:

```bash
cd "$SCRATCH/torchspec/torchspec"
salloc --account=brics.s6p --partition=workq --reservation=brics_s6p \
  --nodes=1 --gpus=1 --cpus-per-task=16 --time=00:30:00
./examples/uccl-isambard/install_uccl.sh
```

Defaults can be overridden with `IMAGE`, `WORK`, `UCCL_SRC`, `UCCL_PYDEPS`,
or `JOBS`. `IMAGE` defaults to the cached
`$SCRATCH/torchspec-qwen35-9b/images/torchspec-latest.sif`.

The installer can continue to use the cached SIF. CXI runtime jobs must use the
`podman-hpc` helper described below: mounting the same libraries into Apptainer
still makes Cray `fi_getinfo(cxi)` return `ENODATA` on Isambard.

The runtime mounts the host Cray libfabric 1.22 tree, `libcxi`, `libnl`, CXI
devices, and CUDA 13 forward-compatibility libraries. It selects:

```bash
UCCL_P2P_TRANSPORT=cxi
UCCL_P2P_DISABLE_IPC=1
UCCL_SOCKET_IFNAME=hsn0
UCCL_LIBFABRIC_SO=/opt/libfabric/lib/libfabric.so.1
```

## Raw CXI smoke tests

The raw smoke deliberately uses UCCL's low-level `regv`, `advertisev`,
`readv_async`, and `poll_async` API. UCCL's high-level convenience transfer API
does not currently provide the metadata needed by the CXI path.

```bash
cd "$SCRATCH/torchspec/torchspec"
sbatch examples/uccl-isambard/raw_single_node.sbatch
sbatch examples/uccl-isambard/raw_two_node.sbatch
sbatch examples/uccl-isambard/baseline_single_node.sbatch
sbatch examples/uccl-isambard/regression_single_node.sbatch
sbatch examples/uccl-isambard/torchspec_single_node.sbatch
sbatch examples/uccl-isambard/torchspec_two_node.sbatch
```

Set `PAYLOAD_MIB=256` with `sbatch --export=ALL,PAYLOAD_MIB=256 ...` for a
larger check. Start at 1 MiB: Cray libfabric 1.22 has known large-registration
performance limitations, so a short correctness test should not begin with a
multi-GiB buffer. Success prints a producer release and a consumer byte-for-byte
verification. Each task also prints its node, local GPU, CXI transport, and
`hsn0` address; address discovery uses a socket ioctl and does not require `ip`.
The `torchspec_*` jobs repeat the transfer through `UcclP2PBackend` and verify
`TransferRef` serialization plus the consumer ACK/producer-registration lifecycle.

## Canonical podman-hpc runtime

`podman_common.sh` implements the working CXI container recipe from
`doublewordai/isambard-skill` commit
`7bb5ca56e08368955f867c7d09665f450e2efa1a`. It uses a persistent container
instead of repeatedly creating one-shot containers, and includes host networking,
host PID/IPC namespaces, supplementary groups, `/dev/cxi*`, GDRCopy, Cray
libfabric, the site aws-ofi-nccl plugin, and the production `FI_CXI_*` settings.

Prepare the image and CUDA 13 compatibility directory once on a compute node:

```bash
source examples/uccl-isambard/podman_common.sh
podman-hpc --squash-dir "$PODMANHPC_SQUASH_DIR" pull "$UCCL_OCI_IMAGE"
uccl_stage_podman_cuda_compat
```

Then create one container per allocated node, execute any number of commands,
and stop it explicitly:

```bash
source examples/uccl-isambard/podman_common.sh
name="$(uccl_podman_name training)"
uccl_podman_start "$name"
uccl_podman_exec "$name" python3 -c \
  'import torch; from uccl import p2p; torch.cuda.init(); print(p2p.Endpoint(0))'
uccl_podman_stop "$name"
```

For a single bounded probe, `uccl_podman_smoke <command> [args...]` performs the
same start/exec/stop lifecycle. Set `UCCL_PODMAN_EXTRA_MOUNTS` as a Bash array
before starting a container when a model directory needs an additional mount;
set `UCCL_PODMAN_FORWARD_ENV` to a space-separated list of extra environment
variable names. The default shared image store is
`/projects/s6p/$USER/podman-store`; override `PODMANHPC_SQUASH_DIR` if the image
has already been migrated elsewhere.

## TorchSpec selection

Choose UCCL in a normal HF or patched-SGLang TorchSpec launch with config
overrides:

```bash
source examples/uccl-isambard/podman_common.sh
export UCCL_P2P_TRANSPORT=cxi
export UCCL_SOCKET_IFNAME=hsn0
export UCCL_P2P_USE_GPU_DIRECT=0

name="$(uccl_podman_name training)"
uccl_podman_start "$name"
uccl_podman_exec "$name" python3 -m torchspec.train_entry \
  --config configs/sglang_qwen3_8b_dflash_smoke.yaml \
  transfer.backend=uccl-p2p \
  uccl.transport=cxi \
  uccl.use_gpu_direct=false \
  uccl.ack_advertise_host="$(uccl_hsn_address)" \
  training.num_train_steps=1
uccl_podman_stop "$name"
```

For multiple nodes, start one uniquely named container on each node and then
start the existing Ray cluster inside those containers. Use a
unique Ray head/GCS port derived from the Slurm job ID; do not reuse that port
for the Ray client/dashboard. For example:

```bash
RAY_GCS_PORT=$((20000 + SLURM_JOB_ID % 20000))
RAY_CLIENT_PORT=$((RAY_GCS_PORT + 1))
RAY_DASHBOARD_PORT=$((RAY_GCS_PORT + 2))
```

The UCCL producer advertises its endpoint through each `TransferRef`, so no
fixed data-plane port is required. Its acknowledgement listener chooses an
ephemeral port.

UCCL currently supports TorchSpec's HF path and the bundled patched-SGLang
spec-training path. The vLLM connector is still Mooncake-specific. Ulysses/USP
sharded transfers are not yet supported by the UCCL backend. Host-pinned staging
(`uccl.use_gpu_direct=false`) is the default because CXI GPU-memory registration
has not yet been validated on Isambard; enable GPU-direct only after a separate
fabric validation.
