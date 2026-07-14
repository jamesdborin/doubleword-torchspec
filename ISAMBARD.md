# Qwen3.5-9B DFlash on Isambard

## Goal

Launch the current W&B Qwen3.5-9B DFlash recipe on Isambard through Slurm and
Apptainer, with reusable model, dataset, image, compilation, and tokenization
caches under `$SCRATCH`.

## Persistent layout

- Source: `$SCRATCH/torchspec/torchspec`
- Runtime root: `$SCRATCH/torchspec-qwen35-9b`
- Apptainer image: `$SCRATCH/torchspec-qwen35-9b/images/torchspec-latest.sif`
- Hugging Face model/dataset cache: `$SCRATCH/torchspec-qwen35-9b/hf-cache`
- TorchSpec/tokenized dataset cache: `$SCRATCH/torchspec-qwen35-9b/cache`
- Checkpoints and outputs: `$SCRATCH/torchspec-qwen35-9b/outputs`
- Slurm and TorchSpec logs: `$SCRATCH/torchspec-qwen35-9b/logs`
- W&B secret (mode 0600, never committed): `$SCRATCH/torchspec-qwen35-9b/secrets/wandb.env`

## Working log

### 2026-07-14

- Confirmed SSH access to `s6p.aip2.isambard` as `jamie.s6p`.
- Confirmed `$SCRATCH=/scratch/s6p/jamie.s6p`, Slurm access, and Apptainer 1.4.1.
- Confirmed there were no pre-existing jobs in the user queue.
- Selected the Hugging Face dataset `jamesdborin/qwen9B-500k-ultrachat-magpie`
  rather than transferring the large local JSONL tree. Hugging Face and derived
  tokenization caches are placed under the persistent runtime root.
- Added an Apptainer preparation script, compute-node execution wrapper, and
  two-GPU Slurm submission script. Runtime overrides match W&B run `u4140ivh`:
  micro-batch 8, accumulation 1, 256 anchors, Liger enabled, sample pool 32,
  and inference threshold 16.

## Reproduction

On Isambard, after syncing the repository and provisioning the W&B secret:

```bash
cd "$SCRATCH/torchspec/torchspec"
# The training job builds the SIF first when it is absent, avoiding a second
# queue wait. For image-only preparation, use prepare.sbatch instead.
sbatch ./examples/qwen35-9b-isambard/submit.sbatch
```

Inspect the job with:

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES%60
tail -f "$SCRATCH/torchspec-qwen35-9b/logs/slurm-JOB_ID.out"
```

The first interactive image build attempt downloaded all OCI layers but the
login-node process disappeared during extraction before producing the SIF. The
downloaded layers remained reusable in the scratch Apptainer cache. Image
assembly was therefore moved to `prepare.sbatch` so it runs on an allocated
compute node. Isambard schedules the supported jobs as full four-GPU node
allocations. The training wrapper explicitly exposes only GPUs `0,1` inside the
container, retaining the original run's one-inference/one-training topology.
The training submission also performs preparation when the SIF is absent. This
avoids completing an image-prep allocation only to return to the queue for a
second full-node allocation.

- Slurm rejected the initial 72-hour training request with
  `QOSMaxWallDurationPerJobLimit`. `sbatch --test-only` confirmed that 24 hours
  is accepted for this account, so the committed training wall time is 24h.
- Submitted training job `5653206`. Slurm accepted it with four GPUs and a
  24-hour wall time; it initially entered `PENDING (Priority)`.
- Liger preparation checks the image and persistent dependency layer first. If
  installation is required it uses `--no-deps`, avoiding a duplicate PyTorch
  and CUDA stack under scratch.
- Found the active account reservation `brics_s6p` (four nodes, through
  2026-08-01). The initial jobs omitted the reservation and consequently sat in
  the general priority queue. Both Slurm scripts now request it explicitly.
- Job `5653320` started immediately on reserved node `nid010318`, but the image
  build failed after about 17 minutes: expanding the ~30 GB OCI rootfs alongside
  the 13 GB reusable layer cache exceeded scratch quota. Persistent blobs and
  the final SIF remain on scratch; Slurm builds now use short-lived node-local
  `/tmp/apptainer-$SLURM_JOB_ID` for extraction.
- Retry job `5653798` built and cached the 11 GB SIF successfully, created W&B
  run `f4jihu3j`, started Ray/Mooncake with the correct 1+1 GPU placement, and
  loaded all 500,000 rows. It then OOM-killed during tokenization because the
  previous hard-coded default spawned 64 tokenizer processes. Dataset worker
  count is now a structured config field, and Isambard caps it at 8.
