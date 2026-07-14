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
- Job `5654028` successfully wrote the 500,000-row tokenized dataset cache
  (21 GB) and downloaded Qwen3.5-9B, then failed trainer initialization because
  the recipe inherited the generic `model.embed_tokens.weight` key. The source
  W&B run used Qwen3.5's `model.language_model.embed_tokens.weight`; that key is
  now explicit in the model recipe.
- Job `5654541` initialized the trainer and SGLang model and entered the
  training loop, but SGLang's Qwen3.5 wrapper crashed on the first inference
  batch. Its generic auxiliary-state setup wrote a `layers_to_capture` field
  that Qwen3.5 does not consume, so no aux states were returned. The launcher
  now binds a narrow patch over `qwen3_vl.py` that delegates supplied layer IDs
  to Qwen3.5's per-layer capture setter while leaving the cached SIF immutable.
- Corrected the patch hunk length after the first dry-run validation caught a
  malformed unified-diff header; no training job was submitted with that draft.
- Submitted retry job `5654710` with reservation `brics_s6p`; it started on
  `nid010318` and reused the persistent 11 GB SIF, Qwen3.5-9B weights, and the
  500,000-row tokenized dataset cache.
- The patched SGLang Qwen3.5 auxiliary-state path passed the previous first-
  inference failure. The job completed its first optimizer step at 16:55:28
  and continued through at least step 15 while this log was updated. The first
  step reported loss 12.856; step 15 reported loss 11.395.
- W&B run `rh12fzxc` is live and receiving optimizer metrics at
  <https://wandb.ai/doubleword/qwen35-9b-dflash/runs/rh12fzxc>. API verification
  showed state `running`, `train/step=6`, `train/avg_loss=12.977486610412598`,
  and performance timing metrics. This is the first fully successful Isambard
  launch of the replicated run.
- At the user's request, cancelled job `5654710` after confirming the two-GPU
  launch. Began scaling the same node to one inference GPU plus three training
  GPUs. The launcher now defaults to FSDP2 `FULL_SHARD`, three training ranks,
  and micro-batch 8 (global batch 24), with environment knobs for repeatable
  batch-size probes and a `REPLICATE` data-parallel fallback.
- Scaled the node to Ray bundles `[1,2,3]` for training and GPU `0` for
  inference. FSDP2 logs confirm `FULL_SHARD` on all three training ranks.
- Batch-size search results (per-rank micro-batch; global batch is 3x): 8/24
  passed five steps; 16/48 passed three; 20/60 passed five; 22/66 passed two;
  23/69 passed step 1 then OOMed on step 2; 24, 25, 26, 28, and 32 OOMed.
  Therefore 22 is the largest tested multi-step-stable micro-batch, giving
  global batch 66 with accumulation 1.
- Larger pools initially deadlocked below one global batch or exhausted the
  16 GiB Mooncake segment when doubled. The launcher now sizes the inference
  pool and threshold to exactly one global batch. It also enables expandable
  CUDA allocator segments to reduce fragmentation.
- Launched the unrestricted three-epoch run as Slurm job `5655199` on
  `nid010318`: three FSDP2 `FULL_SHARD` training GPUs, one SGLang inference GPU,
  micro-batch 22, accumulation 1, and global batch 66. The resulting schedule
  is 22,725 optimizer steps (7,575 per epoch).
- The full run passed the batch-boundary check and continued beyond step 3.
  W&B run `m58w9g79` is live at
  <https://wandb.ai/doubleword/qwen35-9b-dflash/runs/m58w9g79>. API verification
  reported state `running`, `train/step=3`, `train/avg_loss=12.875295639038086`,
  and `perf/train_capacity=9.16365372635988`.
