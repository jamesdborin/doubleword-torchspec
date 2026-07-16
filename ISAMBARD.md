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

### 2026-07-15

- Slurm job `5655199` ultimately failed after 11:27:22 while attempting
  optimizer step 4234 (step 4233 completed, 18.6% of 22,725). W&B independently
  reports state `failed`, `train/step=4233`, `train/avg_loss=2.813598394393921`,
  and `train/avg_acc=0.4066530168056488`. The exception came from training rank
  0 on GPU 1 during the compiled FlexAttention backward Triton kernel:
  `RuntimeError: CUDA driver error: out of memory`. Slurm reported ordinary
  `FAILED (1:0)`, not a host-memory OOM; the batch step peaked at about 189 GB
  host RSS, below the 460 GB allocation. This was also not Mooncake `code=-200`.
- Reconstruction of the failed step from the exact 500,000-row tokenized cache
  overturned the initial “rare worst-case long batch” hypothesis. For a fresh
  epoch-0 stream, step 4234 consumes shuffled cache offsets 279,378–279,443.
  Stable longest-first partitioning gives rank maxima 7,657 / 7,265 / 7,008
  tokens, rounded by the collator to 7,680 / 7,424 / 7,168. Rank 0 has 67,167
  unpadded source tokens and 168,454 raw padded tokens; the global batch has
  201,413 source tokens, lighter than 99.8% of the preceding 4,233 batches.
  The top rank-0 sample is `magpie_055498` at 7,657 tokens.
- Crucially, every preceding batch put at least one 8,192-token sample on every
  rank, so every prior rank used the same 8,192 padded context shape. Step 4234
  was the first shorter shape on all ranks. TorchInductor generated new
  FlexAttention forward/backward artifacts at 05:33, exactly when the run
  failed. The failing generated graph independently encodes rank-0 context
  7,680 and KV length 9,728, confirming the reconstructed membership and rank.
  Its tensors are smaller than the previously successful 8,192/10,240 shape,
  but the driver OOM occurred at the first launch of the newly compiled Triton
  backward kernel, after its output buffers had been allocated. The strongest
  explanation is therefore shape-transition compile/module-load overhead or
  allocator fragmentation at an already marginal mbs22 high-water mark—not
  workload size. A CUDA module or new specialization needed additional device
  memory while the old 8,192-shape allocations/modules remained resident.
  The driver-level exception omitted allocator counters, so module load versus
  fragmentation cannot be separated retrospectively. The last valid checkpoint
  is `outputs/qwen35-9b-dflash-ultrachat-liger-a256-full-fsdp-mbs22-gbs66/checkpoints/iter_0004001`.
- Before the requested mbs20 restart, found job `5660233` already running an
  unintended/superseded mbs22 resume from `iter_0004001` into the old output and
  W&B run. Cancelled it after roughly 22 minutes (around step 4110) so it would
  not continue consuming a reserved node or race the requested replacement.
- Launched the requested resume as Slurm job `5660404` on `nid010542` with one
  inference GPU plus three FSDP2 `FULL_SHARD` training GPUs, micro-batch 20,
  accumulation 1, global batch 60, and inference pool/threshold 60. It restores
  model, optimizer, LR scheduler, RNG, and global step 4000 from
  `iter_0004001`; the new output directory is
  `outputs/qwen35-9b-dflash-ultrachat-liger-a256-resume-iter4001-full-fsdp-mbs20-gbs60`.
- Job `5660404` completed optimizer steps through at least 4010 and remains
  running. W&B run `etgj1ax8` is live at
  <https://wandb.ai/doubleword/qwen35-9b-dflash/runs/etgj1ax8>; API verification
  showed state `running`, `train/step=4010`, loss `3.0169689655303955`, accuracy
  `0.3803236484527588`, grad norm `0.22154049575328827`, and train capacity
  `7.76517521352067` samples/s.

## 2026-07-15: Qwen3.5-9B DFlash SPEED-Bench qualitative evaluation

- Evaluated the latest durable DFlash checkpoint,
  `$SCRATCH/torchspec-qwen35-9b/outputs/qwen35-9b-dflash-ultrachat-liger-a256-full-fsdp-mbs22-gbs66/checkpoints/iter_0004001`,
  against the same target/base model used for training, `Qwen/Qwen3.5-9B`.
  The active resumed training job had advanced beyond step 4600 but had not yet
  produced its next scheduled checkpoint, so `iter_0004001` remained the newest
  complete checkpoint at evaluation launch time.
- Added reusable checkpoint export and evaluation tooling in
  `scripts/tools/export_dflash_for_sglang.py`,
  `scripts/tools/evaluate_speed_bench_sglang.py`, and
  `examples/qwen35-9b-isambard/{run_speed_bench_eval.sh,submit_speed_bench_eval.sbatch}`.
  The exporter maps TorchSpec's projection/norm keys to native SGLang DFlash
  names and records the trained 256-dimensional attention head. A narrowly
  scoped bind-mounted SGLang patch records the acceptance histogram for DFlash,
  matching the existing Eagle/ngram instrumentation without modifying the SIF.
- Probe job `5662228` completed successfully after diagnosing and fixing four
  launch issues: direct script import path, Qwen3.5 speculative radix-cache
  incompatibility, an unrecognized draft `model_type`, and the trained draft's
  `head_dim=256`. The probe loaded `Qwen3_5ForConditionalGeneration` plus
  `DFlashDraftModel` across tensor parallel size 4 and emitted real DFlash
  verification/histogram metadata.
- Full Slurm job `5662290` ran on `nid010318` and completed successfully in
  `00:09:21` with four GH200 GPUs. Invocation:
  `sbatch --export=ALL,RUN_SUFFIX=iter4001-speed-bench-qualitative examples/qwen35-9b-isambard/submit_speed_bench_eval.sbatch`.
  It used SGLang DFLASH, TP=4, block size / verify window 8 (seven proposed
  draft tokens plus the current/bonus token), FlashInfer target and draft
  attention, greedy decoding, concurrency 32, and at most 4096 generated tokens
  per turn. The benchmark was `nvidia/SPEED-Bench`, config `qualitative`, split
  `test`: all 880 conversations and 1,036 turns.
- Results are in
  `$SCRATCH/torchspec-qwen35-9b/outputs/eval-iter4001-speed-bench-qualitative/results.json`
  (per-turn responses and metadata plus aggregate summary); server logs are in
  the same directory and Slurm logs are
  `$SCRATCH/torchspec-qwen35-9b/logs/speed-eval-5662290.{out,err}`.
- Across 636,973 DFlash verify steps, the accepted-draft histogram for accepting
  0 through 7 draft tokens was `[152465, 144462, 99566, 60601, 43091, 36934,
  32837, 67017]`. Independent validation confirmed that this sums to all verify
  steps and its weighted sum is exactly the 1,548,572 accepted drafts reported
  by SGLang.
- **Average acceptance length: 3.4327687359 tokens per verify step, including
  the mandatory target/bonus token.** Equivalently, the model accepted an
  average of 2.4311422933 speculative draft tokens per verify step. The overall
  accepted fraction among all seven proposed draft-token slots was 34.730604%.
- Conditional-by-position reporting requested for each draft slot (the
  percentage of verify steps in which that position was reached/accepted):
  draft token 1 **76.064135%**, token 2 **53.384680%**, token 3 **37.753563%**,
  token 4 **28.239659%**, token 5 **21.474694%**, token 6 **15.676332%**, and
  token 7 **10.521168%**.

## 2026-07-15: DFlash checkpoint acceptance sweep and z-lab reference

- Enumerated all durable DFlash checkpoint directories. The actual long-run
  trajectory contains `iter_0002001` and `iter_0004001`; the other `iter_3/4/6`
  directories came from short batch-size probes and were intentionally excluded
  from the over-time series.
- Reused the complete `iter_0004001` SPEED-Bench result from job `5662290` and
  ran the identical full protocol for `iter_0002001` as job `5662940` on
  `nid010318` (`COMPLETED`, `00:12:07`). Ran the public reference
  `z-lab/Qwen3.5-9B-DFlash` as job `5662941` on `nid010764` (`COMPLETED`,
  `00:13:05`), paired with the same `Qwen/Qwen3.5-9B` base model. Every result
  contains all 880 qualitative conversations / 1,036 turns.
- Independently validated every result: acceptance histograms sum to verify
  steps and histogram-weighted accepted drafts equal SGLang's reported accepted
  draft count. `iter_0002001` used 740,412 verify steps, `iter_0004001` used
  636,973, and z-lab used 543,618.
- Average acceptance length (including the mandatory target/bonus token)
  improved from **2.956061** at step 2001 to **3.432769** at step 4001. The
  z-lab reference measured **4.088571**.
- Position 1–7 acceptance percentages were:
  - `iter_0002001`: **72.1790, 47.7590, 30.7981, 19.9785, 12.8905, 7.7053,
    4.1558**; overall seven-slot acceptance **27.9237%**.
  - `iter_0004001`: **76.0641, 53.3847, 37.7536, 28.2397, 21.4747, 15.6763,
    10.5212**; overall seven-slot acceptance **34.7306%**.
  - `z-lab/Qwen3.5-9B-DFlash`: **80.6955, 58.7333, 42.6649, 32.0876,
    24.4059, 18.8907, 14.7383**. Its native window has 15 draft slots and its
    overall 15-slot acceptance is **20.5778%**, so the CSV records the window
    size explicitly while the requested plot compares the common first seven
    positions.
- Aggregated CSV:
  `$SCRATCH/torchspec-qwen35-9b/outputs/dflash-speed-bench-checkpoint-sweep.csv`.
  Single plot with exactly nine lines (positions 1–7, overall acceptance %, and
  average acceptance length on a secondary axis):
  `$SCRATCH/torchspec-qwen35-9b/outputs/dflash-speed-bench-checkpoint-sweep.png`.
  Local copies are under `artifacts/isambard/` with the same filenames.

## 2026-07-15: Added `iter_0006001` to the acceptance sweep

- The resumed mbs20 training trajectory produced a complete checkpoint at
  `$SCRATCH/torchspec-qwen35-9b/outputs/qwen35-9b-dflash-ultrachat-liger-a256-resume-iter4001-full-fsdp-mbs20-gbs60/checkpoints/iter_0006001`.
  It contains complete model, optimizer, scheduler, RNG, and metadata state.
- Full SPEED-Bench qualitative evaluation job `5666626` completed successfully
  on `nid010318` in `00:11:54`, using the same Qwen3.5-9B base model and protocol
  as the earlier sweep. It processed all 880 conversations / 1,036 turns and
  622,407 DFlash verify steps with no fatal SGLang errors.
- Independent arithmetic validation showed both the histogram-weighted total
  and SGLang's reported total equal **1,564,265 accepted draft tokens**.
  Average acceptance length including the bonus token was **3.5149154813**;
  average accepted speculative drafts were **2.5132509756**; overall seven-slot
  acceptance was **35.903585%**.
- Position 1–7 acceptance percentages were **76.773076%, 54.410217%,
  39.291975%, 29.562650%, 22.874582%, 17.048009%, and 11.364589%**.
- Regenerated the sweep CSV and nine-series plot with four rows/x-axis points:
  steps 2001, 4001, 6001, and the z-lab reference. Updated remote artifacts
  remain under `$SCRATCH/torchspec-qwen35-9b/outputs/`; updated local copies are
  under `artifacts/isambard/`.

## 2026-07-16: Extended acceptance sweep through step 12000

- Renewed the Isambard SSH certificate and inventoried the continuing mbs20
  trajectory. Found three new complete checkpoints: `iter_0008334` (global step
  8333, epoch-boundary save), `iter_0010001` (step 10000), and
  `iter_0012001` (step 12000). Each contains complete three-rank model state and
  metadata. Short `iter_3/4/6` batch-size probes remain excluded.
- Ran the full `nvidia/SPEED-Bench` qualitative/test evaluation with
  `Qwen/Qwen3.5-9B` for each new point. Jobs `5673634` (`iter_0008334`,
  `nid010764`, `00:12:00`), `5673635` (`iter_0010001`, `nid010318`,
  `00:11:38`), and `5673636` (`iter_0012001`, `nid010764`, `00:11:34`) all
  completed exit 0. Each processed all 880 conversations / 1,036 turns with no
  fatal SGLang markers.
- Independently validated each acceptance histogram against its verify count
  and SGLang accepted-draft total. New results:
  - `iter_0008334`: average acceptance length **3.583840**, overall seven-slot
    acceptance **36.887854%**; positions 1–7 **77.422340%, 55.278125%,
    40.258827%, 30.737558%, 24.073340%, 18.042969%, 12.401818%**.
  - `iter_0010001`: average acceptance length **3.652230**, overall seven-slot
    acceptance **37.864406%**; positions 1–7 **78.441046%, 56.533873%,
    41.539873%, 31.879016%, 24.859798%, 18.807614%, 12.989625%**.
  - `iter_0012001`: average acceptance length **3.693922**, overall seven-slot
    acceptance **38.459396%**; positions 1–7 **79.133171%, 57.336623%,
    42.200306%, 32.189104%, 25.302444%, 19.426029%, 13.628099%**.
- Regenerated the validated CSV with seven rows (six trajectory checkpoints plus
  z-lab) and the same exactly-nine-series plot. Remote artifacts:
  `$SCRATCH/torchspec-qwen35-9b/outputs/dflash-speed-bench-checkpoint-sweep.{csv,png}`.
  Updated local copies are under `artifacts/isambard/`.
