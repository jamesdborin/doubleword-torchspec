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
./examples/qwen35-9b-isambard/prepare_isambard.sh
sbatch ./examples/qwen35-9b-isambard/submit.sbatch
```

Inspect the job with:

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES%60
tail -f "$SCRATCH/torchspec-qwen35-9b/logs/slurm-JOB_ID.out"
```
