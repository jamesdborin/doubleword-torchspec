# Nemotron Prompt Extractors

These scripts extract one prompt JSONL row per sample from the 49 datasets listed in
`../nemotron_post_training_v3_dataset_survey.md`.

Each `extract_*.py` wrapper targets one dataset. The shared extraction logic is in
`nemotron_prompt_extraction.py`.

Example:

```bash
python3 examples/doubleword/data/nemotron_prompt_extractors/extract_sft_agentic_v2.py \
  --output /tmp/nemotron_sft_agentic_v2_prompts.jsonl \
  --split train
```

For a dependency-isolated run when `datasets` is not installed in the active interpreter:

```bash
uv run --no-project --with datasets \
  examples/doubleword/data/nemotron_prompt_extractors/extract_rlhf_genrm_v1.py \
  --output /tmp/nemotron_rlhf_genrm_v1_prompts.jsonl \
  --split train
```

Output rows include:

- `dataset`
- `config`
- `split`
- `row_index`
- `prompt`
- `prompt_source`
- `prompt_source_detail`

The extractor always writes one output record for each input sample it iterates. If a row
cannot be normalized into a prompt, the output record is still written with `prompt: null`
and extraction metadata so JSONL line count remains aligned with dataset sample count.

For multi-turn chat/tool trajectories, only the first user message is extracted.
