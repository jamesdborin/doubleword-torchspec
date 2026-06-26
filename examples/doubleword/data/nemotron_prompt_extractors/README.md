# Nemotron Prompt Extractors

These scripts extract one prompt CSV row per sample from the 49 datasets listed in
`../nemotron_post_training_v3_dataset_survey.md`.

Each `extract_*.py` wrapper targets one dataset. The shared extraction logic is in
`nemotron_prompt_extraction.py`.

Example:

```bash
python3 examples/doubleword/data/nemotron_prompt_extractors/extract_sft_agentic_v2.py \
  --output /tmp/nemotron_sft_agentic_v2_prompts.csv \
  --split train
```

For a dependency-isolated run when `datasets` is not installed in the active interpreter:

```bash
uv run --no-project --with datasets \
  examples/doubleword/data/nemotron_prompt_extractors/extract_rlhf_genrm_v1.py \
  --output /tmp/nemotron_rlhf_genrm_v1_prompts.csv \
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
- `system_prompt`
- `system_source`
- `tools`
- `tools_source`

The extractor always writes one output record for each input sample it iterates. If a row
cannot be normalized into a prompt, the output record is still written with `prompt: null`
and extraction metadata so CSV row count remains aligned with dataset sample count.
Nested values such as `tools` are JSON-encoded inside CSV cells.

For multi-turn chat/tool trajectories, the extractor preserves the first non-empty
user message in `prompt`. Non-empty leading system/developer context is preserved
separately in `system_prompt`, and available tool definitions are preserved
separately in `tools`, so downstream model-specific renderers can pass both through
the target chat/template and tool-calling APIs. Later assistant/tool turns are still
treated as completion/trajectory data and are not included in the prompt-only output.

## Full prompt-only export and upload

To launch one tmux pane per dataset in the Nemotron Post-Training v3 collection:

```bash
python3 examples/doubleword/data/nemotron_prompt_extractors/launch_prompt_only_tmux.py
```

This creates the `nemotron-prompts` tmux session with one worker pane per dataset.
By default, workers run through `uv` with `datasets==5.0.0` so repos with HF
`Json` feature metadata load correctly. Three workers may extract at once and
two workers may upload at once.
Outputs are written under `/tmp/nemotron_prompt_only_exports`:

- `<dataset>/prompts.csv`
- `<dataset>/summary.md`
- `<dataset>/null_or_empty_rows.md`
- `summary.md` with one aggregate row per dataset
- `dataset_manifest.csv` with tmux pane and target repo mapping

Each worker isolates Hugging Face cache under the output root and deletes that cache
after extraction. Uploads wait until Hugging Face is authenticated as `jamesdborin`,
then create or update `jamesdborin/<original-dataset-title>-prompt-only`, upload
`prompts.csv` plus Markdown diagnostics, and add dataset-card metadata that points
the Hugging Face dataset viewer at that CSV as the `train` split. The repo is then added to the
`Nemotron-Post-Training-v3 Prompt-Only` collection.

## Legacy local JSONL viewer

The local viewer is still available for older JSONL exports:

```bash
python3 examples/doubleword/data/nemotron_prompt_extractors/jsonl_viewer.py \
  --output-root /tmp/nemotron_prompt_only_exports
```

Open `http://127.0.0.1:8766`. The viewer scans for old `prompts.jsonl` and
`prompt.jsonl` files. New prompt-only exports are written as `prompts.csv` for
the Hugging Face dataset viewer.
