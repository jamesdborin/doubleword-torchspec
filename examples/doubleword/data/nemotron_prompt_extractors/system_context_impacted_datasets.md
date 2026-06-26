# Initial Context and Tool Definition Impacted Datasets

Audited on 2026-06-25 after reviewing the Nemotron prompt extractor logic with
subagents across all `DATASET_SPECS` entries.

The shared extractor previously returned only the first user message for chat-style
`messages` and `responses_create_params.input` values. The datasets below had sampled
rows where non-empty system/developer messages, equivalent initial chat context, or
available tool definitions are part of the prompt and would be lost by that behavior.

## Covered by the Shared Initial-Context and Tool Fix

These datasets should be re-exported because the patched extractor now preserves
the first non-empty user message in `prompt`, with leading system/developer context
preserved separately in `system_prompt` and available tool definitions preserved
separately in `tools`.

| Dataset | Evidence |
| --- | --- |
| `nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1` | Sampled rows start with system instructions before user turns. |
| `nvidia/Nemotron-RL-Instruction-Following-Calendar-v2` | Sampled rows include long system/history context before the extracted user message. |
| `nvidia/Nemotron-SFT-Agentic-v2` | Valid rows start with non-empty domain policy system prompts before the user request. |
| `nvidia/Nemotron-SFT-OpenCode-v1` | Sampled rows start with system prompts and agent instructions. |
| `nvidia/Nemotron-3-Nano-RL-Training-Blend` | Sampled rows include system context before the user request. |
| `nvidia/Nemotron-Agentic-v1` | Sampled rows start with system policy/context before user turns. |
| `nvidia/Nemotron-SWE-v1` | Sampled rows include OpenHands system policy before the user task. |
| `nvidia/Nemotron-SFT-SWE-v2` | `openhands_swe` rows include system context before the user task. |
| `nvidia/Nemotron-SFT-Competitive-Programming-v2` | `exercism` and `text_to_sql` samples include system instructions. |
| `nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1` | Sampled rows include system coding-agent instructions before user/task turns. |
| `nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1` | Sampled rows include system policy before conversation turns. |
| `nvidia/Nemotron-RL-Instruction-Following-MultiTurnChat-v1` | Sampled rows include substantial system identity/context. |
| `nvidia/Nemotron-SFT-ARC-AGI-v1` | Sampled rows include system ARC instructions/output-format guidance. |
| `nvidia/Nemotron-SFT-Instruction-Following-Chat-v3` | `instruction_following` samples include system context; some `chat` rows need the first non-empty user fallback. |
| `nvidia/Nemotron-RL-QA-Abstention-v1` | Sampled rows include system context before the user question. |
| `nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1` | Sampled rows include system instructions before user input. |
| `nvidia/Nemotron-RL-ARC-AGI-v1` | Sampled rows include system framing before puzzle input. |
| `nvidia/Nemotron-RL-SysBench-v1` | System-following rows include system context before user turns. |
| `nvidia/Nemotron-RL-CFBench-v1` | Sampled rows include system context and sometimes prior turns. |
| `nvidia/Nemotron-RL-Multichallenge-v1` | Multi-turn challenge rows include system context. |
| `nvidia/Nemotron-RL-Ultra-Training-Blends` | Mixed blend samples include rows with system or multi-turn prompt context. |
| `nvidia/Nemotron-SFT-SWE-v3` | Sampled rows include system context before user task turns. |
| `nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2` | Some sampled rows include tool schemas or tool-call prompt context in Responses-style params. |
| `nvidia/Nemotron-Math-v2` | Tool-integrated rows can include top-level tool definitions. |
| `nvidia/Nemotron-SFT-CUDA-v1` | Sampled rows include top-level `tools` definitions for filesystem/shell actions. |
| `nvidia/Nemotron-SFT-Science-v2` | Tool-use rows can include top-level tool definitions. |
| `nvidia/Nemotron-RL-Science-v1` | Responses params can include `tools` definitions outside `input`. |
| `nvidia/Nemotron-SFT-Math-v3` | Mixed tool rows can include tool metadata/context outside the first user message. |

## Datasets With Sampled System Prompts

These datasets had sampled rows with non-empty initial system/developer context and
should be re-exported so that context moves from the old concatenated `prompt` form
into the separate `system_prompt` field.

| Dataset | Evidence |
| --- | --- |
| `nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1` | Sampled rows start with system instructions before user turns. |
| `nvidia/Nemotron-RL-Instruction-Following-Calendar-v2` | Sampled rows include long system/history context before the extracted user message. |
| `nvidia/Nemotron-SFT-Agentic-v2` | Valid rows start with non-empty domain policy system prompts before the user request. |
| `nvidia/Nemotron-SFT-OpenCode-v1` | Sampled rows start with system prompts and agent instructions. |
| `nvidia/Nemotron-3-Nano-RL-Training-Blend` | Sampled rows include system context before the user request. |
| `nvidia/Nemotron-Agentic-v1` | Sampled rows start with system policy/context before user turns. |
| `nvidia/Nemotron-SWE-v1` | Sampled rows include OpenHands system policy before the user task. |
| `nvidia/Nemotron-SFT-SWE-v2` | `openhands_swe` rows include system context before the user task. |
| `nvidia/Nemotron-SFT-Competitive-Programming-v2` | `exercism` and `text_to_sql` samples include system instructions. |
| `nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1` | Sampled rows include system coding-agent instructions before user/task turns. |
| `nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1` | Sampled rows include system policy before conversation turns. |
| `nvidia/Nemotron-RL-Instruction-Following-MultiTurnChat-v1` | Sampled rows include substantial system identity/context. |
| `nvidia/Nemotron-SFT-ARC-AGI-v1` | Sampled rows include system ARC instructions/output-format guidance. |
| `nvidia/Nemotron-SFT-Instruction-Following-Chat-v3` | `instruction_following` samples include system context. |
| `nvidia/Nemotron-RL-QA-Abstention-v1` | Sampled rows include system context before the user question. |
| `nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1` | Sampled rows include system instructions before user input. |
| `nvidia/Nemotron-RL-ARC-AGI-v1` | Sampled rows include system framing before puzzle input. |
| `nvidia/Nemotron-RL-SysBench-v1` | System-following rows include system context before user turns. |
| `nvidia/Nemotron-RL-CFBench-v1` | Sampled rows include system context and sometimes prior turns. |
| `nvidia/Nemotron-RL-Multichallenge-v1` | Multi-turn challenge rows include system context. |
| `nvidia/Nemotron-RL-Ultra-Training-Blends` | Mixed blend samples include rows with system or multi-turn prompt context. |
| `nvidia/Nemotron-SFT-SWE-v3` | Sampled rows include system context before user task turns. |
| `nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2` | `tool_calling_extraction` samples include system instruction context in `responses_create_params.input`. |

## Broader Context Risks Still Not Fully Covered

These datasets were also flagged by the audit, but the observed missing context was
primarily later conversation history or non-tool, non-message fields. These datasets
may need a separate prompt schema decision if that context must be preserved too.

| Dataset | Missing context risk |
| --- | --- |
| `nvidia/Nemotron-RL-Super-Training-Blends` | Mixed blend rows can include multi-turn context beyond the first user message. |
| `nvidia/Nemotron-Math-Proofs-v1` | Lean rows can need `formal_statement` and `lean_header` when `messages` is empty. |
| `nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2` | Some rows have multiple user messages or assistant/follow-up turns beyond the initial prompt. |
