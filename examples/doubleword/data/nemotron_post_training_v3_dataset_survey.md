# Nemotron Post-Training v3 Dataset Survey

Source collection: [nvidia/Nemotron-Post-Training-v3](https://huggingface.co/collections/nvidia/nemotron-post-training-v3). Inspected on 2026-06-24 with the Hugging Face collection API, dataset cards, dataset-server `info` / `first-rows` endpoints, and streamed first JSONL records for preview-only repos.

Count note: when the HF dataset viewer is partial or unavailable, I report the dataset-card total and note the indexed viewer count when it differs. "Prompt path" is the field to read for the model input; "completion / label" is the supervised target, reference answer, ranking, or verifier target.

## Summary

| # | Dataset | Samples | Format | Prompt path | Multi-turn? | Completion / label | Eval? |
|---:|---|---:|---|---|---|---|---|
| 1 | [Structured Outputs v2](https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2) | 62,696 | Parquet, Responses API request rows | `responses_create_params.input` | No, mostly single task | No completion; schema fields | Yes, schema/output-format validation |
| 2 | [Citation Formatting v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Citation-Formatting-v1) | 9,540 | JSON, Responses API request rows | `responses_create_params.input` | No | No completion; citation marker expectations | Yes, string/regex verifier |
| 3 | [Free-Form Formatting v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Free-Form-Formatting-v1) | 9,037 | JSON, Responses API request rows | `responses_create_params.input` | No | No completion; regex constraints | Yes, regex verifier |
| 4 | [Function Calling Pivot v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1) | 9,620 | JSON, tool-use state/action rows | `responses_create_params.input` | Yes | `expected_action` | Yes, next action comparison |
| 5 | [Calendar v2](https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Calendar-v2) | 9,915 | JSON, calendar task rows | `responses_create_params.input` | Usually no | `exp_cal_state` | Yes, calendar-state verifier |
| 6 | [SFT Agentic v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2) | 991,900 | JSONL chat/tool trajectories | `messages` | Mixed single and multi-turn | Assistant/tool turns in `messages` | No embedded eval; filtered by LLM judge upstream |
| 7 | [RL litmus-bench v0.1](https://huggingface.co/datasets/nvidia/Nemotron-RL-litmus-bench-v0.1) | 5,714 | JSON, chemistry QA rows | `responses_create_params.input` | No | `expected_answer` | Yes, chemistry/boxed-answer verifier |
| 8 | [RL Super Training Blends](https://huggingface.co/datasets/nvidia/Nemotron-RL-Super-Training-Blends) | 479,303 | JSONL blend files | Usually `responses_create_params.input` | Mixed | Depends on blend: `expected_action`, references, or preference labels | Yes, blend-specific verifiers/rewards |
| 9 | [SFT OpenCode v1](https://huggingface.co/datasets/nvidia/Nemotron-SFT-OpenCode-v1) | approx 459K | JSONL agent traces | `messages`; also `question` / `agent_prompt` | Yes | Assistant/tool turns in `messages` | No embedded eval |
| 10 | [Nano RL Training Blend](https://huggingface.co/datasets/nvidia/Nemotron-3-Nano-RL-Training-Blend) | 93,244 | JSONL RL blend | `responses_create_params.input` | Mixed | `ground_truth`, reward/pass-rate fields | Yes, environment/tool verifiers |
| 11 | [Math Proofs v1](https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v1) | 1,376,666 card total | JSON, viewer partial | `problem` or `messages` when present | Usually no | Verified Lean proof attempts in `messages` | Yes, Lean compiler verification |
| 12 | [Agentic v1](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1) | 335,122 | JSONL chat/tool trajectories | `messages` | Yes | Assistant/tool turns in `messages` | No embedded eval; filtered upstream |
| 13 | [Competitive Programming v1](https://huggingface.co/datasets/nvidia/Nemotron-Competitive-Programming-v1) | 3,927,984 | JSONL SFT code traces | `messages` | Usually no | Assistant solution in `messages` | Reference/test metadata only, no embedded runner |
| 14 | [Math v2](https://huggingface.co/datasets/nvidia/Nemotron-Math-v2) | 7,085,839 | Parquet SFT math rows | `messages` or `problem` | No | Assistant solution in `messages`; `expected_answer` | Reference answer metadata |
| 15 | [SWE v1](https://huggingface.co/datasets/nvidia/Nemotron-SWE-v1) | 51,029 card total | JSON, viewer partial; stringified JSON fields | `json.loads(messages)` | Yes | Assistant/tool trajectory turns | No embedded eval |
| 16 | [SFT SWE v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v2) | 256,254 | JSONL SFT SWE rows | `messages` | Mixed | Assistant/tool turns in `messages` | No embedded eval |
| 17 | [SFT Safety v1](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Safety-v1) | 45,145 | JSON SFT chat rows | `messages` | No | Assistant safe response in `messages` | No embedded eval |
| 18 | [SFT Competitive Programming v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Competitive-Programming-v2) | 844,935 | JSONL SFT code/SQL rows | `messages` | Usually no | Assistant solution in `messages` | Reference/test metadata only |
| 19 | [SpecializedDomains Finance v1](https://huggingface.co/datasets/nvidia/Nemotron-SpecializedDomains-Finance-v1) | 326,698 card total | JSON, viewer partial | `messages` | Usually no | Assistant answer in `messages` | No embedded eval |
| 20 | [SFT Instruction Following Chat v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v2) | 1,998,568 | JSONL SFT chat rows | `messages` | Mixed | Assistant turns in `messages` | No embedded eval |
| 21 | [RLHF GenRM v1](https://huggingface.co/datasets/nvidia/Nemotron-RLHF-GenRM-v1) | 299,517 card total | JSON preference/judge rows, viewer partial | `messages[0][0].content` | No | `score_1`, `score_2`, `ranking` | Yes, GenRM preference judging target |
| 22 | [RL ReasoningGym v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-ReasoningGym-v1) | 15,000 | JSONL procedural RL rows | `responses_create_params.input` | No | `answer` | Yes, task-specific deterministic verifier |
| 23 | [SFT Multilingual v1](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Multilingual-v1) | 3,065,255 | JSONL multilingual SFT rows | `messages` | No | Assistant solution in `messages` | No embedded eval |
| 24 | [RL Safety v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Safety-v1) | 89,068 hosted rows | JSON preference rows | `prompt` | No | `response1`, `response2`, scores/ranking | Yes, preference ranking label |
| 25 | [RL Identity Following v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Identity-Following-v1) | 21,660 | JSON Responses API rows | `responses_create_params.input` | No | Principle/rubric in `principle` | Yes, GenRM-style identity/rubric judge |
| 26 | [RL Agentic SWE Pivot v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1) | 50,308 indexed rows; card says 6,436 train tasks | JSON SWE step/action rows, viewer partial | `responses_create_params.input` | Yes | `expected_action`, `ref_patch`, `ref_message` | Yes, action/patch verifier |
| 27 | [RL Agentic Conversational Tool Use Pivot v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1) | 96,968 indexed rows | JSON tool-use step/action rows | `responses_create_params.input` | Yes | `expected_action` | Yes, next action comparison |
| 28 | [Instruction Following Adversarial v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Adversarial-v1) | 1,000 | JSON instruction/rubric rows | `prompt` or `responses_create_params.input` | No | `reference_response`, `rubric` | Yes, LLM judge rubric |
| 29 | [Instruction Following MultiTurnChat v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-MultiTurnChat-v1) | 2,011 | JSON multi-turn rubric rows | `responses_create_params.input` | Yes | `rubric`; context metadata | Yes, LLM judge rubric |
| 30 | [SFT ARC-AGI v1](https://huggingface.co/datasets/nvidia/Nemotron-SFT-ARC-AGI-v1) | 121,614 | JSONL SFT ARC traces | `messages` | Usually no | Assistant solution in `messages` | Generated rows were filtered by exact grid match |
| 31 | [SFT CUDA v1](https://huggingface.co/datasets/nvidia/Nemotron-SFT-CUDA-v1) | 2,276 | JSON SFT coding agent traces | `messages` | Yes | Assistant/tool turns in `messages` | No embedded eval; hidden tests described in prompt |
| 32 | [SFT Instruction Following Chat v3](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v3) | 887K card total | JSON chat rows, viewer partial | `messages` | Mixed | Assistant turns; chat split trains only last assistant | No embedded eval |
| 33 | [SFT Math v4](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Math-v4) | 545,431 | JSONL SFT math rows | `messages` or `problem` | Mixed for tool rows | Assistant solution in `messages`; `expected_answer` | Reference answer verified during filtering |
| 34 | [Math Proofs v2](https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v2) | 82,737 card total | JSON proof/verification SFT rows, viewer partial | `messages` or `problem` | No | Assistant proof/evaluation text in `messages` | Reference/rubric style, no standalone runner in row |
| 35 | [SFT Multilingual v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Multilingual-v2) | 370,081 | JSON SFT multilingual rows | `messages` | No | Assistant solution in `messages` | No embedded eval |
| 36 | [SFT Safety v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Safety-v2) | 130,350 | JSONL SFT safety rows | `messages` | No | Assistant safe response in `messages` | No embedded eval |
| 37 | [SFT Science v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Science-v2) | 2,837,712 card total | JSON science SFT rows, viewer partial | `messages` | Mixed for tool rows | Assistant answer in `messages` | No embedded eval |
| 38 | [RL Science v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Science-v1) | 150,644 | JSON RL science QA rows | `responses_create_params.input` | No | `expected_answer` | Yes, answer verifier/judge |
| 39 | [RL Math v2](https://huggingface.co/datasets/nvidia/Nemotron-RL-Math-v2) | 7,732 | JSON RL math rows | `responses_create_params.input` or `question` | No | `expected_answer` | Yes, `math_with_judge` verifier |
| 40 | [RL QA Abstention v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-QA-Abstention-v1) | 3,150 | JSON QA-abstention rows | `responses_create_params.input` or `messages` | No | `answer` | Yes, exact/boxed answer or `[IDK]` target |
| 41 | [RL Agentic Indirect Prompt Injection v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1) | 1,272 | JSON agentic security rows | `responses_create_params.input` | Usually no | `verifier_config`, injected environment state | Yes, trace-analysis verifier |
| 42 | [RL ARC-AGI v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-ARC-AGI-v1) | 21,028 | JSON ARC puzzle rows | `responses_create_params.input` | No | `expected_output` | Yes, exact grid match |
| 43 | [RL SysBench v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-SysBench-v1) | 1,010 | JSONL system-following rows | `responses_create_params.input` | Yes, 2 to 20 messages | `instructions`, `llm_judge` | Yes, instruction checks plus LLM judge |
| 44 | [RL CFBench v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-CFBench-v1) | 1,121 | JSON condition-following rows | `responses_create_params.input` | Yes | `instructions`, `llm_judge` | Yes, instruction checks plus LLM judge |
| 45 | [RL Multichallenge v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Multichallenge-v1) | 2,118 | JSON multi-turn challenge rows | `responses_create_params.input` | Yes | `llm_judge` and optional `instructions` | Yes, LLM judge checks |
| 46 | [RL InverseIFEval v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-InverseIFEval-v1) | 1,000 | JSON instruction-following rows | `responses_create_params.input` or `messages` | No | `llm_judge` with reference criteria | Yes, LLM judge checks |
| 47 | [RL Ultra Training Blends](https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends) | 337,721 card total; 54,201 viewer-indexed | JSON blend files | Usually `responses_create_params.input`; IFBench also `prompt` | Mixed | Depends on subset: verifier fields, references, rankings | Yes, subset-specific rewards/verifiers |
| 48 | [SFT SWE v3](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v3) | 237,970 | Parquet SFT SWE trajectories | `messages` | Yes | Assistant/tool turns in `messages` | No embedded eval |
| 49 | [SFT Math v3](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Math-v3) | 3,638,783 | JSONL SFT math rows | `messages` or `problem` | Mixed for tool rows | Assistant solution in `messages`; `expected_answer` | Reference answer metadata |

## Access Patterns

Most RL/RLVR datasets use a Responses API-style request:

```python
prompt_messages = row["responses_create_params"]["input"]
```

Most SFT datasets use chat messages:

```python
messages = row["messages"]
prompt = messages[:-1]
completion = messages[-1]  # usually the assistant target
```

Some agentic traces have many assistant/tool turns. In those, the "completion" is not one string but the assistant/tool trajectory inside `messages`. Some older viewer rows, notably `Nemotron-SWE-v1`, expose `messages` and `tools` as JSON strings, so parse them first:

```python
import json
messages = json.loads(row["messages"])
tools = json.loads(row["tools"])
```

Preference datasets are different:

```python
prompt = row["prompt"]
candidates = [row["response1"], row["response2"]]
label = row["preference_ranking"]
```

## Dataset Notes

### 1. nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2

- Sample shape: `responses_create_params`, `schema_str`, `schema_type`, `schema_fields_count`, `agent_ref`.
- Prompt access: `row["responses_create_params"]["input"]`, a list of chat messages containing output-schema instructions and source text.
- Completion access: none. The model is expected to generate a response matching the requested schema.
- Multi-turn: no conversational back-and-forth in the sampled row.
- Eval: yes. The verifier/agent validates the generated structured output against `schema_str` and related schema metadata.

### 2. nvidia/Nemotron-RL-Instruction-Following-Citation-Formatting-v1

- Sample shape: `license`, `responses_create_params`, `verifier`, `agent_ref`, `used_in`.
- Prompt access: `row["responses_create_params"]["input"][0]["content"]`.
- Completion access: none.
- Multi-turn: no.
- Eval: yes. `verifier` contains `type`, `patterns`, and `expected_markers`; the response is checked for required citation markers such as `[ref:N]`.

### 3. nvidia/Nemotron-RL-Instruction-Following-Free-Form-Formatting-v1

- Sample shape: `license`, `responses_create_params`, `verifier`, `agent_ref`, `used_in`.
- Prompt access: `row["responses_create_params"]["input"]`.
- Completion access: none.
- Multi-turn: no.
- Eval: yes. `verifier.verify_regex` and `verifier.verify_min_matches` specify regex checks over the model output.

### 4. nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1

- Sample shape: `trajectory_id`, `info`, `responses_create_params`, `expected_action`, `agent_ref`.
- Prompt access: `row["responses_create_params"]["input"]`; tools are in `row["responses_create_params"]["tools"]`.
- Completion access: `row["expected_action"]`.
- Multi-turn: yes. Sampled rows include prior user, assistant, reasoning, and tool state.
- Eval: yes. The next model action is compared with `expected_action`, usually a function call name/arguments or a message.

### 5. nvidia/Nemotron-RL-Instruction-Following-Calendar-v2

- Sample shape: `responses_create_params`, `exp_cal_state`, `agent_ref`.
- Prompt access: `row["responses_create_params"]["input"]`; calendar tools are under `row["responses_create_params"]["tools"]`.
- Completion access: no text completion; target state is `row["exp_cal_state"]`.
- Multi-turn: generally single task with calendar instructions.
- Eval: yes. The generated calendar is checked against expected event state, constraints, and conflicts.

### 6. nvidia/Nemotron-SFT-Agentic-v2

- Sample shape: streamed JSONL rows have `model`, `messages`, `tools`, `parallel_tool_calls`, `domain`, and metadata.
- Prompt access: `row["messages"]`; the initial user/system context is the prompt.
- Completion access: assistant and tool-call turns inside `row["messages"]`.
- Multi-turn: mixed. The card describes single-turn, multi-turn, and multi-step tool-use trajectories.
- Eval: no embedded per-row verifier. The card says trajectories were scored/filtered by an LLM judge upstream.

### 7. nvidia/Nemotron-RL-litmus-bench-v0.1

- Sample shape: `responses_create_params`, `expected_answer`, chemistry fields such as `smiles`, plus metadata.
- Prompt access: `row["responses_create_params"]["input"]`.
- Completion access: no completion; `row["expected_answer"]` is the target.
- Multi-turn: no.
- Eval: yes. The sampled task asks for a final boxed/integer answer, and `agent_ref` points to an RDKit chemistry verifier.

### 8. nvidia/Nemotron-RL-Super-Training-Blends

- Sample shape: varies by file. `rlvr*.jsonl` rows look like `responses_create_params`, `expected_action`, `pass_rate*`; `rlhf.jsonl` rows look like GenRM/rubric prompts.
- Prompt access: usually `row["responses_create_params"]["input"]`.
- Completion access: depends on blend, often `expected_action`, `ground_truth`, preference ranking, or rubric metadata.
- Multi-turn: mixed.
- Eval: yes. The blend combines RLVR, SWE, and RLHF rows, so evaluation is delegated to the corresponding tool/action verifier, reward function, or preference/rubric judge.

### 9. nvidia/Nemotron-SFT-OpenCode-v1

- Sample shape: streamed JSONL rows include `question`, `agent_prompt`, `messages`, `tools`, `enabled_tools`, `skills_path`.
- Prompt access: `row["messages"]`; `row["question"]` is the originating user task and `row["agent_prompt"]` is a synthetic high-level solution guide.
- Completion access: assistant/tool turns in `row["messages"]`.
- Multi-turn: yes, agentic CLI traces.
- Eval: no embedded evaluator. It is SFT trajectory data.

### 10. nvidia/Nemotron-3-Nano-RL-Training-Blend

- Sample shape: `id`, `responses_create_params`, `ground_truth`, `category`, `environment_name`, `agent_ref`, `pass_rate*`, `dataset`.
- Prompt access: `row["responses_create_params"]["input"]`.
- Completion access: `row["ground_truth"]` for tool/action targets when present.
- Multi-turn: mixed by source environment.
- Eval: yes. Rows include environment names, ground-truth actions, and pass-rate reward statistics for NeMo Gym-style RL.

### 11. nvidia/Nemotron-Math-Proofs-v1

- Sample shape: `problem`, `formal_statement`, `lean_header`, `messages`, `tools`, source/license fields.
- Prompt access: `row["problem"]` plus Lean fields; if `messages` is non-empty, use `row["messages"]`.
- Completion access: verified proof attempts in `row["messages"]`; sampled first row had an empty `messages` list.
- Multi-turn: no for proof prompts; not conversational.
- Eval: yes. The card describes verification by Lean 4 compilation; final verification depends on compiler success.

### 12. nvidia/Nemotron-Agentic-v1

- Sample shape: JSONL rows include `uuid`, `messages`, `license`, `used_in`, `tools`; interactive rows also include `reasoning`.
- Prompt access: `row["messages"]`.
- Completion access: assistant/tool turns in `row["messages"]`.
- Multi-turn: yes. The card focuses on multi-turn tool-use conversations.
- Eval: no embedded row-level verifier, though the card says trajectories were quality-filtered by a separate language-model judge.

### 13. nvidia/Nemotron-Competitive-Programming-v1

- Sample shape: `uuid`, `messages`, `tools`, dataset/source/split/index metadata.
- Prompt access: `row["messages"]`.
- Completion access: assistant solution in `row["messages"]`.
- Multi-turn: usually no; it is primarily problem-to-solution SFT.
- Eval: no embedded runner in the row. Some metadata links back to source benchmark indices, so evaluation would require reconstructing/running those benchmark tests externally.

### 14. nvidia/Nemotron-Math-v2

- Sample shape: `uuid`, `problem`, `expected_answer`, `messages`, `metadata`, `tools`, source/license fields.
- Prompt access: `row["messages"][0]` or `row["problem"]`.
- Completion access: assistant solution in `row["messages"]`; final reference in `row["expected_answer"]`.
- Multi-turn: no for standard rows, though tool-enabled rows can include tool role turns.
- Eval: reference-answer metadata is present. The row is SFT data, but math evaluation can compare the model's boxed final answer with `expected_answer`.

### 15. nvidia/Nemotron-SWE-v1

- Sample shape: `uuid`, `messages`, `tools`, `dataset`, `repo`; in the viewer sample, `messages` and `tools` were JSON strings.
- Prompt access: `json.loads(row["messages"])`, then use the initial system/user turns.
- Completion access: assistant/tool turns in the parsed `messages`.
- Multi-turn: yes, OpenHands-style software engineering trajectories.
- Eval: no embedded evaluator. SWE-Bench/R2E-style evaluation would require running generated patches in the corresponding repo/test environment.

### 16. nvidia/Nemotron-SFT-SWE-v2

- Sample shape: agentless rows have `messages`, `uuid`, `license`; OpenHands rows also include `metadata`, `tools`, and processing info.
- Prompt access: `row["messages"]`.
- Completion access: assistant/tool trajectory turns in `row["messages"]`.
- Multi-turn: mixed. Agentless rows can be shorter; OpenHands rows are multi-turn.
- Eval: no embedded evaluator in the row.

### 17. nvidia/Nemotron-SFT-Safety-v1

- Sample shape: `messages`, `uuid`, `license`, `used_in`.
- Prompt access: `row["messages"][0]`.
- Completion access: safe assistant response in `row["messages"][1]`.
- Multi-turn: no in the sampled row.
- Eval: no embedded evaluator. This is prompt/completion safety SFT data.

### 18. nvidia/Nemotron-SFT-Competitive-Programming-v2

- Sample shape: `messages`, `license`, `used_in`, `tools`, and source metadata; Text-to-SQL rows add `complexity` and `dialect`.
- Prompt access: `row["messages"]`.
- Completion access: assistant solution in `row["messages"]`.
- Multi-turn: usually no.
- Eval: no embedded runner. Source/test metadata may allow external benchmark-style evaluation.

### 19. nvidia/Nemotron-SpecializedDomains-Finance-v1

- Sample shape: `messages`, `uuid`, `license`, `used_in`.
- Prompt access: `row["messages"]`, usually system plus user financial document/question.
- Completion access: assistant answer in `row["messages"]`.
- Multi-turn: usually no.
- Eval: no embedded evaluator. It is prompt/completion finance SFT data.

### 20. nvidia/Nemotron-SFT-Instruction-Following-Chat-v2

- Sample shape: `messages`, `uuid`, `license`, `used_in`, `reasoning`.
- Prompt access: `row["messages"]`.
- Completion access: assistant turns in `row["messages"]`.
- Multi-turn: mixed; chat data can be multi-turn, instruction-following examples can be single-turn.
- Eval: no embedded evaluator.

### 21. nvidia/Nemotron-RLHF-GenRM-v1

- Sample shape: `messages`, `num_responses`, `score_1`, `score_2`, `ranking`, license fields.
- Prompt access: `row["messages"][0][0]["content"]` in the sampled row; it contains the judge prompt with conversation context, two candidate responses, criteria, and scoring guidelines.
- Completion access: there is no completion target. Labels are `score_1`, `score_2`, and `ranking`.
- Multi-turn: no as a model prompt; it is a comparative judging example.
- Eval: yes. Train/eval a reward/judge model to reproduce the provided scores and comparative ranking.

### 22. nvidia/Nemotron-RL-ReasoningGym-v1

- Sample shape: `responses_create_params`, `question`, `answer`, `metadata`, `agent_ref`, `uuid`, `license`.
- Prompt access: `row["responses_create_params"]["input"]` or `row["question"]`.
- Completion access: no completion; `row["answer"]` is the target.
- Multi-turn: no.
- Eval: yes. Reasoning Gym tasks are procedurally generated and algorithmically verifiable using task-specific metadata and answer checks.

### 23. nvidia/Nemotron-SFT-Multilingual-v1

- Sample shape: `domain`, `language`, `license`, `messages`, `used_in`, `uuid`.
- Prompt access: `row["messages"]`.
- Completion access: assistant answer/reasoning in `row["messages"]`.
- Multi-turn: no in sampled translated math/code rows.
- Eval: no embedded evaluator.

### 24. nvidia/Nemotron-RL-Safety-v1

- Sample shape: `prompt`, `response1`, `response2`, `principle`, `score1`, `score2`, `preference_ranking`, source/model metadata.
- Prompt access: `row["prompt"]`.
- Completion access: candidate responses are `row["response1"]` and `row["response2"]`; label is `row["preference_ranking"]`.
- Multi-turn: no.
- Eval: yes as preference/ranking data. The "eval" is not run as a generative benchmark; reward/judge models learn or are tested against the ranking and scores.

### 25. nvidia/Nemotron-RL-Identity-Following-v1

- Sample shape: `responses_create_params`, `agent_ref`, `dataset`, `principle`.
- Prompt access: `row["responses_create_params"]["input"]`.
- Completion access: no completion; `principle` contains the judging rubric/principle.
- Multi-turn: no in sampled row.
- Eval: yes. `agent_ref` points to a GenRM-style agent; outputs are judged against the identity-following principle.

### 26. nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1

- Sample shape: `trajectory_id`, `info`, `responses_create_params`, `ref_patch`, `ref_message`, `expected_action`, metadata, `pass_rate*`.
- Prompt access: `row["responses_create_params"]["input"]`, with tools under `row["responses_create_params"]["tools"]`.
- Completion access: `row["expected_action"]`; references may include `ref_patch` and `ref_message`.
- Multi-turn: yes, step-level SWE agent state.
- Eval: yes. The next action/tool call is compared to `expected_action`; SWE success can also be checked by applying/running the reference patch/test flow in the source environment.

### 27. nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1

- Sample shape: `trajectory_id`, `responses_create_params`, `expected_action`, scenario/meta fields, `pass_rate*`.
- Prompt access: `row["responses_create_params"]["input"]`.
- Completion access: `row["expected_action"]`.
- Multi-turn: yes.
- Eval: yes. The expected next message or tool call is compared with the model action and arguments.

### 28. nvidia/Nemotron-RL-Instruction-Following-Adversarial-v1

- Sample shape: `prompt`, `responses_create_params`, `reference_response`, `rubric`, judge prompt/system prompt, metadata.
- Prompt access: `row["prompt"]` or `row["responses_create_params"]["input"]`.
- Completion access: no training completion; `row["reference_response"]` is the standard/reference answer for judging.
- Multi-turn: no.
- Eval: yes. The judge prompt evaluates generated responses against each item in `rubric`.

### 29. nvidia/Nemotron-RL-Instruction-Following-MultiTurnChat-v1

- Sample shape: `responses_create_params`, `rubric`, `context`, metadata, `pass_rate*`.
- Prompt access: `row["responses_create_params"]["input"]`.
- Completion access: no completion; evaluation criteria are in `row["rubric"]`.
- Multi-turn: yes.
- Eval: yes. An LLM judge checks the model response against multi-turn rubric questions/pass criteria.

### 30. nvidia/Nemotron-SFT-ARC-AGI-v1

- Sample shape: streamed rows include `license`, `messages`, `tools`, `uuid`, `used_in`, `metadata`.
- Prompt access: `row["messages"]`.
- Completion access: assistant solution in `row["messages"]`.
- Multi-turn: usually no for no-tool configs; reasoning/tool configs may include tool turns.
- Eval: rows were retained only when the generating agent's submitted grid matched the ground-truth solution exactly. The released SFT rows do not carry a separate evaluator field.

### 31. nvidia/Nemotron-SFT-CUDA-v1

- Sample shape: `messages`, `tools`, `uuid`, `metadata`, license fields.
- Prompt access: `row["messages"]`.
- Completion access: assistant/tool trajectory turns in `row["messages"]`.
- Multi-turn: yes.
- Eval: no embedded evaluator. Prompts describe hidden tests; actual evaluation would compile/run CUDA tests outside the row.

### 32. nvidia/Nemotron-SFT-Instruction-Following-Chat-v3

- Sample shape: `messages`, `used_in`, `uuid`, `metadata`; some seed prompts are null/redacted in viewer samples, with assistant target retained.
- Prompt access: `row["messages"]`.
- Completion access: assistant turns in `row["messages"]`. The card notes that for the chat split only the last assistant turn should be used for training.
- Multi-turn: mixed.
- Eval: no embedded evaluator.

### 33. nvidia/Nemotron-SFT-Math-v4

- Sample shape: `messages`, `tools`, `problem`, `expected_answer`, source/subset/license fields.
- Prompt access: `row["messages"]` or `row["problem"]`.
- Completion access: assistant solution in `row["messages"]`; final answer reference in `row["expected_answer"]`.
- Multi-turn: mixed when Python tool-integrated reasoning rows include tool turns.
- Eval: not embedded as a runnable verifier, but the card says final answers were verified against reference answers and only matching samples were retained.

### 34. nvidia/Nemotron-Math-Proofs-v2

- Sample shape: `messages`, `tools`, `problem`, `source`, `dataset`, `subset`, license fields.
- Prompt access: `row["messages"]` or `row["problem"]`.
- Completion access: assistant proof, verification, or meta-verification text in `row["messages"]`.
- Multi-turn: no in sampled proof rows.
- Eval: contains proof/rubric-style prompts and subsets (`proof`, `verification`, `meta-verification`), but no standalone row-level runner.

### 35. nvidia/Nemotron-SFT-Multilingual-v2

- Sample shape: `messages`, `tools`, `license`, `uuid`, `used_in`, `metadata`.
- Prompt access: `row["messages"]`.
- Completion access: assistant solution in `row["messages"]`.
- Multi-turn: no in sampled code/math rows.
- Eval: no embedded evaluator.

### 36. nvidia/Nemotron-SFT-Safety-v2

- Sample shape: `uuid`, `messages`, `used_in`, `prompt_source`, `response_policy`, `translation_languages`, `metadata`.
- Prompt access: `row["messages"][0]`.
- Completion access: safe assistant response in `row["messages"]`.
- Multi-turn: no in sampled row.
- Eval: no embedded evaluator. This is safety SFT data; filtering was upstream.

### 37. nvidia/Nemotron-SFT-Science-v2

- Sample shape: `uuid`, `messages`, `tools`, `license`, `used_in`, `metadata`.
- Prompt access: `row["messages"]`.
- Completion access: assistant answer in `row["messages"]`.
- Multi-turn: mixed; tool-enabled rows can include tool turns.
- Eval: no embedded evaluator in sampled rows.

### 38. nvidia/Nemotron-RL-Science-v1

- Sample shape: `problem`, `expected_answer`, `responses_create_params`, `verifier_type`, `question_type`, metadata.
- Prompt access: `row["responses_create_params"]["input"]` or `row["problem"]`.
- Completion access: no completion; `row["expected_answer"]` is the target.
- Multi-turn: no.
- Eval: yes. The model output is checked against `expected_answer`, with verifier type/agent metadata indicating the answer-verification route.

### 39. nvidia/Nemotron-RL-Math-v2

- Sample shape: `question`, `expected_answer`, `responses_create_params`, `verifier_type`, `agent_ref`, `uuid`.
- Prompt access: `row["responses_create_params"]["input"]` or `row["question"]`.
- Completion access: no completion; `row["expected_answer"]` is the target.
- Multi-turn: no.
- Eval: yes. `verifier_type` is `math_with_judge`; evaluation checks the boxed mathematical answer against the verified reference, with judge support for equivalence.

### 40. nvidia/Nemotron-RL-QA-Abstention-v1

- Sample shape: `question`, `answer`, `responses_create_params`, `messages`, `tools`, source/domain metadata.
- Prompt access: `row["responses_create_params"]["input"]` or `row["messages"]`.
- Completion access: no completion; `row["answer"]` is the target.
- Multi-turn: no.
- Eval: yes. The expected output is the boxed answer or boxed `[IDK]`, encouraging abstention when unsure.

### 41. nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1

- Sample shape: `domain`, `attack_category`, `target_tool`, `responses_create_params`, `environment`, `injection`, `verifier_config`.
- Prompt access: `row["responses_create_params"]["input"]`, with tools and environment state in adjacent fields.
- Completion access: no completion; target behavior is encoded in `verifier_config` and attack metadata.
- Multi-turn: usually no in the prompt, but the task is agentic.
- Eval: yes. `verifier_config` uses trace analysis to check whether the agent resists or follows the indirect injection.

### 42. nvidia/Nemotron-RL-ARC-AGI-v1

- Sample shape: `responses_create_params`, `train`, `test_input`, `expected_output`, puzzle metadata.
- Prompt access: `row["responses_create_params"]["input"]`.
- Completion access: no natural-language completion; `row["expected_output"]` is the target grid.
- Multi-turn: no.
- Eval: yes. Execute/parse the model's transformation and compare the produced test grid exactly against `expected_output`.

### 43. nvidia/Nemotron-RL-SysBench-v1

- Sample shape: `instructions`, `llm_judge`, `responses_create_params`, `messages`, `tools`, metadata.
- Prompt access: `row["responses_create_params"]["input"]` or `row["messages"]`.
- Completion access: no completion; constraints live in `instructions` and judge checks in `llm_judge`.
- Multi-turn: yes. The card says rows contain 2 to 20 messages.
- Eval: yes. Deterministic instruction checks are combined with LLM-judge prompts for system-message-following constraints.

### 44. nvidia/Nemotron-RL-CFBench-v1

- Sample shape: `instructions`, `llm_judge`, `responses_create_params`, `messages`, `tools`.
- Prompt access: `row["responses_create_params"]["input"]`.
- Completion access: no completion; instruction and judge fields define success.
- Multi-turn: yes.
- Eval: yes. The response is checked against structured condition-following instructions and LLM-judge criteria.

### 45. nvidia/Nemotron-RL-Multichallenge-v1

- Sample shape: `instructions`, `llm_judge`, `responses_create_params`, `language`, `messages`, `tools`.
- Prompt access: `row["responses_create_params"]["input"]` or `row["messages"]`.
- Completion access: no completion; judge prompts define pass/fail.
- Multi-turn: yes.
- Eval: yes. The LLM judge evaluates memory, instruction retention, self-coherence, and other multi-turn criteria.

### 46. nvidia/Nemotron-RL-InverseIFEval-v1

- Sample shape: `instructions`, `llm_judge`, `responses_create_params`, `messages`, `tools`.
- Prompt access: `row["responses_create_params"]["input"]`.
- Completion access: no completion; standard/reference response is embedded inside the judge prompt.
- Multi-turn: no.
- Eval: yes. Each `llm_judge` item asks a strict yes/no criterion against the generated response, often based on a reference answer.

### 47. nvidia/Nemotron-RL-Ultra-Training-Blends

- Sample shape: depends on subset. IFBench rows include `prompt`, `kwargs`, `responses_create_params`, pass-rate stats, and nullable fields for other blend schemas; other files include RLHF, SWE, RLVR, and MOPD rows.
- Prompt access: usually `row["responses_create_params"]["input"]`; for IFBench also `row["prompt"]`.
- Completion access: subset-dependent: expected answers/actions, rubric/reference fields, or preference labels.
- Multi-turn: mixed.
- Eval: yes. The blend reuses each source dataset's reward/verifier: instruction checks, math/science answer checks, SWE/action checks, or GenRM/preference judging.

### 48. nvidia/Nemotron-SFT-SWE-v3

- Sample shape: `messages`, `uuid`, `license`.
- Prompt access: `row["messages"]`.
- Completion access: assistant/tool trajectory turns in `row["messages"]`.
- Multi-turn: yes, OpenHands-style software engineering traces.
- Eval: no embedded evaluator in the row.

### 49. nvidia/Nemotron-SFT-Math-v3

- Sample shape: `uuid`, `problem`, `expected_answer`, `messages`, `tools`, source/license fields, `tool_usage`.
- Prompt access: `row["messages"]` or `row["problem"]`.
- Completion access: assistant solution in `row["messages"]`; `row["expected_answer"]` is the reference final answer.
- Multi-turn: mixed when Python tool-integrated reasoning rows include tool turns.
- Eval: reference-answer metadata is present, but the released row is SFT data. Evaluation would compare the final answer with `expected_answer`, typically after extracting the boxed/final answer.

## Source Links

- Collection page: [nvidia/Nemotron-Post-Training-v3](https://huggingface.co/collections/nvidia/nemotron-post-training-v3)
- Collection API: `https://huggingface.co/api/collections/nvidia/nemotron-post-training-v3`
- Dataset-server info pattern: `https://datasets-server.huggingface.co/info?dataset=<dataset_id>`
- Dataset-server first rows pattern: `https://datasets-server.huggingface.co/first-rows?dataset=<dataset_id>&config=<config>&split=<split>`
- Dataset cards and raw files: each dataset link in the summary table above.
