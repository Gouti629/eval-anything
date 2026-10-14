# eval-harness

Lightweight, pip-installable evaluation harness for HuggingFace LLM models.

## Installation

```bash
pip install -e .

# With test dependencies
pip install -e ".[test]"
```

## Quick start

Run all built-in benchmarks:

```bash
eval-harness --model Qwen/Qwen2.5-0.5B --eval_config default
```

Run a single benchmark inline:

```bash
eval-harness --model Qwen/Qwen2.5-0.5B \
  --dataset openai/gsm8k --config main --split test \
  --question_field question --answer_field answer \
  --scorer number_extract --delimiter "####"
```

Compare multiple models:

```bash
eval-harness --model Qwen/Qwen2.5-0.5B --model Qwen/Qwen2.5-1.5B --eval_config default
```

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `--model` | (required) | HuggingFace model name or path. Repeat for multiple. |
| `--eval_config` | -- | Path to YAML config, or `default` for built-in benchmarks. |
| `--dataset` | -- | HuggingFace dataset name (inline mode). |
| `--config` | -- | Dataset config name. Repeat for multiple configs. |
| `--split` | `test` | Dataset split. |
| `--question_field` | -- | Field containing the prompt/question. |
| `--answer_field` | -- | Field containing the ground truth. |
| `--scorer` | `exact_match` | Scoring method: `exact_match`, `number_extract`, `boxed_latex`, `letter_match`, `code_exec`. |
| `--delimiter` | -- | Delimiter for `number_extract` (e.g. `####`). |
| `--formatter` | `plain` | Prompt format: `plain`, `multiple_choice`, `code_completion`. |
| `--choices_field` | -- | Field for multiple-choice options. |
| `--filter` | -- | Row filter expression (e.g. `task_id >= 11 and task_id <= 510`). |
| `--system_prompt` | `""` | System prompt string. |
| `--prepend_prompt` | `false` | Prepend question to output before scoring (for code completion). |
| `--batch_size` | `8` | Inference batch size. Auto-halved on OOM. |
| `--max_new_tokens` | `512` | Generation token limit. |
| `--temperature` | `0.0` | Sampling temperature. 0.0 = greedy. |
| `--pass_k` | -- | Samples for pass@k estimation (code evals only). |
| `--device` | `cuda` | Inference device. |
| `--think` | `false` | Keep `<think>` token in prompt for reasoning models. |
| `--output` | `./results` | Output directory for results. |
| `--comment` | -- | Free-text comment for the run. |

## YAML configuration

Each benchmark is a YAML dict with these fields:

```yaml
- name: gsm8k
  dataset: openai/gsm8k
  config: ['main']          # list or string
  split: test
  question_field: question
  answer_field: answer
  system_prompt: "Solve step by step. Put your answer after ####."
  scorer: number_extract
  delimiter: "####"
```

See `eval_harness/default_evals.yaml` for all 5 built-in benchmarks.

## Built-in benchmarks

| Name | Dataset | Scorer | Notes |
|---|---|---|---|
| gsm8k | openai/gsm8k | number_extract | Uses `####` delimiter |
| math | EleutherAI/hendrycks_math | boxed_latex | 7 algebra/geometry/etc configs |
| mbpp | google-research-datasets/mbpp | code_exec | Filtered to task_id 11-510 |
| humaneval | openai/openai_humaneval | code_exec | Prepends prompt to completion |
| arc | allenai/ai2_arc | letter_match | ARC-Challenge, multiple choice |

## Scoring semantics

Each scorer returns a `(score, info)` tuple:

- **score**: `1.0` (correct) or `0.0` (wrong/missing/ambiguous)
- **info.status**: `parsed`, `missing`, `ambiguous`, `fallback_number`
- **info.parsed**: The extracted answer used for comparison

### Scorer details

- **exact_match**: Case-insensitive, whitespace-normalized string match.
- **number_extract**: Extracts last number from output (or after delimiter). Handles commas, decimals, negatives, fractions, scientific notation.
- **boxed_latex**: Extracts `\boxed{...}` content. Falls back to last number if no `\boxed{}` found.
- **letter_match**: Multi-stage extraction: "the answer is X" patterns, standalone letter on a line, single-char response, single word-boundary letter. Returns `ambiguous` when multiple candidates conflict.
- **code_exec**: Runs extracted code + ground-truth assertions in a subprocess.

### Think block handling

All scorers strip `<think>...</think>` blocks before extraction. Unclosed `<think>` blocks preserve content (the side with substance).

## pass@k

For code evals, use `--pass_k K` to estimate pass@k with the unbiased estimator. Temperature defaults to 0.8 when pass_k is set.

## Resume behavior

Runs are fingerprinted using a SHA256 hash of: model names, eval configs (including dataset, fields, scorer, formatter, filter, delimiter, system prompt), generation parameters, and think mode. A partial result file is saved after each batch.

On restart, if a partial file with a matching fingerprint exists, the run resumes from where it left off. Examples are matched by content-hash IDs, not row indices -- reordering the dataset won't cause skips or duplicates.

If the config changes (different scorer, different filter, etc.), a new fingerprint is generated and the run starts fresh.

## Result schema

Results are saved as JSON with:

- **manifest**: Effective configuration, library versions, device, precision
- **results**: Per-model, per-eval scores with counts: `correct`, `incorrect`, `timed_out`, `oom`, `errored`, `scored`, `infra_failures`
- **examples**: Per-example records with `question`, `ground_truth`, `model_output`, `extracted`, `parse_status`, `score`, `status`
- **timing**: Generation time, scoring time, tokens/second

Infrastructure failures (timeout, OOM, errors) are reported separately from incorrect answers.

## Security limitations

### Code execution

Model-generated code runs in a subprocess with:
- Restricted environment variables (no PYTHONPATH)
- Working directory set to temp
- Resource limits on Linux (512MB memory, 32 processes, 10MB files)
- Timeout enforcement

**This is NOT a secure sandbox.** The subprocess runs on the host without OS-level isolation. Do not run untrusted model outputs on sensitive systems. For production use, consider Docker or firejail.

### Filter expressions

Filter expressions use a safe AST-based evaluator (not `eval()`). Only field comparisons, boolean operators, and literals are supported.

## Token throughput

Token counts are estimated by re-tokenizing the decoded output text. This is an approximation -- the true generated token count may differ slightly due to tokenizer round-trip losses. The metric is labeled `tokens_per_second` in timing data.

## Running tests

```bash
pip install -e ".[test]"
pytest
```

Tests run on Python 3.9+ and do not require GPU or network access.
