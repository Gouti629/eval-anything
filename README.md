# eval-anything

Lightweight LLM evaluation harness with vLLM-accelerated inference, resumable runs, and a Python API for training-loop integration.

## Installation

```bash
pip install -e .              # HuggingFace backend (CPU/Mac/CUDA)
pip install -e ".[vllm]"      # + vLLM backend (CUDA required)
pip install -e ".[test]"      # + test dependencies
```

## Quick start

```bash
# Run all built-in benchmarks with vLLM (default)
eval-anything --model Qwen/Qwen2.5-0.5B --eval_config default

# Use HuggingFace backend (no CUDA needed)
eval-anything --model Qwen/Qwen2.5-0.5B --eval_config default --backend hf --device cpu

# Single benchmark inline
eval-anything --model Qwen/Qwen2.5-0.5B \
  --dataset openai/gsm8k --config main --split test \
  --question_field question --answer_field answer \
  --scorer number_extract --delimiter "####"

# Auto-tune: given batch size, maximize token length from GPU memory
eval-anything --model Qwen/Qwen2.5-7B --eval_config default --batch_size 32 --max_new_tokens auto

# Auto-tune: given token length, maximize batch size from GPU memory
eval-anything --model Qwen/Qwen2.5-7B --eval_config default --max_new_tokens 2048 --batch_size auto

# Multi-GPU with tensor parallelism
eval-anything --model meta-llama/Llama-3-70B --eval_config default --tensor_parallel 4

# 5-shot prompting
eval-anything --model Qwen/Qwen2.5-0.5B --eval_config default --few_shot 5

# Compare two result files
eval-anything --compare results/run1.json results/run2.json
```

## Python API

`quick_eval` accepts a pre-loaded model for use inside training loops:

```python
from eval_harness import quick_eval

# vLLM — tokenizer auto-extracted
results = quick_eval(model=llm, eval_configs="default", limit=50)

# HuggingFace
results = quick_eval(model=hf_model, tokenizer=tokenizer, eval_configs="default", limit=50)

# Returns: {"gsm8k": {"score": 0.42, "n": 50, "correct": 21, "timing": {...}}, ...}
```

### WandB integration

```python
import wandb

wandb.init(project="evals")

def log_progress(info):
    wandb.log({f"{info['eval']}/score": info["score_so_far"],
               f"{info['eval']}/progress": info["done"] / info["total"]})

results = quick_eval(model=llm, callback=log_progress)
for name, r in results.items():
    wandb.log({f"{name}/final_score": r["score"],
               f"{name}/tokens_per_second": r["timing"]["tokens_per_second"]})
```

## Built-in benchmarks

| Name | Dataset | Scorer | Notes |
|---|---|---|---|
| gsm8k | openai/gsm8k | number_extract | `####` delimiter |
| math | EleutherAI/hendrycks_math | boxed_latex | 7 subject configs |
| mmlu | cais/mmlu | letter_match | 57 subjects, knowledge retention |
| mbpp | google-research-datasets/mbpp | code_exec | task_id 11–510 |
| humaneval | openai/openai_humaneval | code_exec | Prepends prompt |
| livecodebench_v5 | PrimeIntellect/LiveCodeBench-v5 | code_exec | stdin/func test types |
| arc | allenai/ai2_arc | letter_match | ARC-Challenge |

## CLI reference

**Inference**

| Flag | Default | Description |
|---|---|---|
| `--model` | (required) | Model name or path. Repeat for multiple. |
| `--backend` | `vllm` | `vllm` (GPU) or `hf` (CPU/Mac/CUDA). |
| `--device` | `cuda` | Inference device. |
| `--batch_size` | `8` | Batch size, or `auto` to maximize from GPU memory. |
| `--max_new_tokens` | `512` | Token limit, or `auto` to maximize from GPU memory. |
| `--temperature` | `0.0` | Sampling temperature. 0 = greedy. |
| `--tensor_parallel` | `1` | Number of GPUs for tensor parallelism (vLLM only). |
| `--think` | off | Keep `<think>` token for reasoning models. |

**Eval specification**

| Flag | Default | Description |
|---|---|---|
| `--eval_config` | — | YAML config path, or `default`. |
| `--dataset` | — | HuggingFace dataset (inline mode). |
| `--config` | — | Dataset config name. Repeat for multiple. |
| `--split` | `test` | Dataset split. |
| `--question_field` | — | Prompt field. |
| `--answer_field` | — | Ground truth field. |
| `--scorer` | `exact_match` | `exact_match`, `number_extract`, `boxed_latex`, `letter_match`, `code_exec`. |
| `--formatter` | `plain` | `plain`, `multiple_choice`, `code_completion`. |
| `--few_shot` | — | Number of few-shot examples from train split. |
| `--few_shot_split` | `train` | Split to sample few-shot examples from. |
| `--few_shot_seed` | `42` | Random seed for few-shot sampling. |
| `--pass_k` | — | Samples for pass@k (code evals). Sets temperature to 0.8. |
| `--filter` | — | Row filter expression (e.g. `task_id >= 11`). |
| `--sandbox` | `auto` | Code execution sandbox: `auto`, `docker`, `bwrap`, `none`. |
| `--limit` | — | Evaluate only first N examples. |

**Output**

| Flag | Default | Description |
|---|---|---|
| `--output` | `./results` | Results directory. |
| `--comment` | — | Free-text run comment. |
| `--compare` | — | Compare 2+ result files instead of running evals. |

## YAML configuration

```yaml
- name: gsm8k
  dataset: openai/gsm8k
  config: ['main']
  split: test
  question_field: question
  answer_field: answer
  system_prompt: "Solve step by step. Put your answer after ####."
  scorer: number_extract
  delimiter: "####"
  few_shot: 5                # optional
  few_shot_split: train      # optional
  max_new_tokens: 1024       # optional per-eval override
```

## Sandboxed code execution

Code evals (`code_exec` scorer) run model-generated code. Sandbox options:

| Mode | Platform | Isolation |
|---|---|---|
| `bwrap` | Linux | Read-only FS, no network, PID namespace, auto-detected |
| `docker` | Any | `--network=none --memory=512m --pids-limit=64`, explicit opt-in |
| `none` | Any | Subprocess with restricted env vars and resource limits |

`auto` selects bwrap if available, otherwise falls back to `none` with a warning. Use `--sandbox docker` to opt into Docker isolation.

## Resume and fingerprinting

Runs are fingerprinted (model, eval config, generation params, backend, think mode). Partial results are saved after each batch. On restart, matching partial files resume from where they left off. Examples are matched by content hash, not row index.

## Running tests

```bash
pip install -e ".[test]"
pytest
```

Tests run on Python 3.10+ without GPU, vLLM, or network access.
