# Contributing to eval-anything

## Adding a benchmark

1. Add the eval definition to `eval_harness/default_evals.yaml`:

```yaml
- name: my_benchmark
  dataset: huggingface/dataset-name
  config: default
  split: test
  question_field: question
  answer_field: answer
  scorer: exact_match
  system_prompt: "Answer concisely."
```

2. Run it to verify: `eval-harness --model <small-model> --eval_config default --limit 5`
3. Add a row to the "Built-in benchmarks" table in `README.md`.

## Adding a scorer

1. Define a function in `eval_harness/rewards/scorers.py`:

```python
def my_scorer(output: str, ground_truth: str, **kwargs) -> tuple[float, dict]:
    # Return (score, info_dict)
    parsed = extract_answer(output)
    correct = parsed == ground_truth
    return (1.0 if correct else 0.0), {"parsed": parsed, "status": "parsed" if parsed else "missing"}
```

2. Register it in the `SCORERS` dict at the top of the file.
3. Add tests in `tests/test_scorers.py`.

## Running tests

```bash
pip install -e ".[test]"
pytest -v
```

Tests run without GPU, vLLM, or network access (all datasets are mocked).

## Code style

- No type stubs needed — the package ships `py.typed`.
- Use `from __future__ import annotations` in all modules.
- Keep imports lazy for heavy dependencies (vLLM, torch) in modules that tests touch.
