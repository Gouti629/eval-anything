import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from eval_harness.evals.generic import GenericEval
from eval_harness.models import load_model
from eval_harness.plotting.compare import plot_comparison, sanitize_model_name
from eval_harness.utils.generation import batched_generate


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _partial_path(output_dir: str, timestamp: str) -> str:
    return os.path.join(output_dir, f"{timestamp}.partial.json")


def _final_path(output_dir: str, timestamp: str) -> str:
    return os.path.join(output_dir, f"{timestamp}.json")


def _save_state(path: str, state: dict):
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def _find_existing_partial(output_dir: str) -> tuple[str, dict] | None:
    if not os.path.exists(output_dir):
        return None
    for fname in sorted(os.listdir(output_dir), reverse=True):
        if fname.endswith(".partial.json"):
            ts = fname.replace(".partial.json", "")
            path = os.path.join(output_dir, fname)
            with open(path) as f:
                return ts, json.load(f)
    return None


def _is_example_done(state: dict, model_key: str, eval_name: str, idx: int) -> bool:
    examples = state.get("examples", {})
    model_examples = examples.get(model_key, {})
    eval_examples = model_examples.get(eval_name, [])
    return any(ex.get("idx") == idx for ex in eval_examples)


def _print_summary(results: dict):
    if not results:
        return

    model_names = list(results.keys())
    eval_names = list(results[model_names[0]].keys())

    col_width = max(len(sanitize_model_name(m)) for m in model_names) + 2
    header = f"{'Model':<{col_width}}"
    for e in eval_names:
        header += f" | {e:>8}"
    print()
    print(header)
    print("-" * len(header))
    for model in model_names:
        row = f"{sanitize_model_name(model):<{col_width}}"
        for e in eval_names:
            score = results[model][e]["score"]
            row += f" | {score:>7.1%}"
        print(row)
    print()


def evaluate(
    models: list[str],
    eval_configs: list[dict],
    comment: str | None = None,
    output_dir: str = "./results",
    batch_size: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    pass_k: int | None = None,
    device: str = "cuda",
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    partial = _find_existing_partial(output_dir)
    if partial is not None:
        timestamp, state = partial
        print(f"Resuming from partial run: {timestamp}")
    else:
        state = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "comment": comment,
            "results": {},
            "config": {
                "batch_size": batch_size,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "pass_k": pass_k,
            },
            "examples": {},
        }

    partial_file = _partial_path(output_dir, timestamp)
    final_file = _final_path(output_dir, timestamp)

    eval_objects = []
    for cfg in eval_configs:
        eval_objects.append(GenericEval(cfg))

    try:
        for model_path in models:
            model_key = model_path
            print(f"\nLoading model: {model_path}")
            model, tokenizer = load_model(model_path, device=device)

            if model_key not in state["results"]:
                state["results"][model_key] = {}
            if model_key not in state["examples"]:
                state["examples"][model_key] = {}

            for eval_obj in eval_objects:
                eval_name = eval_obj.name

                if eval_name in state["results"].get(model_key, {}):
                    result = state["results"][model_key][eval_name]
                    if result.get("_complete", False):
                        print(f"  Skipping {eval_name} (already complete)")
                        continue

                print(f"  Running {eval_name}...")
                data = eval_obj.load_data()

                if eval_name not in state["examples"][model_key]:
                    state["examples"][model_key][eval_name] = []

                use_pass_k = pass_k is not None and eval_obj.is_code_eval
                num_samples = pass_k if use_pass_k else 1
                sample_temp = temperature if temperature > 0 else (0.8 if use_pass_k else 0.0)

                total_score = 0.0
                correct = 0
                failed = 0
                n_done = 0

                for existing_ex in state["examples"][model_key][eval_name]:
                    idx = existing_ex["idx"]
                    if idx < len(data):
                        n_done += 1
                        if existing_ex["status"] != "ok":
                            failed += 1
                        else:
                            total_score += existing_ex["score"]
                            if existing_ex["score"] >= 1.0:
                                correct += 1

                remaining_indices = [
                    idx for idx in range(len(data))
                    if not _is_example_done(state, model_key, eval_name, idx)
                ]

                start_time = time.time()
                pbar = tqdm(total=len(data), initial=n_done, desc=f"    {eval_name}")

                for batch_start in range(0, len(remaining_indices), batch_size):
                    batch_indices = remaining_indices[batch_start : batch_start + batch_size]
                    batch_examples = [data[i] for i in batch_indices]
                    prompts = [eval_obj.format_prompt(ex, tokenizer) for ex in batch_examples]

                    try:
                        outputs = batched_generate(
                            model,
                            tokenizer,
                            prompts,
                            max_new_tokens=max_new_tokens,
                            temperature=sample_temp,
                            batch_size=len(prompts),
                            timeout=2400,
                            num_return_sequences=num_samples,
                        )
                    except TimeoutError:
                        for idx in batch_indices:
                            ex = data[idx]
                            state["examples"][model_key][eval_name].append({
                                "idx": idx,
                                "question": str(ex["question"])[:200],
                                "ground_truth": str(ex["ground_truth"])[:200],
                                "model_output": None,
                                "extracted": None,
                                "score": 0.0,
                                "status": "timeout",
                            })
                            failed += 1
                            pbar.update(1)
                        _save_state(partial_file, state)
                        continue
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        for idx in batch_indices:
                            ex = data[idx]
                            state["examples"][model_key][eval_name].append({
                                "idx": idx,
                                "question": str(ex["question"])[:200],
                                "ground_truth": str(ex["ground_truth"])[:200],
                                "model_output": None,
                                "extracted": None,
                                "score": 0.0,
                                "status": "oom",
                            })
                            failed += 1
                            pbar.update(1)
                        _save_state(partial_file, state)
                        continue

                    for j, idx in enumerate(batch_indices):
                        ex = data[idx]
                        sample_outputs = outputs[j]

                        try:
                            if use_pass_k:
                                sample_scores = []
                                for sample in sample_outputs:
                                    s = eval_obj.score(sample, ex["ground_truth"], example=ex)
                                    sample_scores.append(s)
                                c = sum(1 for s in sample_scores if s >= 1.0)
                                example_score = pass_at_k(len(sample_scores), c, pass_k)
                                extracted = eval_obj.extract_answer(sample_outputs[0])
                                status = "ok"
                            else:
                                output_text = sample_outputs[0]
                                example_score = eval_obj.score(output_text, ex["ground_truth"], example=ex)
                                extracted = eval_obj.extract_answer(output_text)
                                status = "ok"
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            example_score, extracted, status = 0.0, None, "oom"
                            sample_outputs = [None]
                        except Exception:
                            example_score, extracted, status = 0.0, None, "error"
                            sample_outputs = [None]

                        state["examples"][model_key][eval_name].append({
                            "idx": idx,
                            "question": str(ex["question"])[:200],
                            "ground_truth": str(ex["ground_truth"])[:200],
                            "model_output": (sample_outputs[0] or "")[:500] if sample_outputs[0] else None,
                            "extracted": str(extracted)[:200] if extracted else None,
                            "score": example_score,
                            "status": status,
                        })

                        if status != "ok":
                            failed += 1
                        else:
                            total_score += example_score
                            if example_score >= 1.0:
                                correct += 1

                        pbar.update(1)

                    _save_state(partial_file, state)

                pbar.close()

                elapsed = time.time() - start_time
                n_total = len(data)
                avg_score = total_score / n_total if n_total > 0 else 0.0

                metric = eval_obj.metric_name
                if use_pass_k:
                    metric = f"pass@{pass_k}"

                state["results"][model_key][eval_name] = {
                    "score": round(avg_score, 4),
                    "n": n_total,
                    "correct": correct,
                    "failed": failed,
                    "metric": metric,
                    "_complete": True,
                }

                tokens_per_sec = (n_total * max_new_tokens) / elapsed if elapsed > 0 else 0
                print(f"    {eval_name}: {avg_score:.1%} ({correct}/{n_total}) "
                      f"[{elapsed:.1f}s, ~{tokens_per_sec:.0f} tok/s]")

                _save_state(partial_file, state)

            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\nInterrupted. Saving partial results...")
        _save_state(partial_file, state)
        print(f"Partial results saved to {partial_file}")
        return partial_file

    for model_key in state["results"]:
        for eval_name in state["results"][model_key]:
            state["results"][model_key][eval_name].pop("_complete", None)

    _save_state(final_file, state)

    if os.path.exists(partial_file):
        os.remove(partial_file)

    _print_summary(state["results"])

    if len(models) > 1:
        chart_path = os.path.join(output_dir, f"{timestamp}_comparison.png")
        plot_comparison(state["results"], chart_path)
        print(f"Comparison chart saved to {chart_path}")

    print(f"Results saved to {final_file}")
    return final_file
