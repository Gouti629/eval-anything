from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from datetime import datetime

import torch
from tqdm import tqdm

from eval_harness.evals.generic import GenericEval, config_fingerprint, validate_config
from eval_harness.models import load_model
from eval_harness.plotting.compare import plot_comparison, sanitize_model_name
from eval_harness.utils.generation import batched_generate


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _save_state_atomic(path: str, state: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    if platform.system() == "Windows":
        if os.path.exists(path):
            os.remove(path)
    os.rename(tmp, path)


def _partial_path(output_dir: str, fingerprint: str) -> str:
    return os.path.join(output_dir, f"{fingerprint}.partial.json")


def _final_path(output_dir: str, timestamp: str) -> str:
    return os.path.join(output_dir, f"{timestamp}.json")


def _run_fingerprint(models: list[str], eval_configs: list[dict], gen_params: dict) -> str:
    parts = {
        "models": sorted(models),
        "evals": [config_fingerprint(c, gen_params) for c in eval_configs],
        "gen": gen_params,
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def _find_matching_partial(output_dir: str, fingerprint: str) -> dict | None:
    path = _partial_path(output_dir, fingerprint)
    if os.path.exists(path):
        try:
            with open(path) as f:
                state = json.load(f)
            if state.get("fingerprint") == fingerprint:
                return state
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def _is_example_done(state: dict, model_key: str, eval_name: str, example_id: str) -> bool:
    examples = state.get("examples", {})
    model_examples = examples.get(model_key, {})
    eval_examples = model_examples.get(eval_name, [])
    return any(ex.get("example_id") == example_id for ex in eval_examples)


def _build_manifest(models, eval_configs, gen_params, device, think):
    return {
        "models": models,
        "eval_configs": [{k: v for k, v in c.items() if k != "filter"} for c in eval_configs],
        "generation": gen_params,
        "device": device,
        "think": think,
        "precision": "bfloat16",
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
    }


def _print_summary(results: dict):
    if not results:
        return

    model_names = list(results.keys())
    all_evals = set()
    for m in model_names:
        all_evals.update(results[m].keys())
    eval_names = sorted(all_evals)

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
            if e in results[model]:
                score = results[model][e]["score"]
                row += f" | {score:>7.1%}"
            else:
                row += f" | {'N/A':>7}"
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
    think: bool = False,
) -> str:
    for cfg in eval_configs:
        errors = validate_config(cfg)
        if errors:
            raise ValueError(f"Invalid eval config {cfg.get('name', '?')}: {'; '.join(errors)}")

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
    if pass_k is not None and pass_k < 1:
        raise ValueError(f"pass_k must be >= 1, got {pass_k}")

    os.makedirs(output_dir, exist_ok=True)

    gen_params = {
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "pass_k": pass_k,
    }

    fingerprint = _run_fingerprint(models, eval_configs, gen_params)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    existing = _find_matching_partial(output_dir, fingerprint)
    if existing is not None:
        state = existing
        timestamp = state.get("timestamp", timestamp)
        print(f"Resuming run (fingerprint {fingerprint})")
    else:
        state = {
            "timestamp": timestamp,
            "fingerprint": fingerprint,
            "comment": comment,
            "manifest": _build_manifest(models, eval_configs, gen_params, device, think),
            "results": {},
            "config": gen_params,
            "examples": {},
            "timing": {},
        }

    partial_file = _partial_path(output_dir, fingerprint)
    final_file = _final_path(output_dir, timestamp)

    eval_objects = []
    for cfg in eval_configs:
        eval_objects.append(GenericEval(cfg, think=think))

    current_batch_size = batch_size

    try:
        for model_path in models:
            model_key = model_path
            print(f"\nLoading model: {model_path}")
            model, tokenizer = load_model(model_path, device=device)

            if model_key not in state["results"]:
                state["results"][model_key] = {}
            if model_key not in state["examples"]:
                state["examples"][model_key] = {}
            if model_key not in state.get("timing", {}):
                state.setdefault("timing", {})[model_key] = {}

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
                total_gen_tokens = 0
                total_gen_time = 0.0
                total_score_time = 0.0

                for existing_ex in state["examples"][model_key][eval_name]:
                    n_done += 1
                    if existing_ex["status"] not in ("ok", "incorrect"):
                        failed += 1
                    else:
                        total_score += existing_ex["score"]
                        if existing_ex["score"] >= 1.0:
                            correct += 1

                done_ids = {
                    ex["example_id"]
                    for ex in state["examples"][model_key][eval_name]
                    if "example_id" in ex
                }
                done_indices = {
                    ex["idx"]
                    for ex in state["examples"][model_key][eval_name]
                }
                remaining = [
                    (i, ex) for i, ex in enumerate(data)
                    if ex["example_id"] not in done_ids and i not in done_indices
                ]

                start_time = time.time()
                pbar = tqdm(total=len(data), initial=n_done, desc=f"    {eval_name}")

                batch_idx = 0
                while batch_idx < len(remaining):
                    batch_slice = remaining[batch_idx : batch_idx + current_batch_size]
                    batch_indices = [item[0] for item in batch_slice]
                    batch_examples = [item[1] for item in batch_slice]
                    prompts = [eval_obj.format_prompt(ex, tokenizer) for ex in batch_examples]

                    gen_start = time.time()
                    try:
                        outputs = batched_generate(
                            model,
                            tokenizer,
                            prompts,
                            max_new_tokens=max_new_tokens,
                            temperature=sample_temp,
                            batch_size=len(prompts),
                            timeout=120,
                            num_return_sequences=num_samples,
                        )
                        gen_elapsed = time.time() - gen_start
                        total_gen_time += gen_elapsed
                    except TimeoutError:
                        for idx, ex in zip(batch_indices, batch_examples):
                            state["examples"][model_key][eval_name].append({
                                "idx": idx,
                                "example_id": ex["example_id"],
                                "question": str(ex["question"])[:500],
                                "ground_truth": _serialize_gt(ex["ground_truth"]),
                                "model_output": None,
                                "extracted": None,
                                "parse_status": "timeout",
                                "score": 0.0,
                                "status": "timeout",
                            })
                            failed += 1
                            pbar.update(1)
                        _save_state_atomic(partial_file, state)
                        batch_idx += current_batch_size
                        continue
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if current_batch_size > 1:
                            new_bs = max(1, current_batch_size // 2)
                            print(f"\n    OOM with batch_size={current_batch_size}, "
                                  f"reducing to {new_bs}")
                            current_batch_size = new_bs
                            continue
                        for idx, ex in zip(batch_indices, batch_examples):
                            state["examples"][model_key][eval_name].append({
                                "idx": idx,
                                "example_id": ex["example_id"],
                                "question": str(ex["question"])[:500],
                                "ground_truth": _serialize_gt(ex["ground_truth"]),
                                "model_output": None,
                                "extracted": None,
                                "parse_status": "oom",
                                "score": 0.0,
                                "status": "oom",
                            })
                            failed += 1
                            pbar.update(1)
                        _save_state_atomic(partial_file, state)
                        batch_idx += current_batch_size
                        continue

                    for out_seqs in outputs:
                        for seq in out_seqs:
                            total_gen_tokens += len(tokenizer.encode(seq, add_special_tokens=False))

                    score_start = time.time()
                    for j, (idx, ex) in enumerate(zip(batch_indices, batch_examples)):
                        sample_outputs = outputs[j]

                        try:
                            if use_pass_k:
                                sample_results = []
                                for sample in sample_outputs:
                                    s, info = eval_obj.score(
                                        sample, ex["ground_truth"], example=ex
                                    )
                                    sample_results.append({
                                        "output": sample[:1000],
                                        "score": s,
                                        "parse_status": info.get("status", "unknown"),
                                        "parsed": info.get("parsed"),
                                    })
                                c = sum(1 for sr in sample_results if sr["score"] >= 1.0)
                                example_score = pass_at_k(len(sample_results), c, pass_k)
                                extracted = eval_obj.extract_answer(sample_outputs[0])
                                parse_status = sample_results[0]["parse_status"]
                                status = "ok"
                            else:
                                output_text = sample_outputs[0]
                                example_score, info = eval_obj.score(
                                    output_text, ex["ground_truth"], example=ex
                                )
                                extracted = info.get("parsed") or eval_obj.extract_answer(output_text)
                                parse_status = info.get("status", "unknown")
                                sample_results = None
                                status = "ok" if example_score > 0 else "incorrect"
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            example_score, extracted, parse_status = 0.0, None, "oom"
                            status = "oom"
                            sample_results = None
                            sample_outputs = [None]
                        except Exception as exc:
                            example_score, extracted = 0.0, None
                            parse_status = "error"
                            status = "error"
                            sample_results = None
                            sample_outputs = [None]
                            tb = traceback.format_exception_only(type(exc), exc)

                        record = {
                            "idx": idx,
                            "example_id": ex["example_id"],
                            "question": str(ex["question"])[:500],
                            "ground_truth": _serialize_gt(ex["ground_truth"]),
                            "model_output": _safe_trunc(sample_outputs[0], 1000),
                            "extracted": str(extracted)[:500] if extracted else None,
                            "parse_status": parse_status,
                            "score": example_score,
                            "status": status,
                        }
                        if sample_results is not None:
                            record["samples"] = sample_results
                        if status == "error":
                            record["error"] = "".join(tb)[:500] if "tb" in dir() else "unknown"

                        state["examples"][model_key][eval_name].append(record)

                        if status in ("ok", "incorrect"):
                            total_score += example_score
                            if example_score >= 1.0:
                                correct += 1
                        else:
                            failed += 1

                        pbar.update(1)

                    total_score_time += time.time() - score_start
                    _save_state_atomic(partial_file, state)
                    batch_idx += current_batch_size

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

                gen_tok_per_sec = total_gen_tokens / total_gen_time if total_gen_time > 0 else 0
                state.setdefault("timing", {}).setdefault(model_key, {})[eval_name] = {
                    "total_seconds": round(elapsed, 1),
                    "generation_seconds": round(total_gen_time, 1),
                    "scoring_seconds": round(total_score_time, 1),
                    "generated_tokens": total_gen_tokens,
                    "tokens_per_second": round(gen_tok_per_sec, 1),
                    "examples_per_second": round(n_total / elapsed, 2) if elapsed > 0 else 0,
                }

                print(f"    {eval_name}: {avg_score:.1%} ({correct}/{n_total}) "
                      f"[{elapsed:.1f}s, {gen_tok_per_sec:.0f} tok/s gen, "
                      f"{total_score_time:.1f}s scoring]")

                _save_state_atomic(partial_file, state)

            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            current_batch_size = batch_size

    except KeyboardInterrupt:
        print("\nInterrupted. Saving partial results...")
        _save_state_atomic(partial_file, state)
        print(f"Partial results saved to {partial_file}")
        return partial_file

    for model_key in state["results"]:
        for eval_name in state["results"][model_key]:
            state["results"][model_key][eval_name].pop("_complete", None)

    state["completed"] = datetime.now().isoformat(timespec="seconds")
    _save_state_atomic(final_file, state)

    if os.path.exists(partial_file):
        os.remove(partial_file)

    _print_summary(state["results"])

    if len(models) > 1:
        chart_path = os.path.join(output_dir, f"{timestamp}_comparison.png")
        try:
            plot_comparison(state["results"], chart_path)
            print(f"Comparison chart saved to {chart_path}")
        except Exception as e:
            print(f"Warning: chart generation failed: {e}")

    print(f"Results saved to {final_file}")
    return final_file


def _serialize_gt(ground_truth) -> str | list:
    if isinstance(ground_truth, list):
        return ground_truth
    return str(ground_truth)


def _safe_trunc(text, limit: int) -> str | None:
    if text is None:
        return None
    return str(text)[:limit]
