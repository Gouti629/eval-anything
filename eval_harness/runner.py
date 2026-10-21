from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
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


def _run_fingerprint(models: list[str], eval_configs: list[dict], gen_params: dict, think: bool = False) -> str:
    parts = {
        "models": sorted(models),
        "evals": [config_fingerprint(c, gen_params) for c in eval_configs],
        "gen": gen_params,
        "think": think,
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
    gpu_name = None
    if device.startswith("cuda") and torch.cuda.is_available():
        idx = 0
        if ":" in device:
            try:
                idx = int(device.split(":")[1])
            except ValueError:
                pass
        gpu_name = torch.cuda.get_device_name(idx)

    return {
        "models": models,
        "eval_configs": eval_configs,
        "generation": gen_params,
        "device": device,
        "gpu": gpu_name,
        "think": think,
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
    limit: int | None = None,
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
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")

    os.makedirs(output_dir, exist_ok=True)

    gen_params = {
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "pass_k": pass_k,
        "limit": limit,
    }

    fingerprint = _run_fingerprint(models, eval_configs, gen_params, think=think)
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

    try:
        for model_path in models:
            model_key = model_path
            print(f"\nLoading model: {model_path}")
            llm, tokenizer = load_model(model_path, device=device)

            model_info = state["manifest"].setdefault("model_info", {})
            try:
                dtype_str = str(llm.llm_engine.model_config.dtype).replace("torch.", "")
            except Exception:
                dtype_str = "unknown"
            model_info[model_key] = {"dtype": dtype_str}

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
                if limit is not None:
                    data = data[:limit]

                if eval_name not in state["examples"][model_key]:
                    state["examples"][model_key][eval_name] = []

                eval_max_tokens = eval_obj.max_new_tokens or max_new_tokens
                use_pass_k = pass_k is not None and eval_obj.is_code_eval
                num_samples = pass_k if use_pass_k else 1
                sample_temp = temperature if temperature > 0 else (0.8 if use_pass_k else 0.0)

                ctr = {"total_score": 0.0, "correct": 0, "incorrect": 0,
                       "timed_out": 0, "oom_count": 0, "errored": 0,
                       "truncated_count": 0}
                n_done = 0
                total_gen_tokens = 0
                total_gen_time = 0.0
                total_score_time = 0.0

                for existing_ex in state["examples"][model_key][eval_name]:
                    n_done += 1
                    st = existing_ex.get("status", "unknown")
                    if st == "ok":
                        ctr["total_score"] += existing_ex["score"]
                        ctr["correct"] += 1
                    elif st == "incorrect":
                        ctr["incorrect"] += 1
                    elif st == "timeout":
                        ctr["timed_out"] += 1
                    elif st == "oom":
                        ctr["oom_count"] += 1
                    elif st == "error":
                        ctr["errored"] += 1
                    else:
                        ctr["errored"] += 1

                done_ids = {
                    ex["example_id"]
                    for ex in state["examples"][model_key][eval_name]
                    if "example_id" in ex
                }
                remaining = [
                    (i, ex) for i, ex in enumerate(data)
                    if ex["example_id"] not in done_ids
                ]

                remaining_with_prompts = [
                    (i, ex, eval_obj.format_prompt(ex, tokenizer))
                    for i, ex in remaining
                ]

                start_time = time.time()
                pbar = tqdm(total=len(data), initial=n_done, desc=f"    {eval_name}")

                executor = ThreadPoolExecutor(max_workers=1)
                pending_future = None
                pending_gen_tokens = 0

                chunk_idx = 0
                while chunk_idx < len(remaining_with_prompts):
                    if pending_future is not None:
                        score_start = time.time()
                        records = pending_future.result()
                        total_score_time += time.time() - score_start
                        _apply_records(records, state, model_key, eval_name, ctr, pbar)
                        total_gen_tokens += pending_gen_tokens
                        _save_state_atomic(partial_file, state)
                        pending_future = None

                    chunk = remaining_with_prompts[chunk_idx : chunk_idx + batch_size]
                    batch_indices = [item[0] for item in chunk]
                    batch_examples = [item[1] for item in chunk]
                    prompts = [item[2] for item in chunk]

                    gen_start = time.time()
                    try:
                        outputs = batched_generate(
                            llm,
                            prompts,
                            max_new_tokens=eval_max_tokens,
                            temperature=sample_temp,
                            num_return_sequences=num_samples,
                        )
                        total_gen_time += time.time() - gen_start
                    except Exception as e:
                        total_gen_time += time.time() - gen_start
                        print(f"\n    Generation error: {e}")
                        for idx, ex in zip(batch_indices, batch_examples):
                            state["examples"][model_key][eval_name].append({
                                "idx": idx,
                                "example_id": ex["example_id"],
                                "question": str(ex["question"])[:500],
                                "ground_truth": _serialize_gt(ex["ground_truth"]),
                                "model_output": None,
                                "extracted": None,
                                "parse_status": "error",
                                "score": 0.0,
                                "status": "error",
                            })
                            ctr["errored"] += 1
                            pbar.update(1)
                        _save_state_atomic(partial_file, state)
                        chunk_idx += batch_size
                        continue

                    pending_gen_tokens = sum(r.token_count for r in outputs)
                    pending_future = executor.submit(
                        _score_batch, eval_obj, batch_indices, batch_examples,
                        outputs, use_pass_k, pass_k,
                    )
                    chunk_idx += batch_size

                if pending_future is not None:
                    score_start = time.time()
                    records = pending_future.result()
                    total_score_time += time.time() - score_start
                    _apply_records(records, state, model_key, eval_name, ctr, pbar)
                    total_gen_tokens += pending_gen_tokens
                    _save_state_atomic(partial_file, state)
                executor.shutdown(wait=False)

                pbar.close()

                elapsed = time.time() - start_time
                n_total = len(data)
                correct = ctr["correct"]
                incorrect = ctr["incorrect"]
                timed_out = ctr["timed_out"]
                oom_count = ctr["oom_count"]
                errored = ctr["errored"]
                truncated_count = ctr["truncated_count"]
                total_score = ctr["total_score"]
                scored_count = correct + incorrect
                infra_failures = timed_out + oom_count + errored
                avg_score = total_score / n_total if n_total > 0 else 0.0

                metric = eval_obj.metric_name
                if use_pass_k:
                    metric = f"pass@{pass_k}"

                state["results"][model_key][eval_name] = {
                    "score": round(avg_score, 4),
                    "n": n_total,
                    "correct": correct,
                    "incorrect": incorrect,
                    "timed_out": timed_out,
                    "oom": oom_count,
                    "errored": errored,
                    "scored": scored_count,
                    "infra_failures": infra_failures,
                    "truncated": truncated_count,
                    "max_new_tokens": eval_max_tokens,
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
                    "batch_size": batch_size,
                }

                infra_msg = ""
                if infra_failures > 0:
                    infra_msg = f" [{infra_failures} infra failures: {timed_out} timeout, {oom_count} oom, {errored} error]"
                print(f"    {eval_name}: {avg_score:.1%} ({correct}/{n_total}){infra_msg} "
                      f"[{elapsed:.1f}s, {gen_tok_per_sec:.0f} tok/s gen]")

                _save_state_atomic(partial_file, state)

            del llm
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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


def _score_batch(eval_obj, batch_indices, batch_examples, outputs, use_pass_k, pass_k_val):
    records = []
    for j, (idx, ex) in enumerate(zip(batch_indices, batch_examples)):
        gen_result = outputs[j]
        sample_outputs = gen_result.text
        was_truncated = gen_result.truncated
        sample_results = None
        error_tb = None

        try:
            if use_pass_k:
                sample_results = []
                for sample in sample_outputs:
                    s, info = eval_obj.score(sample, ex["ground_truth"], example=ex)
                    sample_results.append({
                        "output": sample[:1000],
                        "score": s,
                        "parse_status": info.get("status", "unknown"),
                        "parsed": info.get("parsed"),
                    })
                c = sum(1 for sr in sample_results if sr["score"] >= 1.0)
                example_score = pass_at_k(len(sample_results), c, pass_k_val)
                extracted = eval_obj.extract_answer(sample_outputs[0])
                parse_status = sample_results[0]["parse_status"]
                status = "ok" if example_score > 0 else "incorrect"
            else:
                output_text = sample_outputs[0]
                example_score, info = eval_obj.score(
                    output_text, ex["ground_truth"], example=ex
                )
                extracted = info.get("parsed") or eval_obj.extract_answer(output_text)
                parse_status = info.get("status", "unknown")
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
            error_tb = "".join(traceback.format_exception_only(type(exc), exc))[:500]

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
            "truncated": was_truncated,
        }
        if sample_results is not None:
            record["samples"] = sample_results
        if error_tb is not None:
            record["error"] = error_tb
        records.append(record)
    return records


def _apply_records(records, state, model_key, eval_name, ctr, pbar):
    for record in records:
        state["examples"][model_key][eval_name].append(record)
        s = record["status"]
        if s == "ok":
            ctr["total_score"] += record["score"]
            ctr["correct"] += 1
        elif s == "incorrect":
            ctr["incorrect"] += 1
        elif s == "timeout":
            ctr["timed_out"] += 1
        elif s == "oom":
            ctr["oom_count"] += 1
        else:
            ctr["errored"] += 1
        if record.get("truncated"):
            ctr["truncated_count"] += 1
        pbar.update(1)
