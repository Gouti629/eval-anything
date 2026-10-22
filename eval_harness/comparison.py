from __future__ import annotations

import json
import os


def compare_runs(paths: list[str]) -> dict:
    """Load result files and compute per-eval score deltas."""
    runs = []
    for path in paths:
        with open(path) as f:
            runs.append(json.load(f))

    all_evals = set()
    for run in runs:
        for model_results in run.get("results", {}).values():
            all_evals.update(model_results.keys())
    eval_names = sorted(all_evals)

    if not eval_names:
        return {"evals": [], "runs": []}

    labels = []
    run_data = []
    for path, run in zip(paths, runs):
        models = list(run.get("results", {}).keys())
        label = run.get("timestamp", os.path.basename(path))
        if len(models) == 1:
            short = os.path.basename(models[0].rstrip("/\\")) or models[0]
            label = f"{short} ({label})"
        labels.append(label)

        scores: dict = {}
        for model_results in run.get("results", {}).values():
            for e in eval_names:
                if e in model_results and e not in scores:
                    scores[e] = model_results[e]
        run_data.append(scores)

    result: dict = {
        "evals": eval_names,
        "runs": [
            {
                "label": label,
                "scores": {e: data.get(e, {}).get("score") for e in eval_names},
            }
            for label, data in zip(labels, run_data)
        ],
    }

    if len(runs) == 2:
        deltas = {}
        for e in eval_names:
            s0 = run_data[0].get(e, {}).get("score")
            s1 = run_data[1].get(e, {}).get("score")
            if s0 is not None and s1 is not None:
                deltas[e] = round(s1 - s0, 4)
            else:
                deltas[e] = None
        result["deltas"] = deltas

    return result


def print_comparison(paths: list[str]) -> None:
    """Print a formatted comparison table to stdout."""
    data = compare_runs(paths)
    eval_names = data["evals"]
    runs = data["runs"]

    if not eval_names or not runs:
        print("No eval results found in the provided files.")
        return

    col_width = max(len(r["label"]) for r in runs) + 2
    if "deltas" in data:
        col_width = max(col_width, len("Delta") + 2)

    eval_col = 10
    header = f"{'Run':<{col_width}}"
    for e in eval_names:
        w = max(eval_col, len(e))
        header += f" | {e:>{w}}"

    print()
    print(header)
    print("-" * len(header))

    for run in runs:
        row = f"{run['label']:<{col_width}}"
        for e in eval_names:
            w = max(eval_col, len(e))
            score = run["scores"].get(e)
            if score is not None:
                row += f" | {score:>{w}.1%}"
            else:
                row += f" | {'N/A':>{w}}"
        print(row)

    if "deltas" in data:
        print("-" * len(header))
        delta_row = f"{'Delta':<{col_width}}"
        for e in eval_names:
            w = max(eval_col, len(e))
            d = data["deltas"].get(e)
            if d is not None:
                sign = "+" if d >= 0 else ""
                delta_row += f" | {sign}{d:>{w - 1}.1%}"
            else:
                delta_row += f" | {'N/A':>{w}}"
        print(delta_row)

    print()
