import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sanitize_model_name(name: str) -> str:
    return os.path.basename(name.rstrip("/\\")) or name


def plot_comparison(results: dict, output_path: str):
    model_names = list(results.keys())
    if len(model_names) < 2:
        return

    all_evals = set()
    for m in model_names:
        all_evals.update(results[m].keys())
    eval_names = sorted(all_evals)
    if not eval_names:
        return

    short_names = [sanitize_model_name(m) for m in model_names]

    x = np.arange(len(eval_names))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(max(8, len(eval_names) * 2), 6))

    for i, (model, short) in enumerate(zip(model_names, short_names)):
        scores = [results[model].get(e, {}).get("score", 0) for e in eval_names]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, scores, width, label=short)
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.1%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylabel("Score")
    ax.set_xlabel("Benchmark")
    ax.set_title("Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(eval_names)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
