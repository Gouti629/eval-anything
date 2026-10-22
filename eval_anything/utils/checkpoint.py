from __future__ import annotations

import json
import os
import platform


def save_state_atomic(path: str, state: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    if platform.system() == "Windows":
        if os.path.exists(path):
            os.remove(path)
    os.rename(tmp, path)


def partial_path(output_dir: str, fingerprint: str) -> str:
    return os.path.join(output_dir, f"{fingerprint}.partial.json")


def final_path(output_dir: str, timestamp: str) -> str:
    return os.path.join(output_dir, f"{timestamp}.json")


def find_matching_partial(output_dir: str, fingerprint: str) -> dict | None:
    path = partial_path(output_dir, fingerprint)
    if os.path.exists(path):
        try:
            with open(path) as f:
                state = json.load(f)
            if state.get("fingerprint") == fingerprint:
                return state
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def is_example_done(state: dict, model_key: str, eval_name: str, example_id: str) -> bool:
    examples = state.get("examples", {})
    model_examples = examples.get(model_key, {})
    eval_examples = model_examples.get(eval_name, [])
    return any(ex.get("example_id") == example_id for ex in eval_examples)
