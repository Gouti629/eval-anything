from __future__ import annotations

import ast
import hashlib
import json
import operator
import re

from datasets import load_dataset

from eval_harness.rewards.scorers import SCORERS, _strip_think


_REQUIRED_FIELDS = {"name", "dataset", "question_field", "answer_field"}
_VALID_SCORERS = set(SCORERS.keys())
_VALID_FORMATTERS = {"plain", "multiple_choice", "code_completion"}


class ConfigError(ValueError):
    pass


_COMPARE_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

_BOOL_OPS = {
    ast.And: all,
    ast.Or: any,
}


def _safe_eval_filter(expr: str, row: dict) -> bool:
    """Evaluate a simple filter expression safely without exec/eval.

    Supports: field comparisons (field > 5, field == "x"), boolean
    combinations (and, or), and 'in' / 'not in' membership tests.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ConfigError(f"Invalid filter expression: {expr!r}: {e}") from e
    return _eval_node(tree.body, row)


def _eval_node(node: ast.AST, row: dict):
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, row)
        result = True
        current = left
        for op_node, comparator in zip(node.ops, node.comparators):
            right = _eval_node(comparator, row)
            if isinstance(op_node, ast.In):
                result = result and (current in right)
            elif isinstance(op_node, ast.NotIn):
                result = result and (current not in right)
            elif type(op_node) in _COMPARE_OPS:
                result = result and _COMPARE_OPS[type(op_node)](current, right)
            else:
                raise ConfigError(f"Unsupported comparison: {type(op_node).__name__}")
            current = right
        return result
    if isinstance(node, ast.BoolOp):
        fn = _BOOL_OPS.get(type(node.op))
        if fn is None:
            raise ConfigError(f"Unsupported boolean op: {type(node.op).__name__}")
        return fn(_eval_node(v, row) for v in node.values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval_node(node.operand, row)
    if isinstance(node, ast.Name):
        if node.id not in row:
            raise ConfigError(f"Filter references unknown field {node.id!r}. Available: {list(row.keys())}")
        return row[node.id]
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_eval_node(el, row) for el in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(el, row) for el in node.elts)
    raise ConfigError(
        f"Unsupported expression node {type(node).__name__} in filter. "
        f"Only field comparisons, and/or/not, and literals are allowed."
    )


def validate_config(config: dict) -> list[str]:
    errors = []
    missing = _REQUIRED_FIELDS - set(config.keys())
    if missing:
        errors.append(f"Missing required fields: {missing}")
    scorer = config.get("scorer", "exact_match")
    if scorer not in _VALID_SCORERS:
        errors.append(f"Unknown scorer {scorer!r}. Valid: {sorted(_VALID_SCORERS)}")
    formatter = config.get("formatter", "plain")
    if formatter not in _VALID_FORMATTERS:
        errors.append(f"Unknown formatter {formatter!r}. Valid: {sorted(_VALID_FORMATTERS)}")
    if config.get("formatter") == "multiple_choice" and not config.get("choices_field"):
        errors.append("multiple_choice formatter requires choices_field")
    return errors


def config_fingerprint(config: dict, gen_params: dict) -> str:
    canonical = json.dumps({
        "dataset": config.get("dataset"),
        "config": config.get("config"),
        "split": config.get("split", "test"),
        "filter": config.get("filter"),
        "question_field": config.get("question_field"),
        "answer_field": config.get("answer_field"),
        "choices_field": config.get("choices_field"),
        "scorer": config.get("scorer", "exact_match"),
        "formatter": config.get("formatter", "plain"),
        "delimiter": config.get("delimiter"),
        "prepend_prompt": config.get("prepend_prompt", False),
        "system_prompt": config.get("system_prompt", ""),
        "max_new_tokens_override": config.get("max_new_tokens"),
        **gen_params,
    }, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class GenericEval:
    def __init__(self, config: dict, think: bool = False):
        errors = validate_config(config)
        if errors:
            raise ConfigError(f"Invalid eval config {config.get('name', '?')}: {'; '.join(errors)}")

        self.name = config["name"]
        self.dataset_name = config["dataset"]
        self.dataset_config = config.get("config", None)
        self.split = config.get("split", "test")
        self.question_field = config["question_field"]
        self.answer_field = config["answer_field"]
        self.system_prompt = config.get("system_prompt", "")
        self.scorer_name = config.get("scorer", "exact_match")
        self.formatter_name = config.get("formatter", "plain")
        self.filter_expr = config.get("filter", None)
        self.delimiter = config.get("delimiter", None)
        self.choices_field = config.get("choices_field", None)
        self.prepend_prompt = config.get("prepend_prompt", False)
        self.max_new_tokens = config.get("max_new_tokens", None)
        self.think = think
        self._scorer_fn = SCORERS[self.scorer_name]

    @property
    def is_code_eval(self) -> bool:
        return self.scorer_name == "code_exec"

    @property
    def metric_name(self) -> str:
        return "pass@1" if self.is_code_eval else "accuracy"

    def load_data(self) -> list[dict]:
        from datasets import concatenate_datasets

        kwargs = {"split": self.split}
        configs = self.dataset_config
        if configs is None:
            configs = [None]
        elif isinstance(configs, str):
            configs = [configs]

        datasets = []
        for cfg in configs:
            if cfg is not None:
                ds = load_dataset(self.dataset_name, cfg, **kwargs)
            else:
                ds = load_dataset(self.dataset_name, **kwargs)
            datasets.append(ds)

        if len(datasets) == 1:
            combined = datasets[0]
        else:
            combined = concatenate_datasets(datasets)

        if self.filter_expr:
            expr = self.filter_expr
            combined = combined.filter(lambda row: _safe_eval_filter(expr, row))

        results = []
        for i, row in enumerate(combined):
            raw = dict(row)
            entry = self._normalize_row(raw, i)
            results.append(entry)
        return results

    def _normalize_row(self, raw: dict, idx: int) -> dict:
        try:
            question = self._get_field(raw, self.question_field)
            answer = self._get_field(raw, self.answer_field)
        except (KeyError, TypeError) as e:
            raise ConfigError(
                f"Dataset {self.dataset_name} row {idx}: field extraction failed: {e}. "
                f"Available fields: {list(raw.keys())}"
            ) from e

        if self.formatter_name == "multiple_choice" and self.choices_field:
            choices_raw = self._get_field(raw, self.choices_field)
            if isinstance(choices_raw, dict) and "label" in choices_raw and "text" in choices_raw:
                choices = list(zip(choices_raw["label"], choices_raw["text"]))
            elif isinstance(choices_raw, list):
                labels = "ABCDEFGHIJKLMNOP"
                choices = [(labels[i], c) for i, c in enumerate(choices_raw)]
            else:
                choices = []
            raw["_choices_normalized"] = choices

        content_hash = hashlib.md5(
            f"{question}||{answer}".encode(errors="replace")
        ).hexdigest()[:12]

        return {
            "question": question,
            "ground_truth": answer,
            "example_id": f"{self.name}_{idx}_{content_hash}",
            "_raw": raw,
        }

    def _get_field(self, row: dict, field: str):
        if "." in field:
            parts = field.split(".")
            val = row
            for p in parts:
                if isinstance(val, dict):
                    val = val[p]
                else:
                    raise KeyError(f"Cannot traverse into {type(val).__name__} with key {p!r}")
            return val
        return row[field]

    def format_prompt(self, example: dict, tokenizer) -> str:
        question = str(example["question"])

        if self.formatter_name == "multiple_choice":
            choices = example["_raw"].get("_choices_normalized", [])
            if choices:
                choice_text = "\n".join(f"{label}. {text}" for label, text in choices)
                question = question + "\n\n" + choice_text

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": question})

        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = ""
            for msg in messages:
                prompt += f"{msg['role']}: {msg['content']}\n"
            prompt += "assistant: "

        if not self.think:
            prompt = prompt.replace(
                "<|im_start|>assistant\n<think>\n",
                "<|im_start|>assistant\n",
            )
        return prompt

    def score(self, generated: str, ground_truth, example: dict | None = None) -> tuple[float, dict]:
        kwargs = {}
        if self.delimiter:
            kwargs["delimiter"] = self.delimiter
        if self.prepend_prompt and example is not None:
            kwargs["prepend"] = str(example["question"])

        if self.scorer_name == "code_exec":
            return self._scorer_fn(generated, ground_truth, **kwargs)

        gt = ground_truth if isinstance(ground_truth, str) else str(ground_truth)
        return self._scorer_fn(generated, gt, **kwargs)

    def extract_answer(self, generated: str) -> str | None:
        clean = _strip_think(generated)
        if self.scorer_name == "number_extract":
            if self.delimiter and self.delimiter in clean:
                after = clean.split(self.delimiter)[-1].strip()
                from eval_harness.rewards.scorers import _NUMBER_RE
                nums = _NUMBER_RE.findall(after)
                return nums[0] if nums else after[:50] or None
            from eval_harness.rewards.scorers import _NUMBER_RE
            numbers = _NUMBER_RE.findall(clean)
            return numbers[-1] if numbers else None
        if self.scorer_name == "boxed_latex":
            from eval_harness.rewards.scorers import _extract_boxed
            return _extract_boxed(clean)
        if self.scorer_name == "letter_match":
            _, info = self._scorer_fn(generated, "X")
            return info.get("parsed")
        if self.scorer_name == "code_exec":
            from eval_harness.rewards.scorers import _extract_code
            return _extract_code(clean)
        return clean.strip()[:100] if clean else None
