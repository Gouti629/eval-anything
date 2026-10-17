"""Tests that default benchmark configs work with representative row shapes.

These tests do NOT download real datasets. They verify that GenericEval's
normalization, formatting, and scoring work correctly with the field
structures that each benchmark actually provides.
"""
import pytest
import yaml
import importlib.resources

from eval_harness.evals.generic import GenericEval, ConfigError


def _load_default_configs():
    ref = importlib.resources.files("eval_harness").joinpath("default_evals.yaml")
    text = ref.read_text(encoding="utf-8")
    return yaml.safe_load(text)


class TestDefaultConfigsValid:
    """All default benchmark configs should pass validation."""

    def test_all_configs_load(self):
        configs = _load_default_configs()
        assert len(configs) == 6

    @pytest.mark.parametrize("idx", range(6))
    def test_config_creates_eval(self, idx):
        configs = _load_default_configs()
        ev = GenericEval(configs[idx])
        assert ev.name in ("gsm8k", "math", "mbpp", "humaneval", "livecodebench_v5", "arc")


class TestGSM8KRowShape:
    """GSM8K rows have: question (str), answer (str with #### delimiter)."""

    def _make_eval(self):
        configs = _load_default_configs()
        cfg = next(c for c in configs if c["name"] == "gsm8k")
        return GenericEval(cfg)

    def test_normalize_row(self):
        ev = self._make_eval()
        raw = {
            "question": "Janet buys 3 pounds of broccoli for $4 a pound. How much did she pay?",
            "answer": "She paid 3*4=$12\n#### 12",
        }
        entry = ev._normalize_row(raw, 0)
        assert entry["question"] == raw["question"]
        assert entry["ground_truth"] == raw["answer"]
        assert "example_id" in entry

    def test_score_correct(self):
        ev = self._make_eval()
        gt = "She paid 3*4=$12\n#### 12"
        score, info = ev.score("Let me compute: 3*4=12\n#### 12", gt)
        assert score == 1.0

    def test_score_wrong(self):
        ev = self._make_eval()
        gt = "#### 12"
        score, info = ev.score("#### 15", gt)
        assert score == 0.0


class TestMATHRowShape:
    """MATH rows have: problem (str), solution (str with \\boxed{})."""

    def _make_eval(self):
        configs = _load_default_configs()
        cfg = next(c for c in configs if c["name"] == "math")
        return GenericEval(cfg)

    def test_normalize_row(self):
        ev = self._make_eval()
        raw = {
            "problem": "What is 2 + 2?",
            "solution": "We compute $2+2=\\boxed{4}$.",
        }
        entry = ev._normalize_row(raw, 0)
        assert entry["question"] == "What is 2 + 2?"
        assert "\\boxed{4}" in entry["ground_truth"]

    def test_score_correct_boxed(self):
        ev = self._make_eval()
        score, info = ev.score("The answer is \\boxed{4}", "\\boxed{4}")
        assert score == 1.0

    def test_score_fallback_number(self):
        ev = self._make_eval()
        score, info = ev.score("The answer is 4", "\\boxed{4}")
        assert score == 1.0
        assert info["status"] == "fallback_number"


class TestMBPPRowShape:
    """MBPP rows have: text (str), test_list (list of assertion strings)."""

    def _make_eval(self):
        configs = _load_default_configs()
        cfg = next(c for c in configs if c["name"] == "mbpp")
        return GenericEval(cfg)

    def test_normalize_row(self):
        ev = self._make_eval()
        raw = {
            "task_id": 100,
            "text": "Write a function to add two numbers.",
            "test_list": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
            "code": "def add(a, b): return a + b",
        }
        entry = ev._normalize_row(raw, 0)
        assert entry["question"] == "Write a function to add two numbers."
        assert isinstance(entry["ground_truth"], list)
        assert len(entry["ground_truth"]) == 2

    def test_score_preserves_list_gt(self):
        ev = self._make_eval()
        code = "def add(a, b):\n    return a + b"
        tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
        score, info = ev.score(code, tests)
        assert score == 1.0

    def test_score_wrong_code(self):
        ev = self._make_eval()
        code = "def add(a, b):\n    return a - b"
        tests = ["assert add(1, 2) == 3"]
        score, _ = ev.score(code, tests)
        assert score == 0.0


class TestHumanEvalRowShape:
    """HumanEval rows have: prompt (str), test (str), with prepend_prompt=true."""

    def _make_eval(self):
        configs = _load_default_configs()
        cfg = next(c for c in configs if c["name"] == "humaneval")
        return GenericEval(cfg)

    def test_normalize_row(self):
        ev = self._make_eval()
        raw = {
            "prompt": "def has_close_elements(numbers, threshold):\n",
            "test": "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False",
            "task_id": "HumanEval/0",
            "canonical_solution": "    pass",
            "entry_point": "has_close_elements",
        }
        entry = ev._normalize_row(raw, 0)
        assert entry["question"] == raw["prompt"]
        assert entry["ground_truth"] == raw["test"]

    def test_prepend_prompt_enabled(self):
        ev = self._make_eval()
        assert ev.prepend_prompt is True

    def test_score_with_prepend(self):
        ev = self._make_eval()
        prompt = "def f(x):"
        completion = "```python\n    return x + 1\n```"
        test = "assert f(1) == 2"
        example = {"question": prompt, "ground_truth": test, "example_id": "test", "_raw": {}}
        score, _ = ev.score(completion, test, example=example)
        assert score == 1.0


class TestARCRowShape:
    """ARC rows have: question (str), answerKey (str), choices (dict with label/text)."""

    def _make_eval(self):
        configs = _load_default_configs()
        cfg = next(c for c in configs if c["name"] == "arc")
        return GenericEval(cfg)

    def test_normalize_row_dict_choices(self):
        ev = self._make_eval()
        raw = {
            "question": "Which is largest?",
            "answerKey": "C",
            "choices": {
                "label": ["A", "B", "C", "D"],
                "text": ["ant", "bee", "elephant", "dog"],
            },
        }
        entry = ev._normalize_row(raw, 0)
        assert entry["ground_truth"] == "C"
        choices = entry["_raw"]["_choices_normalized"]
        assert ("C", "elephant") in choices

    def test_normalize_row_list_choices(self):
        ev = self._make_eval()
        raw = {
            "question": "Which is largest?",
            "answerKey": "C",
            "choices": ["ant", "bee", "elephant", "dog"],
        }
        entry = ev._normalize_row(raw, 0)
        choices = entry["_raw"]["_choices_normalized"]
        assert choices[2] == ("C", "elephant")

    def test_score_correct_letter(self):
        ev = self._make_eval()
        score, info = ev.score("The answer is C", "C")
        assert score == 1.0
        assert info["parsed"] == "C"

    def test_score_wrong_letter(self):
        ev = self._make_eval()
        score, info = ev.score("The answer is A", "C")
        assert score == 0.0
        assert info["parsed"] == "A"
