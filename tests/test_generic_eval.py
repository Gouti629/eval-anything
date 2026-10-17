"""Tests for GenericEval — config validation, ARC normalization, scoring dispatch, safe filter."""
import json

import pytest

from eval_harness.evals.generic import (
    GenericEval, validate_config, config_fingerprint, ConfigError,
    _safe_eval_filter, _transform_livecodebench_tests, _extract_func_name,
)


class TestValidateConfig:
    def test_valid_config(self):
        cfg = {
            "name": "test",
            "dataset": "test/test",
            "question_field": "q",
            "answer_field": "a",
        }
        assert validate_config(cfg) == []

    def test_missing_required(self):
        cfg = {"name": "test"}
        errors = validate_config(cfg)
        assert len(errors) > 0
        assert "Missing required" in errors[0]

    def test_invalid_scorer(self):
        cfg = {
            "name": "test",
            "dataset": "test/test",
            "question_field": "q",
            "answer_field": "a",
            "scorer": "nonexistent",
        }
        errors = validate_config(cfg)
        assert any("Unknown scorer" in e for e in errors)

    def test_invalid_formatter(self):
        cfg = {
            "name": "test",
            "dataset": "test/test",
            "question_field": "q",
            "answer_field": "a",
            "formatter": "nonexistent",
        }
        errors = validate_config(cfg)
        assert any("Unknown formatter" in e for e in errors)

    def test_mc_without_choices_field(self):
        cfg = {
            "name": "test",
            "dataset": "test/test",
            "question_field": "q",
            "answer_field": "a",
            "formatter": "multiple_choice",
        }
        errors = validate_config(cfg)
        assert any("choices_field" in e for e in errors)

    def test_constructor_raises_on_invalid(self):
        with pytest.raises(ConfigError):
            GenericEval({"name": "bad"})


class TestConfigFingerprint:
    def test_same_config_same_fingerprint(self):
        cfg = {
            "name": "test", "dataset": "ds", "question_field": "q",
            "answer_field": "a", "scorer": "exact_match",
        }
        gen = {"batch_size": 8, "max_new_tokens": 512, "temperature": 0.0}
        fp1 = config_fingerprint(cfg, gen)
        fp2 = config_fingerprint(cfg, gen)
        assert fp1 == fp2

    def test_different_scorer_different_fingerprint(self):
        gen = {"batch_size": 8}
        cfg1 = {"dataset": "ds", "scorer": "exact_match"}
        cfg2 = {"dataset": "ds", "scorer": "letter_match"}
        assert config_fingerprint(cfg1, gen) != config_fingerprint(cfg2, gen)

    def test_different_gen_params(self):
        cfg = {"dataset": "ds"}
        assert config_fingerprint(cfg, {"batch_size": 8}) != config_fingerprint(cfg, {"batch_size": 16})

    def test_different_question_field_different_fingerprint(self):
        gen = {"batch_size": 8}
        cfg1 = {"dataset": "ds", "question_field": "q1"}
        cfg2 = {"dataset": "ds", "question_field": "q2"}
        assert config_fingerprint(cfg1, gen) != config_fingerprint(cfg2, gen)

    def test_different_answer_field_different_fingerprint(self):
        gen = {"batch_size": 8}
        cfg1 = {"dataset": "ds", "answer_field": "a1"}
        cfg2 = {"dataset": "ds", "answer_field": "a2"}
        assert config_fingerprint(cfg1, gen) != config_fingerprint(cfg2, gen)

    def test_different_choices_field_different_fingerprint(self):
        gen = {"batch_size": 8}
        cfg1 = {"dataset": "ds", "choices_field": "c1"}
        cfg2 = {"dataset": "ds", "choices_field": "c2"}
        assert config_fingerprint(cfg1, gen) != config_fingerprint(cfg2, gen)

    def test_different_filter_different_fingerprint(self):
        gen = {"batch_size": 8}
        cfg1 = {"dataset": "ds", "filter": "x > 5"}
        cfg2 = {"dataset": "ds", "filter": "x > 10"}
        assert config_fingerprint(cfg1, gen) != config_fingerprint(cfg2, gen)


class TestScoreDispatch:
    def _make_eval(self, scorer, **extra):
        cfg = {
            "name": "test",
            "dataset": "test/test",
            "question_field": "q",
            "answer_field": "a",
            "scorer": scorer,
            **extra,
        }
        return GenericEval(cfg)

    def test_code_exec_preserves_list(self):
        """The critical MBPP bug — ground_truth list must reach code_exec as list."""
        ev = self._make_eval("code_exec")
        gt = ["assert 1 + 1 == 2"]
        # score() should NOT str() the list before passing to code_exec
        score, info = ev.score("x = 1", gt)
        # The code is wrong but the point is it doesn't crash with a stringified list
        assert isinstance(score, float)

    def test_code_exec_correct(self):
        ev = self._make_eval("code_exec")
        code = "def f(x): return x + 1"
        gt = ["assert f(1) == 2", "assert f(0) == 1"]
        score, _ = ev.score(code, gt)
        assert score == 1.0

    def test_number_extract_with_delimiter(self):
        ev = self._make_eval("number_extract", delimiter="####")
        score, _ = ev.score("reasoning #### 42", "#### 42")
        assert score == 1.0

    def test_letter_match_dispatch(self):
        ev = self._make_eval("letter_match")
        score, info = ev.score("The answer is C", "C")
        assert score == 1.0


class TestArcNormalization:
    def test_dict_choices_normalized(self):
        cfg = {
            "name": "arc",
            "dataset": "test/test",
            "question_field": "question",
            "answer_field": "answerKey",
            "formatter": "multiple_choice",
            "choices_field": "choices",
        }
        ev = GenericEval(cfg)
        raw = {
            "question": "What is 1+1?",
            "answerKey": "B",
            "choices": {
                "label": ["A", "B", "C", "D"],
                "text": ["1", "2", "3", "4"],
            },
        }
        entry = ev._normalize_row(raw, 0)
        assert entry["_raw"]["_choices_normalized"] == [
            ("A", "1"), ("B", "2"), ("C", "3"), ("D", "4"),
        ]
        assert entry["example_id"].startswith("arc_0_")
        assert entry["ground_truth"] == "B"

    def test_list_choices_normalized(self):
        cfg = {
            "name": "test",
            "dataset": "test/test",
            "question_field": "q",
            "answer_field": "a",
            "formatter": "multiple_choice",
            "choices_field": "choices",
        }
        ev = GenericEval(cfg)
        raw = {"q": "What?", "a": "B", "choices": ["cat", "dog", "fish", "bird"]}
        entry = ev._normalize_row(raw, 0)
        choices = entry["_raw"]["_choices_normalized"]
        assert choices[0] == ("A", "cat")
        assert choices[1] == ("B", "dog")

    def test_missing_field_raises(self):
        cfg = {
            "name": "test",
            "dataset": "test/test",
            "question_field": "nonexistent",
            "answer_field": "a",
        }
        ev = GenericEval(cfg)
        with pytest.raises(ConfigError, match="field extraction failed"):
            ev._normalize_row({"a": "x"}, 0)

class TestSafeFilter:
    def test_simple_comparison(self):
        assert _safe_eval_filter("x > 5", {"x": 10}) is True
        assert _safe_eval_filter("x > 5", {"x": 3}) is False

    def test_chained_comparison(self):
        assert _safe_eval_filter("x >= 11 and x <= 510", {"x": 100, "task_id": 100}) is True
        assert _safe_eval_filter("x >= 11 and x <= 510", {"x": 5, "task_id": 5}) is False

    def test_mbpp_filter(self):
        assert _safe_eval_filter("task_id >= 11 and task_id <= 510", {"task_id": 100}) is True
        assert _safe_eval_filter("task_id >= 11 and task_id <= 510", {"task_id": 5}) is False

    def test_equality(self):
        assert _safe_eval_filter("name == 'test'", {"name": "test"}) is True
        assert _safe_eval_filter("name == 'test'", {"name": "other"}) is False

    def test_in_operator(self):
        assert _safe_eval_filter("x in [1, 2, 3]", {"x": 2}) is True
        assert _safe_eval_filter("x in [1, 2, 3]", {"x": 5}) is False

    def test_not_operator(self):
        assert _safe_eval_filter("not x > 5", {"x": 3}) is True

    def test_rejects_function_calls(self):
        with pytest.raises(ConfigError, match="Unsupported"):
            _safe_eval_filter("__import__('os').system('echo pwned')", {})

    def test_rejects_attribute_access(self):
        with pytest.raises(ConfigError, match="Unsupported"):
            _safe_eval_filter("x.__class__", {"x": 1})

    def test_unknown_field_raises(self):
        with pytest.raises(ConfigError, match="unknown field"):
            _safe_eval_filter("nonexistent > 5", {"x": 1})

    def test_syntax_error_raises(self):
        with pytest.raises(ConfigError, match="Invalid filter"):
            _safe_eval_filter("x >>> 5", {"x": 1})


class TestAnswerTransformValidation:
    def test_valid_transform_accepted(self):
        cfg = {
            "name": "test", "dataset": "d", "question_field": "q",
            "answer_field": "a", "answer_transform": "livecodebench_tests",
            "scorer": "code_exec", "formatter": "code_completion",
        }
        assert validate_config(cfg) == []

    def test_unknown_transform_rejected(self):
        cfg = {
            "name": "test", "dataset": "d", "question_field": "q",
            "answer_field": "a", "answer_transform": "nonexistent",
        }
        errors = validate_config(cfg)
        assert any("Unknown answer_transform" in e for e in errors)

    def test_transform_changes_fingerprint(self):
        gen = {"batch_size": 8}
        cfg1 = {"dataset": "d", "answer_transform": None}
        cfg2 = {"dataset": "d", "answer_transform": "livecodebench_tests"}
        assert config_fingerprint(cfg1, gen) != config_fingerprint(cfg2, gen)


class TestLiveCodeBenchTransform:
    def test_func_name_style(self):
        verification = json.dumps({
            "test_cases": [
                {"input": "[1, 2, 3]", "output": "6", "func_name": "sum_list"},
            ]
        })
        result = _transform_livecodebench_tests(verification)
        assert len(result) == 1
        assert result[0] == "assert sum_list([1, 2, 3]) == 6"

    def test_default_func_name(self):
        verification = json.dumps({
            "test_cases": [
                {"input": "5", "output": "25"},
            ]
        })
        result = _transform_livecodebench_tests(verification)
        assert result[0] == "assert solution(5) == 25"

    def test_stdin_style(self):
        verification = json.dumps({
            "test_cases": [
                {"input": "hello", "output": "HELLO", "testtype": "stdin"},
            ]
        })
        result = _transform_livecodebench_tests(verification)
        assert len(result) == 1
        code = result[0]
        assert "sys.stdin" in code
        assert "main()" in code
        assert "HELLO" in code

    def test_multiple_cases(self):
        verification = json.dumps({
            "test_cases": [
                {"input": "1", "output": "1", "func_name": "f"},
                {"input": "2", "output": "4", "func_name": "f"},
            ]
        })
        result = _transform_livecodebench_tests(verification)
        assert len(result) == 2

    def test_dict_input_without_wrapper(self):
        verification = json.dumps([
            {"input": "3", "output": "9", "func_name": "sq"},
        ])
        result = _transform_livecodebench_tests(verification)
        assert result[0] == "assert sq(3) == 9"

    def test_output_is_valid_python(self):
        verification = json.dumps({
            "test_cases": [
                {"input": "[1, 2]", "output": "3", "func_name": "add"},
            ]
        })
        result = _transform_livecodebench_tests(verification)
        for line in result:
            compile(line, "<test>", "exec")

    def test_normalize_row_applies_transform(self):
        cfg = {
            "name": "lcb", "dataset": "d", "question_field": "q",
            "answer_field": "a", "answer_transform": "livecodebench_tests",
            "scorer": "code_exec", "formatter": "code_completion",
        }
        ev = GenericEval(cfg)
        verification = json.dumps({
            "test_cases": [{"input": "1", "output": "2", "func_name": "f"}]
        })
        row = {"q": "Write f", "a": verification}
        entry = ev._normalize_row(row, 0)
        assert isinstance(entry["ground_truth"], list)
        assert entry["ground_truth"][0] == "assert f(1) == 2"


class TestExtractFuncName:
    def test_from_assert_list(self):
        gt = ["assert remove_dirty_chars('abc', 'a') == 'bc'"]
        assert _extract_func_name(gt) == "remove_dirty_chars"

    def test_from_string(self):
        assert _extract_func_name("assert f(1) == 2") == "f"

    def test_no_assert_returns_none(self):
        assert _extract_func_name("print('hello')") is None

    def test_empty_list(self):
        assert _extract_func_name([]) is None

    def test_first_match_wins(self):
        gt = ["assert add(1,2) == 3", "assert add(0,0) == 0"]
        assert _extract_func_name(gt) == "add"


class TestCodeCompletionFuncNamePrompt:
    def _make_eval(self, **extra):
        cfg = {
            "name": "test", "dataset": "d", "question_field": "q",
            "answer_field": "a", "scorer": "code_exec",
            "formatter": "code_completion",
            "system_prompt": "Write a Python function.",
            **extra,
        }
        return GenericEval(cfg)

    def _dummy_tokenizer(self):
        class Tok:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return "\n".join(m["content"] for m in messages) + "\nassistant:"
        return Tok()

    def test_func_name_appended_to_prompt(self):
        ev = self._make_eval()
        example = {
            "question": "Write a function to add two numbers.",
            "ground_truth": ["assert add(1, 2) == 3"],
            "_raw": {},
        }
        prompt = ev.format_prompt(example, self._dummy_tokenizer())
        assert "`add`" in prompt

    def test_no_func_name_when_prepend_prompt(self):
        ev = self._make_eval(prepend_prompt=True)
        example = {
            "question": "def add(a, b):",
            "ground_truth": ["assert add(1, 2) == 3"],
            "_raw": {},
        }
        prompt = ev.format_prompt(example, self._dummy_tokenizer())
        assert "should be named" not in prompt

    def test_no_func_name_when_not_extractable(self):
        ev = self._make_eval()
        example = {
            "question": "Write something.",
            "ground_truth": "print('hello')",
            "_raw": {},
        }
        prompt = ev.format_prompt(example, self._dummy_tokenizer())
        assert "should be named" not in prompt


class TestExampleId:
    def test_example_id_has_content_hash(self):
        cfg = {
            "name": "t",
            "dataset": "d",
            "question_field": "q",
            "answer_field": "a",
        }
        ev = GenericEval(cfg)
        e1 = ev._normalize_row({"q": "hello", "a": "world"}, 0)
        e2 = ev._normalize_row({"q": "different", "a": "data"}, 0)
        assert e1["example_id"] != e2["example_id"]
