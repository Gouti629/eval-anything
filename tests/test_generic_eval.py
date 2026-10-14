"""Tests for GenericEval — config validation, ARC normalization, scoring dispatch."""
import pytest

from eval_harness.evals.generic import GenericEval, validate_config, config_fingerprint, ConfigError


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
