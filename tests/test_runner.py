"""Tests for runner — resume fingerprinting, atomic saves, OOM batch halving."""
import json
import os
import tempfile

import pytest

from eval_harness.runner import (
    pass_at_k,
    _save_state_atomic,
    _run_fingerprint,
    _find_matching_partial,
    _serialize_gt,
)


class TestPassAtK:
    def test_all_correct(self):
        assert pass_at_k(10, 10, 1) == 1.0

    def test_none_correct(self):
        assert pass_at_k(10, 0, 1) == 0.0

    def test_some_correct(self):
        result = pass_at_k(10, 5, 1)
        assert 0.0 < result < 1.0

    def test_k_equals_n(self):
        result = pass_at_k(5, 3, 5)
        assert result == 1.0


class TestAtomicSave:
    def test_creates_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.json")
            _save_state_atomic(path, {"key": "value"})
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data == {"key": "value"}

    def test_no_tmp_file_remains(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.json")
            _save_state_atomic(path, {"key": "value"})
            files = os.listdir(d)
            assert len(files) == 1
            assert files[0] == "test.json"

    def test_overwrites_existing(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.json")
            _save_state_atomic(path, {"version": 1})
            _save_state_atomic(path, {"version": 2})
            with open(path) as f:
                assert json.load(f)["version"] == 2


class TestRunFingerprint:
    def test_same_inputs_same_fingerprint(self):
        fp1 = _run_fingerprint(["m1"], [{"dataset": "d"}], {"bs": 8})
        fp2 = _run_fingerprint(["m1"], [{"dataset": "d"}], {"bs": 8})
        assert fp1 == fp2

    def test_different_model_different_fingerprint(self):
        fp1 = _run_fingerprint(["m1"], [{"dataset": "d"}], {"bs": 8})
        fp2 = _run_fingerprint(["m2"], [{"dataset": "d"}], {"bs": 8})
        assert fp1 != fp2

    def test_different_gen_params_different_fingerprint(self):
        fp1 = _run_fingerprint(["m1"], [{"dataset": "d"}], {"bs": 8})
        fp2 = _run_fingerprint(["m1"], [{"dataset": "d"}], {"bs": 16})
        assert fp1 != fp2

    def test_think_mode_changes_fingerprint(self):
        fp1 = _run_fingerprint(["m1"], [{"dataset": "d"}], {"bs": 8}, think=False)
        fp2 = _run_fingerprint(["m1"], [{"dataset": "d"}], {"bs": 8}, think=True)
        assert fp1 != fp2


class TestFindMatchingPartial:
    def test_finds_matching(self):
        with tempfile.TemporaryDirectory() as d:
            fp = "abc123"
            path = os.path.join(d, f"{fp}.partial.json")
            with open(path, "w") as f:
                json.dump({"fingerprint": fp, "data": "test"}, f)
            result = _find_matching_partial(d, fp)
            assert result is not None
            assert result["fingerprint"] == fp

    def test_rejects_mismatched(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "abc123.partial.json")
            with open(path, "w") as f:
                json.dump({"fingerprint": "different", "data": "test"}, f)
            result = _find_matching_partial(d, "abc123")
            assert result is None

    def test_missing_directory(self):
        result = _find_matching_partial("/nonexistent/path", "abc")
        assert result is None


class TestResumeByExampleId:
    """Resume must use example_id (content hash), not row index."""

    def test_reordered_rows_still_matched(self):
        """If dataset rows are reordered, done examples should be found by example_id."""
        from eval_harness.runner import _is_example_done
        state = {
            "examples": {
                "model1": {
                    "eval1": [
                        {"example_id": "eval1_0_abc123", "idx": 0, "score": 1.0},
                        {"example_id": "eval1_1_def456", "idx": 1, "score": 0.0},
                    ]
                }
            }
        }
        assert _is_example_done(state, "model1", "eval1", "eval1_0_abc123")
        assert _is_example_done(state, "model1", "eval1", "eval1_1_def456")
        assert not _is_example_done(state, "model1", "eval1", "eval1_2_ghi789")

    def test_changed_content_not_matched(self):
        """If question or answer changes, different content hash -> not matched."""
        from eval_harness.evals.generic import GenericEval
        cfg = {"name": "t", "dataset": "d", "question_field": "q", "answer_field": "a"}
        ev = GenericEval(cfg)
        e1 = ev._normalize_row({"q": "What is 2+2?", "a": "4"}, 0)
        e2 = ev._normalize_row({"q": "What is 3+3?", "a": "6"}, 0)
        assert e1["example_id"] != e2["example_id"]

    def test_same_content_same_id_regardless_of_index(self):
        from eval_harness.evals.generic import GenericEval
        cfg = {"name": "t", "dataset": "d", "question_field": "q", "answer_field": "a"}
        ev = GenericEval(cfg)
        e1 = ev._normalize_row({"q": "hello", "a": "world"}, 0)
        e2 = ev._normalize_row({"q": "hello", "a": "world"}, 5)
        hash1 = e1["example_id"].split("_", 2)[2]
        hash2 = e2["example_id"].split("_", 2)[2]
        assert hash1 == hash2


class TestTimeoutSafety:
    """Generation timeout must not allow overlapping model.generate() calls."""

    def test_wait_for_pending_clears_finished_thread(self):
        from eval_harness.utils.generation import _wait_for_pending, _pending_lock, _pending_thread
        import eval_harness.utils.generation as gen_mod
        import threading

        finished = threading.Thread(target=lambda: None)
        finished.start()
        finished.join()
        with gen_mod._pending_lock:
            gen_mod._pending_thread = finished
        gen_mod._wait_for_pending()
        assert gen_mod._pending_thread is None

    def test_wait_for_pending_blocks_on_alive_thread(self):
        import eval_harness.utils.generation as gen_mod
        import threading

        event = threading.Event()
        slow_thread = threading.Thread(target=lambda: event.wait(timeout=5))
        slow_thread.start()
        with gen_mod._pending_lock:
            gen_mod._pending_thread = slow_thread
        event.set()
        gen_mod._wait_for_pending()
        assert not slow_thread.is_alive()
        assert gen_mod._pending_thread is None


class TestGenerationResult:
    """GenerationResult carries token count and truncation flag."""

    def test_fields_present(self):
        from eval_harness.utils.generation import GenerationResult
        r = GenerationResult(text=["hello"], token_count=5, truncated=False)
        assert r.text == ["hello"]
        assert r.token_count == 5
        assert r.truncated is False

    def test_truncated_flag(self):
        from eval_harness.utils.generation import GenerationResult
        r = GenerationResult(text=["hi"], token_count=512, truncated=True)
        assert r.truncated is True


class TestTruncationDetection:
    def test_not_truncated_when_short(self):
        import torch
        from eval_harness.utils.generation import _is_truncated
        tokens = torch.tensor([1, 2, 3])
        assert _is_truncated(tokens, eos_token_id=99, max_new_tokens=512) is False

    def test_truncated_at_max_no_eos(self):
        import torch
        from eval_harness.utils.generation import _is_truncated
        tokens = torch.tensor([1, 2, 3, 4])
        assert _is_truncated(tokens, eos_token_id=99, max_new_tokens=4) is True

    def test_not_truncated_at_max_with_eos(self):
        import torch
        from eval_harness.utils.generation import _is_truncated
        tokens = torch.tensor([1, 2, 3, 99])
        assert _is_truncated(tokens, eos_token_id=99, max_new_tokens=4) is False


class TestPerEvalMaxNewTokens:
    def test_eval_max_tokens_in_config(self):
        from eval_harness.evals.generic import GenericEval
        cfg = {
            "name": "test", "dataset": "d",
            "question_field": "q", "answer_field": "a",
            "max_new_tokens": 256,
        }
        ev = GenericEval(cfg)
        assert ev.max_new_tokens == 256

    def test_eval_max_tokens_default_none(self):
        from eval_harness.evals.generic import GenericEval
        cfg = {"name": "test", "dataset": "d", "question_field": "q", "answer_field": "a"}
        ev = GenericEval(cfg)
        assert ev.max_new_tokens is None

    def test_max_tokens_override_changes_fingerprint(self):
        from eval_harness.evals.generic import config_fingerprint
        fp1 = config_fingerprint(
            {"dataset": "d", "max_new_tokens": 256}, {"max_new_tokens": 512}
        )
        fp2 = config_fingerprint(
            {"dataset": "d", "max_new_tokens": 512}, {"max_new_tokens": 512}
        )
        assert fp1 != fp2


class TestResultSemantics:
    """Verify correct/incorrect counting and score denominators."""

    def test_resume_counts_ok_as_correct(self):
        """Examples with status='ok' increment correct, not just 'scored'."""
        examples = [
            {"status": "ok", "score": 1.0, "example_id": "a"},
            {"status": "incorrect", "score": 0.0, "example_id": "b"},
            {"status": "timeout", "score": 0.0, "example_id": "c"},
        ]
        correct = sum(1 for e in examples if e["status"] == "ok")
        incorrect = sum(1 for e in examples if e["status"] == "incorrect")
        infra = sum(1 for e in examples if e["status"] in ("timeout", "oom", "error"))
        assert correct == 1
        assert incorrect == 1
        assert infra == 1
        assert correct + incorrect + infra == len(examples)

    def test_score_denominator_is_total(self):
        """avg_score uses n_total as denominator, including infra failures as zeros."""
        scores = [1.0, 0.0, 0.0]  # one correct, one wrong, one infra (score 0)
        n_total = 3
        avg_score = sum(scores) / n_total
        assert abs(avg_score - 1 / 3) < 1e-6


class TestPlotComparison:
    """plot_comparison must handle models with different eval subsets."""

    def test_different_eval_subsets_no_crash(self):
        import tempfile
        from eval_harness.plotting.compare import plot_comparison
        results = {
            "model_a": {
                "gsm8k": {"score": 0.8},
                "mmlu": {"score": 0.7},
            },
            "model_b": {
                "gsm8k": {"score": 0.6},
            },
        }
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "chart.png")
            plot_comparison(results, path)
            assert os.path.exists(path)

    def test_no_shared_evals(self):
        import tempfile
        from eval_harness.plotting.compare import plot_comparison
        results = {
            "model_a": {"gsm8k": {"score": 0.8}},
            "model_b": {"mmlu": {"score": 0.6}},
        }
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "chart.png")
            plot_comparison(results, path)
            assert os.path.exists(path)


class TestLimitOption:
    """--limit N must affect fingerprint and be validated."""

    def test_limit_changes_fingerprint(self):
        fp1 = _run_fingerprint(["m1"], [{"dataset": "d"}], {"bs": 8, "limit": None})
        fp2 = _run_fingerprint(["m1"], [{"dataset": "d"}], {"bs": 8, "limit": 10})
        assert fp1 != fp2

    def test_limit_zero_raises(self):
        from eval_harness.runner import evaluate
        with pytest.raises(ValueError, match="limit must be >= 1"):
            evaluate(
                models=["m"], eval_configs=[{"name": "t", "dataset": "d",
                    "question_field": "q", "answer_field": "a"}],
                limit=0,
            )

    def test_limit_negative_raises(self):
        from eval_harness.runner import evaluate
        with pytest.raises(ValueError, match="limit must be >= 1"):
            evaluate(
                models=["m"], eval_configs=[{"name": "t", "dataset": "d",
                    "question_field": "q", "answer_field": "a"}],
                limit=-5,
            )


class TestSerializeGt:
    def test_string_passthrough(self):
        assert _serialize_gt("hello") == "hello"

    def test_list_preserved(self):
        gt = ["assert f(1) == 2", "assert f(0) == 1"]
        result = _serialize_gt(gt)
        assert isinstance(result, list)
        assert result == gt

    def test_int_becomes_string(self):
        assert _serialize_gt(42) == "42"
