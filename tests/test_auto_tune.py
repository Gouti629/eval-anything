"""Tests for auto_tune — memory-based batch/token estimation."""
import pytest

from eval_anything.utils.auto_tune import _snap_down, _STANDARD_BATCH_SIZES, _STANDARD_TOKEN_SIZES


class TestSnapDown:
    def test_exact_match(self):
        assert _snap_down(32, _STANDARD_BATCH_SIZES) == 32

    def test_between_values(self):
        assert _snap_down(50, _STANDARD_BATCH_SIZES) == 48

    def test_below_minimum(self):
        assert _snap_down(4, _STANDARD_BATCH_SIZES) is None

    def test_above_maximum(self):
        assert _snap_down(200, _STANDARD_BATCH_SIZES) == 128

    def test_token_snap(self):
        assert _snap_down(3000, _STANDARD_TOKEN_SIZES) == 2048

    def test_token_exact(self):
        assert _snap_down(4096, _STANDARD_TOKEN_SIZES) == 4096

    def test_token_large(self):
        assert _snap_down(10000, _STANDARD_TOKEN_SIZES) == 8192


class TestAutoTuneNoGPU:
    def test_both_given_passthrough(self):
        from eval_anything.utils.auto_tune import auto_tune_params

        class FakeModel:
            pass

        bs, mt = auto_tune_params(FakeModel(), "hf", batch_size=16, max_new_tokens=1024)
        assert bs == 16
        assert mt == 1024

    def test_fallback_when_no_cuda(self):
        from eval_anything.utils.auto_tune import auto_tune_params
        import torch

        class FakeModel:
            pass

        if not torch.cuda.is_available():
            bs, mt = auto_tune_params(FakeModel(), "hf", batch_size=None, max_new_tokens=None)
            assert bs == 8
            assert mt == 512

    def test_fallback_batch_auto(self):
        from eval_anything.utils.auto_tune import auto_tune_params
        import torch

        class FakeModel:
            pass

        if not torch.cuda.is_available():
            bs, mt = auto_tune_params(FakeModel(), "hf", batch_size=None, max_new_tokens=1024)
            assert bs == 8
            assert mt == 1024

    def test_fallback_tokens_auto(self):
        from eval_anything.utils.auto_tune import auto_tune_params
        import torch

        class FakeModel:
            pass

        if not torch.cuda.is_available():
            bs, mt = auto_tune_params(FakeModel(), "hf", batch_size=32, max_new_tokens=None)
            assert bs == 32
            assert mt == 512
