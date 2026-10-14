"""Tests for CLI argument parsing."""
import pytest
from unittest.mock import patch

from eval_harness.cli import main


class TestCliParsing:
    def test_config_flag_is_string_not_list(self):
        """The old type=list bug split strings into characters."""
        import sys
        with patch("eval_harness.cli.evaluate") as mock_eval:
            saved = sys.argv
            try:
                sys.argv = ["eval-harness", "--model", "m", "--dataset", "d",
                             "--config", "ARC-Challenge",
                             "--question_field", "q", "--answer_field", "a"]
                main()
                call_args = mock_eval.call_args
                eval_configs = call_args.kwargs.get("eval_configs") or call_args[1].get("eval_configs")
                if eval_configs is None:
                    eval_configs = call_args[0][1]
                cfg = eval_configs[0]
                assert cfg["config"] == "ARC-Challenge"
                assert cfg["config"] != list("ARC-Challenge")
            finally:
                sys.argv = saved

    def test_config_action_append(self):
        """Multiple --config flags should produce a list."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, action="append", default=None)
        args = parser.parse_args(["--config", "main", "--config", "socratic"])
        assert args.config == ["main", "socratic"]

    def test_config_single_is_list(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, action="append", default=None)
        args = parser.parse_args(["--config", "main"])
        assert args.config == ["main"]

    def test_scorer_choices_validated(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--scorer", type=str,
                            choices=["exact_match", "number_extract", "boxed_latex",
                                     "letter_match", "code_exec"])
        with pytest.raises(SystemExit):
            parser.parse_args(["--scorer", "nonexistent"])

    def test_default_batch_size_is_8(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", type=int, default=8)
        args = parser.parse_args([])
        assert args.batch_size == 8
