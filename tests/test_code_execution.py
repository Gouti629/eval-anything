"""Tests for sandboxed code execution."""
import pytest

from eval_harness.rewards.code_execution import execute_code


class TestExecuteCode:
    def test_passing_code(self):
        assert execute_code("x = 1 + 1", "assert x == 2") == 1.0

    def test_failing_code(self):
        assert execute_code("x = 1 + 1", "assert x == 3") == 0.0

    def test_syntax_error(self):
        assert execute_code("def f(:", "assert True") == 0.0

    def test_timeout(self):
        code = "while True: pass"
        assert execute_code(code, "", timeout=2) == 0.0

    def test_runtime_error(self):
        assert execute_code("1/0", "") == 0.0

    def test_import_works(self):
        code = "import math\nx = math.sqrt(4)"
        assert execute_code(code, "assert x == 2.0") == 1.0

    def test_list_test_code_joined(self):
        """code_exec scorer joins list ground_truth before calling execute_code,
        but execute_code itself takes a string."""
        tests = "\n".join(["assert 1 == 1", "assert 2 == 2"])
        assert execute_code("", tests) == 1.0

    def test_empty_code_empty_test(self):
        assert execute_code("", "") == 1.0
