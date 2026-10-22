"""Tests for sandboxed code execution."""
import os
import platform
import pytest

from eval_harness.rewards.code_execution import (
    execute_code, set_sandbox_mode, _detect_sandbox,
)


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


class TestCodeExecutionIsolation:
    """Verify intended restrictions of the execution environment.

    These tests document restrictions, NOT proof of secure sandboxing.
    A subprocess with restricted env vars does not prevent a determined
    adversary from accessing the host. Real sandbox security requires
    OS-level isolation (containers, seccomp, etc.).
    """

    def test_no_pythonpath_leaks(self):
        code = "import os\nassert os.environ.get('PYTHONPATH', '') == ''"
        assert execute_code(code, "") == 1.0

    def test_cwd_is_temp(self):
        import tempfile
        tmp = tempfile.gettempdir()
        code = f"import os\nassert os.getcwd().startswith({tmp!r})"
        assert execute_code(code, "") == 1.0

    def test_cannot_read_project_files_by_relative_path(self):
        code = "open('pyproject.toml').read()"
        assert execute_code(code, "") == 0.0

    def test_infinite_loop_times_out(self):
        code = "while True: pass"
        assert execute_code(code, "", timeout=2) == 0.0

    @pytest.mark.skipif(platform.system() == "Windows", reason="os.fork not available on Windows")
    def test_fork_bomb_contained(self):
        code = "import os\nfor _ in range(1000): os.fork()"
        assert execute_code(code, "", timeout=3) == 0.0


class TestSandboxMode:
    def setup_method(self):
        set_sandbox_mode("auto")

    def teardown_method(self):
        set_sandbox_mode("auto")

    def test_set_mode_none(self):
        set_sandbox_mode("none")
        assert _detect_sandbox() == "none"

    def test_set_mode_docker(self):
        set_sandbox_mode("docker")
        assert _detect_sandbox() == "docker"

    def test_set_mode_bwrap(self):
        set_sandbox_mode("bwrap")
        assert _detect_sandbox() == "bwrap"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="sandbox must be one of"):
            set_sandbox_mode("invalid")

    def test_auto_returns_valid(self):
        set_sandbox_mode("auto")
        result = _detect_sandbox()
        assert result in ("docker", "bwrap", "none")

    def test_unsandboxed_still_works(self):
        set_sandbox_mode("none")
        assert execute_code("x = 1 + 1", "assert x == 2") == 1.0
