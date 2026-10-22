import os
import platform
import shutil
import subprocess
import sys
import tempfile
import warnings


_sandbox_mode = "auto"
_sandbox_warned = False


def set_sandbox_mode(mode: str):
    """Set sandbox mode: 'auto', 'docker', 'bwrap', or 'none'."""
    valid = ("auto", "docker", "bwrap", "none")
    if mode not in valid:
        raise ValueError(f"sandbox must be one of {valid}, got {mode!r}")
    global _sandbox_mode, _sandbox_warned
    _sandbox_mode = mode
    _sandbox_warned = False


def _detect_sandbox() -> str:
    if _sandbox_mode != "auto":
        return _sandbox_mode
    if platform.system() != "Windows" and shutil.which("bwrap"):
        return "bwrap"
    return "none"


def execute_code(code: str, test_code: str, timeout: int = 10) -> float:
    full_code = code + "\n" + test_code
    sandbox = _detect_sandbox()
    if sandbox == "docker":
        return _execute_docker(full_code, timeout)
    if sandbox == "bwrap":
        return _execute_bwrap(full_code, timeout)

    global _sandbox_warned
    if not _sandbox_warned:
        print(
            "\n  WARNING: Running model-generated code WITHOUT sandbox.\n"
            "  Install bubblewrap (Linux: apt install bubblewrap) or use --sandbox docker.\n"
            "  No network blocking, no filesystem isolation.\n",
            flush=True,
        )
        _sandbox_warned = True
    return _execute_subprocess(full_code, timeout)


def _execute_subprocess(full_code: str, timeout: int) -> float:
    fd, temp_path = tempfile.mkstemp(suffix=".py", prefix="eval_exec_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(full_code)

        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": "",
            "HOME": tempfile.gettempdir(),
            "TMPDIR": tempfile.gettempdir(),
        }
        if platform.system() == "Windows":
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
            env["TEMP"] = tempfile.gettempdir()
            env["TMP"] = tempfile.gettempdir()

        kwargs = {}
        if platform.system() != "Windows":
            import resource

            def _set_limits():
                resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
                resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
                try:
                    resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
                except (ValueError, OSError):
                    pass

            kwargs["preexec_fn"] = _set_limits

        result = subprocess.run(
            [sys.executable, "-u", temp_path],
            capture_output=True,
            timeout=timeout,
            env=env,
            cwd=tempfile.gettempdir(),
            **kwargs,
        )
        return 1.0 if result.returncode == 0 else 0.0
    except subprocess.TimeoutExpired:
        return 0.0
    except Exception:
        return 0.0
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def _execute_bwrap(full_code: str, timeout: int) -> float:
    fd, temp_path = tempfile.mkstemp(suffix=".py", prefix="eval_exec_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(full_code)

        result = subprocess.run(
            [
                "bwrap",
                "--ro-bind", "/", "/",
                "--tmpfs", "/tmp",
                "--tmpfs", "/home",
                "--tmpfs", "/run",
                "--dev", "/dev",
                "--proc", "/proc",
                "--unshare-all",
                "--die-with-parent",
                sys.executable, "-I", "-u", temp_path,
            ],
            capture_output=True,
            timeout=timeout,
            cwd="/tmp",
        )
        return 1.0 if result.returncode == 0 else 0.0
    except subprocess.TimeoutExpired:
        return 0.0
    except Exception:
        return 0.0
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def _execute_docker(full_code: str, timeout: int) -> float:
    fd, temp_path = tempfile.mkstemp(suffix=".py", prefix="eval_exec_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(full_code)

        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "--network=none",
                "--memory=512m",
                "--cpus=1",
                "--pids-limit=64",
                "-v", f"{temp_path}:/code.py:ro",
                "python:3.11-slim",
                "python", "-u", "/code.py",
            ],
            capture_output=True,
            timeout=timeout + 10,
        )
        return 1.0 if result.returncode == 0 else 0.0
    except subprocess.TimeoutExpired:
        return 0.0
    except Exception:
        return 0.0
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
