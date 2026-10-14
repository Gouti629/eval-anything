import os
import platform
import subprocess
import sys
import tempfile


def execute_code(code: str, test_code: str, timeout: int = 10) -> float:
    full_code = code + "\n" + test_code
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
