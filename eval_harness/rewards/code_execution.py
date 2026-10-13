import os
import subprocess
import sys
import tempfile


def execute_code(code: str, test_code: str, timeout: int = 10) -> float:
    full_code = code + "\n" + test_code
    fd, temp_path = tempfile.mkstemp(suffix=".py", prefix="eval_exec_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(full_code)
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            timeout=timeout,
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
