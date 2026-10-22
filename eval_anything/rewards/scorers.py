from __future__ import annotations

import re

from eval_anything.rewards.code_execution import execute_code_detailed


def _strip_think(text: str) -> str:
    result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" not in result:
        return result.strip()

    after = result[result.index("<think>") + len("<think>"):]
    return after.strip()


def exact_match(output: str, ground_truth: str, **kwargs) -> tuple[float, dict]:
    clean = _strip_think(output)
    if not clean.strip():
        return 0.0, {"parsed": None, "status": "missing", "failure_type": "empty"}
    a = " ".join(clean.strip().lower().split())
    b = " ".join(ground_truth.strip().lower().split())
    score = 1.0 if a == b else 0.0
    ft = "correct" if score == 1.0 else "wrong_answer"
    return score, {"parsed": a, "status": "parsed", "failure_type": ft}


def _normalize_number(s: str) -> float | None:
    s = s.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    s = re.sub(r"\s+", "", s)
    if not s:
        return None
    sci = re.match(r"^(-?\d+\.?\d*)[eE]([+-]?\d+)$", s)
    if sci:
        try:
            return float(s)
        except ValueError:
            return None
    frac = re.match(r"^(-?\d+)\s*/\s*(\d+)$", s)
    if frac:
        try:
            return float(frac.group(1)) / float(frac.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


_NUMBER_RE = re.compile(
    r"-?\d+/\d+"
    r"|-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?",
)


def _extract_gt_number(ground_truth: str, delimiter: str | None) -> str:
    gt_text = ground_truth
    if delimiter and delimiter in ground_truth:
        gt_text = ground_truth.split(delimiter)[-1].strip()
    gt_nums = _NUMBER_RE.findall(gt_text)
    return gt_nums[0] if gt_nums else gt_text


def number_extract(output: str, ground_truth: str, delimiter: str | None = None, **kwargs) -> tuple[float, dict]:
    clean = _strip_think(output)

    extracted = None
    method = "missing"

    if delimiter and delimiter in clean:
        after_delim = clean.split(delimiter)[-1].strip()
        nums = _NUMBER_RE.findall(after_delim)
        if nums:
            extracted = nums[0]
            method = "delimiter"

    if extracted is None:
        numbers = _NUMBER_RE.findall(clean)
        if numbers:
            extracted = numbers[-1]
            method = "fallback_last_number"

    if extracted is None:
        ft = "empty" if not clean.strip() else "format_error"
        return 0.0, {"parsed": None, "status": "missing", "method": method, "failure_type": ft}

    gen_val = _normalize_number(extracted)
    if gen_val is None:
        return 0.0, {"parsed": extracted, "status": "parse_error", "method": method, "failure_type": "format_error"}

    gt_text = _extract_gt_number(ground_truth, delimiter)
    gt_val = _normalize_number(gt_text)
    if gt_val is None:
        return 0.0, {"parsed": extracted, "status": "parsed", "method": method, "failure_type": "format_error"}

    score = 1.0 if abs(gen_val - gt_val) < 1e-6 else 0.0
    ft = "correct" if score == 1.0 else "wrong_answer"
    return score, {"parsed": extracted, "status": "parsed", "method": method, "failure_type": ft}


def _extract_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i]
            depth -= 1
    return None


def _normalize_latex(s: str) -> str:
    s = s.strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\%", "%").replace("\\$", "$")
    frac = re.match(r"^\\frac\{([^}]+)\}\{([^}]+)\}$", s)
    if frac:
        try:
            return str(float(frac.group(1)) / float(frac.group(2)))
        except (ValueError, ZeroDivisionError):
            pass
    s = s.replace(",", "").replace(" ", "")
    return s


def boxed_latex(output: str, ground_truth: str, **kwargs) -> tuple[float, dict]:
    clean = _strip_think(output)
    if not clean.strip():
        return 0.0, {"parsed": None, "status": "missing", "failure_type": "empty"}

    extracted = _extract_boxed(clean)
    status = "parsed"
    if extracted is None:
        numbers = _NUMBER_RE.findall(clean)
        extracted = numbers[-1] if numbers else ""
        if not extracted:
            return 0.0, {"parsed": None, "status": "missing", "failure_type": "format_error"}
        status = "fallback_number"

    gt = _extract_boxed(ground_truth) or ground_truth

    na = _normalize_latex(extracted)
    nb = _normalize_latex(gt)
    if na.lower() == nb.lower():
        return 1.0, {"parsed": extracted, "status": status, "failure_type": "correct"}
    try:
        score = 1.0 if abs(float(na) - float(nb)) < 1e-6 else 0.0
        ft = "correct" if score == 1.0 else "wrong_answer"
        return score, {"parsed": extracted, "status": status, "failure_type": ft}
    except ValueError:
        return 0.0, {"parsed": extracted, "status": status, "failure_type": "wrong_answer"}


_FINAL_ANSWER_RE = re.compile(
    r"(?:the\s+answer\s+is|answer:\s*|my\s+answer\s+is)\s*\(?([A-Da-d])\)?",
    re.IGNORECASE,
)
_STANDALONE_LETTER_RE = re.compile(
    r"(?:^|\n)\s*\(?([A-Da-d])\)?\s*\.?\s*$",
    re.MULTILINE,
)


def letter_match(output: str, ground_truth: str, **kwargs) -> tuple[float, dict]:
    clean = _strip_think(output)

    if not clean.strip():
        return 0.0, {"parsed": None, "status": "missing", "failure_type": "empty"}

    for pattern in (_FINAL_ANSWER_RE, _STANDALONE_LETTER_RE):
        m = pattern.search(clean)
        if m:
            letter = m.group(1).upper()
            score = 1.0 if letter == ground_truth.strip().upper() else 0.0
            ft = "correct" if score == 1.0 else "wrong_answer"
            return score, {"parsed": letter, "status": "parsed", "failure_type": ft}

    stripped = clean.strip()
    if len(stripped) == 1 and stripped.upper() in "ABCD":
        letter = stripped.upper()
        score = 1.0 if letter == ground_truth.strip().upper() else 0.0
        ft = "correct" if score == 1.0 else "wrong_answer"
        return score, {"parsed": letter, "status": "parsed", "failure_type": ft}

    return 0.0, {"parsed": None, "status": "missing", "failure_type": "format_error"}


def _extract_code(text: str) -> str:
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].lstrip("\n").rstrip()
    blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].lstrip("\n").rstrip()
    return ""


def code_exec(output: str, ground_truth, prepend: str | None = None, timeout: int = 10, **kwargs) -> tuple[float, dict]:
    clean = _strip_think(output)
    if not clean.strip():
        return 0.0, {"parsed": None, "status": "missing", "failure_type": "empty"}
    code = _extract_code(clean)
    if not code and not prepend:
        return 0.0, {"parsed": None, "status": "missing", "failure_type": "format_error"}
    if prepend:
        code = prepend + "\n" + code
    if isinstance(ground_truth, list):
        test_code = "\n".join(ground_truth)
    else:
        test_code = str(ground_truth)
    score, reason = execute_code_detailed(code, test_code, timeout=timeout)
    ft_map = {"pass": "correct", "fail": "wrong_answer", "timeout": "timeout", "error": "wrong_answer"}
    ft = ft_map.get(reason, "wrong_answer")
    return score, {"parsed": code[:500], "status": "parsed", "failure_type": ft}


SCORERS = {
    "exact_match": exact_match,
    "number_extract": number_extract,
    "boxed_latex": boxed_latex,
    "letter_match": letter_match,
    "code_exec": code_exec,
}
