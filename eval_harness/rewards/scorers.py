import re

from eval_harness.rewards.code_execution import execute_code


def _strip_think(text: str) -> str:
    result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in result:
        result = re.sub(r"<think>.*", "", result, flags=re.DOTALL)
    return result.strip()


def exact_match(output: str, ground_truth: str, **kwargs) -> float:
    a = " ".join(output.strip().lower().split())
    b = " ".join(ground_truth.strip().lower().split())
    return 1.0 if a == b else 0.0


def _normalize_number(s: str) -> float | None:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def number_extract(output: str, ground_truth: str, delimiter: str | None = None, **kwargs) -> float:
    clean = _strip_think(output)
    extracted = None
    if delimiter and delimiter in clean:
        extracted = clean.split(delimiter)[-1].strip()
    if extracted is None:
        numbers = re.findall(r"-?\d+\.?\d*", clean)
        extracted = numbers[-1] if numbers else ""

    gt_text = ground_truth
    if delimiter and delimiter in ground_truth:
        gt_text = ground_truth.split(delimiter)[-1].strip()

    gen_val = _normalize_number(extracted)
    gt_val = _normalize_number(gt_text)
    if gen_val is not None and gt_val is not None:
        return 1.0 if abs(gen_val - gt_val) < 1e-6 else 0.0
    return 1.0 if extracted.strip().lower() == gt_text.strip().lower() else 0.0


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


def boxed_latex(output: str, ground_truth: str, **kwargs) -> float:
    clean = _strip_think(output)
    extracted = _extract_boxed(clean)
    if extracted is None:
        numbers = re.findall(r"-?\d+\.?\d*", clean)
        extracted = numbers[-1] if numbers else ""

    gt = _extract_boxed(ground_truth) or ground_truth

    na = _normalize_latex(extracted)
    nb = _normalize_latex(gt)
    if na.lower() == nb.lower():
        return 1.0
    try:
        return 1.0 if abs(float(na) - float(nb)) < 1e-6 else 0.0
    except ValueError:
        return 0.0


def letter_match(output: str, ground_truth: str, **kwargs) -> float:
    clean = _strip_think(output)
    match = re.search(r"[A-D]", clean.upper())
    if match is None:
        return 0.0
    return 1.0 if match.group(0) == ground_truth.strip().upper() else 0.0


def _extract_code(text: str) -> str:
    if "```python" in text:
        parts = text.split("```python")
        code = parts[-1].split("```")[0]
        return code.strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            return parts[1].split("```")[0].strip()
    return text.strip()


def code_exec(output: str, ground_truth: str, prepend: str | None = None, timeout: int = 10, **kwargs) -> float:
    code = _extract_code(output)
    if prepend:
        code = prepend + "\n" + code
    test_code = ground_truth
    if isinstance(ground_truth, list):
        test_code = "\n".join(ground_truth)
    return execute_code(code, test_code, timeout=timeout)


SCORERS = {
    "exact_match": exact_match,
    "number_extract": number_extract,
    "boxed_latex": boxed_latex,
    "letter_match": letter_match,
    "code_exec": code_exec,
}
