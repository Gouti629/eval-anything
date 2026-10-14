"""Tests for all scoring functions — the most bug-prone part of the harness."""
import pytest

from eval_harness.rewards.scorers import (
    _strip_think,
    _extract_boxed,
    _extract_code,
    _normalize_number,
    exact_match,
    number_extract,
    boxed_latex,
    letter_match,
    code_exec,
)


# === _strip_think ===

class TestStripThink:
    def test_closed_think(self):
        assert _strip_think("<think>reasoning</think>The answer is 42") == "The answer is 42"

    def test_unclosed_think(self):
        result = _strip_think("<think>reasoning about stuff 123 456")
        assert "reasoning" in result

    def test_no_think(self):
        assert _strip_think("plain text") == "plain text"

    def test_multiple_closed(self):
        text = "<think>a</think>middle<think>b</think>end"
        result = _strip_think(text)
        assert "middle" in result
        assert "end" in result

    def test_unclosed_preserves_content(self):
        text = "<think>The answer is clearly 3. So the total is 2 + 1 = 3."
        result = _strip_think(text)
        assert "3" in result

    def test_empty_after_closed_think(self):
        text = "<think>all reasoning no answer</think>"
        result = _strip_think(text)
        assert result == ""

    def test_content_before_unclosed_think(self):
        text = "The answer is 42<think>let me reconsider..."
        result = _strip_think(text)
        assert "42" in result


# === number_extract ===

class TestNumberExtract:
    def test_with_delimiter(self):
        score, info = number_extract("step 1... step 2... #### 42", "#### 42", delimiter="####")
        assert score == 1.0
        assert info["status"] == "parsed"

    def test_without_delimiter(self):
        score, info = number_extract("The answer is 42.", "42")
        assert score == 1.0

    def test_wrong_answer(self):
        score, info = number_extract("The answer is 43.", "42")
        assert score == 0.0

    def test_no_number(self):
        score, info = number_extract("I don't know", "42")
        assert score == 0.0
        assert info["status"] == "missing"

    def test_comma_number(self):
        score, _ = number_extract("The population is 1,234,567", "1234567")
        assert score == 1.0

    def test_decimal(self):
        score, _ = number_extract("The rate is 3.14", "3.14")
        assert score == 1.0

    def test_negative(self):
        score, _ = number_extract("Temperature is -5", "-5")
        assert score == 1.0

    def test_dollar_sign(self):
        score, _ = number_extract("The cost is $100", "$100")
        assert score == 1.0

    def test_scientific_notation(self):
        score, _ = number_extract("Result: 1.5e3", "1500")
        assert score == 1.0

    def test_gsm8k_full_solution_gt(self):
        """Ground truth is a full solution with #### at the end."""
        gt = "It takes 2/2=1 bolt of white fiber\nSo total is 2+1=3\n#### 3"
        score, info = number_extract("#### 3", gt, delimiter="####")
        assert score == 1.0

    def test_think_block_with_numbers(self):
        """Numbers inside <think> should not be extracted."""
        output = "<think>Let me check 7 + 8 = 15, then 15 * 2 = 30</think>#### 30"
        score, info = number_extract(output, "30", delimiter="####")
        assert score == 1.0

    def test_unclosed_think_extracts_last_number(self):
        output = "<think>The answer is 2 + 1 = 3"
        score, info = number_extract(output, "3")
        assert score == 1.0


# === boxed_latex ===

class TestBoxedLatex:
    def test_simple_boxed(self):
        score, _ = boxed_latex("\\boxed{42}", "\\boxed{42}")
        assert score == 1.0

    def test_nested_braces(self):
        score, _ = boxed_latex("\\boxed{\\frac{1}{2}}", "\\boxed{\\frac{1}{2}}")
        assert score == 1.0

    def test_fraction_equals_decimal(self):
        score, _ = boxed_latex("\\boxed{\\frac{1}{2}}", "\\boxed{0.5}")
        assert score == 1.0

    def test_no_boxed_falls_back(self):
        score, info = boxed_latex("The answer is 7", "\\boxed{7}")
        assert score == 1.0
        assert info["status"] == "fallback_number"

    def test_no_answer_at_all(self):
        score, info = boxed_latex("I cannot solve this", "\\boxed{42}")
        assert score == 0.0
        assert info["status"] == "missing"


# === letter_match ===

class TestLetterMatch:
    def test_standalone_letter(self):
        score, info = letter_match("B", "B")
        assert score == 1.0
        assert info["parsed"] == "B"

    def test_the_answer_is_pattern(self):
        score, info = letter_match("After analyzing, the answer is C", "C")
        assert score == 1.0
        assert info["parsed"] == "C"

    def test_does_not_grab_from_reasoning(self):
        """Should NOT pick 'A' from 'Analyzing' or 'B' from reasoning text."""
        output = "Analyzing option B and option C, the answer is D"
        score, info = letter_match(output, "D")
        assert score == 1.0
        assert info["parsed"] == "D"

    def test_reasoning_mentions_multiple_letters_ambiguous(self):
        """Multiple standalone letters without a clear final answer -> ambiguous."""
        output = "Option A is about cats. Option B is about dogs. Both are valid."
        score, info = letter_match(output, "A")
        assert info["status"] == "ambiguous"
        assert score == 0.0

    def test_no_letter_at_all(self):
        score, info = letter_match("I'm not sure about this question.", "A")
        assert score == 0.0
        assert info["status"] == "missing"

    def test_answer_colon_pattern(self):
        score, info = letter_match("Answer: B", "B")
        assert score == 1.0
        assert info["parsed"] == "B"

    def test_single_letter_on_line(self):
        score, info = letter_match("Let me think...\nC\n", "C")
        assert score == 1.0
        assert info["parsed"] == "C"

    def test_wrong_answer(self):
        score, info = letter_match("The answer is A", "B")
        assert score == 0.0
        assert info["parsed"] == "A"

    def test_think_block_letters_ignored(self):
        output = "<think>A seems right but actually B is better</think>The answer is B"
        score, info = letter_match(output, "B")
        assert score == 1.0

    def test_lowercase_accepted(self):
        score, info = letter_match("the answer is c", "C")
        assert score == 1.0


# === code_exec ===

class TestCodeExec:
    def test_correct_code_with_list_gt(self):
        """MBPP ground_truth is a list — must NOT be stringified."""
        code = "def add(a, b):\n    return a + b"
        tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
        score, info = code_exec(code, tests)
        assert score == 1.0

    def test_incorrect_code(self):
        code = "def add(a, b):\n    return a - b"
        tests = ["assert add(1, 2) == 3"]
        score, _ = code_exec(code, tests)
        assert score == 0.0

    def test_string_gt(self):
        code = "def f():\n    return 1"
        score, _ = code_exec(code, "assert f() == 1")
        assert score == 1.0

    def test_timeout(self):
        code = "import time\ntime.sleep(100)"
        score, _ = code_exec(code, "", timeout=1)
        assert score == 0.0

    def test_prepend(self):
        prompt = "def add(a, b):"
        completion = "```python\n    return a + b\n```"
        tests = "assert add(2, 3) == 5"
        score, _ = code_exec(completion, tests, prepend=prompt)
        assert score == 1.0


# === _extract_code ===

class TestExtractCode:
    def test_single_python_block(self):
        text = "Here's my solution:\n```python\ndef f():\n    return 1\n```\nDone."
        assert "def f():" in _extract_code(text)

    def test_multiple_python_blocks_takes_last(self):
        text = "```python\nwrong\n```\nActually:\n```python\ndef f():\n    return 2\n```"
        result = _extract_code(text)
        assert "return 2" in result

    def test_generic_code_block(self):
        text = "```\ndef f():\n    return 1\n```"
        assert "def f():" in _extract_code(text)

    def test_no_code_block(self):
        text = "def f():\n    return 1"
        assert "def f():" in _extract_code(text)


# === exact_match ===

class TestExactMatch:
    def test_exact(self):
        score, _ = exact_match("hello", "hello")
        assert score == 1.0

    def test_case_insensitive(self):
        score, _ = exact_match("Hello", "hello")
        assert score == 1.0

    def test_whitespace_normalized(self):
        score, _ = exact_match("  hello   world  ", "hello world")
        assert score == 1.0

    def test_mismatch(self):
        score, _ = exact_match("hello", "world")
        assert score == 0.0


# === End-to-end scorer agreement ===

class TestScorerAgreement:
    """Verify that score, parsed value, and status are consistent."""

    def test_number_extract_parsed_matches_scored_value(self):
        score, info = number_extract("The answer is 42.", "42")
        assert score == 1.0
        assert info["parsed"] == "42"
        assert info["status"] == "parsed"

    def test_number_extract_wrong_parsed_matches_scored_value(self):
        score, info = number_extract("The answer is 43.", "42")
        assert score == 0.0
        assert info["parsed"] == "43"
        assert info["status"] == "parsed"

    def test_boxed_latex_parsed_matches_scored_value(self):
        score, info = boxed_latex("Therefore \\boxed{7}", "\\boxed{7}")
        assert score == 1.0
        assert info["parsed"] == "7"
        assert info["status"] == "parsed"

    def test_letter_match_parsed_matches_scored_value(self):
        score, info = letter_match("The answer is C", "C")
        assert score == 1.0
        assert info["parsed"] == "C"

    def test_letter_match_wrong_parsed_matches_scored_value(self):
        score, info = letter_match("The answer is A", "C")
        assert score == 0.0
        assert info["parsed"] == "A"

    def test_missing_status_means_zero_score(self):
        score, info = number_extract("no numbers here", "42")
        assert score == 0.0
        assert info["status"] == "missing"

    def test_ambiguous_means_zero_score(self):
        score, info = letter_match("A or B, hard to say", "A")
        assert score == 0.0
        assert info["status"] == "ambiguous"


class TestNumberExtractEdgeCases:
    def test_percentage(self):
        score, _ = number_extract("The rate is 50%", "50")
        assert score == 1.0

    def test_trailing_period(self):
        score, _ = number_extract("The answer is 42.", "42")
        assert score == 1.0

    def test_distractor_numbers_in_reasoning(self):
        """Last number in text is used when no delimiter."""
        output = "First I computed 10, then 20, final answer is 30"
        score, info = number_extract(output, "30")
        assert score == 1.0

    def test_delimiter_ignores_reasoning_numbers(self):
        output = "I see 10 and 20 in the problem #### 30"
        score, _ = number_extract(output, "30", delimiter="####")
        assert score == 1.0


class TestLetterMatchEdgeCases:
    def test_parenthesized_answer(self):
        score, info = letter_match("The answer is (B)", "B")
        assert score == 1.0
        assert info["parsed"] == "B"

    def test_my_answer_is_pattern(self):
        score, info = letter_match("After analysis, my answer is D", "D")
        assert score == 1.0
        assert info["parsed"] == "D"

    def test_only_letter_d_at_end(self):
        score, info = letter_match("Blah blah\nD", "D")
        assert score == 1.0

    def test_empty_input(self):
        score, info = letter_match("", "A")
        assert score == 0.0
        assert info["status"] == "missing"


# === _normalize_number ===

class TestNormalizeNumber:
    def test_integer(self):
        assert _normalize_number("42") == 42.0

    def test_float(self):
        assert _normalize_number("3.14") == pytest.approx(3.14)

    def test_commas(self):
        assert _normalize_number("1,234,567") == 1234567.0

    def test_dollar(self):
        assert _normalize_number("$100") == 100.0

    def test_empty(self):
        assert _normalize_number("") is None

    def test_nonsense(self):
        assert _normalize_number("abc") is None

    def test_negative(self):
        assert _normalize_number("-5") == -5.0

    def test_fraction(self):
        assert _normalize_number("3/4") == pytest.approx(0.75)

    def test_scientific(self):
        assert _normalize_number("1.5e3") == 1500.0
