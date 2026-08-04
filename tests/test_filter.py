"""Leakage detection: the constraint-aware signals in filter.detect_leakage."""
from __future__ import annotations

from synthgen.pipeline.filter import MAX_WORDS, detect_leakage


def test_empty_prompt_is_leak():
    assert detect_leakage({"generated_prompt": "   "}) == (True, "empty")


def test_error_row_is_leak():
    leaked, reason = detect_leakage({"generated_prompt": "ok prompt", "error": "boom"})
    assert leaked and reason == "error"


def test_clean_prompt_passes():
    row = {"generated_prompt": "Write me a short poem about the sea.",
           "constraint_name": "", "constraint_text": ""}
    assert detect_leakage(row) == (False, None)


def test_too_long_is_leak():
    row = {"generated_prompt": "word " * (MAX_WORDS + 1)}
    leaked, reason = detect_leakage(row)
    assert leaked and reason.startswith("too_long")


def test_start_with_phrase_repeated_is_leak():
    # the phrase appears in the instruction AND again in a leaked answer → count >= 2
    row = {"generated_prompt": 'Begin with "Here we go:". Here we go: the answer follows.',
           "constraint_name": "start_with",
           "constraint_text": 'Begin your answer with the exact phrase: "Here we go:".'}
    leaked, reason = detect_leakage(row)
    assert leaked and reason == "answer_includes_start_with_phrase"


def test_start_with_phrase_once_passes():
    row = {"generated_prompt": 'Begin your answer with "Here we go:".',
           "constraint_name": "start_with",
           "constraint_text": 'Begin your answer with the exact phrase: "Here we go:".'}
    assert detect_leakage(row) == (False, None)


def test_format_json_answer_is_leak():
    row = {"generated_prompt": 'Give JSON. {"title": "x", "answer": "y"}',
           "constraint_name": "format_json", "constraint_text": ""}
    leaked, reason = detect_leakage(row)
    assert leaked and reason == "contains_json_answer"


def test_format_bullets_three_bullets_is_leak():
    text = "List things:\n- one\n- two\n- three"
    leaked, reason = detect_leakage(
        {"generated_prompt": text, "constraint_name": "format_bullets"})
    assert leaked and "bullets" in reason


def test_format_numbered_three_items_is_leak():
    text = "Answer:\n1. one\n2) two\n3. three"
    leaked, reason = detect_leakage(
        {"generated_prompt": text, "constraint_name": "format_numbered"})
    assert leaked and "numbered" in reason


def test_all_caps_prompt_is_leak():
    row = {"generated_prompt": "WRITE A POEM ABOUT THE OCEAN PLEASE",
           "constraint_name": "casing_all_caps"}
    leaked, reason = detect_leakage(row)
    assert leaked and reason == "prompt_is_all_caps"
