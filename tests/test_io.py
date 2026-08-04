"""I/O helpers: JSONL round-trip, resume IDs, model slug, JSON extraction."""
from __future__ import annotations

from synthgen.io import (
    extract_json,
    iter_jsonl,
    load_done_ids,
    load_done_ids_ok,
    model_slug,
    next_id_index,
    write_jsonl,
)


def test_write_then_iter_roundtrips(tmp_path):
    rows = [{"id": "a", "x": 1}, {"id": "b", "x": 2, "u": "café"}]
    p = tmp_path / "out.jsonl"
    n = write_jsonl(p, rows)
    assert n == 2
    assert list(iter_jsonl(p)) == rows


def test_iter_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "out.jsonl"
    p.write_text('{"a": 1}\n\n   \n{"a": 2}\n', encoding="utf-8")
    assert [r["a"] for r in iter_jsonl(p)] == [1, 2]


def test_write_jsonl_append(tmp_path):
    p = tmp_path / "out.jsonl"
    write_jsonl(p, [{"a": 1}])
    write_jsonl(p, [{"a": 2}], append=True)
    assert [r["a"] for r in iter_jsonl(p)] == [1, 2]


def test_load_done_ids(tmp_path):
    p = tmp_path / "out.jsonl"
    # one good row, one row missing the key, one malformed — only the good id counts
    p.write_text('{"id": "x"}\n{"nope": 1}\nnot-json\n', encoding="utf-8")
    assert load_done_ids(p) == {"x"}


def test_load_done_ids_missing_file(tmp_path):
    assert load_done_ids(tmp_path / "nope.jsonl") == set()


def test_load_done_ids_ok_excludes_error_records(tmp_path):
    p = tmp_path / "gen.jsonl"
    # "a" succeeded, "b" only errored, "c" errored then succeeded
    p.write_text(
        '{"id": "a", "generated_prompt": "x"}\n'
        '{"id": "b", "error": "boom"}\n'
        '{"id": "c", "error": "boom"}\n'
        '{"id": "c", "generated_prompt": "y"}\n',
        encoding="utf-8",
    )
    assert load_done_ids(p) == {"a", "b", "c"}       # any record present
    assert load_done_ids_ok(p) == {"a", "c"}         # only non-error records


def test_load_done_ids_ok_missing_file(tmp_path):
    assert load_done_ids_ok(tmp_path / "nope.jsonl") == set()


def test_model_slug():
    assert model_slug("google/gemma-4-26b-a4b-it") == "google__gemma-4-26b-a4b-it"
    assert model_slug("openai/gpt-oss:120b") == "openai__gpt-oss_120b"


def test_extract_json_plain():
    assert extract_json('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_extract_json_strips_think_block():
    text = '<think>reasoning about the answer</think>\n{"score": 7, "reason": "ok"}'
    assert extract_json(text) == {"score": 7, "reason": "ok"}


def test_extract_json_strips_code_fences():
    text = '```json\n{"instruction": "hi", "response": "hello"}\n```'
    assert extract_json(text) == {"instruction": "hi", "response": "hello"}


def test_extract_json_picks_last_balanced_object():
    # a stray brace earlier in the text must not break parsing of the real object
    text = 'note: use {curly braces} sparingly\n{"score": 3}'
    assert extract_json(text) == {"score": 3}


def test_extract_json_none_or_empty():
    assert extract_json(None) is None
    assert extract_json("") is None
    assert extract_json("no json here") is None


def test_extract_json_malformed_returns_none():
    assert extract_json('{"score": 3,}') is None


def test_next_id_index_missing_file(tmp_path):
    assert next_id_index(tmp_path / "nope.jsonl", "es") == 0


def test_next_id_index_continues_from_max(tmp_path):
    p = tmp_path / "loc.jsonl"
    write_jsonl(p, [
        {"id": "es-000000", "lang": "es"},
        {"id": "es-000004", "lang": "es"},
        {"id": "fr-000009", "lang": "fr"},   # different language, ignored
    ])
    assert next_id_index(p, "es") == 5
    assert next_id_index(p, "fr") == 10
    assert next_id_index(p, "de") == 0
