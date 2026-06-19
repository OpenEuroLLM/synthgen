"""I/O helpers: JSONL round-trip, resume IDs, model slug."""
from __future__ import annotations

from synthgen.io import iter_jsonl, load_done_ids, model_slug, write_jsonl


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


def test_model_slug():
    assert model_slug("google/gemma-4-26b-a4b-it") == "google__gemma-4-26b-a4b-it"
    assert model_slug("openai/gpt-oss:120b") == "openai__gpt-oss_120b"
