"""Export to open-instruct per-language parquet sources."""
from __future__ import annotations

import pytest

pq = pytest.importorskip("pyarrow.parquet")

from synthgen.pipeline import to_open_instruct
from synthgen.io import write_jsonl


def _rows():
    return [
        {"lang": "fr", "prompt": "Bonjour", "response": "Salut"},
        {"lang": "fr", "prompt": "Ça va?", "response": "Oui"},
        {"lang": "de", "prompt": "Hallo", "response": "Hi"},
        {"lang": "", "prompt": "x", "response": "y"},          # skipped: no lang
        {"lang": "it", "prompt": "  ", "response": "ok"},      # skipped: empty prompt
    ]


def test_export_writes_per_language_parquet(tmp_path):
    src = tmp_path / "sft.jsonl"
    write_jsonl(src, _rows())
    out_dir = tmp_path / "by_language"

    summary = to_open_instruct.run(input=src, out_dir=out_dir, source="synthgen-if")

    assert summary["skipped"] == 2
    assert summary["total"] == 3
    assert summary["by_lang"] == {"de": 1, "fr": 2}

    fr = out_dir / "synthgen-if" / "fr.parquet"
    assert fr.exists()
    table = pq.read_table(fr)
    assert table.schema.names == ["messages", "language"]
    assert table.num_rows == 2

    first = table.to_pylist()[0]
    assert first["language"] == "fr"
    assert first["messages"] == [
        {"content": "Bonjour", "role": "user"},
        {"content": "Salut", "role": "assistant"},
    ]


def test_export_respects_small_batch_size(tmp_path):
    # batch_size smaller than the per-lang count forces multiple flushes/row-groups.
    src = tmp_path / "sft.jsonl"
    write_jsonl(src, [{"lang": "fr", "prompt": f"p{i}", "response": f"r{i}"}
                      for i in range(5)])
    out_dir = tmp_path / "by_language"
    summary = to_open_instruct.run(input=src, out_dir=out_dir, batch_size=2)
    assert summary["by_lang"] == {"fr": 5}
    table = pq.read_table(out_dir / "synthgen-if" / "fr.parquet")
    assert table.num_rows == 5
