"""Dedupe: (lang, normalized-prompt) keying, empty/short drops, counts."""
from __future__ import annotations

import json

from synthgen.pipeline import dedupe
from synthgen.io import iter_jsonl, write_jsonl


def _run(tmp_path, rows, **kw):
    src = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    write_jsonl(src, rows)
    summary = dedupe.run(input=src, output=out, **kw)
    return summary, list(iter_jsonl(out))


def test_dedupe_drops_exact_duplicates(tmp_path):
    rows = [
        {"lang": "fr", "prompt": "Bonjour", "response": "Salut le monde"},
        {"lang": "fr", "prompt": "Bonjour", "response": "Une autre réponse"},
    ]
    summary, kept = _run(tmp_path, rows)
    assert summary["kept"] == 1
    assert summary["dropped_duplicate"] == 1
    assert len(kept) == 1


def test_dedupe_key_is_whitespace_and_case_insensitive(tmp_path):
    rows = [
        {"lang": "de", "prompt": "Hallo Welt", "response": "antwort eins"},
        {"lang": "de", "prompt": "  hallo   WELT ", "response": "antwort zwei"},
    ]
    summary, _ = _run(tmp_path, rows)
    assert summary["kept"] == 1
    assert summary["dropped_duplicate"] == 1


def test_dedupe_same_prompt_different_lang_both_kept(tmp_path):
    rows = [
        {"lang": "es", "prompt": "Hola", "response": "respuesta"},
        {"lang": "it", "prompt": "Hola", "response": "risposta"},
    ]
    summary, _ = _run(tmp_path, rows)
    assert summary["kept"] == 2
    assert summary["dropped_duplicate"] == 0


def test_dedupe_drops_empty_and_short_responses(tmp_path):
    rows = [
        {"lang": "pl", "prompt": "p1", "response": ""},
        {"lang": "pl", "prompt": "p2", "response": "   "},
        {"lang": "pl", "prompt": "p3", "response": "tiny"},          # < 8 chars
        {"lang": "pl", "prompt": "p4", "response": "long enough response"},
    ]
    summary, kept = _run(tmp_path, rows, min_response_chars=8)
    assert summary["dropped_empty"] == 2
    assert summary["dropped_short"] == 1
    assert summary["kept"] == 1
    assert kept[0]["prompt"] == "p4"


def test_dedupe_writes_report_and_per_lang_counts(tmp_path):
    rows = [
        {"lang": "ro", "prompt": "a", "response": "valid answer"},
        {"lang": "ro", "prompt": "a", "response": "dup answer"},
        {"lang": "el", "prompt": "b", "response": "valid answer"},
    ]
    report = tmp_path / "report.json"
    summary, _ = _run(tmp_path, rows, report=report)
    assert report.exists()
    disk = json.loads(report.read_text())
    assert disk == summary
    assert summary["kept_by_lang"] == {"ro": 1, "el": 1}
    assert summary["dropped_by_lang"] == {"ro": 1}
