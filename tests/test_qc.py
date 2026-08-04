"""QC stats over an SFT JSONL: per-language counts, length quantiles, axis tallies."""
from __future__ import annotations

import json

from synthgen.pipeline import qc
from synthgen.io import write_jsonl


def test_qc_basic_stats(tmp_path):
    rows = [
        {"lang": "fr", "prompt": "abc", "response": "x" * 10,
         "persona": "p1", "constraint_name": "format_json", "topic": "qa",
         "usage": {"completion_tokens": 1100}},
        {"lang": "fr", "prompt": "abcd", "response": "y" * 20,
         "persona": None, "constraint_name": None, "topic": "math",
         "usage": {"completion_tokens": 500}},
        {"lang": "de", "prompt": "z", "response": "w" * 5, "topic": "qa"},
    ]
    src = tmp_path / "sft.jsonl"
    out = tmp_path / "qc.json"
    write_jsonl(src, rows)

    summary = qc.run(input=src, output=out)
    assert summary == json.loads(out.read_text())

    assert summary["totals"] == {"n": 3, "n_langs": 2}
    fr = summary["by_lang"]["fr"]
    assert fr["n"] == 2
    assert fr["response_chars"]["max"] == 20
    assert fr["completion_tokens"]["at_max_1024"] == 1  # only the 1100-token row
    # missing persona/constraint fall back to "<none>"
    assert fr["personas"]["<none>"] == 1
    assert fr["constraints"]["<none>"] == 1
    # de row has no usage → completion_tokens default to 0
    assert summary["by_lang"]["de"]["completion_tokens"]["median"] == 0
