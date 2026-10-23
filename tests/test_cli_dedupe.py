"""`synthgen dedupe` CLI wiring for the GPU embedding flags."""
from __future__ import annotations

import json

import pytest

from synthgen import cli
from synthgen.pipeline import dedupe
from synthgen.io import write_jsonl


def _capture_run(monkeypatch):
    calls = {}

    def fake_run(**kw):
        calls.update(kw)
        return {"kept": 0}
    monkeypatch.setattr(dedupe, "run", fake_run)
    return calls


def test_dedupe_cli_passes_device_and_overrides(tmp_path, monkeypatch):
    calls = _capture_run(monkeypatch)
    monkeypatch.setattr("multiprocessing.set_start_method", lambda *a, **k: None)
    src = tmp_path / "in.jsonl"
    write_jsonl(src, [])
    ov = tmp_path / "ov.json"
    ov.write_text(json.dumps({"ga": {"threshold": 0.96}}))
    cli.main(["dedupe", "--input", str(src), "--output", str(tmp_path / "out.jsonl"),
              "--embedding-dedupe", "--embedding-device", "cuda",
              "--embedding-workers", "4", "--lang-overrides", str(ov)])
    assert calls["embedding_device"] == "cuda"
    assert calls["embedding_workers"] == 4
    assert calls["embedding_lang_overrides"] == {"ga": {"threshold": 0.96}}


def test_dedupe_cli_defaults_to_cpu_without_overrides(tmp_path, monkeypatch):
    calls = _capture_run(monkeypatch)
    src = tmp_path / "in.jsonl"
    write_jsonl(src, [])
    cli.main(["dedupe", "--input", str(src), "--output", str(tmp_path / "out.jsonl")])
    assert calls["embedding_device"] is None
    assert calls["embedding_lang_overrides"] is None


def test_dedupe_cli_overrides_require_cuda(tmp_path, monkeypatch):
    _capture_run(monkeypatch)
    ov = tmp_path / "ov.json"
    ov.write_text(json.dumps({"ga": {"threshold": 0.96}}))
    with pytest.raises(SystemExit, match="requires --embedding-device cuda"):
        cli.main(["dedupe", "--input", "x", "--output", "y", "--lang-overrides", str(ov)])
