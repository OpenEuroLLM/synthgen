"""generate.py's localized/judge modes: JSON parsing, error->retry, resume."""
from __future__ import annotations

import asyncio
import collections

import pytest

from synthgen.pipeline import generate
from synthgen.config import Paths, SynthConfig
from synthgen.io import load_done_ids_ok


def _cfg(tmp_path):
    cfg = SynthConfig(paths=Paths.from_root(tmp_path))
    cfg.paths.ensure()
    return cfg


# ----- _record_for -----------------------------------------------------------

def test_record_for_localized_parses_json():
    row = {"id": "es-000000", "lang": "es", "domain": "food & drink",
          "intent": "ask (factual question)", "role": None, "salt": 1234}
    content = '{"instruction":"hola","response":"mundo","role_used":"none","intent_used":"ask"}'
    rec = generate._record_for("localized", row, "gen/model", content, {}, None)
    assert rec["instruction"] == "hola"
    assert rec["response"] == "mundo"
    assert rec["domain"] == "food & drink"
    assert "error" not in rec


def test_record_for_localized_unparseable_is_error():
    row = {"id": "es-000000", "lang": "es"}
    rec = generate._record_for("localized", row, "gen/model", "not json at all", {}, None)
    assert rec["error"] == "unparseable_json"


def test_record_for_localized_missing_fields_is_error():
    row = {"id": "es-000000", "lang": "es"}
    rec = generate._record_for("localized", row, "gen/model",
                               '{"instruction": "only half"}', {}, None)
    assert rec["error"] == "unparseable_json"


def test_record_for_localized_backend_error_passed_through():
    row = {"id": "es-000000", "lang": "es"}
    rec = generate._record_for("localized", row, "gen/model", None, None, "timeout")
    assert rec == {"id": "es-000000", "lang": "es", "model": "gen/model", "error": "timeout"}


def test_record_for_judge_parses_score():
    row = {"id": "es-000000", "lang": "es"}
    rec = generate._record_for("judge", row, "judge/model",
                               '{"score": 8, "reason": "good"}', {}, None)
    assert rec["score"] == 8
    assert rec["reason"] == "good"


def test_record_for_judge_coerces_string_score():
    row = {"id": "es-000000", "lang": "es"}
    rec = generate._record_for("judge", row, "judge/model",
                               '{"score": "7", "reason": "ok"}', {}, None)
    assert rec["score"] == 7


def test_record_for_judge_unparseable_is_error():
    row = {"id": "es-000000", "lang": "es"}
    rec = generate._record_for("judge", row, "judge/model", "garbage", {}, None)
    assert rec["error"] == "unparseable_score"


def test_messages_for_judge_builds_prompt_with_lang_and_country():
    row = {"id": "es-000000", "lang": "es", "instruction": "hola", "response": "mundo"}
    msgs = generate._messages_for("judge", row)
    content = msgs[0]["content"]
    assert "Spanish" in content and "Spain" in content
    assert "hola" in content and "mundo" in content


# ----- path helpers ------------------------------------------------------

def test_judge_mode_requires_explicit_in_path(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="requires an explicit in_path"):
        generate._input_path_for("judge", cfg, "judge/model", None)


def test_output_path_for_localized_and_judge(tmp_path):
    cfg = _cfg(tmp_path)
    assert generate.output_path_for("localized", cfg, "gen/model").name == "loc_gen__model.jsonl"
    assert generate.output_path_for("judge", cfg, "judge/model").name == "judged_judge__model.jsonl"


# ----- resume/retry via _run_backend --------------------------------------

class FakeJSONBackend:
    """Echoes a JSON payload built from the row id; garbles it on the first
    call for ids in `garble_first`, so parse failures are retried like any
    other error."""

    name = "fake"

    def __init__(self, model="fake/model", garble_first=(), payload_fn=None):
        self.model = model
        self.garble_first = set(garble_first)
        self.calls: collections.Counter = collections.Counter()
        self.payload_fn = payload_fn or (
            lambda rid: f'{{"instruction":"i-{rid}","response":"r-{rid}"}}')

    async def chat(self, client, *, messages, sampling):
        rid = messages[0]["content"]
        self.calls[rid] += 1
        if rid in self.garble_first and self.calls[rid] == 1:
            return {"content": "not json", "usage": None, "error": None}
        return {"content": self.payload_fn(rid), "usage": {"completion_tokens": 3},
                "error": None}


def _loc_rows(ids):
    return [{"id": i, "lang": "es", "domain": "d", "intent": "i",
             "role": None, "salt": 1, "meta_prompt": i} for i in ids]


def test_localized_parse_failure_is_retried(tmp_path):
    be = FakeJSONBackend(garble_first={"es-000001"})
    out = tmp_path / "loc.jsonl"
    cfg = _cfg(tmp_path)
    missing = asyncio.run(generate._run_backend(
        be, "localized", _loc_rows(["es-000000", "es-000001", "es-000002"]), out, cfg,
        max_attempts=2, retry_delay=0))
    assert missing == 0
    assert load_done_ids_ok(out) == {"es-000000", "es-000001", "es-000002"}
    assert be.calls["es-000001"] == 2   # garbled once, then succeeded


def test_localized_single_attempt_leaves_parse_failure_missing(tmp_path):
    be = FakeJSONBackend(garble_first={"es-000001"})
    out = tmp_path / "loc.jsonl"
    cfg = _cfg(tmp_path)
    missing = asyncio.run(generate._run_backend(
        be, "localized", _loc_rows(["es-000000", "es-000001"]), out, cfg,
        max_attempts=1, retry_delay=0))
    assert missing == 1
    assert load_done_ids_ok(out) == {"es-000000"}


def test_judge_mode_resumes_across_calls(tmp_path):
    cfg = _cfg(tmp_path)
    out = tmp_path / "judged.jsonl"
    rows = [{"id": "es-000000", "lang": "es", "instruction": "hola", "response": "mundo"}]

    be = FakeJSONBackend(payload_fn=lambda rid: '{"score": 9, "reason": "great"}')
    asyncio.run(generate._run_backend(be, "judge", rows, out, cfg,
                                      max_attempts=1, retry_delay=0))
    assert load_done_ids_ok(out) == {"es-000000"}

    be2 = FakeJSONBackend(payload_fn=lambda rid: '{"score": 9, "reason": "great"}')
    missing = asyncio.run(generate._run_backend(be2, "judge", rows, out, cfg,
                                                max_attempts=1, retry_delay=0))
    assert missing == 0
    assert sum(be2.calls.values()) == 0   # nothing left to do, backend never called
