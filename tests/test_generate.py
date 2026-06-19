"""Retry/resume: a short run re-runs only the missing rows up to max_attempts."""
from __future__ import annotations

import asyncio
import collections

from synthgen import generate
from synthgen.config import Paths, SynthConfig
from synthgen.io import load_done_ids_ok


class FakeBackend:
    """Echoes content per row; fails the given ids on their first call only."""

    name = "fake"

    def __init__(self, model="fake/model", fail_ids_first=()):
        self.model = model
        self.fail_first = set(fail_ids_first)
        self.calls: collections.Counter = collections.Counter()

    async def chat(self, client, *, messages, sampling):
        rid = messages[0]["content"]            # meta_prompt is set to the row id
        self.calls[rid] += 1
        if rid in self.fail_first and self.calls[rid] == 1:
            return {"content": None, "usage": None, "error": "transient"}
        return {"content": f"ok-{rid}", "usage": {"completion_tokens": 3}, "error": None}


def _rows(ids):
    return [{"id": i, "lang": "fr", "topic": "qa", "meta_prompt": i} for i in ids]


def _cfg(tmp_path):
    cfg = SynthConfig(paths=Paths.from_root(tmp_path))
    cfg.paths.ensure()
    return cfg


def test_single_attempt_leaves_failed_rows_missing(tmp_path):
    be = FakeBackend(fail_ids_first={"b"})
    out = tmp_path / "gen.jsonl"
    missing = asyncio.run(generate._run_backend(
        be, "prompt", _rows(["a", "b", "c"]), out, _cfg(tmp_path),
        max_attempts=1, retry_delay=0))
    assert missing == 1
    assert load_done_ids_ok(out) == {"a", "c"}      # b errored → not "ok"


def test_retry_fills_the_gap(tmp_path):
    be = FakeBackend(fail_ids_first={"b"})
    out = tmp_path / "gen.jsonl"
    missing = asyncio.run(generate._run_backend(
        be, "prompt", _rows(["a", "b", "c"]), out, _cfg(tmp_path),
        max_attempts=3, retry_delay=0))
    assert missing == 0
    assert load_done_ids_ok(out) == {"a", "b", "c"}
    assert be.calls["b"] == 2                        # retried exactly once
    assert be.calls["a"] == 1                        # finished rows not re-run


def test_resume_skips_already_successful_across_calls(tmp_path):
    out = tmp_path / "gen.jsonl"
    cfg = _cfg(tmp_path)
    rows = _rows(["a", "b", "c"])

    first = FakeBackend()
    asyncio.run(generate._run_backend(first, "prompt", rows, out, cfg,
                                      max_attempts=1, retry_delay=0))
    # second invocation: nothing left to do, backend never called
    second = FakeBackend()
    missing = asyncio.run(generate._run_backend(second, "prompt", rows, out, cfg,
                                                max_attempts=1, retry_delay=0))
    assert missing == 0
    assert sum(second.calls.values()) == 0
