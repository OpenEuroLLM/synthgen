"""End-to-end integration checks for topup.py's two safety-critical
guarantees, using a FakeBackend (deterministic, in-process, no GPU/network)
so they run in seconds -- but going through the REAL generate.py/dedupe.py/
near_dup.py/quality_filter.py code, not stubs, so a real regression in any
of them would actually be caught here. Built in lieu of a live smoke test
(no time for one before the 36-language production run).

  1. Stop-resume: killing mid-round and calling topup.run() again continues
     from on-disk checkpoint state -- prior successful rows are preserved
     untouched, and the resumed call never re-requests generation for prompts
     that already succeeded.
  2. Dynamic dedup via judge: a near-duplicate (not exact-string-identical,
     so only the embedding check catches it -- exact-hash dedupe.py alone
     would let it through) gets judged, kept by quality_filter, then DROPPED
     by the real near_dup.py embedding check inside dedupe.py -- and that
     drop shrinks the round's survivor count enough that topup's own deficit
     calculation triggers a real replacement generation next round, ending
     with target_per_lang genuinely-distinct rows, not just
     target_per_lang exact-string-unique ones.

Note: synthgen.pipeline.generate.py's backend.chat() only ever receives the
fully-rendered prompt text (via _messages_for), never the row dict itself --
so a fake backend can't key off a row's `id`. Distinguish gen vs. judge calls
by a substring unique to JUDGE_QUALITY_PROMPT ("holistic quality score",
absent from GENERATION_PROMPT); track continuation-across-resume by
comparing which rendered prompts were actually sent, not by row id.
"""
from __future__ import annotations

import json

from synthgen.config import Paths, SynthConfig
from synthgen.io import iter_jsonl, load_done_ids_ok
from synthgen.localized import topup
from synthgen.pipeline import near_dup


class FakeBackend:
    """Cycles through a fixed instruction list for mode="localized"; always
    returns a fixed passing score for mode="judge" (detected by a substring
    unique to JUDGE_QUALITY_PROMPT)."""

    model = "fake"
    name = "fake"

    def __init__(self, instructions: list[str]):
        self.instructions = instructions
        self._counter = 0
        self.calls: list[str] = []  # rendered prompt text for every call

    async def chat(self, client, *, messages, sampling):
        content = messages[0]["content"]
        self.calls.append(content)
        if "holistic quality score" in content:  # JUDGE_QUALITY_PROMPT only
            return {"content": '{"score": 9, "reason": "fine"}', "usage": {}, "error": None}
        instr = self.instructions[self._counter % len(self.instructions)]
        self._counter += 1
        return {"content": json.dumps({"instruction": instr, "response": f"resp-{instr}",
                                       "role_used": "none", "intent_used": "ask"}),
               "usage": {}, "error": None}


def _cfg(tmp_path):
    cfg = SynthConfig(paths=Paths.from_root(tmp_path))
    cfg.paths.ensure()
    return cfg


class _FakeEmbedder:
    """Two instructions sharing the same first token get identical (fake)
    embeddings -- deterministic, no real model needed, same approach as
    tests/test_near_dup.py."""

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        import numpy as np
        groups: dict[str, "np.ndarray"] = {}
        vecs = []
        for t in texts:
            key = t.split()[0]
            if key not in groups:
                groups[key] = np.random.RandomState(abs(hash(key)) % (2**31)).randn(8)
                groups[key] /= np.linalg.norm(groups[key])
            vecs.append(groups[key])
        return np.array(vecs)


def test_dynamic_dedup_via_judge_triggers_real_replacement_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(near_dup, "_embedder", None)
    monkeypatch.setattr(near_dup, "_load_embedder", lambda model_name=None: _FakeEmbedder())

    # "dup" prefix shared by the first two -> near-duplicate pair under the
    # fake embedder's grouping; every other instruction has a unique,
    # non-colliding first token so it's never mistaken for a duplicate.
    instructions = ["dup variant-one", "dup variant-two", "alpha-item",
                    "beta-item", "gamma-item", "delta-item", "epsilon-item"]
    gen = FakeBackend(instructions)
    judge = FakeBackend(instructions)

    cfg = _cfg(tmp_path)
    summary = topup.run(
        cfg, gen_backends=[gen], judge_backend=judge,
        target_per_lang=5, thresholds={"global": 0}, lang_codes=["es"],
        seed=0, max_rounds=5, embedding_dedupe=True, overgen_factor_default=1.0,
    )

    assert summary["met_target"], f"never reached target: {summary}"
    assert summary["survivors_by_lang"]["es"] >= 5
    # the near-dup pair should have cost at least one real extra generation
    # beyond the bare minimum (5 asked for at overgen_factor_default=1.0)
    assert summary["total_raw_generated"] > 5, (
        "expected the near-dup drop to trigger a real replacement generation, "
        f"but total_raw_generated={summary['total_raw_generated']}")
    assert summary["embedding_near_dup_dropped_total"] >= 1

    # confirm the FINAL kept output has no near-duplicate pair left in it
    final_rows = list(iter_jsonl(cfg.paths.outputs / "loc_full.dedup.jsonl"))
    final_instructions = [r["prompt"] for r in final_rows]
    first_tokens = [t.split()[0] for t in final_instructions]
    assert len(first_tokens) == len(set(first_tokens)), (
        f"a near-duplicate pair survived into the final output: {final_instructions}")


def test_stop_resume_does_not_redo_completed_work(tmp_path):
    """Round 1 (capped at max_rounds=1) can't fully reach target on its own
    given overgen_factor_default=1.0 and a low target vs. what's requested --
    inspect state, then "resume" with a fresh backend instance (as a real
    restarted process would have) and confirm: prior successful rows are
    preserved untouched, and the resumed call's backend is never asked to
    regenerate a prompt that already succeeded."""
    instructions = [f"instruction-{i}" for i in range(20)]
    gen = FakeBackend(instructions)
    judge = FakeBackend(instructions)
    cfg = _cfg(tmp_path)

    topup.run(cfg, gen_backends=[gen], judge_backend=judge,
             target_per_lang=5, thresholds={"global": 0}, lang_codes=["es"],
             seed=0, max_rounds=1, max_attempts=1, overgen_factor_default=1.0)

    gen_out = cfg.paths.outputs / "loc_fake.jsonl"
    done_after_round1 = load_done_ids_ok(gen_out)
    assert len(done_after_round1) > 0, "round 1 should have produced some real output"
    rows_after_round1 = {r["id"]: r for r in iter_jsonl(gen_out) if r["id"] in done_after_round1}
    prompts_sent_in_round1 = set(gen.calls)

    gen2 = FakeBackend(instructions)
    judge2 = FakeBackend(instructions)
    summary = topup.run(cfg, gen_backends=[gen2], judge_backend=judge2,
                        target_per_lang=5, thresholds={"global": 0}, lang_codes=["es"],
                        seed=0, max_rounds=5, max_attempts=3, overgen_factor_default=1.0)

    assert summary["met_target"]
    done_final = load_done_ids_ok(gen_out)
    assert done_after_round1 <= done_final, "an id done before the kill went missing after resume"
    # every row present after round 1 is byte-identical after resume -- never
    # regenerated/overwritten
    rows_final = {r["id"]: r for r in iter_jsonl(gen_out)}
    for rid, row in rows_after_round1.items():
        assert rows_final[rid] == row, f"row {rid} was overwritten by the resumed call"
    # the resumed backend never re-received a prompt that already succeeded
    assert not (set(gen2.calls) & prompts_sent_in_round1), (
        "resume re-requested a prompt that had already succeeded before the kill")
