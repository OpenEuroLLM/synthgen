"""topup.run()'s gen_concurrency/judge_concurrency overrides: cfg.concurrency
must be set correctly for each phase and restored after -- these two
generate.run() calls are sequential (never concurrent) within one
single-threaded asyncio process, which is what makes the mutate-in-place
approach safe; this test exists specifically to pin that down.
"""
from __future__ import annotations

import json

from synthgen.config import Paths, SynthConfig
from synthgen.localized import topup
from synthgen.io import write_jsonl


class _FakeBackend:
    def __init__(self, model):
        self.model = model


def _cfg(tmp_path, concurrency=16):
    cfg = SynthConfig(paths=Paths.from_root(tmp_path))
    cfg.concurrency = concurrency
    cfg.paths.ensure()
    return cfg


def _patch_pipeline(monkeypatch, cfg, concurrency_log: list[tuple[str, int]]):
    """Stubs generate.run/quality_filter.run/dedupe.run so one round completes
    immediately with enough survivors to hit target_per_lang=1, while
    recording cfg.concurrency at the moment each phase's generate.run() call
    happens.
    """
    from synthgen.localized import topup as T

    def fake_generate_run(cfg_arg, *, backends, mode, **kwargs):
        concurrency_log.append((mode, cfg_arg.concurrency))
        out = cfg_arg.paths.outputs / f"{mode}_fake.jsonl"
        if mode == "localized":
            write_jsonl(out, [{"id": "es-000000", "lang": "es", "domain": "d",
                               "instruction": "hola", "response": "mundo"}])
        else:  # judge
            write_jsonl(out, [{"id": "es-000000", "lang": "es", "score": 9, "reason": "ok"}])
        return [out]

    def fake_quality_filter_run(*, gen, judged, thresholds, output, report=None):
        write_jsonl(output, [{"id": "es-000000", "lang": "es", "prompt": "hola",
                              "response": "mundo", "quality_score": 9}])
        return {"kept": 1}

    def fake_dedupe_run(*, input, output, **kwargs):
        with open(input) as fin, open(output, "w") as fout:
            fout.write(fin.read())
        return {"kept": 1, "dropped_near_dup_embedding": 0}

    monkeypatch.setattr(T, "generate", type("G", (), {
        "run": staticmethod(fake_generate_run),
        "output_path_for": staticmethod(lambda mode, cfg_arg, model: cfg_arg.paths.outputs / f"{mode}_fake.jsonl"),
    }))
    monkeypatch.setattr(T, "quality_filter", type("Q", (), {
        "run": staticmethod(fake_quality_filter_run),
    }))
    monkeypatch.setattr(T, "dedupe", type("D", (), {
        "run": staticmethod(fake_dedupe_run),
    }))


def test_gen_and_judge_concurrency_overrides_applied_per_phase(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, concurrency=16)
    calls: list[tuple[str, int]] = []
    _patch_pipeline(monkeypatch, cfg, calls)

    topup.run(cfg, gen_backends=[_FakeBackend("gen")], judge_backend=_FakeBackend("judge"),
             target_per_lang=1, thresholds={"global": 0}, lang_codes=["es"],
             gen_concurrency=256, judge_concurrency=8)

    modes = dict(calls)
    assert modes["localized"] == 256
    assert modes["judge"] == 8


def test_concurrency_restored_after_round(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, concurrency=16)
    calls: list[tuple[str, int]] = []
    _patch_pipeline(monkeypatch, cfg, calls)

    topup.run(cfg, gen_backends=[_FakeBackend("gen")], judge_backend=_FakeBackend("judge"),
             target_per_lang=1, thresholds={"global": 0}, lang_codes=["es"],
             gen_concurrency=256, judge_concurrency=8)

    assert cfg.concurrency == 16


def test_concurrency_restored_even_if_gen_phase_raises(tmp_path, monkeypatch):
    from synthgen.localized import topup as T

    cfg = _cfg(tmp_path, concurrency=16)

    def raising_generate_run(cfg_arg, *, backends, mode, **kwargs):
        raise RuntimeError("simulated failure mid gen-phase")

    monkeypatch.setattr(T, "generate", type("G", (), {
        "run": staticmethod(raising_generate_run),
        "output_path_for": staticmethod(lambda mode, cfg_arg, model: cfg_arg.paths.outputs / f"{mode}_fake.jsonl"),
    }))

    try:
        topup.run(cfg, gen_backends=[_FakeBackend("gen")], judge_backend=_FakeBackend("judge"),
                 target_per_lang=1, thresholds={"global": 0}, lang_codes=["es"],
                 gen_concurrency=256, judge_concurrency=8)
    except RuntimeError:
        pass

    assert cfg.concurrency == 16


def test_unset_overrides_leave_concurrency_unchanged(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, concurrency=16)
    calls: list[tuple[str, int]] = []
    _patch_pipeline(monkeypatch, cfg, calls)

    topup.run(cfg, gen_backends=[_FakeBackend("gen")], judge_backend=_FakeBackend("judge"),
             target_per_lang=1, thresholds={"global": 0}, lang_codes=["es"])

    modes = dict(calls)
    assert modes["localized"] == 16
    assert modes["judge"] == 16
