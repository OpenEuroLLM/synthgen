"""Backend-agnostic generator with checkpoint/resume.

Four modes:
    prompt      meta_prompt -> generated_prompt
                Input  : prompts/prompts.jsonl  (from `synthgen build-prompts`)
                Output : outputs/gen_<model_slug>.jsonl
                Feeds  : filter / verify / back-translate

    response    generated_prompt -> assistant response
                Input  : outputs/gen_<model_slug>.filtered.jsonl
                Output : outputs/sft_<model_slug>.jsonl
                One SFT pair per row.

    localized   meta_prompt -> {instruction, response} in one call (JSON out)
                Input  : prompts/localized_prompts.jsonl (from
                         `synthgen.prompts_localized`)
                Output : outputs/loc_<model_slug>.jsonl
                A parse failure is recorded as an error (not a silent drop) so
                it is retried like any other failure.
                Feeds  : mode="judge", synthgen.topup

    judge       {instruction, response} -> {score, reason}
                Input  : outputs/loc_<model_slug>.jsonl (rows from mode="localized")
                Output : outputs/judged_<model_slug>.jsonl
                Feeds  : synthgen.quality_filter

Two backends (selected at the CLI):
    openrouter  hosted API, requires OPENROUTER_API_KEY
    vllm        local OpenAI-compatible servers, addresses discovered from a
                directory of `*.endpoint` files (round-robin pool)

Sharding (`split=True`) round-robins the input rows across multiple backends
for diversity when running multiple models — useful when calling several
OpenRouter models on a single shared prompt set.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Iterable

import httpx

from synthgen.backends import Backend
from synthgen.config import LANG_COUNTRY, LANGUAGES_PHASE3, SynthConfig
from synthgen.io import extract_json, iter_jsonl, load_done_ids_ok, model_slug
from synthgen.log import get_logger

log = get_logger("synthgen.generate")


# ----- mode helpers --------------------------------------------------------

def _input_path_for(mode: str, cfg: SynthConfig, model: str,
                    explicit: Path | None) -> Path:
    if explicit:
        return explicit
    if mode == "prompt":
        return cfg.paths.prompts / "prompts.jsonl"
    if mode == "localized":
        return cfg.paths.prompts / "localized_prompts.jsonl"
    if mode == "judge":
        # Judge input is keyed by the *generator's* model slug, not the judge
        # backend's — there's no way to derive that from the judge backend
        # alone, so this mode always requires an explicit `in_path`.
        raise ValueError("mode='judge' requires an explicit in_path "
                         "(outputs/loc_<generator_model_slug>.jsonl)")
    s = model_slug(model)
    p = cfg.paths.outputs / f"gen_{s}.filtered.jsonl"
    return p if p.exists() else cfg.paths.outputs / f"gen_{s}.jsonl"


def _output_path_for(mode: str, cfg: SynthConfig, model: str,
                     explicit: Path | None) -> Path:
    if explicit:
        return explicit
    s = model_slug(model)
    if mode == "prompt":
        return cfg.paths.outputs / f"gen_{s}.jsonl"
    if mode == "localized":
        return cfg.paths.outputs / f"loc_{s}.jsonl"
    if mode == "judge":
        return cfg.paths.outputs / f"judged_{s}.jsonl"
    return cfg.paths.outputs / f"sft_{s}.jsonl"


def _load_rows(mode: str, path: Path) -> list[dict]:
    if mode in ("prompt", "localized"):
        return list(iter_jsonl(path))
    if mode == "judge":
        return [r for r in iter_jsonl(path)
                if not r.get("error") and r.get("instruction") and r.get("response")]
    rows: list[dict] = []
    for r in iter_jsonl(path):
        if r.get("error") or not r.get("generated_prompt"):
            continue
        rows.append(r)
    return rows


def _messages_for(mode: str, row: dict) -> list[dict]:
    if mode == "judge":
        from synthgen.localized.prompts import judge_quality_prompt
        content = judge_quality_prompt(
            lang_name=LANGUAGES_PHASE3[row["lang"]], country=LANG_COUNTRY[row["lang"]],
            instruction=row["instruction"], response=row["response"])
        return [{"role": "user", "content": content}]
    key = "meta_prompt" if mode in ("prompt", "localized") else "generated_prompt"
    return [{"role": "user", "content": row[key]}]


def _record_for(mode: str, row: dict, model: str,
                content: str | None, usage: dict | None, error: str | None) -> dict:
    if error:
        return {"id": row["id"], "lang": row["lang"], "model": model, "error": error}
    if mode == "prompt":
        return {
            "id": row["id"], "lang": row["lang"], "topic": row["topic"],
            "constraint_name": row.get("constraint_name"),
            "constraint_text": row.get("constraint_text"),
            "persona": row.get("persona"),
            "model": model,
            "generated_prompt": content,
            "usage": usage or {},
        }
    if mode == "localized":
        ex = extract_json(content) or {}
        instr, resp = ex.get("instruction"), ex.get("response")
        if not instr or not resp:
            return {"id": row["id"], "lang": row["lang"], "model": model,
                    "error": "unparseable_json" if content else "empty_content"}
        return {
            "id": row["id"], "lang": row["lang"],
            "domain": row.get("domain"), "is_local": row.get("is_local"),
            "intent": row.get("intent"), "role": row.get("role"),
            "salt": row.get("salt"),
            "intent_used": ex.get("intent_used"), "role_used": ex.get("role_used"),
            "model": model,
            "instruction": instr, "response": resp,
            "usage": usage or {},
        }
    if mode == "judge":
        j = extract_json(content) or {}
        sc = j.get("score")
        if isinstance(sc, str) and sc.strip().lstrip("-").isdigit():
            sc = int(sc)
        if not isinstance(sc, (int, float)):
            return {"id": row["id"], "lang": row["lang"], "model": model,
                    "error": "unparseable_score" if content else "empty_content"}
        return {
            "id": row["id"], "lang": row["lang"], "model": model,
            "score": sc, "reason": j.get("reason"),
            "usage": usage or {},
        }
    return {
        "id": row["id"], "lang": row["lang"], "topic": row.get("topic"),
        "constraint_name": row.get("constraint_name"),
        "constraint_text": row.get("constraint_text"),
        "persona": row.get("persona"),
        "model": model,
        "prompt": row["generated_prompt"],
        "response": content,
        "usage": usage or {},
    }


# ----- core runner ---------------------------------------------------------

async def _generate_once(backend: Backend, mode: str, todo: list[dict],
                         out_path: Path, cfg: SynthConfig, sampling: dict) -> None:
    """Run one pass over `todo`, appending one record per row to `out_path`."""
    sem = asyncio.Semaphore(cfg.concurrency)
    progress = {"done": 0, "errs": 0}
    total = len(todo)
    t0 = time.time()

    # httpx.AsyncClient()'s default is Limits(max_connections=100,
    # max_keepalive_connections=20) -- silently caps real concurrent
    # in-flight requests at 100 NO MATTER what cfg.concurrency/this
    # semaphore allows. Found via a 4-node concurrency-scaling smoke test
    # (studies/localized_axes_ablation/13_concurrency_scaling/): throughput
    # only reached ~1.15x a single replica's peak with 4 replicas and
    # concurrency raised proportionally -- this default connection cap was
    # silently overriding cfg.concurrency the entire time, for every run
    # this pipeline has ever done, not just this test. Sized to
    # cfg.concurrency (with a little headroom) so the semaphore's limit is
    # the real one again.
    limits = httpx.Limits(max_connections=cfg.concurrency + 10,
                          max_keepalive_connections=cfg.concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        with open(out_path, "a", encoding="utf-8") as fout:
            async def worker(row: dict):
                async with sem:
                    res = await backend.chat(
                        client,
                        messages=_messages_for(mode, row),
                        sampling=sampling,
                    )
                    rec = _record_for(mode, row, backend.model,
                                      res.get("content"), res.get("usage"),
                                      res.get("error"))
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    progress["done"] += 1
                    if "error" in rec:
                        progress["errs"] += 1
                    if progress["done"] % 1000 == 0 or progress["done"] == total:
                        rate = progress["done"] / max(1e-6, time.time() - t0)
                        eta = (total - progress["done"]) / max(1e-6, rate)
                        log.info("  [%s] %d/%d errs=%d %.1f req/s eta=%.1fmin",
                                 backend.model, progress["done"], total,
                                 progress["errs"], rate, eta / 60)
            await asyncio.gather(*(worker(r) for r in todo))
            log.info("[%s/%s] pass done in %.1fs, errors=%d/%d",
                     backend.name, backend.model,
                     time.time() - t0, progress["errs"], total)


async def _run_backend(backend: Backend, mode: str, rows: list[dict],
                       out_path: Path, cfg: SynthConfig,
                       sampling: dict | None = None, *,
                       max_attempts: int = 1, retry_delay: float = 10.0) -> int:
    """Generate `rows` for `backend`, retrying short runs up to `max_attempts`.

    "Short" = some input rows still lack a successful (non-error) record. Each
    attempt re-runs only the rows still missing, so errored/incomplete rows are
    retried while finished ones are skipped. Returns the count still missing.
    """
    sampling = sampling or cfg.sampling
    target_ids = {r["id"] for r in rows}
    n_target = len(target_ids)

    for attempt in range(1, max_attempts + 1):
        done = load_done_ids_ok(out_path)
        todo = [r for r in rows if r["id"] not in done]
        n_done = n_target - len(todo)
        if not todo:
            log.info("[%s/%s] complete: %d/%d generated",
                     backend.name, backend.model, n_done, n_target)
            return 0
        if attempt == 1:
            log.info("[%s/%s] %d rows to run (skipping %d done) -> %s",
                     backend.name, backend.model, len(todo), n_done, out_path.name)
        else:
            log.warning("[%s/%s] attempt %d/%d: %d/%d done, retrying %d missing "
                        "after %.0fs", backend.name, backend.model, attempt,
                        max_attempts, n_done, n_target, len(todo), retry_delay)
            if retry_delay > 0:
                await asyncio.sleep(retry_delay)
        await _generate_once(backend, mode, todo, out_path, cfg, sampling)

    missing = len(target_ids - load_done_ids_ok(out_path))
    if missing:
        log.error("[%s/%s] INCOMPLETE after %d attempt(s): %d/%d generated, "
                  "%d still missing", backend.name, backend.model, max_attempts,
                  n_target - missing, n_target, missing)
    else:
        log.info("[%s/%s] complete: %d/%d generated",
                 backend.name, backend.model, n_target, n_target)
    return missing


def _max_tokens_default(mode: str) -> int:
    if mode == "prompt":
        return 1024
    if mode == "judge":
        return 512
    return 1536  # response, localized


def _sampling_for(mode: str, cfg: SynthConfig, backend: Backend,
                  max_tokens: int | None) -> dict:
    sampling = {**cfg.sampling}
    sampling["max_tokens"] = max_tokens or _max_tokens_default(mode)
    if mode == "judge":
        # Deterministic scoring, and — critical for reasoning judge models like
        # Qwen3.6 — disable thinking so the JSON comes out directly instead of
        # burning max_tokens on a <think> block and truncating before it.
        sampling["temperature"] = 0.0
        sampling["top_p"] = 1.0
        if backend.name == "vllm":
            sampling["chat_template_kwargs"] = {"enable_thinking": False}
    return sampling


# ----- public API ----------------------------------------------------------

def output_path_for(mode: str, cfg: SynthConfig, model: str) -> Path:
    """Public accessor for a mode's default output path (no explicit override) —
    lets callers (e.g. synthgen.topup) locate a stage's checkpoint file without
    re-running it."""
    return _output_path_for(mode, cfg, model, None)



def run(
    cfg: SynthConfig,
    *,
    backends: Iterable[Backend],
    mode: str = "prompt",
    in_path: Path | None = None,
    split: bool | None = None,
    max_tokens: int | None = None,
    max_attempts: int = 1,
    retry_delay: float = 10.0,
) -> list[Path]:
    """Run generation across `backends`.

    - mode="prompt"     : meta_prompt → generated_prompt
    - mode="response"   : generated_prompt → response (SFT pair)
    - mode="localized"  : meta_prompt → {instruction, response} in one call
    - mode="judge"      : {instruction, response} → {score, reason}
    - split=True       : partition rows across backends (round-robin) for diversity
                         default True iff multiple backends are passed
    - max_attempts>1   : if a run finishes short of its target (some rows have no
                         successful record), re-run the missing rows up to this
                         many times, waiting `retry_delay` seconds between attempts
    """
    if mode not in ("prompt", "response", "localized", "judge"):
        raise ValueError(f"mode must be one of prompt/response/localized/judge, "
                         f"got {mode!r}")
    cfg.paths.ensure()
    backends = list(backends)
    if not backends:
        raise ValueError("at least one backend required")

    do_split = split if split is not None else (len(backends) > 1)
    outputs: list[Path] = []

    for be in backends:
        ip = _input_path_for(mode, cfg, be.model, in_path)
        if not ip.exists():
            raise SystemExit(f"input file not found: {ip}")

    if do_split and len(backends) > 1:
        # Load rows from each backend's input path; if all backends share an input
        # (the common case) we shard once and dispatch.
        # We require all backends to read the same input for split mode.
        shared_in = _input_path_for(mode, cfg, backends[0].model, in_path)
        for be in backends[1:]:
            if _input_path_for(mode, cfg, be.model, in_path) != shared_in:
                raise SystemExit("split mode requires all backends to share an input file")
        rows = _load_rows(mode, shared_in)
        log.info("loaded %d rows from %s", len(rows), shared_in)
        shards: dict[str, list[dict]] = {be.model: [] for be in backends}
        for i, r in enumerate(rows):
            shards[backends[i % len(backends)].model].append(r)
        for be in backends:
            log.info("  [split] %s: %d rows", be.model, len(shards[be.model]))
        for be in backends:
            out = _output_path_for(mode, cfg, be.model, None)
            sampling = _sampling_for(mode, cfg, be, max_tokens)
            asyncio.run(_run_backend(be, mode, shards[be.model], out, cfg, sampling,
                                     max_attempts=max_attempts, retry_delay=retry_delay))
            outputs.append(out)
    else:
        for be in backends:
            ip = _input_path_for(mode, cfg, be.model, in_path)
            out = _output_path_for(mode, cfg, be.model, None)
            rows = _load_rows(mode, ip)
            log.info("loaded %d rows from %s", len(rows), ip)
            sampling = _sampling_for(mode, cfg, be, max_tokens)
            asyncio.run(_run_backend(be, mode, rows, out, cfg, sampling,
                                     max_attempts=max_attempts, retry_delay=retry_delay))
            outputs.append(out)

    return outputs
