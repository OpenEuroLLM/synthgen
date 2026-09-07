"""Real-data anchoring for Pool A (general/dataset-common domains): streams a
real chat dataset, classifies sampled user turns into Pool A domains, and
writes two static artifacts consumed offline by prompts_grounded.py:

  domain_exemplars/<lang>.jsonl      real prompts bucketed by domain, used as
                                      light/deep grounding content (both pools)
  domain_distribution/<lang>.json    Pool-A-ONLY empirical frequency, smoothed
                                      with a floor — NEVER applied to Pool B

Same shape as `synthgen/pipeline/topics.py::run()` (HF `datasets` streaming +
LLM classification into a fixed label set) — deliberately reused rather than
re-derived, just pointed at the broadened Pool A label set instead of
`TOPIC_NAMES`.

**The HF Hub download step still needs a LOGIN NODE with internet access** —
run this from there, before any SLURM job. The classifier step has two
options:

  --classifier-backend openrouter (default)   needs OPENROUTER_API_KEY, runs
                                               from the login node alongside
                                               the download step
  --classifier-backend vllm                   points at an already-running
                                               local vLLM server instead (no
                                               API key needed) — use this if
                                               OPENROUTER_API_KEY isn't set;
                                               trades a purpose-sized
                                               classifier model
                                               (openai/gpt-4.1-nano) for
                                               whatever's already served
                                               (e.g. the same gen/judge
                                               model), which is overkill for
                                               a one-label classification
                                               call but avoids the API key
                                               dependency

  python ground_domains.py --n-per-lang 400 --out .
  python ground_domains.py --n-per-lang 400 --out . \
      --classifier-backend vllm --vllm-endpoints $EP/gen --vllm-model gen
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthgen.backends import OpenRouterBackend, VLLMBackend  # noqa: E402
from synthgen.config import SynthConfig, LANGUAGES_PHASE3  # noqa: E402
from synthgen.log import get_logger  # noqa: E402
import taxonomy_broadened as TB  # noqa: E402

log = get_logger("axes_ablation.ground_domains")

CLASSIFIER_MODEL = "openai/gpt-4.1-nano"

# Language-code mapping between synthgen's ISO codes and WildChat's `language`
# field (WildChat stores full English language names, e.g. "French"). Extend
# as needed if a LANGUAGES_PHASE3 code's WildChat name differs.
WILDCHAT_LANG_NAMES: dict[str, str] = {
    "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
    "pt": "Portuguese", "pl": "Polish", "nl": "Dutch", "cs": "Czech",
    "ro": "Romanian", "el": "Greek", "uk": "Ukrainian",
}

CLASSIFY_PROMPT = """Classify the following user prompt into EXACTLY ONE of these
domain categories. Reply with only the category name, nothing else.

Categories:
{category_list}
- none_of_these: does not fit any category above

USER PROMPT:
\"\"\"{text}\"\"\"

Category:"""


def _category_list() -> str:
    return "\n".join(f"- {d}" for d in TB.POOL_A)


def _first_user_message(example: dict) -> str | None:
    for field in ("conversation", "messages"):
        if field in example:
            for t in example[field]:
                if isinstance(t, dict) and t.get("role") == "user":
                    return t.get("content")
    if "prompt" in example:
        return example["prompt"]
    return None


async def _classify_all(backend, texts: list[str], concurrency: int,
                        *, is_vllm: bool = False) -> list[str]:
    sem = asyncio.Semaphore(concurrency)
    results: list[str] = [""] * len(texts)
    prompt_categories = set(TB.POOL_A) | {"none_of_these"}
    sampling = {"temperature": 0.0, "max_tokens": 20}
    if is_vllm:
        # in case the served model is a reasoning model — disable thinking so
        # the one-word category comes back directly instead of behind a
        # <think> block (see CLAUDE.md's judge-sampling note; same failure
        # mode, different caller).
        sampling["chat_template_kwargs"] = {"enable_thinking": False}
        await backend.ensure_ready()
    async with httpx.AsyncClient() as client:
        async def worker(i: int, text: str):
            t = (text or "")[:1500].strip()
            if not t:
                results[i] = "none_of_these"
                return
            async with sem:
                res = await backend.chat(
                    client,
                    messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(
                        category_list=_category_list(), text=t)}],
                    sampling=sampling)
                cat = (res.get("content") or "none_of_these").strip()
                results[i] = cat if cat in prompt_categories else "none_of_these"
        await asyncio.gather(*(worker(i, t) for i, t in enumerate(texts)))
    return results


def _sample_texts_by_lang(dataset: str, split: str, n_per_lang: int,
                          langs: list[str], seed: int, *,
                          cache_dir: Path | None = None) -> dict[str, list[str]]:
    """Streams real user turns from `dataset` (needs internet — login node
    only). If `cache_dir` is given, raw sampled texts are cached there as
    `<lang>.json`, and reused on a later call instead of re-streaming — so a
    SLURM job (no internet) can rerun classification against already-cached
    texts by pointing --texts-cache at the same directory, without touching
    the network."""
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = {c: json.loads((cache_dir / f"{c}.json").read_text())
                 for c in langs if (cache_dir / f"{c}.json").exists()}
        if all(c in cached for c in langs):
            log.info("using cached raw texts from %s (no network access needed)", cache_dir)
            # honor n_per_lang even against a cache built at a larger n (e.g.
            # a smoke run reusing a full-scale cache) -- deterministic slice,
            # same seed as the streaming path would have used.
            rng = random.Random(seed)
            sliced = {}
            for c in langs:
                texts = list(cached[c])
                rng.shuffle(texts)
                sliced[c] = texts[:n_per_lang]
            return sliced

    from datasets import load_dataset

    log.info("streaming %s (%s)...", dataset, split)
    ds = load_dataset(dataset, split=split, streaming=True)
    rng = random.Random(seed)
    wanted_names = {WILDCHAT_LANG_NAMES.get(c, c): c for c in langs}
    buffers: dict[str, list[str]] = defaultdict(list)
    overshoot = n_per_lang * 3

    for ex in ds:
        lang_field = ex.get("language")
        code = wanted_names.get(lang_field)
        if not code or len(buffers[code]) >= overshoot:
            continue
        t = _first_user_message(ex)
        if t and len(t) > 20:
            buffers[code].append(t)
        if all(len(buffers.get(c, [])) >= overshoot for c in langs):
            break

    out: dict[str, list[str]] = {}
    for code in langs:
        texts = buffers.get(code, [])
        rng.shuffle(texts)
        out[code] = texts[:n_per_lang]
        if len(out[code]) < n_per_lang:
            log.warning("only found %d/%d real prompts for lang=%s in %s — "
                       "coverage may be too thin; consider the LMArena-100k "
                       "fallback or an aggregate cross-language distribution",
                       len(out[code]), n_per_lang, code, dataset)
    if cache_dir is not None:
        for code, texts in out.items():
            (cache_dir / f"{code}.json").write_text(json.dumps(texts, ensure_ascii=False))
        log.info("cached raw texts -> %s (reusable without internet later)", cache_dir)
    return out


def _smoothed_distribution(counts: Counter, *, floor: float = 0.02) -> dict[str, float]:
    """Softens the raw empirical distribution with a per-domain floor so a
    domain absent from the sample (or rare in real chat logs) doesn't get
    rounded to exactly 0 — a hard 0 in sampling weights means it can never be
    drawn again, which is a stronger claim than "rare in this dataset"."""
    domains = TB.POOL_A
    total = sum(counts.get(d, 0) for d in domains) or 1
    raw = {d: counts.get(d, 0) / total for d in domains}
    floored = {d: max(p, floor) for d, p in raw.items()}
    norm = sum(floored.values())
    return {d: round(p / norm, 4) for d, p in floored.items()}


def run(*, dataset: str, split: str, n_per_lang: int, langs: list[str], seed: int,
       classifier_backend: str, classifier_model: str,
       vllm_endpoints: str | None, out_dir: Path, floor: float,
       texts_cache: Path | None) -> None:
    exemplar_dir = out_dir / "domain_exemplars"
    dist_dir = out_dir / "domain_distribution"
    exemplar_dir.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)

    texts_by_lang = _sample_texts_by_lang(dataset, split, n_per_lang, langs, seed,
                                          cache_dir=texts_cache)
    cfg = SynthConfig()
    is_vllm = classifier_backend == "vllm"
    if is_vllm:
        if not vllm_endpoints:
            raise SystemExit("--classifier-backend vllm requires --vllm-endpoints")
        backend = VLLMBackend(model=classifier_model, endpoints_dir=vllm_endpoints)
    else:
        backend = OpenRouterBackend(model=classifier_model, cfg=cfg)

    agg_counts: Counter[str] = Counter()
    for code, texts in texts_by_lang.items():
        if not texts:
            continue
        log.info("classifying %d prompts for lang=%s...", len(texts), code)
        cats = asyncio.run(_classify_all(backend, texts, cfg.concurrency, is_vllm=is_vllm))

        by_domain: dict[str, list[str]] = defaultdict(list)
        for text, cat in zip(texts, cats):
            if cat == "none_of_these":
                continue
            by_domain[cat].append(text)

        exemplar_rows = [
            {"lang": code, "domain": d, "text": t}
            for d, texts_for_d in by_domain.items() for t in texts_for_d
        ]
        with (exemplar_dir / f"{code}.jsonl").open("w", encoding="utf-8") as f:
            for r in exemplar_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        counts = Counter({d: len(v) for d, v in by_domain.items()})
        agg_counts.update(counts)
        dist = _smoothed_distribution(counts, floor=floor)
        (dist_dir / f"{code}.json").write_text(json.dumps({
            "lang": code, "source": dataset, "split": split, "n_classified": len(texts),
            "raw_counts": dict(counts), "smoothed_distribution": dist,
        }, indent=2, ensure_ascii=False))
        log.info("wrote %d exemplars + distribution for lang=%s", len(exemplar_rows), code)

    # aggregate-across-languages fallback, for any LANGUAGES_PHASE3 language
    # whose per-language coverage in this dataset is too thin to trust alone
    # (see the warning in _sample_texts_by_lang).
    agg_dist = _smoothed_distribution(agg_counts, floor=floor)
    (dist_dir / "_aggregate.json").write_text(json.dumps({
        "source": dataset, "split": split, "langs": langs,
        "raw_counts": dict(agg_counts), "smoothed_distribution": agg_dist,
    }, indent=2, ensure_ascii=False))
    log.info("wrote aggregate fallback distribution -> %s", dist_dir / "_aggregate.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="allenai/WildChat-1M",
                    help="fallback if per-language coverage is too thin: "
                         "lmsys/lmarena-human-preference-100k")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-per-lang", type=int, default=400)
    ap.add_argument("--langs", nargs="*", default=list(LANGUAGES_PHASE3))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--classifier-backend", choices=["openrouter", "vllm"], default="openrouter")
    ap.add_argument("--classifier-model", default=None,
                    help=f"defaults to {CLASSIFIER_MODEL!r} for openrouter, "
                         "or the served model name for vllm (e.g. 'gen')")
    ap.add_argument("--vllm-endpoints", default=None,
                    help="endpoints dir for --classifier-backend vllm")
    ap.add_argument("--floor", type=float, default=0.02,
                    help="minimum per-domain sampling weight after smoothing "
                         "(see _smoothed_distribution docstring)")
    ap.add_argument("--texts-cache", default=None,
                    help="dir to cache/reuse raw sampled texts, so a later "
                         "--classifier-backend vllm run (e.g. inside a SLURM "
                         "job with no internet) can reclassify without "
                         "re-streaming from the HF Hub")
    ap.add_argument("--out", default=str(Path(__file__).parent))
    args = ap.parse_args()
    classifier_model = args.classifier_model or (
        "gen" if args.classifier_backend == "vllm" else CLASSIFIER_MODEL)
    run(dataset=args.dataset, split=args.split, n_per_lang=args.n_per_lang,
       langs=args.langs, seed=args.seed, classifier_backend=args.classifier_backend,
       classifier_model=classifier_model, vllm_endpoints=args.vllm_endpoints,
       out_dir=Path(args.out), floor=args.floor,
       texts_cache=Path(args.texts_cache) if args.texts_cache else None)


if __name__ == "__main__":
    main()
