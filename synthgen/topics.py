"""Estimate empirical topic distribution from a real chat dataset.

Loads a HuggingFace dataset, samples N user prompts, classifies each into the
topic taxonomy via a small LM (default `openai/gpt-4.1-nano`), and writes a
frequency table to `topic_distribution.json`. `prompts.load_topics` reads it
when `SynthConfig.topic_source == "empirical"`.
"""
from __future__ import annotations

import asyncio
import json
import random
from collections import Counter
from pathlib import Path

import httpx

from synthgen.backends import OpenRouterBackend
from synthgen.config import SynthConfig
from synthgen.log import get_logger

log = get_logger("synthgen.topics")

CLASSIFIER_MODEL = "openai/gpt-4.1-nano"
TOPIC_NAMES = ["creative_writing", "qa", "translation", "math", "summarization",
               "trivia", "brainstorming", "roleplay", "coding", "data_analysis", "other"]

CLASSIFY_PROMPT = """Classify the following user prompt into EXACTLY ONE of these topic categories. Reply with only the category name, nothing else.

Categories:
- creative_writing: fiction, poetry, stories, songs
- qa: factual questions with clear answers
- translation: translate text between languages
- math: arithmetic, algebra, word problems
- summarization: summarize a passage given in the prompt
- trivia: cultural, historical, scientific facts
- brainstorming: idea generation for a real situation
- roleplay: act-out scenarios, character play
- coding: code snippets, debugging
- data_analysis: analyze tables, lists, datasets
- other: anything else (chat, advice, small talk)

USER PROMPT:
\"\"\"{text}\"\"\"

Category:"""


def _first_user_message(example: dict, text_field: str | None = None) -> str | None:
    if text_field and text_field in example:
        v = example[text_field]
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            for t in v:
                if isinstance(t, dict) and t.get("role") == "user":
                    return t.get("content")
    for field in ("conversation", "messages"):
        if field in example:
            for t in example[field]:
                if t.get("role") == "user":
                    return t.get("content")
    if "prompt" in example:
        return example["prompt"]
    return None


async def _classify_all(backend: OpenRouterBackend, texts: list[str],
                        concurrency: int) -> Counter[str]:
    sem = asyncio.Semaphore(concurrency)
    counts: Counter[str] = Counter()
    total = len(texts)
    done = 0
    sampling = {"temperature": 0.0, "max_tokens": 20}
    async with httpx.AsyncClient() as client:
        async def worker(text: str):
            nonlocal done
            t = (text or "")[:1500].strip()
            if not t:
                counts["other"] += 1
                done += 1
                return
            async with sem:
                res = await backend.chat(
                    client,
                    messages=[{"role": "user",
                               "content": CLASSIFY_PROMPT.format(text=t)}],
                    sampling=sampling,
                )
                cat = (res.get("content") or "other").strip().lower().strip(".,")
                counts[cat if cat in TOPIC_NAMES else "other"] += 1
                done += 1
                if done % 50 == 0 or done == total:
                    log.info("classified %d/%d", done, total)
        await asyncio.gather(*(worker(t) for t in texts))
    return counts


def run(cfg: SynthConfig, *, dataset: str = "allenai/WildChat-1M",
        split: str = "train", text_field: str | None = None,
        n: int = 2000, seed: int = 0, model: str = CLASSIFIER_MODEL) -> Path:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("pip install 'synthgen[topics]' or 'datasets'") from e

    log.info("loading %s (%s)...", dataset, split)
    ds = load_dataset(dataset, split=split, streaming=True)
    rng = random.Random(seed)
    texts: list[str] = []
    for ex in ds:
        t = _first_user_message(ex, text_field)
        if t and len(t) > 20:
            texts.append(t)
            if len(texts) >= n * 3:
                break
    rng.shuffle(texts)
    texts = texts[:n]
    log.info("sampled %d prompts. classifying via %s...", len(texts), model)

    backend = OpenRouterBackend(model=model, cfg=cfg)
    counts = asyncio.run(_classify_all(backend, texts, cfg.concurrency))
    total = sum(counts.values())
    dist = {t: round(counts.get(t, 0) / total, 4) for t in TOPIC_NAMES}

    out = cfg.paths.root / "topic_distribution.json"
    out.write_text(json.dumps({
        "source": dataset, "split": split, "n": total,
        "distribution": dist, "raw_counts": dict(counts),
    }, indent=2))
    log.info("wrote %s", out)
    for t, p in sorted(dist.items(), key=lambda x: -x[1]):
        log.info("  %-18s %5.1f%%", t, 100 * p)
    return out
