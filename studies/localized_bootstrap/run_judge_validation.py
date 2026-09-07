"""Judge-validation study: does the LLM judge separate good from bad?

Generate known-GOOD and known-BAD instruction-tuning examples in English and
Hindi, score each with the holistic quality judge, and compare the score
distributions. If the judge is trustworthy, good >> bad with little overlap.

Talks to two local vLLM servers via the synthgen endpoint-pool backend:
  --gen-endpoints    generator (gemma-4-31b-it)
  --judge-endpoints  judge     (Qwen3.6-27B)

Run inside the vLLM container from the repo root, e.g.:
  python studies/localized_bootstrap/run_judge_validation.py \
      --gen-endpoints endpoints/gen --judge-endpoints endpoints/judge \
      --n-good 5 --n-bad 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from synthgen.backends import VLLMBackend  # noqa: E402
from synthgen.io import write_jsonl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompts as P  # noqa: E402
import taxonomy as T  # noqa: E402

# (code, name, country) for this study — the 11 phase-3 EU languages
# (synthgen.config.LANGUAGES_PHASE3), with a representative country per language
# for localization grounding.
LANGS = [
    ("es", "Spanish", "Spain"),
    ("fr", "French", "France"),
    ("de", "German", "Germany"),
    ("it", "Italian", "Italy"),
    ("pt", "Portuguese", "Portugal"),
    ("pl", "Polish", "Poland"),
    ("nl", "Dutch", "the Netherlands"),
    ("cs", "Czech", "Czechia"),
    ("ro", "Romanian", "Romania"),
    ("el", "Greek", "Greece"),
    ("uk", "Ukrainian", "Ukraine"),
]

# Fraction of GOOD examples that even get a persona *suggested* (the model may
# still drop it). The rest are deliberately persona-free direct requests.
PERSONA_P = 0.5

# Fraction of examples drawn from the GENERAL (professional/creative) domains;
# the rest come from the LOCAL domains that carry the localization signal. Local
# is the point of this dataset, so general is the minority for variety only.
GENERAL_P = 0.2


def extract_json(text: str | None) -> dict | None:
    """Parse the LAST balanced {...} object, after dropping any <think> block
    (reasoning models emit reasoning before the answer) and code fences."""
    if not text:
        return None
    t = text.strip()
    if "</think>" in t:                       # keep only the post-reasoning answer
        t = t.split("</think>")[-1]
    t = re.sub(r"```(?:json)?", "", t).strip()
    end = t.rfind("}")
    while end != -1:
        depth = 0
        for i in range(end, -1, -1):
            if t[i] == "}":
                depth += 1
            elif t[i] == "{":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[i:end + 1])
                    except json.JSONDecodeError:
                        break
        end = t.rfind("}", 0, end)
    return None


async def _chat(backend, client, prompt, sampling):
    res = await backend.chat(client, messages=[{"role": "user", "content": prompt}],
                             sampling=sampling)
    return res.get("content"), res.get("error")


def build_tasks(n_good: int, n_medium: int, n_bad: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    tasks: list[dict] = []

    def localized(code, lang_name, country, label, i, medium_type=None):
        is_local = rng.random() >= GENERAL_P
        domain = rng.choice(T.LOCAL_DOMAINS if is_local else T.GENERAL_DOMAINS)
        intents = T.LOCAL_INTENTS if is_local else T.GENERAL_INTENTS
        return {
            "id": f"{code}-{label}-{i:02d}", "lang": code, "lang_name": lang_name,
            "country": country, "label": label,
            "bad_type": None, "medium_type": medium_type,
            "domain": domain, "is_local": is_local,
            "intent": rng.choice(intents),
            "role": rng.choice(T.ROLES) if rng.random() < PERSONA_P else None,
            "salt": rng.randint(1000, 9999),
        }

    for code, lang_name, country in LANGS:
        for i in range(n_good):
            tasks.append(localized(code, lang_name, country, "good", i))
        med_types = list(P.MEDIUM_DEFECTS)
        for i in range(n_medium):
            tasks.append(localized(code, lang_name, country, "medium", i,
                                   medium_type=med_types[i % len(med_types)]))
        bad_types = P.bad_types_for(code)
        for i in range(n_bad):
            tasks.append({
                "id": f"{code}-bad-{i:02d}", "lang": code, "lang_name": lang_name,
                "country": country, "label": "bad",
                "bad_type": bad_types[i % len(bad_types)], "medium_type": None,
                "salt": rng.randint(1000, 9999),
            })
    return tasks


async def process(t: dict, gen, judge, client, gen_s, judge_s) -> dict:
    rec = {**t}
    if t["label"] == "good":
        gp = P.good_example_prompt(
            lang_name=t["lang_name"], country=t["country"], domain=t["domain"],
            intent=t["intent"], salt=t["salt"], role=t["role"],
            localized=t.get("is_local", True))
    elif t["label"] == "medium":
        gp = P.medium_example_prompt(
            lang_name=t["lang_name"], country=t["country"], domain=t["domain"],
            intent=t["intent"], salt=t["salt"], role=t["role"],
            medium_type=t["medium_type"])
    else:
        gp = P.bad_example_prompt(
            lang_name=t["lang_name"], country=t["country"], bad_type=t["bad_type"],
            salt=t.get("salt", 0))
    txt, err = await _chat(gen, client, gp, gen_s)
    ex = extract_json(txt) or {}
    rec["instruction"] = ex.get("instruction")
    rec["response"] = ex.get("response")
    if t["label"] in ("good", "medium"):  # what the model used after adapt/drop
        rec["role_used"] = ex.get("role_used")
        rec["intent_used"] = ex.get("intent_used")
    rec["gen_err"] = err

    if rec["instruction"] and rec["response"]:
        jp = P.judge_quality_prompt(
            lang_name=t["lang_name"], country=t["country"],
            instruction=rec["instruction"], response=rec["response"])
        txt, _ = await _chat(judge, client, jp, judge_s)
        rec["judge_raw"] = (txt or "")[:800]
        j = extract_json(txt) or {}
        sc = j.get("score")
        if isinstance(sc, str) and sc.strip().lstrip("-").isdigit():
            sc = int(sc)
        rec["score"] = sc if isinstance(sc, (int, float)) else None
        rec["reason"] = j.get("reason")
    return rec


def _hist(scores: list[int]) -> str:
    c = Counter(scores)
    return "\n".join(
        f"    {s:2d} | {'#' * c.get(s, 0)}{'' if c.get(s) else '.'} ({c.get(s, 0)})"
        for s in range(10, -1, -1))


def summarize(records: list[dict]) -> dict:
    def scores(pred):
        return [r["score"] for r in records
                if pred(r) and isinstance(r.get("score"), (int, float))]

    def stats(xs):
        if not xs:
            return {"n": 0, "mean": None, "min": None, "max": None}
        return {"n": len(xs), "mean": round(sum(xs) / len(xs), 2),
                "min": min(xs), "max": max(xs)}

    labels = ("good", "medium", "bad")
    summ: dict = {"overall": {}, "by_language": {}, "by_bad_type": {},
                  "by_medium_type": {}}
    for label in labels:
        summ["overall"][label] = stats(scores(lambda r, l=label: r["label"] == l))
    for code, name, _ in LANGS:
        summ["by_language"][code] = {
            label: stats(scores(lambda r, l=label, c=code:
                                 r["label"] == l and r["lang"] == c))
            for label in labels}
    for bt in P.BAD_DEFECTS:
        st = stats(scores(lambda r, b=bt: r.get("bad_type") == b))
        if st["n"]:
            summ["by_bad_type"][bt] = st
    for mt in P.MEDIUM_DEFECTS:
        st = stats(scores(lambda r, m=mt: r.get("medium_type") == m))
        if st["n"]:
            summ["by_medium_type"][mt] = st
    summ["gen_failures"] = sum(1 for r in records if not r.get("response"))
    return summ


def to_dataset(records: list[dict], gen_id: str, judge_id: str) -> list[dict]:
    """Finalized rows: content + a single `meta` column holding every
    conditioning variable and provenance, for later reporting."""
    rows = []
    for r in records:
        rows.append({
            "id": r["id"],
            "language": r["lang"],
            "instruction": r.get("instruction"),
            "response": r.get("response"),
            "quality_score": r.get("score"),
            "meta": {
                "lang_code": r["lang"], "lang_name": r["lang_name"],
                "country": r["country"],
                "label": r["label"], "bad_type": r.get("bad_type"),
                "medium_type": r.get("medium_type"),
                "domain": r.get("domain"), "is_local": r.get("is_local"),
                "intent_suggested": r.get("intent"), "intent_used": r.get("intent_used"),
                "role_suggested": r.get("role"), "role_used": r.get("role_used"),
                "salt": r.get("salt"),
                "generator": gen_id, "judge": judge_id,
                "judge_reason": r.get("reason"),
            },
        })
    return rows


def print_report(records, summ):
    def sc(label):
        return [r["score"] for r in records if r["label"] == label
                and isinstance(r.get("score"), (int, float))]
    print("\n=== SCORE DISTRIBUTION (0-10) ===")
    for label in ("good", "medium", "bad"):
        o = summ["overall"][label]
        print(f"{label.upper():7s} n={o['n']}  mean={o['mean']}")
        print(_hist(sc(label)))
    print("\n=== medium by type (want these in the MIDDLE, ~4-7) ===")
    for mt, st in summ["by_medium_type"].items():
        print(f"    {mt:20s} mean={st['mean']}  n={st['n']}")
    print("\n=== bad by type (want these LOW) ===")
    for bt, st in summ["by_bad_type"].items():
        print(f"    {bt:20s} mean={st['mean']}  n={st['n']}")
    print("\n=== by language (good / medium / bad means) ===")
    for code, bl in summ["by_language"].items():
        print(f"    {code}: good={bl['good']['mean']}  "
              f"medium={bl['medium']['mean']}  bad={bl['bad']['mean']}")
    print("\n=== per-language score distribution (all labels, for threshold picking) ===")
    for code, name, _ in LANGS:
        xs = [r["score"] for r in records if r["lang"] == code
              and isinstance(r.get("score"), (int, float))]
        print(f"  -- {name} ({code}), n={len(xs)} --")
        print(_hist(xs))
    if summ["gen_failures"]:
        print(f"\n[warn] {summ['gen_failures']} examples failed to generate/parse")


async def main_async(args):
    gen_s = {"temperature": 0.9, "top_p": 0.95, "max_tokens": 1536}
    # Judge is a reasoning model — disable thinking so it emits the JSON directly.
    judge_s = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512,
               "chat_template_kwargs": {"enable_thinking": False}}
    gen = VLLMBackend(model=args.gen_model, endpoints_dir=args.gen_endpoints)
    judge = VLLMBackend(model=args.judge_model, endpoints_dir=args.judge_endpoints)

    tasks = build_tasks(args.n_good, args.n_medium, args.n_bad, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)

    # Heartbeat for the job's stall watchdog: rewrite this file's mtime every time
    # an example finishes. If it stops advancing, generation has stalled and the
    # watchdog (in the slurm script) can kill the job instead of idling to wall.
    hb = out / ".heartbeat"
    hb.write_text("0/{}".format(len(tasks)))  # arm immediately, before first result
    done = 0
    lock = asyncio.Lock()

    async with httpx.AsyncClient() as client:
        await gen.ensure_ready()
        await judge.ensure_ready()

        async def worker(t):
            nonlocal done
            async with sem:
                try:
                    r = await process(t, gen, judge, client, gen_s, judge_s)
                except Exception as e:
                    r = {**t, "fatal_error": f"{type(e).__name__}: {e}"}
            async with lock:
                done += 1
                hb.write_text(f"{done}/{len(tasks)}")
                if done % 5 == 0 or done == len(tasks):
                    print(f"[progress] {done}/{len(tasks)} examples done", flush=True)
            return r

        records = await asyncio.gather(*(worker(t) for t in tasks))

    write_jsonl(out / "records.jsonl", records)
    write_jsonl(out / "dataset.jsonl", to_dataset(records, args.gen_id, args.judge_id))
    summ = summarize(records)
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2))
    print_report(records, summ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-endpoints", required=True)
    ap.add_argument("--judge-endpoints", required=True)
    ap.add_argument("--gen-model", default="gen")
    ap.add_argument("--judge-model", default="judge")
    ap.add_argument("--gen-id", default="google/gemma-4-31b-it",
                    help="generator id recorded in dataset meta")
    ap.add_argument("--judge-id", default="Qwen/Qwen3.6-27B",
                    help="judge id recorded in dataset meta")
    ap.add_argument("--n-good", type=int, default=10, help="good examples per language")
    ap.add_argument("--n-medium", type=int, default=10, help="medium examples per language")
    ap.add_argument("--n-bad", type=int, default=10, help="bad examples per language")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).parent / "outputs"))
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
