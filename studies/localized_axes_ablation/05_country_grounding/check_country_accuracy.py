"""Does '01_domain_grounding's deep-grounding mode actually land on
{country}-specific content, or does the model genericize/hallucinate despite
having a real anchor?

Reuses the judge as a fact-locality checker (a second, different judge call)
rather than inventing a new labeling scheme — same holistic-scoring
philosophy as `synthgen.localized.prompts.judge_quality_prompt`, just a
different question.

Depends on `01_domain_grounding`'s output — run that subfolder first.

  python check_country_accuracy.py \
      --judge-endpoints endpoints/judge \
      --records ../01_domain_grounding/outputs/mode_deep/records.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from synthgen.backends import VLLMBackend  # noqa: E402
from synthgen.config import LANGUAGES_PHASE3, LANG_COUNTRY  # noqa: E402
from synthgen.io import extract_json, iter_jsonl, write_jsonl  # noqa: E402

JUDGE_SAMPLING = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 256,
                  "chat_template_kwargs": {"enable_thinking": False}}

COUNTRY_LOCALITY_PROMPT = """\
You are checking whether a piece of text is actually SPECIFIC to {country}, or
whether it is generic content that happens to be written in {lang_name} and
could equally describe any country.

Instruction:
{instruction}

Response:
{response}

Judge the RESPONSE (the instruction may reasonably be generic if it's a
direct request; what matters is whether the response's content — facts,
names, institutions, prices, customs — is genuinely tied to {country}, or
could be copy-pasted into an answer about a different country without
anyone noticing).

Output JSON only, no commentary, no code fences:
{{"country_specific": true|false, "confidence": "low"|"medium"|"high",
  "reason": "<one sentence>"}}"""


async def _chat(backend, client, prompt, sampling):
    res = await backend.chat(client, messages=[{"role": "user", "content": prompt}],
                             sampling=sampling)
    return res.get("content"), res.get("error")


async def check_one(rec: dict, judge, client) -> dict:
    lang = rec.get("lang")
    lang_name = LANGUAGES_PHASE3.get(lang, lang)
    country = LANG_COUNTRY.get(lang, lang)
    prompt = COUNTRY_LOCALITY_PROMPT.format(
        country=country, lang_name=lang_name,
        instruction=rec.get("instruction", ""), response=rec.get("response", ""))
    txt, err = await _chat(judge, client, prompt, JUDGE_SAMPLING)
    j = extract_json(txt) or {}
    return {
        "id": rec.get("id"), "lang": lang, "domain": rec.get("domain"),
        "effective_grounding_mode": rec.get("effective_grounding_mode"),
        "exemplar_used": bool(rec.get("exemplar_used")),
        "country_specific": j.get("country_specific"),
        "confidence": j.get("confidence"), "reason": j.get("reason"),
        "check_err": err,
    }


async def main_async(args):
    records = [r for r in iter_jsonl(args.records)
              if r.get("instruction") and r.get("response")]
    judge = VLLMBackend(model=args.judge_model, endpoints_dir=args.judge_endpoints)
    sem = asyncio.Semaphore(args.concurrency)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient() as client:
        await judge.ensure_ready()

        async def worker(rec):
            async with sem:
                return await check_one(rec, judge, client)

        results = await asyncio.gather(*(worker(r) for r in records))

    write_jsonl(out / "country_accuracy.jsonl", results)

    with_exemplar = [r for r in results if r["exemplar_used"] and r["country_specific"] is not None]
    without_exemplar = [r for r in results if not r["exemplar_used"] and r["country_specific"] is not None]

    def rate(rows):
        return (sum(1 for r in rows if r["country_specific"]) / len(rows)) if rows else None

    summary = {
        "n_total": len(results),
        "with_real_exemplar": {"n": len(with_exemplar), "country_specific_rate": rate(with_exemplar)},
        "without_real_exemplar": {"n": len(without_exemplar), "country_specific_rate": rate(without_exemplar)},
    }
    print("\n=== COUNTRY-SPECIFICITY RATE ===")
    print(f"  with real grounding exemplar:    {summary['with_real_exemplar']}")
    print(f"  without real grounding exemplar: {summary['without_real_exemplar']}")
    print("\nIf 'with_real_exemplar' rate isn't meaningfully higher, real-data "
         "grounding (01) is improving topical relevance without improving "
         "actual country-specificity — worth flagging back in the top-level REPORT.")

    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-endpoints", required=True)
    ap.add_argument("--judge-model", default="judge")
    ap.add_argument("--records", required=True,
                    help="01_domain_grounding/outputs/mode_deep/records.jsonl")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default=str(Path(__file__).parent / "outputs"))
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
