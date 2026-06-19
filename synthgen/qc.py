"""Per-language QC stats over an SFT JSONL."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from synthgen.io import iter_jsonl
from synthgen.log import get_logger

log = get_logger("synthgen.qc")


def _quantile(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    i = int(q * (len(sorted_vals) - 1))
    return sorted_vals[i]


def run(*, input: Path, output: Path) -> dict:
    n_by_lang: Counter[str] = Counter()
    prompt_lens: dict[str, list[int]] = defaultdict(list)
    resp_lens: dict[str, list[int]] = defaultdict(list)
    persona_by_lang: dict[str, Counter] = defaultdict(Counter)
    constraint_by_lang: dict[str, Counter] = defaultdict(Counter)
    topic_by_lang: dict[str, Counter] = defaultdict(Counter)
    ctok_by_lang: dict[str, list[int]] = defaultdict(list)

    for r in iter_jsonl(input):
        lang = r.get("lang", "")
        n_by_lang[lang] += 1
        prompt_lens[lang].append(len(r.get("prompt", "") or ""))
        resp_lens[lang].append(len(r.get("response", "") or ""))
        persona_by_lang[lang][r.get("persona") or "<none>"] += 1
        constraint_by_lang[lang][r.get("constraint_name") or "<none>"] += 1
        topic_by_lang[lang][r.get("topic") or "<none>"] += 1
        ct = (r.get("usage") or {}).get("completion_tokens")
        if isinstance(ct, int):
            ctok_by_lang[lang].append(ct)

    out: dict = {"by_lang": {},
                 "totals": {"n": sum(n_by_lang.values()), "n_langs": len(n_by_lang)}}
    for lang in sorted(n_by_lang):
        pl = sorted(prompt_lens[lang])
        rl = sorted(resp_lens[lang])
        ct = sorted(ctok_by_lang[lang])
        out["by_lang"][lang] = {
            "n": n_by_lang[lang],
            "prompt_chars": {"median": median(pl), "p10": _quantile(pl, 0.10),
                             "p90": _quantile(pl, 0.90), "max": pl[-1]},
            "response_chars": {"median": median(rl), "p10": _quantile(rl, 0.10),
                               "p90": _quantile(rl, 0.90), "max": rl[-1]},
            "completion_tokens": {
                "median": median(ct) if ct else 0,
                "p90": _quantile(ct, 0.90) if ct else 0,
                "max": ct[-1] if ct else 0,
                "at_max_1024": sum(1 for x in ct if x >= 1024),
            },
            "personas": dict(persona_by_lang[lang].most_common()),
            "constraints": dict(constraint_by_lang[lang].most_common()),
            "topics_top10": dict(topic_by_lang[lang].most_common(10)),
            "n_topics": len(topic_by_lang[lang]),
        }

    output.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log.info("langs=%d  total=%d", len(n_by_lang), sum(n_by_lang.values()))
    for lang in sorted(n_by_lang):
        b = out["by_lang"][lang]
        log.info("  %s: n=%d  resp_med=%d  resp_p90=%d  trunc@1024=%d",
                 lang, b["n"], b["response_chars"]["median"],
                 b["response_chars"]["p90"], b["completion_tokens"]["at_max_1024"])
    return out
