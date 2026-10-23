"""Build a Hugging Face-ready release of the deduped 35-language synthetic SFT set.

Input : loc_full.dedup_embed.jsonl (rows: id, lang, prompt, response, domain,
        role, quality_score, model, intent, is_local)
Output: <out>/data/<lang>/train-00000-of-00001.parquet  (zstd)
        <out>/stats.json                                (counts used by the card)
        <out>/README.md                                 (dataset card, YAML configs)

Columns: id, language, messages [{role: user}, {role: assistant}], domain,
persona (the generation persona, null if none), judge_score (7-10).
Dropped: model (always the internal alias "gen"), intent and is_local (always
null in this set). Streams the input with one ParquetWriter per language, so
memory stays flat. Nothing is filtered here: every input row is written, and
the counts are checked against the input at the end.

usage: python build_hf_release.py <dedup_embed.jsonl> <dedup_embed.report.json> <out_dir>
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

MSG = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))
SCHEMA = pa.schema([
    ("id", pa.string()),
    ("language", pa.string()),
    ("messages", MSG),
    ("domain", pa.string()),
    ("persona", pa.string()),
    ("judge_score", pa.int8()),
])
BATCH = 20_000

LANG_NAMES = {
    "bg": "Bulgarian", "bs": "Bosnian", "ca": "Catalan", "cs": "Czech", "da": "Danish",
    "de": "German", "el": "Greek", "es": "Spanish", "et": "Estonian", "eu": "Basque",
    "fi": "Finnish", "fr": "French", "ga": "Irish", "gl": "Galician", "hr": "Croatian",
    "hu": "Hungarian", "is": "Icelandic", "it": "Italian", "ka": "Georgian", "lt": "Lithuanian",
    "lv": "Latvian", "mk": "Macedonian", "mt": "Maltese", "nl": "Dutch", "no": "Norwegian",
    "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "sk": "Slovak", "sl": "Slovenian",
    "sq": "Albanian", "sr": "Serbian", "sv": "Swedish", "tr": "Turkish", "uk": "Ukrainian",
}


def build(src: Path, out: Path) -> dict:
    data = out / "data"
    writers: dict[str, pq.ParquetWriter] = {}
    buf: dict[str, list[dict]] = defaultdict(list)
    counts: Counter = Counter()
    domains: dict[str, Counter] = defaultdict(Counter)
    personas: Counter = Counter()
    scores: Counter = Counter()
    seen_ids: set[str] = set()
    dup_ids = 0

    def flush(lang: str) -> None:
        rows = buf[lang]
        if not rows:
            return
        if lang not in writers:
            d = data / lang
            d.mkdir(parents=True, exist_ok=True)
            writers[lang] = pq.ParquetWriter(d / "train-00000-of-00001.parquet", SCHEMA,
                                             compression="zstd")
        writers[lang].write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
        rows.clear()

    t0 = time.time()
    with open(src, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            r = json.loads(line)
            lang = r["lang"]
            if r["id"] in seen_ids:
                dup_ids += 1
            seen_ids.add(r["id"])
            buf[lang].append({
                "id": r["id"],
                "language": lang,
                "messages": [{"role": "user", "content": r["prompt"]},
                             {"role": "assistant", "content": r["response"]}],
                "domain": r.get("domain"),
                "persona": r.get("role"),
                "judge_score": int(r["quality_score"]),
            })
            counts[lang] += 1
            domains[lang][r.get("domain")] += 1
            personas[r.get("role")] += 1
            scores[int(r["quality_score"])] += 1
            if len(buf[lang]) >= BATCH:
                flush(lang)
            if n % 500_000 == 0:
                print(f"  {n:,} rows ({time.time() - t0:.0f}s)", flush=True)
    for lang in list(buf):
        flush(lang)
    for w in writers.values():
        w.close()

    total_domains: Counter = Counter()
    for c in domains.values():
        total_domains.update(c)
    return {
        "rows": sum(counts.values()),
        "rows_by_language": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "duplicate_ids": dup_ids,
        "domains": dict(total_domains.most_common()),
        "personas": {str(k): v for k, v in personas.most_common()},
        "judge_scores": {str(k): v for k, v in sorted(scores.items())},
    }


def verify(out: Path, stats: dict) -> None:
    """Re-read every parquet's metadata and one row; counts must match the input."""
    total = 0
    for lang, n in stats["rows_by_language"].items():
        p = out / "data" / lang / "train-00000-of-00001.parquet"
        meta = pq.ParquetFile(p).metadata
        assert meta.num_rows == n, f"{lang}: parquet has {meta.num_rows}, expected {n}"
        got = pq.ParquetFile(p).schema_arrow
        assert got.names == SCHEMA.names, f"{lang}: columns {got.names}"
        assert str(got.field("judge_score").type) == "int8" and pa.types.is_list(got.field("messages").type)
        first = pq.ParquetFile(p).read_row_group(0).slice(0, 1).to_pylist()[0]
        assert first["language"] == lang and len(first["messages"]) == 2
        total += meta.num_rows
    assert total == stats["rows"], f"total {total} != {stats['rows']}"
    print(f"VERIFY OK: {len(stats['rows_by_language'])} files, {total:,} rows, schema matches")


def card(stats: dict, report: dict) -> str:
    langs = sorted(stats["rows_by_language"])
    overrides = report.get("embedding_lang_overrides", {})
    cfgs = ["- config_name: all", "  default: true", "  data_files:", "  - split: train",
            "    path: data/*/train-*.parquet"]
    for l in langs:
        cfgs += [f"- config_name: '{l}'", "  data_files:", "  - split: train",
                 f"    path: data/{l}/train-*.parquet"]
    yaml = "\n".join([
        "---",
        "license: apache-2.0",
        "language:",
        *[f"- '{l}'" for l in langs],
        "task_categories:",
        "- text-generation",
        "tags:",
        "- synthetic",
        "- instruction-tuning",
        "- sft",
        "- multilingual",
        "- llm-as-a-judge",
        "size_categories:",
        "- 1M<n<10M",
        "configs:",
        *cfgs,
        "---",
    ])

    lang_rows = "\n".join(
        f"| `{l}` | {LANG_NAMES.get(l, l)} | {n:,} |"
        + (" LaBSE ≥ 0.96 |" if l in overrides else " MiniLM ≥ 0.97 |")
        for l, n in stats["rows_by_language"].items())
    dom_rows = "\n".join(f"| {d} | {n:,} | {100 * n / stats['rows']:.1f}% |"
                         for d, n in stats["domains"].items())
    score_rows = "\n".join(f"| {s} | {n:,} | {100 * n / stats['rows']:.1f}% |"
                           for s, n in stats["judge_scores"].items())
    personas_named = {k: v for k, v in stats["personas"].items() if k != "None"}
    persona_none = stats["personas"].get("None", 0)

    return f"""{yaml}

# EuroMultiSynth-Instruct: synthetic instruction-tuning data in 35 European languages

{stats['rows']:,} single-turn instruction/response pairs in 35 European (and
neighbouring) languages, written natively in each language (not translated) by
an LLM, scored by an LLM judge, quality-filtered and de-duplicated within each
language. Built with [`synthgen`](https://github.com/OpenEuroLLM/synthgen) for the OpenEuroLLM project.

## Quick start

```python
from datasets import load_dataset

ds = load_dataset("openeurollm/EuroMultiSynth-Instruct", "de", split="train")   # one language
ds = load_dataset("openeurollm/EuroMultiSynth-Instruct", split="train")         # all 35 languages
print(ds[0]["messages"])
```

## Fields

| field | type | description |
|---|---|---|
| `id` | string | unique row id, `<lang>-<number>` |
| `language` | string | ISO 639-1 code |
| `messages` | list of `{{role, content}}` | `[user prompt, assistant response]` |
| `domain` | string | task category the prompt was generated for ({len(stats['domains'])} values) |
| `persona` | string or null | user persona the prompt was written as (null for {100 * persona_none / stats['rows']:.0f}% of rows) |
| `judge_score` | int8 | LLM-judge quality score, 0-10 (only 7-10 kept) |

## How it was made

1. **Generation.** `google/gemma-4-31b-it` (served with vLLM) wrote each
   instruction *and* its response in the target language in one call,
   conditioned on a task domain and, for some prompts, a user persona.
   7,040,000 unique samples were generated (160k or 320k per language).
2. **Judging.** `Qwen/Qwen3.6-27B` scored the samples 0-10 (temperature 0,
   thinking disabled). 2,204 samples (0.03%) were never judged and are not included.
3. **Quality filter.** Kept samples with score ≥ 7: 5,688,783 rows (80.9% of
   scored samples).
4. **Exact de-duplication** on (language, whitespace/case-normalised prompt):
   −93,387 rows.
5. **Near-duplicate removal, within each language only.** Prompts were
   embedded and a prompt was dropped if its cosine similarity to an earlier
   kept prompt in the *same* language was at or above the threshold (greedy,
   first occurrence kept). 31 languages used
   `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` at **0.97**;
   Maltese, Irish, Basque and Icelandic, which that model does not cover, used
   `sentence-transformers/LaBSE` at **0.96** (calibrated on German and Croatian
   to remove the same share as MiniLM at 0.97). Thresholds were chosen from
   sweeps with example pairs per similarity band: at these values, removed
   pairs are rewordings of the same request. −1,434,798 rows.

Final: **{stats['rows']:,} rows**.

## Languages

| code | language | rows | near-dup model / threshold |
|---|---|---|---|
{lang_rows}

## Domains

| domain | rows | share |
|---|---|---|
{dom_rows}

## Judge scores

| score | rows | share |
|---|---|---|
{score_rows}

Personas ({len(personas_named)}): {", ".join(f"{k} ({v:,})" for k, v in personas_named.items())}.

## Limitations

- **Fully synthetic.** Both the prompts and the responses are model-generated;
  facts, figures and local details (laws, prices, places) can be wrong or out of date.
- **LLM-judged.** Quality filtering relies on a single LLM judge, whose reliability
  likely varies by language and is probably weaker for lower-resource languages.
- **Uneven size and quality across languages.** Languages with lower judge pass
  rates are smaller (e.g. Irish: 320,000 generated, {stats['rows_by_language'].get('ga', 0):,} kept).
  A few spot-checked Basque and Irish samples contained malformed words, so
  expect more noise in the lower-resource languages.
- **Residual redundancy.** The near-duplicate thresholds are deliberately
  conservative, so many prompts still share a template or scenario with
  different specifics.
- **Single-turn only.**

## License

Released under the **Apache License 2.0**.

This choice follows from the models used to build the dataset, all of which
are Apache-2.0:

| role | model | license |
|---|---|---|
| generation (wrote every prompt and response) | [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) | [Apache-2.0](https://ai.google.dev/gemma/docs/gemma_4_license) |
| quality judge (scores only, no text in the dataset) | [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B) | [Apache-2.0](https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/LICENSE) |
| near-duplicate filtering | [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), [`sentence-transformers/LaBSE`](https://huggingface.co/sentence-transformers/LaBSE) | Apache-2.0 |

All text in this dataset is model output, attributed to `google/gemma-4-31B-it`.
Please credit both the dataset and the generating model when you use it, and
follow Google's [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy)
in spirit when building on Gemma-generated data.

## Citation

TODO
"""


def main() -> None:
    src, rep, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"FATAL: {out} is not empty; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads(rep.read_text())
    stats = build(src, out)
    print(json.dumps({k: v for k, v in stats.items() if k != "domains"}, indent=1), flush=True)
    if stats["rows"] != report["kept"]:
        raise SystemExit(f"FATAL: wrote {stats['rows']} rows but report says kept={report['kept']}")
    verify(out, stats)
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    (out / "README.md").write_text(card(stats, report))
    print(f"wrote {out / 'README.md'} and stats.json")


if __name__ == "__main__":
    main()
