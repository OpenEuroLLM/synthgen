"""Export an SFT JSONL into open-instruct's per-language parquet *source* layout.

The OELLM `fabio-open-instruct` mixer (`oellm/pipelines/preprocessing/
assemble_mixture.py`) draws training data from per-language parquet files at::

    <by_language_dir>/<source>/<lang>.parquet

where each parquet carries two columns:

    messages  : list<struct<content: string, role: string>>
    language  : string

This module turns ``sft_full.dedup.jsonl`` (rows of ``{lang, prompt, response,
...}``) into one such *source* directory (default name ``synthgen-if``) so the
synthetic corpus can be referenced as a ``sources:`` entry in a mixture config.

Writing is streamed with one ``ParquetWriter`` per language, so memory stays
flat regardless of corpus size.
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from synthgen.log import get_logger

log = get_logger("synthgen.to_open_instruct")

# Field order matches preprocess_datasets.py's final mapping ({"content", "role"}).
_MSG_TYPE = pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())]))
SCHEMA = pa.schema([("messages", _MSG_TYPE), ("language", pa.string())])


def _flush(writer: pq.ParquetWriter, msgs: list, langs: list) -> None:
    table = pa.table({"messages": pa.array(msgs, type=_MSG_TYPE),
                      "language": pa.array(langs, type=pa.string())},
                     schema=SCHEMA)
    writer.write_table(table)


def run(*, input: Path, out_dir: Path, source: str = "synthgen-if",
        batch_size: int = 20_000, report: Path | None = None) -> dict:
    """Convert ``input`` JSONL → ``out_dir/<source>/<lang>.parquet``.

    Returns a summary dict of per-language row counts.
    """
    dest = out_dir / source
    dest.mkdir(parents=True, exist_ok=True)

    writers: dict[str, pq.ParquetWriter] = {}
    buf_msgs: dict[str, list] = {}
    buf_lang: dict[str, list] = {}
    counts: dict[str, int] = {}
    skipped = 0

    def writer_for(lang: str) -> pq.ParquetWriter:
        w = writers.get(lang)
        if w is None:
            w = pq.ParquetWriter(dest / f"{lang}.parquet", SCHEMA,
                                 use_dictionary=False)
            writers[lang] = w
            buf_msgs[lang] = []
            buf_lang[lang] = []
            counts[lang] = 0
        return w

    with open(input, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            if not line.strip():
                continue
            r = json.loads(line)
            lang = r.get("lang", "")
            prompt = (r.get("prompt", "") or "").strip()
            resp = (r.get("response", "") or "").strip()
            if not lang or not prompt or not resp:
                skipped += 1
                continue

            w = writer_for(lang)
            buf_msgs[lang].append([
                {"content": prompt, "role": "user"},
                {"content": resp, "role": "assistant"},
            ])
            buf_lang[lang].append(lang)
            counts[lang] += 1

            if len(buf_msgs[lang]) >= batch_size:
                _flush(w, buf_msgs[lang], buf_lang[lang])
                buf_msgs[lang].clear()
                buf_lang[lang].clear()

            if (i + 1) % 200_000 == 0:
                log.info("read %d rows", i + 1)

    for lang, w in writers.items():
        if buf_msgs[lang]:
            _flush(w, buf_msgs[lang], buf_lang[lang])
        w.close()

    summary = {
        "input": str(input),
        "source_dir": str(dest),
        "source": source,
        "skipped": skipped,
        "total": sum(counts.values()),
        "by_lang": dict(sorted(counts.items())),
    }
    if report:
        report.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("wrote %d langs, %d rows to %s (skipped %d)",
             len(counts), summary["total"], dest, skipped)
    return summary
