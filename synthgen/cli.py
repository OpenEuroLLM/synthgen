"""`synthgen` CLI: a single entrypoint that dispatches to each pipeline stage.

Run `synthgen --help` for subcommands.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from synthgen import __version__
from synthgen.config import Paths, SynthConfig
from synthgen.log import get_logger

log = get_logger("synthgen")


def _build_cfg(args: argparse.Namespace) -> SynthConfig:
    paths = Paths.from_root(getattr(args, "root", None))
    paths.ensure()
    cfg = SynthConfig(
        phase=getattr(args, "phase", 3),
        paths=paths,
    )
    concurrency = getattr(args, "concurrency", None)
    if concurrency is not None:
        cfg.concurrency = concurrency
    return cfg


def _add_root(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", default=None,
                   help="Project root (default: $SYNTHGEN_ROOT or cwd)")
    p.add_argument("--concurrency", type=int, default=None,
                   help="In-flight requests against the endpoint pool "
                        "(default: SynthConfig.concurrency, 16). A pool with "
                        "many replica endpoints needs this raised roughly "
                        "proportional to the replica count to actually "
                        "saturate them -- see topup's --gen-concurrency/"
                        "--judge-concurrency for per-phase overrides.")


# ----- subcommand implementations -----------------------------------------

def _cmd_build_prompts(args):
    from synthgen.pipeline import prompts
    prompts.build(_build_cfg(args), seed=args.seed, n_override=args.n)


def _cmd_build_localized_prompts(args):
    from synthgen.localized import prompts as prompts_localized
    cfg = _build_cfg(args)
    out = cfg.paths.prompts / "localized_prompts.jsonl"
    rows = prompts_localized.build(n_per_lang=args.n, seed=args.seed,
                                   lang_codes=args.langs)
    from synthgen.io import write_jsonl
    n = write_jsonl(out, rows)
    log.info("wrote %d localized prompts to %s", n, out)


def _build_backends(backend: str, models: list[str], cfg: SynthConfig, *,
                    endpoints_dir: str | None = None, min_endpoints: int = 1):
    """Construct Backend instances for an explicit (backend, models, ...) triple."""
    from synthgen.backends import OpenRouterBackend, VLLMBackend
    if backend == "openrouter":
        return [OpenRouterBackend(model=m, cfg=cfg) for m in models]
    if backend == "vllm":
        if not endpoints_dir:
            raise SystemExit("--endpoints-dir is required for --backend vllm")
        return [VLLMBackend(model=m, cfg=cfg, endpoints_dir=endpoints_dir,
                            min_endpoints=min_endpoints)
                for m in models]
    raise SystemExit(f"unknown backend: {backend}")


def _make_backends(args, cfg: SynthConfig):
    """Construct the list of Backend instances per CLI flags."""
    models = args.models or list(cfg.generators)
    return _build_backends(args.backend, models, cfg,
                           endpoints_dir=args.endpoints_dir,
                           min_endpoints=args.min_endpoints)


def _cmd_generate(args):
    from synthgen.pipeline import generate
    cfg = _build_cfg(args)
    backends = _make_backends(args, cfg)
    generate.run(
        cfg, backends=backends,
        mode=args.mode,
        in_path=Path(args.in_path) if args.in_path else None,
        split=args.split,
        max_tokens=args.max_tokens,
        max_attempts=args.max_attempts,
        retry_delay=args.retry_delay,
    )


def _cmd_filter(args):
    from synthgen.pipeline import filter as flt
    cfg = _build_cfg(args)
    gen = [Path(p) for p in args.gen] if args.gen else None
    flt.run(cfg, gen=gen)


def _cmd_verify(args):
    from synthgen.pipeline import verify
    cfg = _build_cfg(args)
    gen = [Path(p) for p in args.gen] if args.gen else None
    verify.run(cfg, gen=gen)


def _cmd_back_translate(args):
    from synthgen.pipeline import back_translate
    cfg = _build_cfg(args)
    gen = [Path(p) for p in args.gen] if args.gen else None
    back_translate.run(cfg, gen=gen, sample=args.sample, seed=args.seed,
                       model=args.model)


def _cmd_dedupe(args):
    import json as _json
    from synthgen.pipeline import dedupe
    summary = dedupe.run(
        input=Path(args.input), output=Path(args.output),
        min_response_chars=args.min_response_chars,
        report=Path(args.report) if args.report else None,
        embedding_dedupe=args.embedding_dedupe,
        embedding_threshold=args.embedding_threshold,
        embedding_model=args.embedding_model,
    )
    print(_json.dumps(summary, indent=2, ensure_ascii=False))


def _cmd_qc(args):
    from synthgen.pipeline import qc
    qc.run(input=Path(args.input), output=Path(args.output))


def _cmd_quality_filter(args):
    import json as _json
    from synthgen.localized import quality_filter
    thresholds = _json.loads(Path(args.thresholds).read_text())
    summary = quality_filter.run(
        gen=Path(args.gen), judged=Path(args.judged), thresholds=thresholds,
        output=Path(args.output), report=Path(args.report) if args.report else None,
    )
    print(_json.dumps(summary, indent=2, ensure_ascii=False))


def _cmd_decide_thresholds(args):
    import json as _json
    from synthgen.localized import thresholds as th
    decision = th.run(
        records=Path(args.records), output=Path(args.output),
        label=args.label, percentile=args.percentile,
        spread_tolerance=args.spread_tolerance,
        report=Path(args.report) if args.report else None,
    )
    print(_json.dumps(decision, indent=2, ensure_ascii=False))


def _cmd_topup(args):
    import json as _json
    from synthgen.localized import topup
    cfg = _build_cfg(args)
    thresholds = _json.loads(Path(args.thresholds).read_text())
    gen_models = args.gen_models or list(cfg.generators)
    gen_backends = _build_backends(args.gen_backend, gen_models, cfg,
                                   endpoints_dir=args.gen_endpoints_dir,
                                   min_endpoints=args.gen_min_endpoints)
    judge_backend = _build_backends(args.judge_backend, [args.judge_model], cfg,
                                    endpoints_dir=args.judge_endpoints_dir,
                                    min_endpoints=args.judge_min_endpoints)[0]
    summary = topup.run(
        cfg, gen_backends=gen_backends, judge_backend=judge_backend,
        target_per_lang=args.target, thresholds=thresholds,
        lang_codes=args.langs, overgen_factor_default=args.overgen_factor,
        max_rounds=args.max_rounds, seed=args.seed,
        max_attempts=args.max_attempts, retry_delay=args.retry_delay,
        split=args.split, embedding_dedupe=args.embedding_dedupe,
        embedding_threshold=args.embedding_threshold,
        embedding_model=args.embedding_model,
        gen_concurrency=args.gen_concurrency, judge_concurrency=args.judge_concurrency,
    )
    print(_json.dumps(summary, indent=2, ensure_ascii=False))


def _cmd_export_openinstruct(args):
    import json as _json
    from synthgen.pipeline import to_open_instruct
    summary = to_open_instruct.run(
        input=Path(args.input), out_dir=Path(args.out_dir),
        source=args.source, batch_size=args.batch_size,
        report=Path(args.report) if args.report else None,
    )
    print(_json.dumps(summary, indent=2, ensure_ascii=False))


def _cmd_topics(args):
    from synthgen.pipeline import topics
    cfg = _build_cfg(args)
    topics.run(cfg, dataset=args.dataset, split=args.split,
               text_field=args.text_field, n=args.n, seed=args.seed,
               model=args.model)


# ----- argparse wiring -----------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="synthgen",
                                 description="Multilingual synthetic IF data generation.")
    ap.add_argument("--version", action="version", version=f"synthgen {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # build-prompts
    p = sub.add_parser("build-prompts", help="Build prompts.jsonl")
    _add_root(p)
    p.add_argument("--phase", type=int, choices=[1, 3], default=3)
    p.add_argument("--n", type=int, default=None, help="examples per language")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_build_prompts)

    # build-localized-prompts
    p = sub.add_parser("build-localized-prompts",
                       help="Build localized_prompts.jsonl (taxonomy-driven, "
                            "one-shot instruction+response generation)")
    _add_root(p)
    p.add_argument("--n", type=int, required=True, help="examples per language")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--langs", nargs="*", default=None,
                   help="Override language codes (default: all LANGUAGES_PHASE3)")
    p.set_defaults(func=_cmd_build_localized_prompts)

    # generate
    p = sub.add_parser("generate",
                       help="Generate via OpenRouter or local vLLM servers")
    _add_root(p)
    p.add_argument("--phase", type=int, choices=[1, 3], default=3)
    p.add_argument("--backend", choices=["openrouter", "vllm"], default="openrouter",
                   help="Generation route (default: openrouter)")
    p.add_argument("--mode", choices=["prompt", "response", "localized", "judge"],
                   default="prompt",
                   help="prompt: meta→generated_prompt; response: generated_prompt→SFT "
                        "response; localized: meta→{instruction,response} in one call; "
                        "judge: {instruction,response}→{score,reason} (requires --in)")
    p.add_argument("--models", nargs="*", default=None,
                   help="Override the generator list (default: cfg.generators).")
    p.add_argument("--in", dest="in_path", default=None,
                   help="Override input JSONL (default: derived from mode + model).")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="Override max_tokens (defaults: 1024 prompt, 1536 response).")
    p.add_argument("--max-attempts", type=int, default=1,
                   help="If the run finishes short of the target, re-run the missing "
                        "rows up to this many times (default: 1, no retry).")
    p.add_argument("--retry-delay", type=float, default=10.0,
                   help="Seconds to wait between retry attempts (default: 10).")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--split", dest="split", action="store_true", default=None,
                     help="Round-robin partition rows across models for diversity.")
    grp.add_argument("--no-split", dest="split", action="store_false",
                     help="Force every row through every model (head-to-head A/B).")
    # vLLM-specific:
    p.add_argument("--endpoints-dir", default=None,
                   help="(vllm) directory of *.endpoint files (host:port per file)")
    p.add_argument("--min-endpoints", type=int, default=1,
                   help="(vllm) wait until at least N servers are ready")
    p.set_defaults(func=_cmd_generate)

    # filter
    p = sub.add_parser("filter", help="Filter leakage in gen_*.jsonl")
    _add_root(p)
    p.add_argument("--gen", nargs="*", default=None)
    p.set_defaults(func=_cmd_filter)

    # verify
    p = sub.add_parser("verify", help="Yield report: lang-id + constraint heuristics")
    _add_root(p)
    p.add_argument("--gen", nargs="*", default=None)
    p.set_defaults(func=_cmd_verify)

    # back-translate
    p = sub.add_parser("back-translate",
                       help="Back-translate prompts to English (OpenRouter); emit review TSV")
    _add_root(p)
    p.add_argument("--gen", nargs="*", default=None)
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default=None,
                   help="OpenRouter model id (default: cfg.back_translator)")
    p.set_defaults(func=_cmd_back_translate)

    # dedupe
    p = sub.add_parser("dedupe", help="Dedupe an SFT JSONL by (lang, prompt)")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-response-chars", type=int, default=8)
    p.add_argument("--report", default=None)
    p.add_argument("--embedding-dedupe", action="store_true",
                   help="also drop paraphrased near-duplicates within each "
                        "language via sentence-embedding cosine similarity "
                        "(catches what exact-match dedup can't; needs "
                        "sentence-transformers, skipped with a warning if "
                        "unavailable)")
    p.add_argument("--embedding-threshold", type=float, default=None,
                   help="cosine-similarity cutoff for --embedding-dedupe "
                        "(default: near_dup.DEFAULT_THRESHOLD, 0.85)")
    p.add_argument("--embedding-model", default=None,
                   help="sentence-transformers model id for --embedding-dedupe "
                        "(default: near_dup.DEFAULT_EMBED_MODEL, a multilingual "
                        "model -- don't swap in an English-only one, see "
                        "near_dup.py)")
    p.set_defaults(func=_cmd_dedupe)

    # qc
    p = sub.add_parser("qc", help="Per-language QC stats over an SFT JSONL")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=_cmd_qc)

    # quality-filter
    p = sub.add_parser("quality-filter",
                       help="Keep rows scoring >= threshold[lang] (localized+judge output)")
    p.add_argument("--gen", required=True, help="outputs/loc_<model>.jsonl")
    p.add_argument("--judged", required=True, help="outputs/judged_<judge_model>.jsonl")
    p.add_argument("--thresholds", required=True,
                   help="JSON file: {\"global\": x} or {lang: x, ...}")
    p.add_argument("--output", required=True)
    p.add_argument("--report", default=None)
    p.set_defaults(func=_cmd_quality_filter)

    # decide-thresholds
    p = sub.add_parser("decide-thresholds",
                       help="Global-vs-per-language threshold from a judge-validation "
                            "pilot's records.jsonl")
    p.add_argument("--records", required=True,
                   help="studies/localized_bootstrap outputs*/records.jsonl")
    p.add_argument("--output", required=True, help="thresholds.json to write")
    p.add_argument("--label", default="good",
                   help="which label's scores to calibrate on (default: good)")
    p.add_argument("--percentile", type=float, default=0.10,
                   help="lower-tail quantile of that label's scores (default: 0.10)")
    p.add_argument("--spread-tolerance", type=float, default=1.5,
                   help="max(cut)-min(cut) across languages below which one global "
                        "threshold is used instead of per-language (default: 1.5)")
    p.add_argument("--report", default=None, help="full decision (mode/spread/cuts) JSON")
    p.set_defaults(func=_cmd_decide_thresholds)

    # topup
    p = sub.add_parser("topup",
                       help="Resumable loop: generate -> judge -> quality-filter -> "
                            "dedupe until every language hits --target survivors")
    _add_root(p)
    p.add_argument("--target", type=int, required=True, help="survivors per language")
    p.add_argument("--thresholds", required=True, help="JSON from decide-thresholds")
    p.add_argument("--langs", nargs="*", default=None,
                   help="Override language codes (default: all LANGUAGES_PHASE3)")
    p.add_argument("--gen-backend", choices=["openrouter", "vllm"], default="openrouter")
    p.add_argument("--gen-models", nargs="*", default=None,
                   help="Override the generator list (default: cfg.generators)")
    p.add_argument("--gen-endpoints-dir", default=None,
                   help="(vllm) directory of *.endpoint files for the generator(s)")
    p.add_argument("--gen-min-endpoints", type=int, default=1)
    p.add_argument("--judge-backend", choices=["openrouter", "vllm"], default="openrouter")
    p.add_argument("--judge-model", required=True)
    p.add_argument("--judge-endpoints-dir", default=None,
                   help="(vllm) directory of *.endpoint files for the judge")
    p.add_argument("--judge-min-endpoints", type=int, default=1)
    p.add_argument("--overgen-factor", type=float, default=1.6,
                   help="Initial raw-rows-per-needed-survivor multiplier, before "
                        "the observed survival rate takes over (default: 1.6)")
    p.add_argument("--max-rounds", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-attempts", type=int, default=3,
                   help="Per-round retry attempts for generate/judge (default: 3)")
    p.add_argument("--retry-delay", type=float, default=15.0)
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--split", dest="split", action="store_true", default=None,
                     help="Round-robin new rows across generator models.")
    grp.add_argument("--no-split", dest="split", action="store_false")
    p.add_argument("--embedding-dedupe", action="store_true",
                   help="also drop paraphrased near-duplicates each round via "
                        "sentence-embedding cosine similarity, and let the "
                        "resulting deficit trigger real replacement generation "
                        "-- see near_dup.py and topup.run()'s docstring. Costs "
                        "extra generate/judge calls proportional to whatever "
                        "collapse rate this actually finds; measure with an "
                        "ablation before enabling on a full production run.")
    p.add_argument("--embedding-threshold", type=float, default=None,
                   help="cosine-similarity cutoff for --embedding-dedupe "
                        "(default: near_dup.DEFAULT_THRESHOLD, 0.85)")
    p.add_argument("--embedding-model", default=None,
                   help="sentence-transformers model id for --embedding-dedupe "
                        "(default: near_dup.DEFAULT_EMBED_MODEL, multilingual)")
    p.add_argument("--gen-concurrency", type=int, default=None,
                   help="override --concurrency for just the gen-phase call. "
                        "Needed when the gen endpoint pool has many more "
                        "replicas than the judge pool -- EndpointPool round-"
                        "robins evenly, so one shared concurrency value under-"
                        "saturates a big pool or oversaturates a small one.")
    p.add_argument("--judge-concurrency", type=int, default=None,
                   help="override --concurrency for just the judge-phase call "
                        "(see --gen-concurrency).")
    p.set_defaults(func=_cmd_topup)

    # export-openinstruct
    p = sub.add_parser("export-openinstruct",
                       help="Export SFT JSONL into open-instruct per-language parquet source")
    p.add_argument("--input", required=True, help="SFT JSONL (e.g. sft_full.dedup.jsonl)")
    p.add_argument("--out-dir", required=True,
                   help="by_language dir; writes <out-dir>/<source>/<lang>.parquet")
    p.add_argument("--source", default="synthgen-if", help="source name (subdir)")
    p.add_argument("--batch-size", type=int, default=20_000)
    p.add_argument("--report", default=None)
    p.set_defaults(func=_cmd_export_openinstruct)

    # topics
    p = sub.add_parser("topics",
                       help="Estimate empirical topic distribution from a HF dataset")
    _add_root(p)
    p.add_argument("--dataset", default="allenai/WildChat-1M")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default=None)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="openai/gpt-4.1-nano",
                   help="Classifier model id (OpenRouter)")
    p.set_defaults(func=_cmd_topics)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
