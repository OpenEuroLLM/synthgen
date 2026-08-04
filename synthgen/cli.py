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
    return SynthConfig(
        phase=getattr(args, "phase", 3),
        paths=paths,
    )


def _add_root(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", default=None,
                   help="Project root (default: $SYNTHGEN_ROOT or cwd)")


# ----- subcommand implementations -----------------------------------------

def _cmd_build_prompts(args):
    from synthgen.pipeline import prompts
    prompts.build(_build_cfg(args), seed=args.seed, n_override=args.n)


def _make_backends(args, cfg: SynthConfig):
    """Construct the list of Backend instances per CLI flags."""
    from synthgen.backends import OpenRouterBackend, VLLMBackend
    models = args.models or list(cfg.generators)
    if args.backend == "openrouter":
        return [OpenRouterBackend(model=m, cfg=cfg) for m in models]
    if args.backend == "vllm":
        if not args.endpoints_dir:
            raise SystemExit("--endpoints-dir is required for --backend vllm")
        return [VLLMBackend(model=m, cfg=cfg,
                            endpoints_dir=args.endpoints_dir,
                            min_endpoints=args.min_endpoints)
                for m in models]
    raise SystemExit(f"unknown backend: {args.backend}")


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
    )
    print(_json.dumps(summary, indent=2, ensure_ascii=False))


def _cmd_qc(args):
    from synthgen.pipeline import qc
    qc.run(input=Path(args.input), output=Path(args.output))


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

    # generate
    p = sub.add_parser("generate",
                       help="Generate via OpenRouter or local vLLM servers")
    _add_root(p)
    p.add_argument("--phase", type=int, choices=[1, 3], default=3)
    p.add_argument("--backend", choices=["openrouter", "vllm"], default="openrouter",
                   help="Generation route (default: openrouter)")
    p.add_argument("--mode", choices=["prompt", "response"], default="prompt",
                   help="prompt: meta→generated_prompt; response: generated_prompt→SFT response")
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
    p.set_defaults(func=_cmd_dedupe)

    # qc
    p = sub.add_parser("qc", help="Per-language QC stats over an SFT JSONL")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=_cmd_qc)

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
