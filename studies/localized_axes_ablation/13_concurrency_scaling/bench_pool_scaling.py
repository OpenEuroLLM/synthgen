"""Multi-node concurrency-scaling smoke test: does throughput actually scale
with replica count when concurrency is raised proportionally, going through
the REAL synthgen.backends.VLLMBackend/EndpointPool code path (not raw HTTP
like studies/localized_bootstrap/bench_throughput.py) -- directly testing the
claim that a single flat `concurrency` value dilutes across however many
`*.endpoint` files are registered in a pool (see synthgen/localized/topup.py's
gen_concurrency/judge_concurrency docstring, added this session).

Sweeps concurrency at a few multiples of the single-replica saturation point
(968 tok/s gen / 795 tok/s judge, per studies/localized_bootstrap's earlier
benchmark) against however many replicas are actually registered in each
pool, and reports achieved aggregate tok/s -- if pooling+concurrency scaling
works as expected, throughput at concurrency=N_replicas*256 should approach
N_replicas*968 (gen) / N_replicas*795 (judge), not plateau at the single-
replica ceiling.

  python bench_pool_scaling.py --gen-endpoints $EP/gen --judge-endpoints $EP/judge
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from synthgen.backends import VLLMBackend  # noqa: E402

PROMPT = (  # same ~90-token prompt as the single-endpoint benchmark, for
            # direct comparability of per-replica numbers
    "I'm planning a week-long trip and need help. Please write a detailed, well-"
    "organized guide covering local food, transport, budgeting, and etiquette, "
    "with specific named recommendations and clear step-by-step advice. Explain "
    "your reasoning and keep it practical for a first-time visitor.")

SINGLE_REPLICA_PEAK = {"gen": 968.0, "judge": 795.0}  # tok/s, from bench_throughput.py


async def one_call(backend, client, n_tokens: int) -> int:
    res = await backend.chat(
        client, messages=[{"role": "user", "content": PROMPT}],
        sampling={"temperature": 0.0, "max_tokens": n_tokens, "min_tokens": n_tokens,
                 "ignore_eos": True})
    usage = res.get("usage") or {}
    return usage.get("completion_tokens", n_tokens)


async def sweep(label: str, backend: VLLMBackend, n_replicas: int, n_tokens: int = 512):
    print(f"\n# {label}: {n_replicas} replica(s) registered")
    concurrency_levels = [n_replicas * c for c in (8, 32, 128, 256)]
    peak = 0.0
    # See synthgen/pipeline/generate.py's _generate_once: httpx.AsyncClient()'s
    # default Limits caps real concurrent connections at 100, silently
    # overriding any higher concurrency this sweep requests -- this is the
    # bug the first run of this exact script (job 21832482) surfaced, sized
    # here to the highest level actually tested below.
    max_c = max(concurrency_levels)
    limits = httpx.Limits(max_connections=max_c + 10, max_keepalive_connections=max_c)
    async with httpx.AsyncClient(timeout=1200, limits=limits) as client:
        await backend.ensure_ready()
        # warmup
        await asyncio.gather(*(one_call(backend, client, n_tokens) for _ in range(min(8, n_replicas * 8))))
        for c in concurrency_levels:
            t0 = time.time()
            toks = await asyncio.gather(*(one_call(backend, client, n_tokens) for _ in range(c)))
            el = time.time() - t0
            total = sum(toks)
            tps = total / el
            peak = max(peak, tps)
            expected = SINGLE_REPLICA_PEAK.get(label, 0) * n_replicas
            print(f"  concurrency={c:5d}  tok/s={tps:8.0f}  "
                 f"(expected ceiling ~{expected:.0f} at {n_replicas}x replicas)  "
                 f"sec={el:.1f}")
    print(f"# {label} PEAK: {peak:.0f} tok/s "
         f"({peak / max(1, SINGLE_REPLICA_PEAK.get(label, 1)):.2f}x single-replica peak, "
         f"{n_replicas} replicas registered)")
    return peak


def count_endpoints(dir_: str) -> int:
    return len(list(Path(dir_).glob("*.endpoint")))


async def main_async(args):
    n_gen = count_endpoints(args.gen_endpoints)
    n_judge = count_endpoints(args.judge_endpoints)
    print(f"registered endpoints: gen={n_gen}  judge={n_judge}")

    gen_backend = VLLMBackend(model=args.gen_model, endpoints_dir=args.gen_endpoints)
    judge_backend = VLLMBackend(model=args.judge_model, endpoints_dir=args.judge_endpoints)

    gen_peak = await sweep("gen", gen_backend, n_gen)
    judge_peak = await sweep("judge", judge_backend, n_judge)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({
        "n_gen_replicas": n_gen, "n_judge_replicas": n_judge,
        "gen_peak_tok_s": gen_peak, "judge_peak_tok_s": judge_peak,
        "gen_scaling_factor": gen_peak / SINGLE_REPLICA_PEAK["gen"],
        "judge_scaling_factor": judge_peak / SINGLE_REPLICA_PEAK["judge"],
    }, indent=2))
    print(f"\nwrote {out / 'summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-endpoints", required=True)
    ap.add_argument("--judge-endpoints", required=True)
    ap.add_argument("--gen-model", default="gen")
    ap.add_argument("--judge-model", default="judge")
    ap.add_argument("--out", default=str(Path(__file__).parent / "outputs"))
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
