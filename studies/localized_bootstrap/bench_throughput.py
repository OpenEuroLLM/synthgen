"""Saturation throughput benchmark for a local vLLM replica.

Sweeps concurrency and, at each level, fires that many requests that each decode
EXACTLY --max-tokens tokens (ignore_eos), then reports aggregate decode tok/s.
The plateau is the replica's saturated throughput — the number to size a budget
against (the study runs at concurrency 8, well below saturation).

  python bench_throughput.py --endpoints endpoints/gen --model gen
"""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import httpx

PROMPT = (  # ~90-token realistic instruction so prefill is representative
    "I'm planning a week-long trip and need help. Please write a detailed, well-"
    "organized guide covering local food, transport, budgeting, and etiquette, "
    "with specific named recommendations and clear step-by-step advice. Explain "
    "your reasoning and keep it practical for a first-time visitor.")


def endpoint(dir_: str) -> str:
    f = sorted(Path(dir_).glob("*.endpoint"))[0]
    return f.read_text().strip()


async def one(client, url, model, n):
    payload = {"model": model, "messages": [{"role": "user", "content": PROMPT}],
               "max_tokens": n, "min_tokens": n, "ignore_eos": True,
               "temperature": 0.0}
    r = await client.post(url, json=payload, timeout=1200)
    d = r.json()
    return d.get("usage", {}).get("completion_tokens", n)


async def sweep(url, model, levels, n):
    print(f"# {model} @ {url}  (max_tokens={n}, ignore_eos)")
    print(f"# {'concurrency':>11} {'tok/s':>10} {'total_tok':>10} {'sec':>7} {'tok/s/req':>10}")
    async with httpx.AsyncClient() as client:
        await one(client, url, model, 8)  # warmup
        best = 0.0
        for c in levels:
            t0 = time.time()
            toks = await asyncio.gather(*(one(client, url, model, n) for _ in range(c)))
            el = time.time() - t0
            total = sum(toks)
            tps = total / el
            best = max(best, tps)
            print(f"  {c:>11} {tps:>10.0f} {total:>10} {el:>7.1f} {tps / c:>10.0f}")
        print(f"# PEAK sustained decode throughput: {best:.0f} tok/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--levels", default="8,16,32,64,128,256")
    args = ap.parse_args()
    url = f"http://{endpoint(args.endpoints)}/v1/chat/completions"
    levels = [int(x) for x in args.levels.split(",")]
    asyncio.run(sweep(url, args.model, levels, args.max_tokens))


if __name__ == "__main__":
    main()
