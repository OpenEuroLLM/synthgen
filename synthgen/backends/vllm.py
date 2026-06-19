"""vLLM backend — fan out across a directory of local OpenAI-compatible servers.

Each vLLM server registers itself by writing `host:port` to
`<endpoints_dir>/<some_name>.endpoint`. The pool re-scans the directory every
~30s, so servers that come up after the client started are absorbed without a
restart, and dead endpoints can be removed by deleting their file.

This is the same wire format as `pilot/vllm/generate_vllm.py` — it talks
`/v1/chat/completions` to plain HTTP, no auth.
"""
from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path

import httpx

from synthgen.backends.base import Backend, GenResult
from synthgen.log import get_logger

log = get_logger("synthgen.vllm")


class EndpointPool:
    """Round-robin pool over `<dir>/*.endpoint` files. Re-reads periodically."""

    def __init__(self, endpoints_dir: Path, refresh_s: float = 30.0) -> None:
        self.dir = Path(endpoints_dir)
        self.refresh_s = refresh_s
        self._endpoints: list[str] = []
        self._cursor = 0
        self._lock = asyncio.Lock()
        self._last_refresh = 0.0

    def _read_dir(self) -> list[str]:
        eps: list[str] = []
        for p in sorted(self.dir.glob("*.endpoint")):
            try:
                ep = p.read_text().strip()
                if ep:
                    eps.append(ep)
            except Exception:
                continue
        return eps

    async def get(self) -> str:
        async with self._lock:
            now = time.time()
            if not self._endpoints or now - self._last_refresh > self.refresh_s:
                new = self._read_dir()
                if new:
                    self._endpoints = new
                    self._last_refresh = now
            if not self._endpoints:
                raise RuntimeError(f"no endpoints in {self.dir}")
            ep = self._endpoints[self._cursor % len(self._endpoints)]
            self._cursor += 1
            return ep

    async def wait_until_ready(self, min_n: int = 1, timeout_s: float = 600.0) -> None:
        t0 = time.time()
        while True:
            eps = self._read_dir()
            if len(eps) >= min_n:
                self._endpoints = eps
                self._last_refresh = time.time()
                log.info("[pool] %d endpoints ready: %s%s",
                         len(eps), eps[:3], "..." if len(eps) > 3 else "")
                return
            if time.time() - t0 > timeout_s:
                raise SystemExit(f"timed out waiting for endpoints in {self.dir}")
            await asyncio.sleep(5.0)


class VLLMBackend(Backend):
    """vLLM backend talking OpenAI chat-completions to a local server pool."""

    name = "vllm"

    def __init__(
        self,
        *,
        model: str,
        endpoints_dir: str | Path,
        cfg=None,
        refresh_s: float = 30.0,
        max_retries: int = 5,
        retry_base_delay_s: float = 2.0,
        timeout_s: float = 600.0,
        min_endpoints: int = 1,
        wait_timeout_s: float = 600.0,
    ) -> None:
        self.model = model
        self.pool = EndpointPool(Path(endpoints_dir), refresh_s=refresh_s)
        self.max_retries = cfg.max_retries if cfg else max_retries
        self.retry_base_delay_s = cfg.retry_base_delay_s if cfg else retry_base_delay_s
        self.timeout_s = timeout_s
        self.min_endpoints = min_endpoints
        self.wait_timeout_s = wait_timeout_s
        self._ready = False

    async def ensure_ready(self) -> None:
        if not self._ready:
            await self.pool.wait_until_ready(
                min_n=self.min_endpoints, timeout_s=self.wait_timeout_s)
            self._ready = True

    async def chat(self, client: httpx.AsyncClient, *,
                   messages: list[dict], sampling: dict) -> GenResult:
        await self.ensure_ready()
        delay = self.retry_base_delay_s
        last_err: Exception | None = None
        for _ in range(self.max_retries):
            try:
                ep = await self.pool.get()
            except Exception as e:
                last_err = e
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)
                continue
            url = f"http://{ep}/v1/chat/completions"
            payload = {"model": self.model, "messages": messages, **sampling}
            try:
                r = await client.post(url, json=payload,
                                      headers={"Content-Type": "application/json"},
                                      timeout=self.timeout_s)
                if r.status_code == 429 or r.status_code >= 500:
                    raise RuntimeError(f"http {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                data = r.json()
                return {
                    "content": data["choices"][0]["message"]["content"],
                    "usage": data.get("usage", {}),
                    "error": None,
                }
            except Exception as e:
                last_err = e
                await asyncio.sleep(delay + random.random() * 0.5)
                delay = min(delay * 2, 30.0)
        return {"content": None, "usage": None, "error": str(last_err)}
