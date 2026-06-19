"""OpenRouter backend (hosted models, single endpoint, API key)."""
from __future__ import annotations

import asyncio
import os

import httpx

from synthgen.backends.base import Backend, GenResult


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def require_openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set")
    return key


class OpenRouterBackend(Backend):
    """Async OpenRouter chat-completions client with retry/backoff."""

    name = "openrouter"

    def __init__(
        self,
        *,
        model: str,
        cfg=None,
        api_key: str | None = None,
        url: str = OPENROUTER_URL,
        max_retries: int = 5,
        retry_base_delay_s: float = 2.0,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        self.url = url
        self.max_retries = cfg.max_retries if cfg else max_retries
        self.retry_base_delay_s = cfg.retry_base_delay_s if cfg else retry_base_delay_s
        self.timeout_s = timeout_s
        self.api_key = api_key or require_openrouter_key()

    async def chat(self, client: httpx.AsyncClient, *,
                   messages: list[dict], sampling: dict) -> GenResult:
        payload = {"model": self.model, "messages": messages, **sampling}
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        delay = self.retry_base_delay_s
        last_err: Exception | None = None
        for _ in range(self.max_retries):
            try:
                r = await client.post(self.url, json=payload, headers=headers,
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
                await asyncio.sleep(delay)
                delay *= 2
        return {"content": None, "usage": None, "error": str(last_err)}
