"""Backend protocol — the only thing the pipeline knows about a remote LM."""
from __future__ import annotations

from typing import Protocol, TypedDict

import httpx


class GenResult(TypedDict, total=False):
    content: str | None
    usage: dict | None
    error: str | None


class Backend(Protocol):
    """Async chat-completions backend.

    Implementations are responsible for retries, timeouts, and pool management.
    `chat` must always return — never raise — so the caller can record errors
    alongside successes in the JSONL output.
    """

    name: str
    model: str

    async def chat(
        self,
        client: httpx.AsyncClient,
        *,
        messages: list[dict],
        sampling: dict,
    ) -> GenResult:
        ...

    async def aclose(self) -> None:
        """Optional teardown hook (e.g. flush pools)."""
        return None
