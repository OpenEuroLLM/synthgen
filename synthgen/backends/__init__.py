"""Generation backends.

A `Backend` wraps a remote chat-completions API (OpenRouter, local vLLM, …)
behind a single async `chat(...)` method. The rest of the pipeline is backend-
agnostic.
"""
from synthgen.backends.base import Backend, GenResult
from synthgen.backends.openrouter import OpenRouterBackend
from synthgen.backends.vllm import VLLMBackend, EndpointPool

__all__ = [
    "Backend", "GenResult",
    "OpenRouterBackend",
    "VLLMBackend", "EndpointPool",
    "build_backend",
]


def build_backend(name: str, *, model: str, cfg=None, **kwargs) -> Backend:
    """Construct a backend by short name.

    name:
        "openrouter" — OpenRouterBackend
        "vllm"       — VLLMBackend
    """
    name = name.lower()
    if name == "openrouter":
        return OpenRouterBackend(model=model, cfg=cfg, **kwargs)
    if name == "vllm":
        return VLLMBackend(model=model, cfg=cfg, **kwargs)
    raise ValueError(f"unknown backend: {name!r}")
