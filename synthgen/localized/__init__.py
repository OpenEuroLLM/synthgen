"""Localized instruction-generation pipeline: taxonomy-driven single-call
{instruction, response} generation, LLM-judge scoring, quality filtering, and
the resumable top-up loop that drives per-language counts to a target.

Distinct from the top-level `synthgen.prompts` (topic-based, two-stage
meta_prompt -> generated_prompt -> response) pipeline, which this package does
not replace — both share `synthgen.generate`'s backend-agnostic runner via its
mode= parameter.
"""
