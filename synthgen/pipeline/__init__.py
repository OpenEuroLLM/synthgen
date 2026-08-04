"""The original topic-based pipeline stages:

    build-prompts -> generate (--mode prompt) -> filter -> verify
                  -> generate (--mode response)
                  -> dedupe -> qc -> export-openinstruct
                  (back-translate at any point for human review)

`generate` and `dedupe` are also reused by `synthgen.localized` (the
taxonomy-driven, single-call generation + judge pipeline) via their mode=
parameter / plain file-in-file-out interface — nothing here is topic-pipeline
specific except `prompts.py`.
"""
