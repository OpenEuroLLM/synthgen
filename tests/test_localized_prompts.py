"""synthgen.localized.prompts: generation_prompt()'s four optional layers,
build_rows()'s weighted domain draw, and the removal of sampled intent/salt
from the prompt text (studies/localized_axes_ablation/'s findings -- see
prompts.py's module docstring)."""
from __future__ import annotations

from synthgen.localized import prompts as P
from synthgen.localized import taxonomy as T


# ----- generation_prompt(): the four optional layers ------------------------

def test_generation_prompt_has_all_four_optional_layers():
    text = " ".join(P.generation_prompt(lang_name="French", country="France",
                                        domain="workplace email & communication").split())
    assert "more specific, non-obvious general-capability angle" in text
    assert "genuine France-specific angle" in text
    assert "editing/rewriting existing text" in text
    assert "higher-difficulty version of this task" in text


def test_generation_prompt_no_suggested_intent():
    text = P.generation_prompt(lang_name="French", country="France", domain="coding")
    assert "suggested" not in text.lower()
    assert "decide a realistic user intent" in text


def test_generation_prompt_no_salt_or_diversity_seed():
    text = P.generation_prompt(lang_name="French", country="France", domain="coding")
    assert "salt" not in text.lower()
    assert "diversity seed" not in text.lower()


def test_generation_prompt_persona_on_and_off():
    off = P.generation_prompt(lang_name="Polish", country="Poland", domain="coding")
    assert "No persona" in off

    on = P.generation_prompt(lang_name="Polish", country="Poland", domain="coding",
                             role="university student")
    assert "a university student" in on


def test_generation_prompt_never_forces_locality_binary():
    # Same call shape regardless of "topic-ish" vs "culture-ish" domain name --
    # there's no is_local/localized parameter left to even branch on.
    for domain in ("coding", "food & drink", "short stories & fiction"):
        text = P.generation_prompt(lang_name="German", country="Germany", domain=domain)
        assert "genuine Germany-specific angle" in text
        assert domain in text


# ----- build_rows(): weighted domain draw, no is_local/intent field ---------

def test_build_rows_output_schema_has_no_is_local_or_sampled_intent():
    rows = P.build_rows(5, "es", seed=0)
    for r in rows:
        assert "is_local" not in r
        assert "intent" not in r
        assert set(r) == {"id", "lang", "domain", "role", "salt", "meta_prompt"}


def test_build_rows_domain_always_from_taxonomy():
    rows = P.build_rows(50, "fr", seed=0)
    domains = {r["domain"] for r in rows}
    assert domains <= set(T.DOMAINS)


def test_build_rows_weighted_domain_draw_respects_domain_weights(monkeypatch):
    # Heavily weight one domain -- with enough draws it should dominate.
    fake_weights = {d: 0.001 for d in T.DOMAINS}
    fake_weights["coding"] = 100.0
    monkeypatch.setattr(T, "DOMAIN_WEIGHTS", fake_weights)

    rows = P.build_rows(200, "es", seed=0)
    coding_share = sum(1 for r in rows if r["domain"] == "coding") / len(rows)
    assert coding_share > 0.8


def test_build_rows_uniform_fallback_when_weights_empty(monkeypatch):
    monkeypatch.setattr(T, "DOMAIN_WEIGHTS", {})
    rows = P.build_rows(500, "es", seed=0)
    domains = {r["domain"] for r in rows}
    # every domain should get sampled at least once at n=500 across 18 domains
    # if the fallback is truly uniform (not "always domain[0]" or similar bug)
    assert len(domains) >= len(T.DOMAINS) - 2


def test_build_rows_deterministic_given_same_seed():
    a = P.build_rows(10, "de", seed=0)
    b = P.build_rows(10, "de", seed=0)
    assert [r["domain"] for r in a] == [r["domain"] for r in b]
    assert [r["role"] for r in a] == [r["role"] for r in b]


def test_build_rows_persona_p_respected():
    rows_low = P.build_rows(300, "es", seed=0, persona_p=0.0)
    assert all(r["role"] is None for r in rows_low)

    rows_high = P.build_rows(300, "es", seed=0, persona_p=1.0)
    assert all(r["role"] is not None for r in rows_high)
