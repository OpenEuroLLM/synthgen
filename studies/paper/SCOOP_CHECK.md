# Scoop check — synthgen.localized recipe

**Verdict: Level 3 — Medium Overlap** (2 of 4 axes matched by the closest prior work).
Driven by UPDESH (arXiv 2509.21294) and Seed-Free Thai (arXiv 2411.15484).

## Delta

> Unlike UPDESH, which grounds bottom-up multilingual instruction synthesis in retrieved
> language-specific Wikipedia passages and gates production data on heuristics alone
> (IndicLID confidence + word-repetition ratio), reserving LLM-as-judge for post-hoc
> analysis on a 2K sample, synthgen emits instruction *and* response in a single
> retrieval-free call conditioned on an explicit country x local/general-domain x intent x
> persona taxonomy, and makes a rubric-free holistic judge the production gate — with
> per-language score thresholds driving a resumable top-up loop that keeps generating
> rounds until every language reaches its target survivor count, delivering balanced
> per-language yields for 11 European languages without needing a per-language retrieval
> corpus.

## Decomposed claim

- **Problem framing** — Produce multilingual instruction-following SFT data for 11 European
  languages (es fr de it pt pl nl cs ro el uk) without translating from English; evaluate by
  fine-tuning against translated-IF and multi-source-mixture baselines.
- **Core mechanism** — Single-call `{instruction, response}` generation conditioned on
  (language, representative country, domain drawn from a two-pool local/general taxonomy,
  intent, optional persona at rate p, integer diversity salt); a rubric-free holistic 0-10
  judge scoring *the weaker of* task-worth and response-execution, with explicit
  loanword/transliteration leniency and code-switching penalties; per-language score
  thresholds; a resumable generate->judge->filter->dedupe top-up loop that sizes each round
  from the observed survival rate until a per-language target is met.
- **Key insight** — Translation carries translationese *and* Anglocentric content; explicit
  cultural-domain conditioning manufactures coverage that scraped and translated corpora
  are structurally missing, and a per-language-calibrated judge threshold is what makes
  single-call generation safe at scale. Per-language survival rates differ, so a
  target-driven loop, not a fixed-N pass, is what actually yields balanced data.
- **Application domain** — 11 European languages; open-weight models (Gemma-4-31B generator,
  Qwen3.6-27B judge) served on public HPC; general chat / instruction-following.

## Comparison

- **Proposed work — synthgen.localized**
  - Title: (this work)
  - Date: 2026-09
  - Source: this repository
  - Problem framing: native (non-translated) multilingual IF data for 11 EU languages, evaluated by SFT vs translated + multi-source baselines
  - Core mechanism: retrieval-free taxonomy-conditioned single-call generation + rubric-free holistic judge as production gate + per-language thresholds + target-driven resumable top-up loop
  - Key insight: explicit cultural-domain conditioning manufactures scarce coverage; per-language judge calibration makes single-call generation safe; balanced yield requires a loop, not a pass
  - Application domain: 11 European languages, open-weight models on HPC

- **Prior work A — UPDESH: Synthesizing Grounded Instruction Tuning Data for 13 Indic Languages**
  - Title: UPDESH
  - Date: 2025-09 (ACL 2026)
  - Source: arXiv:2509.21294
  - Problem framing (verified): bottom-up synthetic IF data for 13 Indic languages + English,
    explicitly positioned against "dominant top-down translation-based approaches"; validated
    by fine-tuning Llama-3.1-8B / Phi4-14B across 13 benchmarks — **MATCH**
  - Core mechanism (verified): two subsets. Reasoning = selective *translation* of
    OrcaAgentInstruct/OrcaMath via Llama-3.1-405B. Generative = instruction-backtranslation
    from target-language Wikipedia with Qwen3-235B-A22B, plus 26.8K cultural artifacts
    harvested by traversing `Category:Culture of India` 2-3 levels deep. Production filter is
    two heuristics: "(1) Language Identification using INDICLID with a 0.75 confidence
    threshold, and (2) word repetition ratio capped at 0.75" — **DIFFER**
  - Key insight (verified): "context-aware, culturally grounded data generation is essential";
    reasoning is language-agnostic and so translation-safe, generative is not — **MATCH**
  - Application domain (verified): 13 Indic languages, >=235B open models, 9.5M points — **DIFFER**
  - Assumptions & scope: requires a substantial target-language Wikipedia and a working
    language-ID model per language. Explicitly reports LLM-judge unreliability on culturally
    nuanced axes (agreement 45.6% linguistic plausibility, 51.4% persona adherence, 52.8%
    repetitiveness vs 98.4% toxicity) — which is *why* the judge is not the filter.
  - Closest-passage evidence: Sec. "Data Filtering" — "manual validation was not feasible,
    therefore ... we employed automated quality checks but use the standard threshold-based
    method ... (1) Language Identification using INDICLID ... (2) word repetition ratio capped
    at 0.75"; Sec. 4.2 — "GPT-4O served as an automated evaluator using identical protocols".
  - **Axes matching: 2 -> Level 3 — Medium Overlap** (*two axes differ*)

- **Prior work B — Seed-Free Synthetic Data Generation Framework: A Case Study in Thai**
  - Title: Seed-Free Synthetic Data Generation Framework for Instruction-Tuning LLMs
  - Date: 2024-11 (ACL SRW 2024)
  - Source: arXiv:2411.15484
  - Problem framing (verified): seed-data-free synthetic IF generation for a low-resource
    language, compared against Thai LLMs trained on translated data — **MATCH**
  - Core mechanism (verified): LLM generates General Topics and Cultural Topics from separate
    prompts, then per topic either retrieves a Wikipedia section via MediaWiki API (top-10,
    random pick, split by section) or writes its own context in one of 13 styles; generates
    instructions for 4 fixed task types directly in Thai; embedding-cosine diversity filter.
    No judge, no per-language thresholds, no yield loop — **DIFFER**
  - Key insight (verified): fluency + diversity + cultural context are the three properties
    that make an IF dataset work; 5k rows with all three match SOTA trained on 10-100x more — **MATCH**
  - Application domain (verified): Thai only, Claude-3 Haiku as generator — **DIFFER**
  - Assumptions & scope: single language; the framework's parameters are perturbed to make 5
    datasets, so it is a controlled study of the three properties rather than a production recipe.
  - Closest-passage evidence: Sec. 3.1 — "uses an LLM ... to first randomly generate a given
    number of topics that either are general topics or relate to a specific culture ... we
    search Wikipedia for a related text and then prompt Haiku to generate instructions ...
    The data then goes through a diversity control step".
  - **Axes matching: 2 -> Level 3 — Medium Overlap** (*two axes differ*)
  - **Note:** this paper already owns the general-topics/cultural-topics split. Our two-pool
    design is not novel against it.

- **Prior work C — MIDB: Multilingual Instruction Data Booster**
  - Title: MIDB
  - Date: 2025-05 (AAAI 2026)
  - Source: arXiv:2505.17671
  - Problem framing (verified): repair the quality of *already machine-translated* synthesized
    multilingual instruction data — **DIFFER**
  - Core mechanism (verified): a booster model trained on ~36.8k human-expert revision examples
    across 16 languages, applied at inference to boost content, translation, and localization — **DIFFER**
  - Key insight (verified): MT-based multilingual synthesis compounds English content errors
    with MT defects and insufficient localization, producing "cultural inequality"; cited
    example — an English-derived model claims Spanish is a popular second language in Greece — **MATCH**
  - Application domain (verified): 16 languages including European ones; +19.5% cultural-question
    accuracy across five non-English cultures — **MATCH**
  - Assumptions & scope: presupposes an MT'd corpus to boost and expert-authored revisions.
  - **Axes matching: 2 -> Level 3 — Medium Overlap** (*two axes differ*)

- **Prior work D — UltraLink**
  - Title: UltraLink: An Open-Source Knowledge-Enhanced Multilingual SFT Dataset
  - Date: 2024-02 (ACL 2024)
  - Source: arXiv:2402.04588
  - Problem framing (verified): build a multilingual SFT dataset that beats simple translation — **MATCH**
  - Core mechanism (verified): Wikipedia knowledge-grounded data augmentation for the
    language-specific half; rule-based filter; language-agnostic half pruned by exploiting
    cross-lingual transfer — **DIFFER**
  - Key insight (verified): split language-specific from language-agnostic ability and treat
    them differently — **MATCH** (this is the two-pool idea, in prior art)
  - Application domain (verified): 5 languages (En/Zh/Ru/Fr/Es), ~1M samples — **DIFFER** (only 2 EU)
  - **Axes matching: 2 -> Level 3 — Medium Overlap** (*two axes differ*)

- **Prior work E — MURI**
  - Title: MURI: High-Quality Instruction Tuning Datasets for Low-Resource Languages via Reverse Instructions
  - Date: 2024-09 (TACL 2025)
  - Source: arXiv:2409.12958
  - Problem framing: IF data for low-resource languages without annotators — **MATCH**
  - Core mechanism: reverse instructions over existing human-written native texts + translation
    pipeline; filters for inappropriate content — **DIFFER**
  - Key insight: sourcing from native domains "ensures cultural relevance and diversity" — **MATCH**
  - Application domain: 200 languages, 2M+ pairs, mT5 evaluation — **DIFFER**
  - **Axes matching: 2 -> Level 3 — Medium Overlap** (*two axes differ*)

- **Prior work F — M2Lingual**
  - Title: M2Lingual: Enhancing Multilingual, Multi-Turn Instruction Alignment
  - Date: 2024-06
  - Source: arXiv:2406.16783
  - Problem framing: fully synthetic multilingual IFT dataset beyond high-resource languages — **MATCH**
  - Core mechanism: two-step Evol prompt taxonomy (19 tasks x 9 conditions, 21 dialogue
    variations) applied to Aya-derived seeds; post-hoc n-gram repetition filter — **DIFFER**
  - Key insight: taxonomy-guided *complexity evolution* of seeds, not cultural grounding — **DIFFER**
  - Application domain: 70 languages, 182K pairs, multi-turn — **DIFFER**
  - **Axes matching: 1 -> Level 4 — Low Overlap** (*three axes differ*)

- **Prior work G — Magpie**
  - Title: Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing
  - Date: 2024-06 (ICLR 2025)
  - Source: arXiv:2406.08464
  - Problem framing: English-centric alignment data synthesis — **DIFFER**
  - Core mechanism: self-synthesis from an aligned model's chat-template prefix, then
    quality/difficulty tagging and filtering — **PARTIAL** (LLM-scored filtering) -> counted as match
  - Key insight: an aligned model will emit its own instruction distribution unprompted — **DIFFER**
  - Application domain: English — **DIFFER**
  - **Axes matching: 1 -> Level 4 — Low Overlap** (*three axes differ*)

## Overall verdict

**Level 3 — Medium Overlap.** UPDESH and Seed-Free Thai each match on problem framing and
key insight: both argue that translation-first multilingual IF data is culturally deficient
and that native, culturally-conditioned generation fixes it, and both validate by fine-tuning.
UltraLink and MURI land in the same band for the same reason. What none of them shares is the
core mechanism: every one of these grounds generation in a retrieved target-language corpus,
and every one gates production data on heuristics, rules, or embeddings — with the LLM judge,
where present, used as an after-the-fact evaluation instrument rather than the filter. UPDESH
in particular *measured* LLM-judge/human agreement and found it weak on culturally nuanced
dimensions, which is precisely the assumption our recipe bets against and must therefore
defend with evidence. The delta is defensible but must be stated explicitly in the write-up:
the contribution is retrieval-free axis conditioning + judge-as-gate + yield-targeted top-up,
not "native beats translated" and not the general/local pool split, both of which are prior art.

## Do not claim as novel

| Claim | Owned by |
|---|---|
| Translation loses cultural content / translationese hurts | MIDB, UPDESH, WangchanThaiInstruct, X-Instruction, "Is It Good Data...?" (EMNLP 2024) |
| General vs cultural/local domain split | Seed-Free Thai (general vs cultural topics), UltraLink (language-agnostic vs language-specific) |
| Taxonomy-conditioned synthetic instruction generation | Self-Instruct lineage, M2Lingual (Evol taxonomy) |
| LLM-judge scoring to filter instruction data | Alpagasus, Magpie, CoachLM, "Survey on Data Selection for LLM IT" |
| Seed-free generation | Seed-Free Thai, Magpie |

## Defensible selling points

1. **Retrieval-free grounding.** Every close neighbour needs a target-language corpus first.
   Ours needs only the axes plus the generator's parametric knowledge of the country. This is
   the sharpest mechanical difference and it is directly testable — `01_domain_grounding`'s
   none/light/deep arms are exactly the experiment, and a Wikipedia-grounded UPDESH-style arm
   is the ablation a reviewer will ask for.
2. **The judge is the production gate, not an evaluation instrument.** Rubric-free, holistic,
   weakest-of-(task, response), with explicit loanword leniency and code-switching penalties,
   calibrated per language. UPDESH's negative agreement result makes this a contested claim —
   which is what makes it worth a paper. The judge-validation pilot plus `07_judge_model_sensitivity`
   are the evidence, and they need human agreement numbers to stand up.
3. **Yield-targeted top-up.** Per-language survival rates differ, so fixed-N generation yields
   imbalanced data. A resumable loop that sizes each round from the observed survival rate
   until each language hits its target is small, unclaimed, and practically the thing that
   makes the recipe usable.
4. **European coverage with open weights on public HPC.** UPDESH uses >=235B models;
   Seed-Free Thai uses Claude-3. Gemma-4-31B + Qwen3.6-27B on LUMI is a reproducibility and
   sovereignty claim that fits OELLM and that no existing EU resource (EuroLLM, Teuken, Lucie,
   PLLuM) currently offers as a culturally-grounded synthetic IF recipe across 11 languages.
5. **The axis ablations.** No prior work ablates persona rate, intent necessity, salt, grounding
   depth, and pool split with *paired* designs on identically-sampled rows. UPDESH offers
   "design considerations" without controlled measurement. This is the honest home for
   `localized_axes_ablation` in the paper.

## Baselines to run

- **Translation:** Bactrian-X, mAlpaca, Aya Collection (translated portion) for the target languages.
- **Multi-source mixtures:** Aya Dataset+Collection, MURI-IT, M2Lingual, UltraLink; PLLuM for Polish.
- **Method baselines re-run in-language (strongest reviewer ask):** Magpie-in-language,
  Self-Instruct-in-language, and a UPDESH/Seed-Free-style Wikipedia-grounded arm — the last one
  isolates exactly the retrieval-free claim.
- **Evaluation caution:** per "Is It Good Data for Multilingual IT or Just Bad Multilingual
  Evaluation?" (EMNLP 2024), translated test sets systematically hide the native/translated gap.
  Use native and generative benchmarks, not translated multiple-choice, or the headline result
  will not appear.

---

## Addendum — Elo / arena evaluation is NOT a differentiator

Checked directly in the PDFs rather than assumed. **UPDESH already does this, in the
configuration we were planning.** §5.2 "Comparative Cultural evaluations (ELO Rankings)":
91,982 pairwise battles across seven checkpoints, GPT-4o as pairwise judge, all pairings
with randomized answer positions for positional-bias control, Elo per Boubdir et al. (2023),
on deliberately culturally-grounded community queries — scored against **Bactrian-X (1311)
and Aya-Collection (1208)**, the same translated/multi-source baselines we plan to use.
MURI reports pairwise win rates across 21 languages; MIDB across 16; GaMS3-12B uses a human
Slovene arena. Elo does not move the novelty verdict off Level 3.

This is not bad news for the eval design — it makes our numbers *comparable* — but it must
not be written up as a contribution. Three consequences:

1. **Circularity risk, and it is serious.** Our headline mechanism is an LLM judge as the
   production *gate*. If a similar LLM judge is also the *arbiter* at evaluation time, a
   reviewer will say we optimized the training filter against the eval metric. Mitigations,
   in descending order of strength: native-speaker human raters per language; a judge from a
   different model family than the filter judge (and never the generator's own family);
   report filter-judge/eval-judge/human agreement explicitly. At minimum the eval judge must
   differ from `Qwen3.6-27B`.
2. **Per-language Elo with confidence intervals is the underused move.** Everyone above
   pools battles or reports a single aggregate table. Our entire thesis is *per-language*
   balance, so per-language Elo (with CIs, and enough battles per language to support them)
   is both the evaluation our claim actually needs and something the prior work does not do.
3. **Elo is fragile.** Boubdir et al. — the very paper UPDESH follows — is about Elo's
   sensitivity to match ordering and pairing, and Elo is only interpretable within a fixed
   model pool. Report the pool, the battle count per language, positional-bias controls, and
   stability under re-ordering, or the numbers will not survive review.

The genuine evaluation-side delta, if we want one, is **whose preferences and which queries**
— native-speaker human preference per language, versus UPDESH's automated GPT-4o arbiter,
which is notable because UPDESH's own §4.2 reports LLM-judge/human agreement collapsing on
culturally nuanced dimensions (45.6% linguistic plausibility, 51.4% persona adherence). Using
an automated judge to rank *cultural* helpfulness while reporting that automated judges are
unreliable on exactly that is the soft spot in the closest prior work. That is worth taking,
but it is an evaluation-quality argument, not a novelty claim.
