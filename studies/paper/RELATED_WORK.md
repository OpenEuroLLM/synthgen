# Related work — multilingual synthetic instruction-following data

Scope: prior art for the `synthgen.localized` recipe (taxonomy-conditioned
single-call {instruction, response} generation -> holistic LLM judge ->
per-language thresholds -> resumable top-up loop), for 11 European languages.

Search: 11 queries across Semantic Scholar / OpenAlex / arXiv / OpenReview /
Crossref / DBLP, 2022-2026, via `.claude/ResearchStudio-Idea/skills/paper_search`.
Raw output in `raw_search/`, fetched abstracts in `arxiv_abs.json`.
Full-text deep dive on 6 papers (PDFs read, not just abstracts).

## 1. How multilingual IF data gets built today — four families

### (a) Translate English IF data (the baseline the paper must beat)
- **Bactrian-X** (Li et al., 2023) — Alpaca + Dolly MT'd into 3.4M pairs, 52 languages.
- **mAlpaca** (Chen et al., 2024) — translated Alpaca.
- **Aya Collection** (Singh et al., ACL 2024) — 513M instances / 114 languages via
  templating + N-way NLLB-3.3B translation, on top of the human-authored **Aya Dataset**
  (65 languages).
- Known failure modes, all documented: translationese, propagated MT errors, and
  Anglocentric *content* that no amount of MT quality fixes.

### (b) Repair or re-localize translated data
- **MIDB** (arXiv 2505.17671, AAAI 2026) — a *booster* model trained on ~36.8k
  human-expert revisions across 16 languages that rewrites MT'd synthetic instruction
  pairs to fix content errors, MT defects, and missing localization. Frames the problem
  as "cultural inequality" in trained LLMs. Post-hoc repair of MT output, not native
  generation.
- **CoachLM** (ICDE 2024) — automatic instruction revision for data quality (English-first).

### (c) Ground generation in target-language text (the dominant "native" route)
- **UPDESH** (arXiv 2509.21294, ACL 2026) — the closest prior work. Bottom-up synthesis
  with >=235B open models grounded in **language-specific Wikipedia**, 9.5M points across
  13 Indic languages. Two subsets: a *reasoning* subset that is still translated
  (Orca-AgentInstruct / OrcaMath via Llama-3.1-405B), and a *generative* subset built
  instruction-backtranslation-style from Wikipedia pages, with India-specific cultural
  artifacts harvested by traversing `Category:Culture of India` 2-3 levels deep.
  Production filtering is **heuristics only** (IndicLID confidence > 0.75, word-repetition
  ratio < 0.75); LLM-as-judge (GPT-4o) is used for *post-hoc quality analysis* on a 2K
  stratified sample against 10K human ratings — and the paper explicitly reports that
  judge-human agreement collapses on the culturally nuanced dimensions
  (linguistic plausibility 45.6%, persona adherence 51.4%, repetitiveness 52.8%) while
  holding on objective ones (toxicity 98.4%).
- **UltraLink** (arXiv 2402.04588, ACL 2024) — ~1M samples, En/Zh/Ru/Fr/Es. Separates
  **language-specific** from **language-agnostic** ability; the language-specific half is
  built by Wikipedia knowledge-grounded augmentation, the language-agnostic half is pruned
  because cross-lingual transfer makes repetition unnecessary. Rule-based filter.
- **Seed-Free Thai** (arXiv 2411.15484, ACL SRW 2024) — seed-data-free framework naming
  three properties: fluency, diversity, cultural context. LLM generates **general topics
  and cultural topics separately**, retrieves a Wikipedia section (or writes its own
  context in one of 13 styles), then generates instructions for 4 task types directly in
  Thai; embedding-similarity diversity filter. 5k rows match Thai SOTA trained on 10-100x more.
- **MURI** (TACL 2025) — reverse instructions from existing human-written native texts,
  2M pairs / 200 languages, no human annotators and no multilingual model required.
- **X-Instruction** (arXiv 2405.19744) — English instruction, target-language response,
  synthesized from native web text; self-curated and diversified, 10 languages.
- **WangchanThaiInstruct** (arXiv 2508.15239) — human-authored Thai IF data; ablations
  isolate the effect of native supervision, and models tuned on it beat translated-data
  models in- and out-of-domain.

### (d) Seed-and-evolve / self-synthesis (mostly English, some multilingual)
- **Self-Instruct** (ACL 2023), **Alpaca**, **Evol-Instruct / WizardLM** (ICLR 2024),
  **Magpie** (ICLR 2025, self-synthesis from an aligned model's template prefix with
  quality/difficulty tagging), **OrcaAgentInstruct** (25.8M pairs).
- **M2Lingual** (arXiv 2406.16783) — the multilingual member of this family: a two-step
  **Evol prompt taxonomy** (19 NLP tasks x 9 conditions + 21 dialogue variations) applied
  to multilingual seeds drawn from Aya; 182K pairs, 70 languages, 17+ tasks. Post-hoc
  n-gram repetition filter; GPT-4 judge used for *evaluation*, not filtering.
- **COIG-CQIA** (NAACL 2024) — Chinese; "quality is all you need", real-world sources +
  rigorous human verification, explicitly rejecting English-distilled data as misaligned
  with native interaction patterns.

## 2. Does native data actually beat translated data?
- **"Is It Good Data for Multilingual IT or Just Bad Multilingual Evaluation?"**
  (EMNLP 2024) — controlled native-vs-translated at *both* the tuning and evaluation
  stage. Key methodological warning for our eval design: only **native or generative**
  benchmarks reveal the native/translated gap; translated test sets hide it, especially
  when model performance is high.
- **"A Fair Comparison without Translationese"** (2025) — English vs target-language
  instructions, controlling for translationese.
- **MedInjection-FR** (LREC 2026) — native vs synthetic vs translated in French biomedical IT.
- **"Multilingual Instruction Tuning With Just a Pinch of Multilinguality"** (2024) —
  how little multilingual data is needed for cross-lingual instruction transfer; the
  natural "how much is enough" counterweight to any large-scale generation claim.
- **"Multilingual Pretraining and IT Improve Cross-Lingual Knowledge Alignment, But Only
  Shallowly"** (NAACL 2024).

## 3. European-language context
- **EuroLLM** (2024; EuroLLM-22B tech report 2026), **Teuken-7B** (2024), **Lucie-7B**
  (2025), **MiniLingua** (2025) — European multilingual models; SFT mixes are largely
  translated or aggregated, not natively synthesized with cultural conditioning.
- **The PLLuM Instruction Corpus** (arXiv 2511.17161) — Polish; a functional typology of
  *organic vs converted vs synthetic* instructions, with observations on human-authored
  vs synthetic data in linguistic adaptation. The nearest European analogue, but
  single-language and typology-first rather than a generation recipe.
- **"Synthetic Instruction Generation for Low-Resource Nordic Languages"** (LREC 2026) —
  viability and limitations of exactly this move in a European setting.
- **"Building a Strong Instruction LM for a Less-Resourced Language"** (GaMS3-12B,
  Slovene, 2026) — the full adaptation recipe (CPT + 2-stage SFT) for one EU language.
- **"Improving Romanian LLM Pretraining Data using Diversity and Quality Filtering"**
  (LoResLM 2026).

## 4. Quality filtering and LLM-as-judge
- **"A Survey on LLM-as-a-Judge"** (2024); **"On LLMs-Driven Synthetic Data Generation,
  Curation, and Evaluation: A Survey"** (2024, ~108 cites) — the two framing surveys.
- **"Cross-Lingual Stability of LLM Judges Under Controlled Generation: Evidence from
  Finno-Ugric Languages"** (2026) and **"Checklist Engineering Empowers Multilingual LLM
  Judges"** (2025) — directly relevant to whether a per-language judge threshold is
  defensible; cite alongside UPDESH's negative agreement result.
- **"A Survey on Data Selection for LLM Instruction Tuning"** (2025); **LIMA**;
  **Selective Reflection-Tuning** (ACL Findings 2024); **LEAD** (VLDB 2025).
- **"Call for Rigor in Reporting Quality of Instruction Tuning Data"** (ACL 2025).

## 5. Diversity / mode collapse in synthetic data
- **"Synthetic Eggs in Many Baskets: The Impact of Synthetic Data Diversity on LLM
  Fine-Tuning"** (ACL Findings 2026) — the headline citation for why the salt/persona
  ablations matter.
- **"Measuring Lexical Diversity of Synthetic Data Generated through Fine-Grained Persona
  Prompting"** (EMNLP Findings 2025) — directly relevant to studies 02 and 10.
- **MetaSynth** (ACL 2025), **"Multi-Sample Prompting and Actor-Critic Prompt Optimization
  for Diverse Synthetic Data Generation"** (2025), **"Attributes as Textual Genes"**
  (EMNLP 2025), **BARE** (2025).

## 6. Where the gap is
Every close neighbour that generates natively **retrieves a target-language corpus first**
(UPDESH, UltraLink, Seed-Free Thai, MURI, X-Instruction) — which presupposes a per-language
corpus and biases coverage toward what that corpus contains. Every one of them filters
with heuristics, embeddings, or rules, and uses an LLM judge (when at all) as an
*evaluation instrument* after the fact. None of them treats per-language survivor count as
the thing the pipeline is driven by. That is the space `synthgen.localized` occupies.

## 7. Elo / arena-style evaluation — prior art (checked, not assumed)

Pairwise-preference evaluation with Elo or win rates is **standard practice** in exactly
this literature, including in the closest prior work. Verified in the PDFs:

- **UPDESH** (arXiv:2509.21294) §5.2 "Comparative Cultural evaluations (ELO Rankings)" —
  **91,982 pairwise battles** across seven Phi-4 checkpoints, GPT-4o as the pairwise judge,
  all model pairings evaluated with randomized answer positions to mitigate positional bias,
  Elo computed following Boubdir et al. (2023). Questions were collected as culturally
  grounded, non-academic, community-driven queries (Samiksha collection process) precisely
  because "such [academic] evaluations do not fully capture how useful these models are for
  real-world user queries". Reported table: UPDESH-32K 1696, IndicAlign Cleaned 1659,
  UPDESH 1607, **Bactrian-X Indic 1311, Aya-Collection Indic Sampled 1208**. They also
  correlate Elo against NLU/NLG averages (Fig. 11).
- **MURI** (TACL 2025) — pairwise **win rates** across 21 languages, Command R+ as judge
  (59% vs mT0); further comparisons vs Aya-1 and Llama-3-70B-Instruct.
- **MIDB** (AAAI 2026) — **win rate** = (#win + #tie/2)/N against multiple baselines across
  16 languages.
- **M2Lingual** — GPT-4 judge ratings; cites Chatbot Arena.
- **GaMS3-12B** (Slovene, 2026) — **human** Slovene LLM arena, >60% win rate vs GPT-4o.
- **compar:IA** (2026) — the French Government's LLM arena collecting French-language human
  prompts and preference data. The closest European analogue to an arena resource.

### Methodological literature to cite (and to defend against)
- **Boubdir et al., "Elo Uncovered: Robustness and Best Practices in Language Model
  Evaluation"** (2023) — the paper UPDESH follows; it is about Elo's *fragility* to match
  ordering and pairing. Elo is only meaningful within a fixed model pool.
- **"How Reliable is Multilingual LLM-as-a-Judge?"** (EMNLP Findings 2025).
- **M-RewardBench** (ACL 2024, ~55 cites), **M-Prometheus** (2025), **mR3** (2025) —
  multilingual judge/reward-model reliability.
- **"Cross-Lingual Stability of LLM Judges Under Controlled Generation"** (Finno-Ugric, 2026).
- **"Rating Roulette: Self-Inconsistency in LLM-As-A-Judge Frameworks"** (EMNLP Findings 2025).
- **am-ELO: A Stable Framework for Arena-based LLM Evaluation** (2025);
  **"Confidence and Stability of Global and Pairwise Scores in NLP Evaluation"** (2025).
- **Chatbot Arena** (2024) — the origin of the protocol.

---

# Second pass

## 8. MultiSynt/MT — the European sibling paper (arXiv:2607.00890)

Read in full. **Not a scoop threat** — it is *pre-training* data, not instruction data — but
it is the most important paper in this survey after UPDESH, for three reasons.

**What it is.** An open synthetic *parallel* corpus, ~4.8T target-language tokens across 36
languages, made by translating 100B high-quality Nemotron-CC tokens with TOWER+ and
OPUS-MT/HPLT-MT. Consortium authorship (Helsinki, Turku, Oslo, Charles University, AI Sweden,
IST/Instituto de Telecomunicações, Prompsit, LAION, JSC) — i.e. the HPLT/EuroLLM
neighbourhood. For many medium- and lower-resource European languages this is the largest
openly available pre-training resource. Reference LLMs trained on it reach HPLT 2.0's final
score with ~72% fewer tokens.

**Why it matters to us — finding 1 (§5.3, Norwegian case study).** They train four 1.7B
models differing *only* in Norwegian pre-training data (HPLT 2.0 native, HPLT 3.0 native,
MultiSynt/MT via TOWER+ 9B, via OPUS-MT) across 11 checkpoints:
- **NorIdiom** (idiom completion): both native-data models beat both translated-data models,
  "with a stable gap throughout training", because idiomatic constructions are a long-tail
  target-language phenomenon translation does not generate at native rates.
- **NorCommonsenseQA**: the four curves overlap; translated data catches up and slightly leads
  by 100B tokens — translated data "lacks local cultural anchoring but preserves the
  structured reasoning patterns of the high-quality English source."

Their conclusion — "native data wins on tasks probing culturally or idiomatically local
knowledge; on tasks probing transferable reasoning structure, models trained on MultiSynt/MT
match or exceed those trained on native data" — is **independent European-language evidence
for the two-pool design**, and it converges with UPDESH's reasoning-vs-generative split
(reasoning is translation-safe, culturally grounded generation is not). Three independent
groups now report the same content-dependent boundary. That is strong support for the
*design* and simultaneously conclusive evidence that the design is **not our novelty**.

**Why it matters to us — finding 2 (§5.1, the evaluation blind spot).** Three MT systems that
human raters and the TOWER leaderboard separate cleanly are **indistinguishable** on standard
multilingual benchmarks, because "those benchmarks reward picking the right token under strong
exploitable cues, while the surface fluency of the model's own free-form generations does not
enter the score." A fluency-sensitive **LLM-as-a-judge** protocol (DeepSeek V3.1 comparing
free-form continuations against a native-data baseline, winrate averaged over prompts and 5
languages, sanity-checked by confirming winrate rises monotonically with Qwen-2.5 model size)
recovers the ranking cleanly. This independently corroborates the EMNLP 2024 "Is It Good Data
...?" warning and, crucially, **legitimises judge-based free-form evaluation** as the protocol
that can see what MCQ benchmarks cannot — which is exactly the evaluation our claim needs.

**Positioning.** MultiSynt/MT is the strongest available answer to "why not just translate?"
*at the pre-training layer*, and it answers it honestly: translate for scale and reasoning,
but you still lose idiomatic and cultural anchoring. Our paper is the instruction-tuning-layer
counterpart to that limitation. Cite it as motivation, not as a competitor. Note also its own
stated limitation — the headline gain confounds source-corpus quality with the translation
step — which is the methodological trap our own translated-baseline comparison must avoid.

## 9. Conditioning axes — prior art for the ablations
- **PersonaHub / "Scaling Synthetic Data Creation with 1,000,000,000 Personas"**
  (arXiv:2406.20094, ~454 cites) — the reference point for persona-conditioned synthesis at
  scale. Directly relevant to `02_persona_rate` and `10_persona_specificity`: our finding that
  persona conditioning may be worth far less than `PERSONA_P = 0.5` assumes is a *negative
  result against this paper's premise*, which makes it publishable rather than incidental.
- **#InsTag** (arXiv:2308.07074) — instruction tagging to analyse SFT data diversity/complexity;
  the closest prior art for treating instruction attributes as an explicit analysable axis.
- **Montessori-Instruct** (arXiv:2410.14208), **CRAFT** (arXiv:2409.02098),
  **Web Reconstruction** (arXiv:2504.15573) — alternative conditioning/sourcing strategies.

## 10. Quality filtering — the judge-as-gate lineage
- **AlpaGasus** (arXiv:2307.08701) — the canonical "LLM scores each example, threshold, train
  on the survivors" result. **This is the closest prior art for our filtering step and was
  missing from the first pass — it must be cited.** Our differences: per-*language* calibrated
  thresholds rather than one global cut, and a loop that regenerates to a target rather than a
  one-shot filter.
- **DEITA** (arXiv:2312.15685) — complexity + quality + diversity scoring for data selection.
- **Superfiltering** (arXiv:2402.00530) — a weak model can do the filtering a strong one would;
  relevant to the cost argument for a 27B judge.

## 11. Evaluation suite — what to actually run
Beyond Elo (§7), the multilingual IF benchmark landscape is now well populated and our eval
section should draw from it rather than invent: **M-IFEval** (arXiv:2502.04688),
**Multi-IF** (arXiv:2410.15553), **XIFBench** (arXiv:2503.07539), **MaXIFE**
(arXiv:2506.01776), **IFBench / Generalizing Verifiable Instruction Following**
(arXiv:2507.02833). Note **IndicIFEval** (arXiv:2602.22125) exists for UPDESH's languages and
has no European equivalent at comparable coverage — a gap we could fill or at least name.

For measuring what Pool B is *for*, the cultural-knowledge benchmarks are the right instrument:
**BLEnD** (arXiv:2406.09948, NeurIPS 2024, ~237 cites), **CulturalBench** (arXiv:2410.02677),
**NormAd** (arXiv:2404.12464), **Cultural Adaptation of Recipes** (arXiv:2310.17353).
**SemEval-2026 Task 7** extends BLEnD's everyday-knowledge framing across languages and
cultures, and there is a Swedish extension (RESOURCEFUL 2026) — evidence that European
cultural-knowledge evaluation is being built out right now and worth tracking.

## 12. Post-training context
**Tulu 3** (arXiv:2411.15124) is the reference open post-training recipe and the format our
`export-openinstruct` step targets. **"A Post-trainer's Guide to Multilingual Training Data"**
(arXiv:2504.16677) covers cross-lingual transfer dynamics — directly relevant to how much
per-language data is actually needed, and a natural pairing with "Just a Pinch of
Multilinguality". **HelpSteer3-Preference** (arXiv:2505.11475) is the open human preference
data across languages, relevant if we run human preference collection.
**WildChat** (arXiv:2405.01470) is the source `01_domain_grounding` draws Pool A weights from
— it must be cited there.
