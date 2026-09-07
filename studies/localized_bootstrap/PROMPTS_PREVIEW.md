# Rendered prompts — judge-validation study (good / medium / bad + judge)

Filled for both languages. 10 good + 10 medium + 10 bad per language. Fluency/mixing norms are decided by the model (no per-language flag); the judge is told to be lenient on loanwords/transliteration.

# ===== English (the United Kingdom) =====

## GOOD generation

```
You are a native speaker and expert writer of English as used in the United Kingdom.
Create ONE high-quality instruction-tuning example: a realistic user
instruction in English and an excellent assistant response.

Localized domain: food & drink
Diversity seed: 4173   (use it to pick a NON-OBVIOUS angle; do not default to
the single most famous example of this domain.)

Silently choose a SPECIFIC sub-aspect of "food & drink" in the United Kingdom. Then decide a
realistic user intent (suggested: ask for a recommendation) — use it only if it fits this domain,
otherwise pick an intent that does.
Suggested persona: a parent — use it ONLY if such a person would plausibly ask about this domain; otherwise adapt to a role who would, or drop the persona. If used, invent a specific realistic situation and goal and let it shape the instruction.

Write the INSTRUCTION so that ALL hold:
  - Native fluency: write the way an educated native speaker of English actually writes, following that language's real usage norms. If native English speakers routinely use English loanwords or technical terms, that is fine; if they do not, keep it fully in English. Avoid translationese, and never switch whole phrases or the instruction itself into English.
  - Realistic & self-contained: something a real person in the United Kingdom would type;
    the user has NOT seen any source text; NOT a textbook/quiz question.
  - Genuinely local & non-trivial: answering well requires knowledge specific to
    the United Kingdom; avoid trivia with an obvious one-word answer.
  - Any constraint must be NATURAL and motivated (a real user would impose it);
    no pointless lexical/format tricks.

Write the RESPONSE so that ALL hold:
  - WRONG-ANSWER GUARD: everything stated must be factually correct. If you are
    not certain of a fact, leave it out — never fabricate.
  - BAD-EXPLANATION GUARD: explanations must be clear, correct, and appropriately
    detailed. No hand-waving, no filler, no padding.
  - Name real, specific entities from the United Kingdom where relevant. No clichés/stereotypes.

META-EVALUATION (do this silently before you output): re-read your example and
confirm — the role/intent/domain fit together naturally; language is consistent
throughout; the instruction is natural and non-trivial; the response is factually
correct and well-explained. Fix any issue.

Report which role and intent you actually used (role "none" if you dropped it).

Output JSON only, no commentary, no code fences:
{"instruction":"...","response":"...","role_used":"...","intent_used":"..."}
```

## MEDIUM generation — slightly_generic

```
You are producing ONE MEDIUM-QUALITY instruction-tuning example in English
(the United Kingdom) — decent and usable, but NOT excellent, because of exactly ONE
modest, non-fatal weakness (below). Everything else should be fine: natural
English, on-topic, coherent. Do NOT make it clearly bad — the weakness must
be subtle, the kind of thing that separates a 6 from a 9.

Localized domain: transport & commuting. Suggested intent: ask how to do something.
No persona: write a direct, standalone request.
Follow the real usage norms of English (natural English loanwords are fine where idiomatic).

Modest weakness to include: The instruction is good, but the RESPONSE, though correct, stays a bit generic — it misses the specific local detail (named dishes, places, schemes, brands) that would make it excellent.

Output JSON only, no commentary, no code fences:
{"instruction":"...","response":"..."}
```

## BAD generation — trivial

```
You are producing ONE deliberately LOW-QUALITY instruction-tuning example in
English (the United Kingdom), of a specific defective kind, so a quality filter can be
tested. Make the defect real, but keep it superficially plausible (as a careless
data pipeline would produce). Do NOT add any note that it is bad.

Defect to inject: A TRIVIAL trivia question with an obvious, universally-known one-word answer (e.g. the capital of a major country, 2+2). Low training value. The response is a bare one-liner.

Output JSON only, no commentary, no code fences:
{"instruction":"...","response":"..."}
```

## JUDGE (holistic 0-10, language-lenient)

```
You are a strict quality filter for multilingual instruction-tuning data. You are
fluent in English and knowledgeable about the United Kingdom. Only the highest-quality
examples should be kept; low-quality data harms training.

Rate the OVERALL QUALITY of the following example as training data, 0 to 10.
Score by the SEVERITY of the single WORST real problem — do NOT count flaws, and
do NOT floor an otherwise-good example for one small blemish.

  9-10  Excellent — fluent, natural, correct, genuinely useful. Tiny nits are ok.
  7-8   Good — solid and usable; only MINOR issues (a little generic, a small
        stylistic slip, one typo, heavy but natural loanwords). KEEP.
  5-6   Fair — usable but clearly imperfect: a shallow explanation, one small
        factual imprecision, or a somewhat basic question. Borderline keep.
  3-4   Weak — a real problem that limits usefulness: a partly-off or thin
        answer, or an unnatural/pointless constraint.
  1-2   Poor — a serious defect: trivial trivia, or a clearly wrong or incoherent
        part of the answer.
  0     Unusable — wrong language or script entirely, or a wholly wrong/empty answer.

IMPORTANT: a single small blemish — a lone typo, one garbled word, natural
loanwords, or phonetic transliteration — should cost only a point or two. NEVER
drop an otherwise-excellent example below 6 for one such blemish. Reserve 0-2 for
genuinely serious defects that undermine the whole example.

LANGUAGE NOTE: judge by the real usage norms of English, which you know better than any fixed rule. If educated English speakers routinely mix in English loanwords/technical terms, do NOT penalize that as code-switching or as spelling/transliteration errors. Phonetic or approximate transliteration is acceptable but rate it somewhat LOWER than clean standard native spelling — a minor reduction, not a 0-1. Be LENIENT on script/loanword/transliteration matters (when unsure whether a word or spelling is acceptable in English, assume it IS and give it the benefit of the doubt); reserve low scores for genuine defects: whole clauses/sentences or the instruction/constraint written in English, or text in the wrong script.

Example (English, the United Kingdom):
Instruction:
<instruction>

Response:
<response>

Output JSON only, no commentary, no code fences:
{"score": <integer 0-10>, "reason": "<one sentence>"}
```

# ===== Hindi (India) =====

## GOOD generation

```
You are a native speaker and expert writer of Hindi as used in India.
Create ONE high-quality instruction-tuning example: a realistic user
instruction in Hindi and an excellent assistant response.

Localized domain: food & drink
Diversity seed: 4173   (use it to pick a NON-OBVIOUS angle; do not default to
the single most famous example of this domain.)

Silently choose a SPECIFIC sub-aspect of "food & drink" in India. Then decide a
realistic user intent (suggested: ask for a recommendation) — use it only if it fits this domain,
otherwise pick an intent that does.
Suggested persona: a parent — use it ONLY if such a person would plausibly ask about this domain; otherwise adapt to a role who would, or drop the persona. If used, invent a specific realistic situation and goal and let it shape the instruction.

Write the INSTRUCTION so that ALL hold:
  - Native fluency: write the way an educated native speaker of Hindi actually writes, following that language's real usage norms. If native Hindi speakers routinely use English loanwords or technical terms, that is fine; if they do not, keep it fully in Hindi. Avoid translationese, and never switch whole phrases or the instruction itself into English.
  - Realistic & self-contained: something a real person in India would type;
    the user has NOT seen any source text; NOT a textbook/quiz question.
  - Genuinely local & non-trivial: answering well requires knowledge specific to
    India; avoid trivia with an obvious one-word answer.
  - Any constraint must be NATURAL and motivated (a real user would impose it);
    no pointless lexical/format tricks.

Write the RESPONSE so that ALL hold:
  - WRONG-ANSWER GUARD: everything stated must be factually correct. If you are
    not certain of a fact, leave it out — never fabricate.
  - BAD-EXPLANATION GUARD: explanations must be clear, correct, and appropriately
    detailed. No hand-waving, no filler, no padding.
  - Name real, specific entities from India where relevant. No clichés/stereotypes.

META-EVALUATION (do this silently before you output): re-read your example and
confirm — the role/intent/domain fit together naturally; language is consistent
throughout; the instruction is natural and non-trivial; the response is factually
correct and well-explained. Fix any issue.

Report which role and intent you actually used (role "none" if you dropped it).

Output JSON only, no commentary, no code fences:
{"instruction":"...","response":"...","role_used":"...","intent_used":"..."}
```

## MEDIUM generation — slightly_generic

```
You are producing ONE MEDIUM-QUALITY instruction-tuning example in Hindi
(India) — decent and usable, but NOT excellent, because of exactly ONE
modest, non-fatal weakness (below). Everything else should be fine: natural
Hindi, on-topic, coherent. Do NOT make it clearly bad — the weakness must
be subtle, the kind of thing that separates a 6 from a 9.

Localized domain: transport & commuting. Suggested intent: ask how to do something.
No persona: write a direct, standalone request.
Follow the real usage norms of Hindi (natural English loanwords are fine where idiomatic).

Modest weakness to include: The instruction is good, but the RESPONSE, though correct, stays a bit generic — it misses the specific local detail (named dishes, places, schemes, brands) that would make it excellent.

Output JSON only, no commentary, no code fences:
{"instruction":"...","response":"..."}
```

## BAD generation — trivial

```
You are producing ONE deliberately LOW-QUALITY instruction-tuning example in
Hindi (India), of a specific defective kind, so a quality filter can be
tested. Make the defect real, but keep it superficially plausible (as a careless
data pipeline would produce). Do NOT add any note that it is bad.

Defect to inject: A TRIVIAL trivia question with an obvious, universally-known one-word answer (e.g. the capital of a major country, 2+2). Low training value. The response is a bare one-liner.

Output JSON only, no commentary, no code fences:
{"instruction":"...","response":"..."}
```

## JUDGE (holistic 0-10, language-lenient)

```
You are a strict quality filter for multilingual instruction-tuning data. You are
fluent in Hindi and knowledgeable about India. Only the highest-quality
examples should be kept; low-quality data harms training.

Rate the OVERALL QUALITY of the following example as training data, 0 to 10.
Score by the SEVERITY of the single WORST real problem — do NOT count flaws, and
do NOT floor an otherwise-good example for one small blemish.

  9-10  Excellent — fluent, natural, correct, genuinely useful. Tiny nits are ok.
  7-8   Good — solid and usable; only MINOR issues (a little generic, a small
        stylistic slip, one typo, heavy but natural loanwords). KEEP.
  5-6   Fair — usable but clearly imperfect: a shallow explanation, one small
        factual imprecision, or a somewhat basic question. Borderline keep.
  3-4   Weak — a real problem that limits usefulness: a partly-off or thin
        answer, or an unnatural/pointless constraint.
  1-2   Poor — a serious defect: trivial trivia, or a clearly wrong or incoherent
        part of the answer.
  0     Unusable — wrong language or script entirely, or a wholly wrong/empty answer.

IMPORTANT: a single small blemish — a lone typo, one garbled word, natural
loanwords, or phonetic transliteration — should cost only a point or two. NEVER
drop an otherwise-excellent example below 6 for one such blemish. Reserve 0-2 for
genuinely serious defects that undermine the whole example.

LANGUAGE NOTE: judge by the real usage norms of Hindi, which you know better than any fixed rule. If educated Hindi speakers routinely mix in English loanwords/technical terms, do NOT penalize that as code-switching or as spelling/transliteration errors. Phonetic or approximate transliteration is acceptable but rate it somewhat LOWER than clean standard native spelling — a minor reduction, not a 0-1. Be LENIENT on script/loanword/transliteration matters (when unsure whether a word or spelling is acceptable in Hindi, assume it IS and give it the benefit of the doubt); reserve low scores for genuine defects: whole clauses/sentences or the instruction/constraint written in English, or text in the wrong script.

Example (Hindi, India):
Instruction:
<instruction>

Response:
<response>

Output JSON only, no commentary, no code fences:
{"score": <integer 0-10>, "reason": "<one sentence>"}
```
