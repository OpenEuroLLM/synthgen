# Judge-validation study

Before trusting the LLM judge, check it separates good from bad. We generate
known-**good** and known-**bad** instruction-tuning examples in **English** and
**Hindi**, score each with the holistic quality judge (0–10), and compare the
good vs bad score distributions.

- **Good** = natural, localized, non-trivial instruction + correct, well-explained
  response (generation prompt carries a meta-eval self-check plus wrong-answer and
  bad-explanation guards).
- **Bad** = deliberately reproduces v1 failure modes: `mixed_language`,
  `unnatural_constraint`, `trivial`, `wrong_answer`, `bad_explanation`.

See `PROMPTS_PREVIEW.md` for the exact rendered prompts.

## Models & container (staged)

| Role | Model | Location |
|---|---|---|
| Container | `vllm-openai-rocm.sif` (vLLM 0.23, tf 5.12) | `/scratch/project_465002530/containers/` |
| Generator | `google/gemma-4-31b-it` | `$HF_HOME` cache |
| Judge | `Qwen/Qwen3.6-27B` | `$HF_HOME` cache |

`HF_HOME=/scratch/project_462001516/cache/huggingface/abhasjha`

## Run

```bash
cd /scratch/project_462001516/abhasjha/synthgen
sbatch studies/localized_bootstrap/serve_and_run.slurm          # 5 good + 5 bad per lang
# override: N_GOOD=10 N_BAD=10 sbatch studies/localized_bootstrap/serve_and_run.slurm
```

Serves gemma (TP=2, GPUs 0–1) and Qwen (TP=2, GPUs 2–3) on one dev-g node,
publishes endpoints, runs `run_judge_validation.py`. Account = `project_462001516`.

## Output

- `outputs/records.jsonl` — one row per example: label, bad_type, instruction,
  response, judge `score`, `reason`. **Read this yourself to sanity-check quality.**
- `outputs/summary.json` + slurm log — good vs bad score stats, per-language and
  per-bad-type means, and a text histogram of the distribution.

## Files

- `prompts.py` — good generation, bad generation (5 defect types), holistic judge.
- `taxonomy.py` — localized domains, role archetypes, intents.
- `run_judge_validation.py` — driver, JSON parsing, distribution report.
- `serve_and_run.slurm` — vLLM launch + study run.
