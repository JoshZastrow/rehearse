---
name: llm-evaluation
description: Action-oriented advisor for evaluating LLMs — choosing and building benchmarks, perplexity, knowledge/reasoning (MMLU/GPQA/HLE), chat/preference eval (Chatbot Arena ELO, AlpacaEval, LLM-as-judge), agentic benchmarks (SWE-bench), safety/jailbreak eval, ecological validity, and decontamination. Use when deciding how to measure a model, picking the right benchmark for a goal, building an eval harness, or debugging contaminated/gamed scores.
metadata:
  source: Stanford CS336 (Spring 2026) Lecture 12 — Evaluation
  promptSignals:
    phrases:
      - "evaluate the model"
      - "which benchmark"
      - "build an eval"
      - "perplexity"
      - "MMLU"
      - "LLM as judge"
      - "chatbot arena"
      - "SWE-bench"
      - "benchmark contamination"
      - "decontamination"
    minScore: 4
---

# llm-evaluation — measuring LLMs well

You help engineers choose, build, and interpret evaluations. Core principle: **evaluation turns an abstract construct into a concrete prompt-and-score procedure, and what you measure is what you optimize** — so design the metric to match the real goal.

## Mental models (hold these first)
1. **Construct → procedure gap.** Every benchmark operationalizes a fuzzy construct ("reasoning," "helpfulness"). Always ask what construct you actually care about, then pick the closest valid procedure.
2. **Four families, increasing realism/cost:** perplexity → multiple-choice knowledge/reasoning → open-ended/chat preference → agentic tasks. Realism rises, automatic-scorability falls.
3. **Contamination is the default failure.** If test data leaked into training, scores are meaningless. Assume contamination until shown otherwise.
4. **Judges have biases.** Human and LLM judges skew toward length, style, and sycophancy.

## Procedure — pick the eval for your goal
1. **State the decision** the eval informs (ship? pick model? measure an algorithm?).
2. **Match family:**
   - *Track training / scaling* → **perplexity** (probability on held-out data; approaches true entropy; underlies cloze tasks like LAMBADA). Guard against invalid "distributions" on leaderboards.
   - *Knowledge & reasoning* → **MMLU** (now saturated), **GPQA** (Google-proof, PhD-level; experts ≈65%), **Humanity's Last Exam**. Prefer benchmarks with **held-out private sets**.
   - *Assistant quality* → **Chatbot Arena** (pairwise human preference → ELO), **AlpacaEval** (win-rate vs baseline). Expect judge bias.
   - *Real capability* → **agentic**: SWE-bench (real GitHub issues), cyber CTF, long-horizon tasks. Score depends on scaffold/delegation/context-management, not just the base model.
   - *Frontier reasoning* → ARC-AGI-style abstract reasoning.
3. **Add safety eval** (jailbreaks, contextual harm) — note safety correlates with capability.
4. **Check ecological validity** — does it resemble real query streams (e.g. medical/economic query-level benchmarks)?

## Procedure — make results trustworthy
- **Decontaminate:** check train/test overlap; use model cutoff dates; for code, prefer private/internal repos not on the web.
- **Read the outputs.** Always inspect raw generations — agentic benchmarks can be **gamed** (e.g. an empty response scoring ~38%).
- **Harden judges:** rubrics/checklists, multiple judges + ensembling, fixed prompts; remember LLM-judge ≠ better unless mimicking a target population.
- **Report honestly:** state the construct, the scoring rule, and known contamination/judge caveats.

## Cheatsheet
| Goal | Use | Watch |
|---|---|---|
| Scaling/training signal | perplexity | invalid distributions |
| Knowledge/reasoning | GPQA / HLE (private split) | MMLU saturation, contamination |
| Assistant preference | Arena ELO / AlpacaEval | length & sycophancy bias |
| Real engineering ability | SWE-bench / agentic | scaffold-dependence, gaming |
| Algorithm (not model) | nanoGPT speedrun-style | apples-to-apples setup |

## Pitfalls
- Optimizing a saturated or contaminated benchmark.
- Trusting an aggregate score without reading transcripts.
- Using multiple-choice as a proxy for real-world open-ended usage.
- Treating an LLM judge as ground truth.

---
*Derived from Stanford CS336 Spring 2026, Lecture 12 (Evaluation). Transcript: `yt2md/docs/transcripts/…lecture-12-evaluation*`; index in `~/Desktop/youtube/`.*
