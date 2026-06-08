---
name: llm-data-curation
description: Action-oriented advisor for designing and implementing an LLM pre-training data pipeline. Use when sourcing and curating training data — choosing sources (Common Crawl, Wikipedia, GitHub, arXiv, books), HTML-to-text extraction, language ID, quality filtering (heuristic and model-based), deduplication, decontamination, data mixing/weighting, and copyright/licensing — to turn hundreds of trillions of raw tokens into a few trillion high-quality tokens.
metadata:
  source: Stanford CS336 (Spring 2026) Lectures 13 & 14 — Data (Sources, Datasets; Transformation, Dedup, Mixing, Synthetic)
  promptSignals:
    phrases:
      - "training data pipeline"
      - "pretraining data"
      - "common crawl"
      - "data curation"
      - "quality filtering"
      - "deduplication"
      - "data mixing"
      - "dataset sources"
      - "html to text"
      - "decontamination"
      - "copyright training data"
    minScore: 4
---

# llm-data-curation — designing & implementing an LLM data pipeline

You are an engineering advisor for **pre-training data**. The job is messy human labor: distill enormous raw crawls (hundreds of trillions of tokens) down to a few trillion **high-quality** tokens. A well-filtered quality subset reliably beats a larger raw dataset — so optimize for *signal per token*, and treat every stage as a place where quality is won or lost.

## Step 0 — Frame the job
- **Which stage?** Pre-training (broad web) → mid-training (targeted, higher-quality, e.g. math/code/proofs) → post-training (task-specific). Boundaries are blurry; decide what mixture each stage needs.
- **Quality scales with curation effort** — it's a long-tail problem. Budget human + classifier effort accordingly.
- **Legal exposure is a first-class design constraint**, not an afterthought (see below).

## The pipeline (each step is an action)

1. **Source selection.** Web backbone = **Common Crawl** (monthly since 2007, billions of pages, stored as raw HTTP/WARC). Layer in high-value specialized sources:
   - **Wikipedia** — cited, high quality; download dumps (don't crawl). *Watch for data-poisoning* in editable wikis.
   - **GitHub** — permissively licensed code + commit history via the **GitHub Archive** (hourly snapshots).
   - **arXiv** — papers as **LaTeX source** (preferred) or PDF→text; mostly Creative Commons.
   - **Books** (e.g. Gutenberg = copyright-cleared) and curated Q&A-format corpora.
2. **Extraction.** **HTML→text quality matters a lot** (per DataComp-LM) — invest in a good extractor; bad boilerplate stripping silently caps your ceiling.
3. **Language ID & routing.** Filter/route by language early (cf. CCNet for low-resource langs).
4. **Quality filtering** — two complementary tools:
   - *Heuristic* (C4-style): keep lines ending in punctuation, min words/sentences per page, drop boilerplate/junk.
   - *Model-based* (the modern default): train a fast classifier (e.g. **fastText** "is this Wikipedia-like / educational?", as in CCNet, Llama 1, Nemotron). Optionally **LLM-rephrase** low-quality text toward high-quality style.
5. **Deduplication.** Exact + fuzzy (MinHash) dedup across and within sources — major quality/efficiency win; balance against diversity.
6. **Toxicity / PII handling** as policy requires.
7. **Decontamination.** Remove overlap with your eval benchmarks before training.
8. **Mixing & weighting.** Up/down-weight domains (web vs code vs math vs books); specialized sources are often a small fraction by token count but high leverage. Add code (balance Python/C vs low-resource langs) and decide how much commit/PR/diff **context** to include.

## Copyright & licensing decision aid
- Copyright **attaches automatically** to any fixed creative expression (low bar) — assume content is protected unless told otherwise.
- Permitted paths: **Creative Commons** licenses, direct **licensing deals**, public-domain works.
- **Fair use** four factors — weigh most heavily: (1) how *transformative* the use is, and (4) *effect on the market*; snippets favored over whole works.
- Copying *expression* is the issue, **not verbatim memorization**; note that *possessing* pirated data can itself be a violation.
- Context: NYT v. OpenAI (2023); the Anthropic ruling that *training* can be fair use while *pirated acquisition* is not (narrow, not blanket permission). When sourcing, prefer permissively licensed corpora (Common Pile-style, ~8 TB achievable) even if quality currently trails the top closed datasets.

## Reference lineage (steal the recipe, not the data)
BERT (BookCorpus+Wiki) → GPT-2 **WebText** (Reddit-outbound, 40 GB; OpenWebText) → **CCNet** → **C4** (heuristic-cleaned CC, ~800 GB) → **GPT-3** (filtered CC + curated, ~400B tok) → **The Pile** (open, diverse) → **Gopher/MassiveWeb** (English dedup + quality) → **Llama 1** (Wiki-reference classifier; **RedPajama** reproduction) → **Dolma / RefinedWeb / DataComp-LM / Nemotron** (model-based filtering at trillion-token scale; Llama 3 ≈ 15T tokens). The arc: from small high-quality sets + heuristic web → **model-based filtering** at scale.

## Quick checklist
- [ ] Sources chosen + licensing/robots.txt/ToS reviewed per source
- [ ] Good HTML→text extractor benchmarked
- [ ] Language ID + heuristic + model-based quality filters
- [ ] Exact + fuzzy dedup
- [ ] Benchmark decontamination
- [ ] Domain mixing weights set, code context decided
- [ ] Token budget: raw → filtered (expect 100T+ → a few T)

## Pitfalls
- Treating Common Crawl as ready-to-use (extraction/filtering is the whole game).
- Skipping decontamination → inflated, untrustworthy evals.
- Over-filtering into an overly narrow subset (lose diversity).
- Ignoring licensing until after training.

## Part 2 deep-dives (Lecture 14) — dedup, mixing, synthetic

**Deduplication mental model.** Dedup is a major quality + efficiency win and reduces memorization. Exact dedup = hashing. **Fuzzy** dedup approximates **Jaccard similarity** with **MinHash**, scaled by **LSH**: split each signature into **bands × rows**, two docs collide if any *band* matches entirely. The match-probability curve is **S-shaped** — more bands → more matches; tune bands/rows to place the threshold where you want. Run dedup **across all sources**, not per-source.

**Data mixing = choosing epochs.** Source weights implicitly set **how many epochs** each source is seen (a small high-quality source may be seen ~50× while a giant source is seen <1×). Mixtures are realized by sampling a source per example into each batch. To *optimize* weights, regress **mixture weights → loss** at small scale (DoReMi-style) and extrapolate — but **don't overfit the proxy**; cap epochs (Omix-style) and simulate data scarcity. Mixing applies across a **2-D grid** of domains.

**Quality filtering is token-budget-dependent.** If you'll train for *many* tokens, low-quality data hurts less; for short runs, demand higher quality. The extractor matters: bad HTML→text silently caps your ceiling.

**Synthetic data.** Increasingly central (esp. post-training): define **environments**, use stronger models as **teachers**, generate **code tasks from repositories** (SWE-style), and pick along a **fully- vs semi-synthetic** spectrum (including filtering your crawl to match a target distribution).

---
*Derived from Stanford CS336 Spring 2026, Lectures 13 & 14 (Data). Underlying transcript: `yt2md/docs/transcripts/…lecture-13-data-source*`; segmented source + searchable index in `~/Desktop/youtube/`.*
