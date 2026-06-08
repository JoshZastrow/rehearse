---
name: llm-scaling-laws
description: Action-oriented advisor for using scaling laws to plan LLM training — predicting loss from compute/data/params, choosing compute-optimal allocation (Chinchilla), fitting power laws, transferring hyperparameters across scale (MuP, learning-rate/batch scaling, WSD schedules), and comparing architectures or optimizers by slope. Use when sizing a run, setting hyperparameters before a big job, deciding tokens-per-parameter, or judging whether an intervention will scale.
metadata:
  source: Stanford CS336 (Spring 2026) Lectures 9 & 11 — Scaling Laws
  promptSignals:
    phrases:
      - "scaling law"
      - "compute optimal"
      - "chinchilla"
      - "tokens per parameter"
      - "how big a model"
      - "hyperparameter transfer"
      - "muP"
      - "learning rate scaling"
      - "critical batch size"
      - "isoflops"
      - "predict loss"
    minScore: 4
---

# llm-scaling-laws — planning runs with scaling laws

You help engineers **de-risk expensive training** by predicting large-scale behavior from cheap small-scale experiments. Never "just run the scary big job" — fit a curve and extrapolate.

## Mental models (hold these first)
1. **Loss is a power law.** `loss ≈ irreducible_entropy + A·N^(-α) + B·D^(-β)` (N=params, D=data). On log–log axes it's a line bending toward an **irreducible-error asymptote**. The same form recurs everywhere.
2. **Data scaling = parametric estimation.** Estimating richer functions gives characteristic slopes (mean-estimation ≈ slope −1). More model than data → irreducible regime.
3. **Two budgets, one frontier.** Compute `C ≈ 6·N·D`. Compute-optimal allocation trades N against D along this constraint.
4. **Scale-invariants must be held fixed** as you scale (aspect ratio, non-embedding param count) or your curves lie.
5. **Upstream ≠ downstream.** Loss predicts loss well; it predicts task accuracy only weakly. Scaling laws give trends, not guarantees.

## Procedure A — Predict loss / pick model+data size (compute-optimal)
1. Fix architecture family and all scale-invariants (aspect ratio, head dims, etc.).
2. Run an **IsoFLOP sweep**: several compute budgets; at each, vary N (and D = C/6N) and record final loss.
3. Fit the joint power law (N, D → loss); find the minimizer per FLOP budget.
4. Read off **compute-optimal N\*, D\***. Default sanity anchor: **~20 tokens/param** (Chinchilla). If Kaplan and Chinchilla seem to disagree, check (a) embedding vs non-embedding param counting, (b) LR schedule, (c) curve-fitting method — re-tuning these reconciles them.
5. **Decide whether to overtrain.** Compute-optimal minimizes *training* loss for a FLOP budget; if the model will serve heavy inference, deliberately **overtrain** (D ≫ 20·N) to get a smaller model at fixed quality.

## Procedure B — Transfer hyperparameters across scale
1. **Learning rate / batch:** use scaling-law fits (e.g. MiniCPM scales LR by fan-out; set **critical batch size** from a steps-vs-examples sweep — the variance-reduction sweet spot).
2. **Schedule:** prefer **WSD (warmup–stable–decay) trapezoid** over cosine so you can *extend/reuse* a run by re-decaying (decay ≈ 10–20% of steps) instead of restarting.
3. **MuP (maximal-update parameterization):** to make a tuned small model transfer to a large one, enforce the two invariants — **activations and updates stay order-one as width grows** — which fixes per-layer init scales and layer-adaptive LRs. Caveat: MuP has known stress-test failures; verify, don't trust blindly.

## Procedure C — Compare an architecture or optimizer
1. Train both across **several scales**, not one point.
2. Compare **slopes**, not single scores — a better intercept with a worse slope loses at scale.
3. Scale *all* learning rates per point; watch for small-scale wins that **invert at larger scale** (e.g. weight-decay / "Cautious Adam" surprises; nanoGPT-speedrun-style optimizer gains; Muon's Newton–Schulz orthogonalization).

## Decision cheatsheet
| Situation | Move |
|---|---|
| "How big a model for C flops?" | IsoFLOP fit → N\*,D\*; anchor ~20 tok/param |
| Heavy inference target | overtrain (smaller N, larger D) |
| Don't want to tune LR per size | MuP or fan-out LR scaling |
| Want to extend a run later | WSD schedule, re-decay |
| New optimizer/arch claim | multi-scale slope comparison |
| MoE | fit scaling vs sparsity & active/total params |

## Pitfalls
- Extrapolating from one scale, or from a non-fixed aspect ratio.
- Trusting upstream loss as a downstream-quality guarantee.
- Over-fitting the scaling fit itself (too few, too-small points; sensitive recipes like crazy warmups).
- Forgetting inference cost when choosing the compute-optimal point.

---
*Derived from Stanford CS336 Spring 2026, Lectures 9 & 11 (Scaling Laws). Transcripts: `yt2md/docs/transcripts/…lecture-9-scaling-laws*`, `…lecture-11-scaling-laws*`; searchable index in `~/Desktop/youtube/`.*
