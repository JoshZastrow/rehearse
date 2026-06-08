---
name: llm-inference
description: Action-oriented advisor for making LLM inference fast and cheap — the repeated production cost. Use when optimizing latency vs throughput, sizing/serving a model, shrinking the KV cache (GQA/MLA/sliding-window), quantization, speculative decoding, continuous batching, paged attention, prefill/decode disaggregation, KV-cache offloading, megakernels, or debugging a slow/garbled serving stack.
metadata:
  source: Stanford CS336 (Spring 2026) Lecture 10 (Inference) + Guest Lecture (Dan Fu, serving systems)
  promptSignals:
    phrases:
      - "inference optimization"
      - "serving latency"
      - "throughput"
      - "KV cache"
      - "speculative decoding"
      - "paged attention"
      - "continuous batching"
      - "prefill decode"
      - "quantize model serving"
      - "vLLM"
      - "time to first token"
    minScore: 4
---

# llm-inference — making serving fast and cheap

You help engineers optimize **inference**, the repeated cost that dominates a deployed model's lifetime (serving, agents, RL rollouts). Small per-token wins compound enormously.

## Mental models (hold these first)
1. **Latency vs throughput are in tension.** Time-to-first-token + per-token latency = what a user feels; throughput = tokens/sec across concurrent requests. Batching raises throughput and latency together, up to a hardware limit.
2. **Arithmetic intensity decides the regime.** Flops-per-byte-moved tells you compute-bound vs memory-bound. **Prefill is compute-bound (efficient); decode's intensity does NOT scale with batch → memory-bandwidth-bound → the bottleneck.**
3. **The KV cache is the central object.** It saves recompute but its size (∝ batch × layers × seq × head-dims) is what limits batch and drives memory traffic. Most algorithmic wins shrink or reuse it.
4. **Prefill and decode are different machines.** Treat them separately — even on different hardware.

## Procedure A — Diagnose before optimizing
1. Measure TTFT, per-token latency, and throughput at several batch sizes.
2. Classify the hot path: prefill (compute-bound) vs decode (memory-bound)?
3. Compute KV-cache bytes; check if you're memory-capacity- or bandwidth-limited.
4. Pick levers by regime (below). Re-measure after each.

## Procedure B — Algorithmic levers (shrink/reuse the KV cache)
- **GQA** (grouped-query attention): share K/V across heads — big KV cut, ~no accuracy loss. Default first move.
- **MLA** (multi-head latent attention): compress K/V into a latent — larger savings.
- **Sliding-window / local + interleaved global** attention; **cross-layer KV sharing**; **linear-attention** variants — each trades expressivity (no free lunch).
- **Quantization:** simulate during training; quantize per-tensor/channel (activation-aware). Lower precision → less memory traffic → higher throughput.
- **Pruning + distillation:** drop unimportant layers, then heal on a target task.
- **Speculative decoding:** cheap **draft** model proposes K tokens; big **target** verifies in parallel and accepts a burst **losslessly** (rejection sampling). Distill the draft for high accept rates.

## Procedure C — Serving-system levers (scheduling & memory)
- **Continuous batching:** evict finished sequences and admit new ones every step to keep the GPU full.
- **Paged attention (vLLM):** manage KV cache like **OS virtual memory** (pages + index) → no fragmentation, enables **prefix/prompt sharing** across requests.
- **Prefill/decode disaggregation:** run flop-heavy prefill and memory-bound decode on **separate, specialized hardware**.
- **KV-cache offloading:** spill cache to CPU/disk when GPU memory runs out; care about SSD bandwidth; predict reuse to decide what to keep.
- **Request routing:** route by request length/shape — a simple split can yield ~40% faster serving.
- **Megakernels:** fuse a whole model pass into one kernel to remove inter-kernel gaps and **tail effects**; can pipeline (start loading next layer before attention finishes).

## Decision cheatsheet
| Symptom | Likely lever |
|---|---|
| KV cache won't fit / low batch | GQA → MLA, quantization, offloading |
| Decode memory-bound | shrink KV, speculative decoding, megakernels |
| GPU underutilized across requests | continuous batching, paged attention |
| Mixed long-prompt + chat traffic | prefill/decode disaggregation, length routing |
| Long-context serving | sliding-window/local attention, paged KV |

## War-story pitfalls (from production)
- Garbled output / wrong language is often an **off-by-one kernel bug**, not "quantization." Read kernels.
- Don't let one workload's KV cache starve others — isolate/offload.
- Cross-GPU interconnects are flaky; minimize cross-device traffic in the decode loop.

---
*Derived from Stanford CS336 Spring 2026, Lecture 10 (Inference) and the Dan Fu guest lecture (production serving, megakernels, loop transformers). Transcripts: `yt2md/docs/transcripts/…lecture-10-inference*`, `…guest-lecture-dan-fu*`; index in `~/Desktop/youtube/`.*
