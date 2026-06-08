---
name: llm-gpu-kernels
description: Action-oriented advisor for writing fast custom GPU kernels for LLM workloads — fusing element-wise ops, writing Triton kernels (softmax, tiled matmul, fused activations), reasoning about occupancy/coalescing/bank-conflicts, profiling, and deciding when to drop to a custom kernel vs torch.compile vs PTX. Use when a hot op is memory-bound, when fusing a sequence of ops, or when authoring/debugging a Triton or CUDA kernel.
metadata:
  source: Stanford CS336 (Spring 2026) Lecture 6 — Kernels, Triton, XLA
  promptSignals:
    phrases:
      - "write a triton kernel"
      - "custom CUDA kernel"
      - "kernel fusion"
      - "fuse operations"
      - "torch.compile"
      - "memory bound op"
      - "occupancy"
      - "memory coalescing"
      - "bank conflict"
      - "tiled matmul"
      - "PTX"
    minScore: 4
---

# llm-gpu-kernels — writing fast kernels

You help engineers make a hot operation fast by writing (or fusing) GPU kernels. First principle: **always measure** — GPU performance is non-obvious, so profile before and after every change.

## Mental models (hold these first)
1. **Memory hierarchy rules everything:** registers (per-thread, fastest) → shared memory/SRAM (per-SM) → HBM (large, slow). **Every HBM round-trip is the cost.** Most kernel wins = fewer HBM reads/writes.
2. **Execution model:** threads → **thread blocks** (run on one SM, share its shared memory) → grid. A block is your unit of cooperation: load a tile into shared memory, compute, write back.
3. **Warps of 32.** Threads run in lockstep warps; `if/else` → **divergence** (both sides run, masked). The warp scheduler swaps warps to hide memory latency.
4. **Occupancy is bounded by resources** — registers/thread and shared-memory/block. Too many registers → fewer resident warps → less latency hiding.
5. **Two memory-access traps:** non-**coalesced** HBM reads (a warp should hit one cache line) and shared-memory **bank conflicts**.

## Procedure A — Decide whether to write a kernel
1. **Profile** the hot path (CUDA events; warmup + average; watch the dim-vs-throughput sawtooth).
2. Classify: is it **memory-bound** (many HBM round-trips over element-wise ops) or already compute-bound?
3. Ladder of effort, cheapest first:
   - **torch.compile** — fuses simple op graphs into one kernel automatically. Try this first.
   - **Existing fused library kernel** (e.g. fused GELU, FlashAttention). Reuse before writing.
   - **Triton kernel** — when you need a fusion/tiling the compiler won't do. The practical sweet spot.
   - **CUDA/PTX** — only when you must control every instruction.

## Procedure B — Fuse to kill HBM round-trips
- A chain of element-wise ops (e.g. GELU = several ops) naively does read→write HBM **per op**.
- **Fuse:** read from HBM once → do all work in registers/shared memory → write once. This is the single biggest, easiest win.

## Procedure C — Write a Triton kernel
1. **Grid:** launch one program per tile; `pid = tl.program_id(0)`.
2. **Offsets + mask:** `offs = pid*BLOCK + tl.arange(0, BLOCK)`; `mask = offs < N` for the ragged final block.
3. **Load → compute → store:** `tl.load(X+offs, mask=mask)` → math → `tl.store(...)`. Operate at the **tile/block** level, not per-thread.
4. **Reductions (softmax):** load the row tile, **subtract the row max**, exponentiate, sum, divide — fusing what the naive version does in multiple HBM passes. For rows wider than BLOCK, **iterate over column tiles** accumulating partials.
5. **Tiled matmul:** naive matmul re-reads HBM for every (M,N,K) → constant, **bad arithmetic intensity**. Instead load **A/B tiles into shared memory**, compute the output tile from shared memory (reuse), then write the tile to HBM once — "globally naive, locally idealized." Index with **strides**.
6. **Fuse the epilogue:** apply activations directly on the output tile before the final store.
7. Inspect the generated **PTX** if you need to understand/scheduling-debug; let the compiler place registers unless you have a specific reason not to.

## Cheatsheet
| Situation | Move |
|---|---|
| Chain of element-wise ops, memory-bound | torch.compile → else fuse in Triton |
| Standard op (GELU, attention) | reuse a fused library kernel |
| Custom fusion/tiling | Triton kernel |
| Matmul re-reading HBM | tile into shared memory |
| Need an activation after matmul | fuse into the epilogue |
| Last-resort total control | hand-written CUDA/PTX |

## Pitfalls
- Optimizing without profiling (or without warmup/averaging).
- Non-coalesced reads / shared-memory bank conflicts silently killing bandwidth.
- Register-heavy kernels tanking occupancy.
- Launching a block count that doesn't tile the SMs (wasted wave).
- Reaching for PTX when torch.compile or Triton would do.

## Related
For *which* memory format and the roofline/arithmetic-intensity framing, see **`gpu-compute-planning`** (memory/compute sizing) and **`llm-inference`** (megakernels, KV-cache kernels). Lecture 5 (GPUs/TPUs hardware model) is covered by **`gpu-compute-planning`**.

---
*Derived from Stanford CS336 Spring 2026, Lecture 6 (Kernels, Triton, XLA). Transcript: `yt2md/docs/transcripts/…lecture-6-kernels-triton-xla*`; index in `~/Desktop/youtube/`.*
