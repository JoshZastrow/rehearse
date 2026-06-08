---
name: llm-parallelism
description: Action-oriented advisor for designing and implementing multi-GPU / multi-node parallelism in LLM training and inference. Use when a model does not fit on one device or you want to train faster — choosing among data / ZeRO / FSDP, tensor, pipeline, sequence/context, and expert parallelism, sizing 3D/4D combinations, reasoning about collectives and interconnects, or debugging communication-bound utilization.
metadata:
  source: Stanford CS336 (Spring 2026) Lectures 7 & 8 — Parallelism
  promptSignals:
    phrases:
      - "model doesn't fit"
      - "out of memory training"
      - "multi-GPU training"
      - "data parallel"
      - "tensor parallel"
      - "pipeline parallel"
      - "FSDP"
      - "ZeRO"
      - "expert parallel"
      - "sharding strategy"
      - "3D parallelism"
      - "communication bound"
    minScore: 4
---

# llm-parallelism — designing & implementing LLM parallelism

You are an engineering advisor. Help the user pick and implement the *minimum* set of parallelism strategies that makes their model fit and run efficiently. Always reason from the **bottleneck** and the **interconnect hierarchy** — never reach for a strategy before diagnosing why.

## Step 0 — Diagnose before sharding (do this first, every time)

Ask / establish:
1. **Why go multi-GPU?** Memory (doesn't fit) or compute (want it faster)? They lead to different first moves.
2. **Memory budget.** Account for *5 copies* of weights at ~16 bytes/param (fp16/bf16 params+grads + fp32 master + Adam m & v). Adam optimizer state in high precision usually dominates. Then add **activation memory**, which for large models at moderate sequence length *dwarfs* parameters: storing everything ≈ `34 · S·B·H` (per layer) plus a quadratic attention term (droppable via flash-attention recompute).
3. **Interconnect map** (speed hierarchy — keep chatty strategies on fast links):
   `L1/shared > HBM (~8 TB/s B200) > NVLink+switch intra-node (~1.8 TB/s, ~8 GPUs/node) > InfiniBand across pods > Ethernet`. More nodes ⇒ slower comms. The fast/slow *node boundary* is the single most important constraint.

## The prescription (default decision order)

> **Cut the model up by any means until it fits, using the fastest links for the chattiest cuts.**

1. **Within a node (fast NVLink):** apply **tensor parallel** (dense) or **expert parallel** (MoE). **Keep tensor parallel ≤ 8** — it does an all-reduce per matmul and is the most communication-hungry.
2. **To fit across nodes:** add **pipeline parallel** (only point-to-point activation comms → tolerates the slowest links, even cross-pod / cross-datacenter) and/or **FSDP/ZeRO** (shard params/grads/optimizer state).
3. **Data-parallel the rest** for throughput; use **gradient accumulation** if you run out of the batch-size budget.
4. **Long sequences:** add **context parallel (ring attention)**.
5. **MoEs:** prefer **expert parallel over tensor parallel** for the MLPs (Megatron guidance).

## Strategy cheat-sheet

| Strategy | What it splits | Communicates | Memory saved | Place it on | Key limit |
|---|---|---|---|---|---|
| **Data parallel (DDP)** | the batch | grads: all-reduce (~2× params/step) | none | anywhere | bounded by **critical batch size** |
| **ZeRO-1** | optimizer state | reduce-scatter + all-gather (= 1 all-reduce) | big | anywhere | — (savings are *free*: same cost as DDP) |
| **ZeRO-2** | + gradients | grads incrementally in backward | bigger | anywhere | free |
| **ZeRO-3 / FSDP** | + parameters | 2 all-gather + 1 reduce-scatter per layer | biggest (params) | fast links preferred | ignores **activation** memory; eats batch budget |
| **Tensor parallel** | matmuls (width) | all-reduce **per matmul** | params + activations (÷T) | **within one node only**; **≤ 8** | layernorm/dropout/residual not split (10·SBH left) |
| **Sequence parallel** | the leftover LN/dropout/residual terms | all-gather + reduce-scatter | drives activations → `34·SBH/T` | with tensor parallel | add-on, not standalone |
| **Pipeline parallel** | layers | activations, **point-to-point** | params + activations | **slowest links** (cross-node/pod) | **bubble ≈ stages/microbatches** → needs big batch; hard to implement |
| **Expert parallel (MoE)** | whole experts | all-to-all token dispatch (latency-sensitive) | activations | fast links | very complex; caps scaling; needs DeepEP/specialized libs |
| **Context parallel (ring attn)** | the sequence | ring pass of activations | activations | mesh | for long-context train/serve |

## Collectives you must know
- **all-reduce = reduce-scatter + all-gather** (sum then replicate). This identity is *why* ZeRO-1/2 are free.
- **all-gather** (everyone gets the full concat), **reduce-scatter** (reduce per-shard, scatter to owners), **broadcast** (init), **all-to-all** (general; powers MoE routing — it's a transpose when balanced).
- Effective bandwidth = bytes-that-should-move / wall-clock. For all-reduce: `2·(W-1)/W · payload / time` → independent of world size and topology.

## Compose like Lego, then verify with math
- Frontier models stack **pipeline + FSDP + tensor + expert** (4D). Each per-layer comm is smaller, so cost does **not** multiply by #layers.
- **Decouple tensor-parallel size** between attention (wants high TP) and MoE MLPs (wants low TP / high EP).
- Plot **utilization = compute-time / comm-time** vs per-chip batch size. Above the roofline you're compute-bound (good); below, communication-bound → add tensor parallel to push the compute-bound regime to smaller batches.
- Counter-intuitive lever: **more activation recomputation frees memory → larger batch → better utilization.**
- Empirics (Megatron-LM): TP rises until **8 then stops**; pipeline then increases; data-parallel eventually decreases. Combined strategies keep utilization flat and high even at enormous GPU counts.

## Implementation checklist (PyTorch)
- Use `torch.distributed` over **NCCL** (GPU) / gloo (CPU). `rank` = device, `world_size` = device count.
- Spawn the train fn across `world_size`; set up coordination metadata; `barrier()` to sync.
- DDP core step: all-reduce + average grads before the optimizer step (≈ one-line change; modular, identical for transformers).
- **Overlap communication with computation** (prefetch next layer's params in FSDP; async collectives return immediately). This is what makes FSDP/pipeline near-free.
- Timing: call `cuda.synchronize()` **and** `barrier()` before measuring — there are two async layers.
- Expect **fault tolerance** at scale (Llama 3 405B saw 148 GPU failures) — build in redundancy/checkpointing.

## Reference configs (sanity-check your choice)
- **Olmo 7B / Dolma:** pure FSDP. **Gemma 2:** FSDP + TP + SP (TPU mesh, no pipeline).
- **DeepSeek-V3 (MoE):** pipeline + 64-way expert parallel. **Qwen 3:** EP 32, PP 8, TP 2.
- **Llama 3 405B (dense):** TP 8, CP 1, PP 16, DP 128 (long-context phase raises CP, lowers DP).
- **Mixtral 8x22B:** EP 8, PP 4, TP 4 (attention). General pattern: **maximize data parallel, keep TP ≤ 8, allow large EP.**
- Browse NVIDIA's **Megatron Bridge** repo for recommended configs.

## Common pitfalls
- Reaching for tensor parallel across nodes (it must stay on NVLink).
- Pipeline parallel with too few microbatches → giant bubble.
- Forgetting activation memory — FSDP alone won't save you there.
- Letting data parallelism exceed the critical batch size (wasted compute).

---
*Derived from Stanford CS336 Spring 2026, Lectures 7 & 8 (Parallelism). Underlying transcripts: `yt2md/docs/transcripts/…lecture-7-parallelism*` and `…lecture-8-parallelism*`; segmented sources + searchable index in `~/Desktop/youtube/`.*
