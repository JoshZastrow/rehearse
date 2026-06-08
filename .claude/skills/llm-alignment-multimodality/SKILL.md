---
name: llm-alignment-multimodality
description: Action-oriented advisor for adding vision (and other modalities) to LLMs. Use when building vision-language models — choosing CLIP/SigLIP contrastive encoders, ViT patching, the encoder-adapter-LLM recipe (LLaVA/Qwen-VL), handling arbitrary image/video resolution and long context, staged multimodal training, image generation via discrete tokens, or deciding between encoder-adapter vs unified any-modality token models.
metadata:
  source: Stanford CS336 (Spring 2026) Lecture 17 — Alignment / Multimodality
  promptSignals:
    phrases:
      - "vision language model"
      - "multimodal model"
      - "CLIP"
      - "vision encoder"
      - "image tokens"
      - "LLaVA"
      - "Qwen-VL"
      - "ViT"
      - "image generation tokens"
      - "video understanding"
    minScore: 4
---

# llm-alignment-multimodality — adding vision to LLMs

You help engineers extend a language model to **ingest (and sometimes generate) images/video**. Core principle: **don't retrain from scratch — align a good vision encoder to a pretrained LLM through an adapter, then stage the data.**

## Mental models (hold these first)
1. **Tokens are the interface.** LLMs consume token embeddings. The whole game is producing *meaningful* image tokens (patch embeddings or discrete codes) the LLM can attend to — a pixel is not meaningful; a patch/caption-aligned vector is.
2. **Contrastive alignment = shared space.** CLIP-style training makes matching image–text pairs have high dot-product and mismatches low, across a batch. Web-scale captioned pairs do the heavy lifting; the contrastive loss is **tied to batch size** (SigLIP-style losses + distributed tricks let it scale).
3. **Encoder + adapter + LLM** is the dominant, stable recipe. Unified "everything is one token stream" models are elegant but historically **unstable** and currently less popular.
4. **Resolution/length is a token-budget problem.** Arbitrary resolutions and video explode token counts; you tile/crop and budget.

## Procedure A — Build a vision encoder
1. Use a **ViT**: split image into patches (e.g. 16×16), embed, run a transformer encoder.
2. Train with **CLIP/SigLIP contrastive** objective on large captioned datasets (LAION-style); preprocess to fixed input (resize to multiple resolutions, center-crop).
3. Validate via **zero-shot classification** as a quick capability proxy.

## Procedure B — Build a vision-language model (LLaVA/Qwen-VL recipe)
1. Take a **pretrained CLIP vision encoder** + a **pretrained LLM** (don't train from scratch).
2. Insert an **adapter** (projection/MLP) mapping image vectors into the LLM's token-embedding space.
3. **Stage the training:**
   - Stage 1 — **alignment**: train the adapter so images map into language space.
   - Stage 2 — **knowledge**: broaden on image-text data.
   - Stage 3 — **task/instruction**: GPT-4-generated Q&A, bounding boxes, captions resembling target tasks.
4. **Handle resolution/video**: crop into tiles for high-res; budget tokens to avoid repetitive-frame domination; for video use frame sampling.
5. **For long video / many tiles** (Qwen-VL style): use **multimodal rotary position embeddings** (height/width/time) and scale context (e.g. 256K) with deeper vision-language fusion.

## Procedure C — Generate images (if needed)
1. Discretize images into **image tokens** via a learned codebook (VQ-style, ~1024 tokens/image).
2. Let one model emit text and image tokens uniformly — but expect **training instability**; consider this path only if unified generation is a hard requirement, else keep encoder-adapter for understanding + a separate generator.

## Cheatsheet
| Goal | Move |
|---|---|
| Image understanding, fast | CLIP encoder + adapter + LLM (LLaVA) |
| High-res / long video | tiling + mRoPE + long context (Qwen-VL) |
| Zero-shot classification | CLIP/SigLIP encoder alone |
| Generate images too | discrete image tokens (unstable; weigh cost) |
| Scale CLIP training | SigLIP loss + distributed batch sharding |

## Pitfalls
- Letting repetitive video frames dominate the token budget.
- Expecting unified any-modality token models to train as stably as encoder-adapter.
- Forgetting the contrastive loss couples to batch size when scaling.
- Treating multimodal data loading like text — it has different normalization/throughput needs.

---
*Derived from Stanford CS336 Spring 2026, Lecture 17 (Alignment — Multimodality). Transcript: `yt2md/docs/transcripts/…lecture-17-alignment---multimodality*`; index in `~/Desktop/youtube/`.*
