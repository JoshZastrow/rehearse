# ADR: Dual-Stream Encoding for Caller Model Fine-Tuning

**Date:** 2026-06-07  
**Status:** Accepted  
**Context:** Establishing the data format for supervised fine-tuning of the Moshi caller model from synthetic call sessions.

---

## Context

Moshi is a full-duplex speech model that processes **two audio streams simultaneously** — a "main" speaker and an "other" speaker — interleaved with a single text token stream. Its vocabulary covers 17 codebooks per frame:

```
codebook 0     : text token (BPE)
codebooks 1–8  : main-speaker audio (8 Mimi codes)
codebooks 9–16 : other-speaker audio (8 Mimi codes)
```

The training objective computes loss only on the main-speaker audio (codebooks 1–8) and the text stream (codebook 0), treating the other-speaker audio as conditioning context.

We need to fine-tune **two separate models**:

- **Provider model** — the AI coach; main speaker = provider (left channel of stereo WAV)
- **Caller model** — the person being coached; main speaker = caller (right channel)

The training pipeline reads stereo WAVs produced by `prepare_stereo.py` or `generate_synthetic_calls.py`, where channel 0 is always provider and channel 1 is always caller.

---

## Decision

**Encode both stereo channels for every training sample, ordered by training target first.**

Concretely, in `InterleavedTokenizer.__call__`:

1. Select `main_audio = stereo[channel]` and `other_audio = stereo[1 - channel]`
2. Stack as a batch of two mono clips: shape `[2, 1, T]`
3. Pass to `mimi.encode()` → shape `[2, 8, T]`
4. Reshape to `[1, 16, T]` via `.view(1, -1, T)`
5. Concatenate with text tokens `[1, 1, T]` → final codes shape `[1, 17, T]`

The `channel` parameter in `TrainArgs` (and `rehearse_smoke.yaml`) controls which stereo channel becomes the main stream. Setting `channel=1` trains the caller model; `channel=0` trains the provider model.

---

## Alternatives Considered

### A: Encode only the target channel (9 codebooks)

Simple but wrong. The Moshi model asserts `K == self.num_codebooks` where `num_codebooks = 17`. Passing 9 codebooks raises `AssertionError: (9, 17)` immediately.

The model architecture does not support single-stream mode — it was designed for full-duplex and uses the other-speaker stream as real-time context during generation.

### B: Always encode in fixed channel order (provider first, caller second)

This would hardcode codebooks 1–8 as provider and 9–16 as caller regardless of which model is being trained. The loss function reads codebooks 1–8 as the training target, so this approach would be correct for the provider model but train the wrong stream for the caller model — computing loss on provider audio instead of caller audio with no error surfaced.

### C: Swap channels in the stereo WAV at data prep time

Produce two separate WAV files per session — one for provider training (unchanged) and one for caller training (channels swapped). This moves the complexity into the data pipeline, which runs once on Modal, rather than into the tokenizer, which runs at every training step. Cleaner conceptually, but doubles storage and requires a separate dataset.py pass per model type.

We rejected C for now because the `channel` parameter makes the tokenizer config self-documenting and avoids re-running the annotation pipeline. If we ever need to cache tokenized tensors for speed, C becomes the right choice.

---

## Consequences

- `TrainArgs` must always carry `channel: int` and `main_speaker_label: str`. These are now explicit in every YAML config rather than defaulting to provider.
- The stereo channel layout (`ch0 = provider, ch1 = caller`) is a project-wide convention enforced at the `prepare_stereo.py` and `generate_synthetic_calls.py` boundary. Any new data source must match this layout or the channel assignment is wrong.
- A future provider-model training run uses `channel: 0` and `main_speaker_label: "provider"` with no other changes.
- Mono audio (single-channel sources) is handled by duplicating the audio into both streams with the other stream zeroed out. This produces valid 17-codebook tensors but provides no cross-speaker context — usable for single-speaker pretraining only.
