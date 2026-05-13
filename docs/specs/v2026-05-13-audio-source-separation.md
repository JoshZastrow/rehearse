# rehearse — Audio Source Separation + Speech Enhancement Pipeline

**Status**: draft  
**Owner**: jz  
**Depends on**: `docs/specs/v2026-05-13-vad-turn-segmentation.md` (runs before this),
  `rehearse/types.py` (EnhancedClipRecord, VoiceTrainingRecord),
  `rehearse/storage.py`  

---

## 0. One-line summary

An offline batch pipeline that takes per-clip WAV files from the VAD
segmentation step and produces clean, 24kHz enhanced audio with DNSMOS
quality filtering — ready for TTS fine-tuning.

---

## 1. Context: the telephony sampling-rate problem

Rehearse uses Twilio Media Streams. Twilio hard-wires that transport to
**G.711 mu-law at 8 kHz** — there is no codec or sample-rate configuration
available on the Media Streams WebSocket. The runtime upsamples to 16 kHz
PCM16 for internal processing (`audio/resample.py: upsample_8k_to_16k`),
but that upsampling is linear interpolation: it adds no information above
4 kHz.

Pocket TTS and the Emilia dataset both operate at **24 kHz**. A naive
resample of Rehearse WAVs produces files that are technically 24 kHz but
spectrally empty above 4 kHz — the model would train on bandwidth-limited
data. Post-hoc speech enhancement (bandwidth extension) fills the upper
spectrum generatively, producing perceptually natural wideband audio from
telephone-origin input. Kyutai used Adobe Enhance Speech in their own
Pocket TTS evaluation for the same reason.

**Options evaluated:**

| Approach | True wideband? | Feasibility |
|---|---|---|
| Config change to Twilio codec | Not possible — Media Streams is locked | — |
| Switch callers to WebRTC (Twilio Voice SDK) | Yes (Opus, 16kHz+) | High effort — requires UI changes |
| Post-hoc speech enhancement (Resemble Enhance) | Perceptually plausible | Low effort — offline batch |
| Accept 16kHz for training | Sub-optimal for Pocket TTS | None |

**Decision**: post-hoc speech enhancement now; WebRTC is the right
long-term fix and this spec does not preclude it. If WebRTC sessions are
added, clips from those sessions skip this step (already wideband clean).

---

## 2. Scope

In scope:
- Offline batch processor that runs after `vad_segment` has produced
  `pipeline/clips/clips.jsonl` and `pipeline/clips/clip_*.wav`.
- For each accepted clip: denoise, bandwidth-extend to 24kHz, normalize.
- DNSMOS P.835 OVRL quality gate (≥ 3.0), matching the Emilia-Pipe bar.
- Output: `pipeline/enhanced/manifest.jsonl` (rows of `EnhancedClipRecord`)
  and per-clip enhanced WAVs at `pipeline/enhanced/clip_NNNN_enhanced.wav`.
- After enhancement, emit the joined `VoiceTrainingRecord` rows: accepted
  `(wav_path, text)` pairs ready for fine-tuning.

Out of scope:
- Real-time in-call processing.
- Coach audio (excluded by the VAD segmentation step upstream).
- VAD segmentation — covered in `v2026-05-13-vad-turn-segmentation.md`.
- Speaker diarization — speaker identity is already encoded in the clips
  produced upstream.

---

## 3. Functional requirements

**F-E1.** The pipeline MUST read `pipeline/clips/clips.jsonl` and skip
clips with `status != "accepted"` or `transcript_missing == true`.
It MUST also skip the session if `Session.consent != GRANTED`.

**F-E2.** For each accepted clip the pipeline MUST apply, in order:
  1. Resample to 24 kHz mono (from 16kHz PCM16 input).
  2. Denoise: remove line hiss, DTMF tones, and background noise using
     `resemble-enhance` in denoiser mode.
  3. Bandwidth extension: generatively fill 4–12 kHz spectrum using
     `resemble-enhance` in enhancer mode.
  4. Loudness normalize to −20 dBFS, capped at ±3 dB adjustment to
     avoid distortion on very quiet or very loud clips.

**F-E3.** After processing, compute **DNSMOS P.835 OVRL** for each output
file. Clips scoring below 3.0 are written with `status = "rejected_quality"`.

**F-E4.** Accepted files MUST be written to
`pipeline/enhanced/clip_{clip_index:04d}_enhanced.wav` inside the session
directory.

**F-E5.** The pipeline MUST write `pipeline/enhanced/manifest.jsonl` with
one `EnhancedClipRecord` (from `rehearse/types.py`) per processed clip,
including rejected ones.

**F-E6.** After writing `manifest.jsonl`, the pipeline MUST produce
`pipeline/enhanced/voice_training.jsonl` — one `VoiceTrainingRecord` per
clip that is accepted in BOTH `clips.jsonl` and `manifest.jsonl`. This is
the final `(wav_path, text)` dataset consumed by the fine-tuning job.
`phone_number_hash` is copied from `session.json` to enable cross-session
speaker grouping.

**F-E7.** Register new artifacts in `session.json`:
  - `artifact_paths["enhanced_audio"] = "pipeline/enhanced/manifest.jsonl"`
  - `artifact_paths["voice_training"] = "pipeline/enhanced/voice_training.jsonl"`

**F-E8.** The pipeline MUST be idempotent: re-running overwrites
`pipeline/enhanced/` and rewrites both manifest files.

**F-E9.** The pipeline MUST be runnable as:
  - `python -m rehearse.pipeline.audio_enhance <session_id>`
  - `python -m rehearse.pipeline.audio_enhance --all`

---

## 4. Non-functional requirements

**N-P1.** Processing MUST complete in under 2× real-time on CPU for a
typical session's accepted clips. `resemble-enhance` benchmarks at
~1.5–2× real-time on CPU.

**N-P2.** GPU acceleration is permitted but not required.

**N-Q1.** Each enhanced WAV MUST be validated with `soundfile.read()`
before the manifest entry is written. Invalid outputs are logged and
skipped rather than crashing the batch.

**N-L1.** Sessions without consent MUST NOT be processed.

---

## 5. Dependencies

| Role | Tool | License |
|---|---|---|
| Denoise + bandwidth extension | `resemble-enhance` (pip) | MIT |
| DNSMOS quality scoring | `DNSMOS` (Microsoft, pip) | MIT |
| Audio I/O | `soundfile` | BSD |
| Pydantic serialization | `rehearse/types.py` | — |

`resemble-enhance` runs both steps in one invocation:
```bash
resemble-enhance input.wav output.wav --denoise --enhance
```

---

## 6. Output layout (inside session directory)

```
sessions/{session_id}/
├── pipeline/
│   ├── clips/
│   │   ├── clips.jsonl                 ← AudioClipRecord rows (upstream input)
│   │   ├── clip_0000.wav               ← 16kHz source clips (upstream input)
│   │   └── ...
│   └── enhanced/
│       ├── manifest.jsonl              ← EnhancedClipRecord rows
│       ├── voice_training.jsonl        ← VoiceTrainingRecord rows (final output)
│       ├── clip_0000_enhanced.wav      ← 24kHz, denoised, bandwidth-extended
│       └── ...
```

---

## 7. Pipeline position

```
[vad_segment — v2026-05-13-vad-turn-segmentation.md]
  Output: pipeline/clips/clips.jsonl + clip_NNNN.wav (16kHz)

[audio_enhance — this spec]
  Input:  pipeline/clips/clips.jsonl + clip_*.wav (16kHz)
  Output: pipeline/enhanced/manifest.jsonl          (EnhancedClipRecord rows)
          pipeline/enhanced/voice_training.jsonl    (VoiceTrainingRecord rows)
          pipeline/enhanced/clip_NNNN_enhanced.wav  (24kHz)

[TTS fine-tuning]
  Input:  voice_training.jsonl — one (wav_path, text) pair per row
```

---

## 8. Open questions

1. **Resemble Enhance vs Voicefixer**: Resemble Enhance is the default
   recommendation (better on telephone speech per published benchmarks).
   Benchmark both on a sample of real Rehearse call audio before committing
   to the production batch job.

2. **DNSMOS threshold**: 3.0 matches Emilia-Pipe. If the rejection rate
   on post-enhanced telephone audio is high (>50%), evaluate lowering to
   2.8. Measure on 20+ sessions before deciding.

3. **WebRTC path**: If Rehearse adds a browser-based session type, clips
   from those sessions can skip steps 2–4 of F-E2 (already wideband clean).
   A `source_type: "pstn" | "webrtc"` field on `AudioClipRecord` would
   route them correctly without changing this pipeline's interface.
