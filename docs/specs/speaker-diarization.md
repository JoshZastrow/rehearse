# Speaker Diarization — Pipeline Spec

## Problem

All words in `audio.json` are currently labeled `"SPEAKER_MAIN"`. The transcript
has `user` and `coach` as distinct speakers, but the annotation step ignores that.
Fine-tuning on unlabeled speaker data limits the model's ability to learn
speaker-conditioned behavior.

## Goals

1. Assign correct speaker labels (`user` / `coach`) to every word alignment in `audio.json`.
2. Handle both new sessions (with per-turn WAVs) and old sessions (mixed mono only).
3. Keep diarization as a rerunnable, skippable pipeline step — not tightly coupled to annotation.

---

## Option A — Stereo WAV from Per-Turn Files (deterministic, no ML)

**Availability:** ~6 of 29 current sessions (those with `audio/user/` and `timing.jsonl`).

### How it works

1. Read `timing.jsonl` to get absolute `t_ms` start offset per turn per role.
2. Read each `audio/user/turn_N.wav` and `audio/coach/turn_N.wav`.
3. Place each turn's PCM bytes at the correct offset in a silent buffer (one per speaker).
4. Interleave into a 2-channel WAV: **ch0 = user, ch1 = coach**.
5. Pass stereo WAV to `annotate.py`; transcribe each channel separately and label accordingly.

### Tradeoffs

| | |
|---|---|
| ✓ | No ML inference — deterministic, free, fast |
| ✓ | Perfect speaker separation (ground truth from the runtime) |
| ✗ | Only works for sessions that recorded per-turn WAVs |
| ✗ | Older sessions (~77%) have no `audio/` dir — needs fallback |
| ✗ | Requires `AudioRecorder` to keep writing per-turn WAVs going forward |

### Pipeline integration

```
audio/user/turn_N.wav  ─┐
audio/coach/turn_N.wav ─┤  build_stereo.py  →  audio_stereo.wav
timing.jsonl           ─┘

audio_stereo.wav  →  annotate.py (ch0=user, ch1=coach)  →  audio.json
```

---

## Option B — pyannote Speaker Diarization (ML inference, universal)

**Availability:** All sessions — only requires `audio.wav`.

Model: [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)

### How it works

1. New pipeline step `diarize.py` runs on Modal GPU before `annotate.py`.
2. Takes `audio.wav`, runs pyannote diarization, outputs `audio_segments.json`:
   ```json
   {"segments": [
     {"speaker": "SPEAKER_00", "start": 0.0, "end": 4.2},
     {"speaker": "SPEAKER_01", "start": 4.5, "end": 9.1},
     ...
   ]}
   ```
3. `annotate.py` loads `audio_segments.json`, overlaps each Whisper word timestamp
   against the segments, assigns the matching speaker label.
4. Speaker IDs (`SPEAKER_00`, `SPEAKER_01`) are mapped to `user` / `coach` using
   the transcript: whichever speaker ID aligns with the first coach utterance in
   `transcript.jsonl` is labeled `coach`, the other `user`.

### Tradeoffs

| | |
|---|---|
| ✓ | Works on all sessions — no per-turn WAVs required |
| ✓ | Retroactively fixes all existing `audio.json` files |
| ✓ | Generalises to future sessions regardless of recording setup |
| ✗ | Requires HuggingFace token + pyannote model agreement |
| ✗ | GPU inference cost per session (~1–2 min on T4) |
| ✗ | Diarization errors possible (overlapping speech, short turns) |
| ✗ | Speaker ID → role mapping heuristic can fail on edge cases |

### Pipeline integration

```
audio.wav  →  diarize.py (Modal GPU)  →  audio_segments.json
                                               ↓
audio.wav  →  annotate.py (Modal GPU)  →  audio.json
              (uses audio_segments.json to label words)
```

Full pipeline order:

```
sessions.jsonl
    │
    ▼
diarize.py          writes audio_segments.json per session
    │
    ▼
annotate.py         writes audio.json (words with speaker labels)
    │
    ▼
dataset.py          writes data/sessions.jsonl manifest for training
```

---

## Option C — Transcript Timestamp Alignment (no ML, universal fallback)

**Availability:** All sessions with `transcript.jsonl`.

### How it works

For each Whisper word, find the transcript utterance whose `[ts_start, ts_end]`
window contains `word.start`. Inherit that utterance's `speaker` label.

### Tradeoffs

| | |
|---|---|
| ✓ | Free, instant, no additional inference |
| ✓ | Uses ground truth speaker labels from the runtime |
| ✗ | Requires Whisper and transcript timestamps to be well-aligned |
| ✗ | Words at turn boundaries may be misattributed |
| ✗ | Transcript timestamps come from STT, not ground truth audio |

---

## Recommendation

**Primary path: Option B (pyannote) with Option C as fallback.**

- Run `diarize.py` on all sessions. It works retroactively and produces a
  reusable `audio_segments.json` artifact regardless of how the session was recorded.
- Where diarization confidence is low (e.g. very short sessions, <10 words),
  fall back to transcript timestamp alignment (Option C).
- Option A (stereo WAV) is valuable for the runtime going forward — `AudioRecorder`
  already writes per-turn WAVs — but it cannot fix existing sessions and adds
  complexity to the annotation step.

---

## Schema Changes

### `schemas.py` additions

```python
class DiarizationSegment(BaseModel):
    speaker: str    # "SPEAKER_00" | "SPEAKER_01"
    start: float
    end: float

class DiarizationOutput(BaseModel):
    segments: list[DiarizationSegment]
```

`AlignmentItem` speaker field changes from `str` to `Literal["user", "coach"]`
after the speaker ID → role mapping is applied.

---

## Integration Points

| File | Change |
|---|---|
| `train/pipeline/diarize.py` | New — Modal pyannote inference, writes `audio_segments.json` |
| `train/pipeline/annotate.py` | Load `audio_segments.json` if present; fall back to transcript alignment |
| `train/pipeline/schemas.py` | Add `DiarizationSegment`, `DiarizationOutput`; tighten `AlignmentItem` speaker type |
| `train/pipeline/dataset.py` | No change — manifest format unchanged |

---

## Open Questions

1. **Speaker ID → role mapping**: Using first coach utterance from `transcript.jsonl`
   as the anchor is a heuristic. Should we validate against prosody or timing data?
2. **Overlap handling**: If pyannote assigns overlapping segments, which speaker wins?
   Last-writer, or split at the midpoint?
3. **Per-turn WAV retention**: Should `AudioRecorder` be updated to always write
   per-turn WAVs so future sessions always have Option A available?
