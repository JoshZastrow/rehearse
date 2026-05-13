# rehearse — VAD Turn Segmentation: Per-Clip Audio Export

**Status**: draft  
**Owner**: jz  
**Depends on**: `rehearse/writers/artifacts.py` (TimingWriter, AudioRecorder),
  `rehearse/types.py` (AudioClipRecord), `rehearse/storage.py`  
**Relates to**: `docs/specs/v2026-05-13-audio-source-separation.md` (runs after this)

---

## 0. One-line summary

An offline pipeline that slices per-speaker session WAV files into
per-conversational-turn clips (3–30 seconds, user-audio-only) aligned
with transcript text, producing `AudioClipRecord` rows in
`pipeline/clips/clips.jsonl` — the input to the enhancement pipeline.

---

## 1. What already exists and what is missing

### What exists

`AudioRecorder` (`writers/artifacts.py:99`) writes per-role files during
the live call, splitting on speaker-role switches:

```
{session_dir}/audio/user/turn_0.wav   ← first contiguous user block
{session_dir}/audio/user/turn_1.wav   ← next user block after coach spoke
{session_dir}/audio/coach/turn_0.wav
...
```

Each `audio/user/turn_N.wav` contains only user audio — no coach overlap
regardless of when the coach spoke — because `AudioRecorder` writes frames
keyed by `frame.speaker`. These are the right raw units: pure user audio,
already split by role.

`TimingWriter` (`writers/artifacts.py:164`) runs independently and applies
a 600ms silence threshold to identify sub-turn boundaries within each role's
audio. It writes to `timing.jsonl`:

```json
{"turn_index": 0, "role": "user", "event": "audio_start", "t_ms": 1200}
{"turn_index": 0, "role": "user", "event": "audio_end", "t_ms": 9800, "duration_ms": 8600}
{"turn_index": 1, "role": "user", "event": "audio_start", "t_ms": 12000}
{"turn_index": 1, "role": "user", "event": "audio_end", "t_ms": 14500, "duration_ms": 2500}
```

`t_ms` is wall-clock milliseconds from the first frame seen by TimingWriter.

### The gap

`TimingWriter` VAD boundaries do not produce WAV files — only events.
`AudioRecorder` WAV files are split by role switch, not by silence. A user
who speaks for 90 seconds straight (no role switch) produces one
`turn_0.wav` that may span multiple TimingWriter turns.

For TTS fine-tuning, Emilia-Pipe requires 3–30 second clips. A 90-second
WAV is too long and contains internal silences. This pipeline closes the gap.

### Resolved design question: slice from `audio/user/turn_*.wav`, not `audio.wav`

The previous draft proposed slicing from `audio.wav` (the mixed session
recording). That introduces a coach-overlap risk: if the coach interrupted
mid-utterance, the mixed WAV contains coach audio in the same time window.

**Decision**: slice from `audio/user/turn_*.wav` (user-only files).

The offset model uses byte arithmetic against the per-role WAV files, where
the start offset of each VAD sub-turn is derived relative to its containing
role-switch block. See §3.

---

## 2. Scope

In scope:
- Offline batch processor consuming `audio/user/turn_*.wav`,
  `timing.jsonl`, `transcript.jsonl`, and `session.json` for a finalized
  session.
- User-side audio clips only.
- Clip length enforcement: discard clips < 3s; split clips > 30s at the
  nearest internal silence boundary using Silero-VAD.
- Transcript alignment: span-match from final `transcript.jsonl` records.
- Output: `pipeline/clips/clips.jsonl` (rows of `AudioClipRecord`) and
  per-clip WAV files at `pipeline/clips/clip_NNNN.wav`.

Out of scope:
- Real-time processing during the call.
- Coach-side clips.
- Source separation or quality filtering — those run after this step in
  `v2026-05-13-audio-source-separation.md`.
- Changing `AudioRecorder` or `TimingWriter` runtime behavior.

---

## 3. Offset model: timing events → WAV byte positions

### Role-switch blocks

`AudioRecorder` increments `turn_index["user"]` each time a coach frame
arrives while user was active, then opens a new `user/turn_N.wav` when the
next user frame arrives. `TimingWriter` increments its `user` turn_index
on role switch AND on 600ms silence.

The mapping between role-switch WAV index N and TimingWriter turn indices:

1. Parse all user `audio_start`/`audio_end` pairs from `timing.jsonl`,
   sorted ascending by `t_ms`.
2. Parse all coach `audio_start`/`audio_end` pairs similarly.
3. Assign each user timing pair to a role-switch block by finding, in
   order, the N-th gap where a coach turn falls between two user timing
   pairs. Each gap boundary increments the role-switch block index.
4. All user timing pairs before the first coach gap belong to block 0
   (i.e. `audio/user/turn_0.wav`), those between the first and second
   coach gaps belong to block 1, and so on.

### Byte offset within a role-switch WAV

Within block N, let `block_anchor_ms` = `t_ms` of the first
`audio_start` event in that block. For any timing sub-turn in block N:

```
relative_start_ms = sub_turn_start_ms - block_anchor_ms
relative_end_ms   = sub_turn_end_ms   - block_anchor_ms
byte_start = int(relative_start_ms / 1000 * 16_000 * 2)
byte_end   = int(relative_end_ms   / 1000 * 16_000 * 2)
```

PCM16 mono at 16kHz: 2 bytes per sample, 16,000 samples per second.

`relative_start_ms` and `relative_end_ms` are stored in `AudioClipRecord`
as `start_ms` and `end_ms` so the enhancement step can re-slice if needed.

### Timing drift caveat

`TimingWriter` uses `time.monotonic()`. Under I/O load, monotonic time and
PCM byte position can diverge by tens of milliseconds per turn. For TTS
fine-tuning, a few frames of silence padding at clip edges is acceptable.
If empirical validation shows drift > 100ms, switch to a PCM-derived
anchor: record total bytes written by `AudioRecorder` at the moment of
each role switch and store that in a `timing_anchors.jsonl` artifact.
That change is not required now.

---

## 4. Transcript alignment

`transcript.jsonl` final records carry `ts_start` / `ts_end` as Unix
timestamps (float seconds). `Session.created_at` in `session.json` is the
call start wall-clock time.

Convert a timing event's `t_ms` to an absolute timestamp:

```python
abs_ts = session.created_at.timestamp() + t_ms / 1000
```

For each accepted clip `(abs_start, abs_end)`, collect all final
`transcript.jsonl` records for `speaker == "user"` whose `[ts_start, ts_end]`
overlaps `[abs_start, abs_end]`. Concatenate their text sorted by `ts_start`.

If no records overlap: `text = None`, `transcript_missing = True`.
The clip is still written to disk; it is excluded downstream.

---

## 5. Functional requirements

**F-V1.** The pipeline MUST skip sessions where `Session.consent != GRANTED`.

**F-V2.** The pipeline MUST parse `timing.jsonl` and assign each user
`(audio_start, audio_end)` pair to a role-switch block (§3). The block
index determines which `audio/user/turn_N.wav` to slice from.

**F-V3.** Clips with `duration_ms < 3_000` MUST be written to the manifest
with `status = "rejected_too_short"` and no WAV file emitted.

**F-V4.** Clips with `duration_ms > 30_000` MUST be recursively split at
the longest internal silence detected by Silero-VAD. Each resulting
sub-clip is independently subject to F-V3 and F-V4. Silero-VAD runs on
the in-memory PCM buffer; no intermediate file is written.

**F-V5.** For each accepted clip, the pipeline MUST:
  1. Seek to `byte_start` in `audio/user/turn_{block}.wav`.
  2. Read `byte_end - byte_start` bytes.
  3. Write a valid 16kHz PCM16 mono WAV to
     `pipeline/clips/clip_{clip_index:04d}.wav`.

**F-V6.** The pipeline MUST write `pipeline/clips/clips.jsonl` with one
`AudioClipRecord` (from `rehearse/types.py`) per processed clip, including
rejected ones. Accepted records have `wav_path` populated; rejected records
have `wav_path = ""`.

**F-V7.** After writing `clips.jsonl`, the pipeline MUST register it in
`session.json` by setting `artifact_paths["clips"] = "pipeline/clips/clips.jsonl"`.

**F-V8.** The pipeline MUST be idempotent: re-running overwrites
`pipeline/clips/` and rewrites `clips.jsonl`.

**F-V9.** The pipeline MUST be runnable as:
  - `python -m rehearse.pipeline.vad_segment <session_id>`
  - `python -m rehearse.pipeline.vad_segment --all`

---

## 6. Non-functional requirements

**N-P1.** For a typical 10-minute session, processing MUST complete in
under 60 seconds on CPU. WAV slicing is I/O-bound; Silero-VAD runs at
~100× real-time.

**N-Q1.** Each output WAV MUST be readable by `soundfile.read()` without
error before its manifest entry is written.

**N-L1.** Sessions without consent MUST NOT be touched. Check
`session.json` before opening any audio file.

---

## 7. Dependencies

| Role | Tool | License |
|---|---|---|
| WAV I/O | `soundfile` | BSD |
| Long-clip splitting (F-V4) | `silero-vad` | MIT |
| Pydantic serialization | `rehearse/types.py AudioClipRecord` | — |

---

## 8. Output layout (inside session directory)

```
sessions/{session_id}/
├── audio/user/turn_0.wav      ← source (existing, untouched)
├── audio/user/turn_1.wav
├── timing.jsonl               ← source (existing, untouched)
├── transcript.jsonl           ← source (existing, untouched)
└── pipeline/
    └── clips/
        ├── clips.jsonl        ← AudioClipRecord rows
        ├── clip_0000.wav      ← 3–30s, 16kHz PCM16 mono, user-only
        ├── clip_0001.wav
        └── ...
```

`pipeline/clips/` is the input directory for the enhancement pipeline
defined in `v2026-05-13-audio-source-separation.md`.

---

## 9. Pipeline position

```
[Live call]
  AudioRecorder  → audio/user/turn_*.wav          (16kHz, role-switch blocks)
  TimingWriter   → timing.jsonl                   (VAD turn boundaries)
  TranscriptWriter → transcript.jsonl

[vad_segment — this spec]
  Input:  audio/user/turn_*.wav + timing.jsonl + transcript.jsonl
  Output: pipeline/clips/clips.jsonl + clip_NNNN.wav (16kHz, 3–30s)

[audio_enhance — v2026-05-13-audio-source-separation.md]
  Input:  pipeline/clips/clip_*.wav
  Output: pipeline/enhanced/manifest.jsonl + clip_NNNN_enhanced.wav (24kHz)

[TTS fine-tuning]
  Input:  accepted (EnhancedClipRecord.wav_path, AudioClipRecord.text) pairs
```
