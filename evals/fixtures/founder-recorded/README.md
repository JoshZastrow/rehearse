# Founder-recorded fixtures

Three self-recorded audio clips used as a smoke fixture for the audio-native
eval path. Stand-in for real MME-Emotion clips until that dataset is vendored.

## How to record

Record each clip on your phone (Voice Memos works) for ~8–15 seconds. Aim
for genuine prosody — the point is to test whether models pick up emotion
from how you say it, not from what you say.

| Filename | Emotion | Suggested line (improvise freely) |
|---|---|---|
| `clip-001-frustration.m4a` | Frustration | "I've told you three times now, and nothing changes. I don't know what I'm supposed to do anymore." |
| `clip-002-sadness.m4a` | Sadness | "I just keep thinking about how things used to be. It feels like we're not the same people we were." |
| `clip-003-hopeful.m4a` | Hopeful | "I think this could actually work. I haven't felt this way about a project in a long time." |

Drop the files into `clips/` (this directory). m4a is fine; Gemini accepts it.

## How to run the eval against these clips

The MME-Emotion eval reads its manifest path from
`MME_EMOTION_MANIFEST_PATH`. Point it at this manifest:

```bash
set -a && source .env && set +a
export MME_EMOTION_MANIFEST_PATH=evals/fixtures/founder-recorded/manifest.json

uv run rehearse-eval run \
  --eval mme-emotion \
  --environment multimodal-llm \
  --provider gemini \
  --limit 3 \
  --concurrency 2
```

You'll see something like:

```
run_id: 20260428T...
examples: 3 (ok=3 error=0 timeout=0)
  mme_recognition_accuracy: 1.000
```

## What's deliberately not here

- **Audio extraction from video.** Clips are audio-only; no video track.
- **Hopeful in the MME-Emotion label set.** The "real" MME-Emotion label set
  is 9 classes (no Hopeful). We added Hopeful here because the third clip
  needed it. When real MME-Emotion clips land, drop the Hopeful slot and
  use the original 9 classes.
- **Calibration.** These three clips are not enough to claim anything about
  any model. They exist to prove the wiring works end-to-end.
