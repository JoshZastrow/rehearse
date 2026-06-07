# Design 3 — Waveform Split

## Visual Layout

```
┌─────────────────────────────────────┐
│ ← designs    Design 3 · Waveform  [End]│  ← nav bar
├─────────────────────────────────────┤
│                                     │
│  YOU                                │
│  What kind of difficult             │
│  conversation would you like        │  ← transcript panel (60%)
│  to practice?                       │     chat-bubble layout
│                                     │     user bubbles right-aligned
│                    COACH            │     agent bubbles left-aligned
│       I'm here to help. Tell        │
│       me about the situation.       │
│                                     │
│  Sure, I need to have a...          │  ← streaming (faded)
│                                     │
├─────────────────────────────────────┤
│  YOU          │    COACH            │
│  ▐▌▐▌▐▐▌▐▌▐▌│  ▐▌▐▌▐▐▌▐▌▐▌    🎙 │  ← waveform panel (40%)
│  ██████████  │  ██████████         │     left = user (purple)
│  ▐▌▐▌▐▐▌▐▌▐▌│  ▐▌▐▌▐▐▌▐▌▐▌       │     right = agent (blue)
└─────────────────────────────────────┘
```

## Waveform Rendering

```
History buffer: 80 samples (FIFO)
Bar width:      canvas.width / 80
Bar height:     max(2px, sample × canvas.height × 0.85)
Bar alpha:      0.3 + sample × 0.7  (louder = more opaque)
Color (user):   #8844ff  (purple, with alpha)
Color (agent):  #4488ff  (blue, with alpha)

Inactive side:  level always 0 → flat 2px baseline bars
Active side:    audioLevel from VoiceSession hook
```

## Transcript Bubble System

```
Speaker  │  Alignment  │  Label color  │  Label text
─────────────────────────────────────────────────────
user     │  flex-end   │  #8844ff      │  "You"
agent    │  flex-start │  #4488ff      │  "Coach"

Streaming entries: opacity 0.6 (in-progress)
Final entries:     opacity 1.0
Max width:         75% of container
```

## Two-Voice Design Rationale

Placing user and agent waveforms side-by-side makes turn-taking legible
at a glance — you see who just stopped speaking and who started. This
maps directly to what Rehearse users care about: am I talking too much?
Is the coach responsive? The transcript above gives content context while
the waveforms give prosodic/energy context. Together they create a
coaching-dashboard feel that differentiates Rehearse from a generic chat UI.

## Comparison to Design 1 & 2

```
Feature               │ D1 Gemini │ D2 Orb  │ D3 Waveform
──────────────────────────────────────────────────────────
Transcript history    │ no (6)    │ none    │ full history
Speaker identity      │ no        │ no      │ labeled
Audio visualization   │ orb       │ blob    │ dual bars
Distraction level     │ low       │ lowest  │ medium
Info density          │ medium    │ low     │ high
Best for              │ mobile    │ ambient │ desktop/coaching
```
