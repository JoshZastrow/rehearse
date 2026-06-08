# Design 1 — Gemini Style

## Visual Layout

```
┌─────────────────────────────────────┐
│ ← designs    Design 1 · Gemini      │  ← nav bar
├─────────────────────────────────────┤
│                                     │
│                                     │
│  Got it. I can tell you all about   │
│  the voice experience you're using. │  ← large transcript text
│                                     │     fades out for old lines
│  I'm a supportive AI coach for      │
│  Rehearse. What would you like to   │
│  practice today?                    │
│                                     │
│  We can chat about—                 │  ← streaming (faded, smaller)
│                                     │
│                                     │
│                                     │
├─────────────────────────────────────┤
│  □   ↑   ●●●●●   🎙   ✕            │  ← control bar
│ cam shr  [orb]  mic  end           │
└─────────────────────────────────────┘
```

## Orb States

```
Idle/user speaking:          Agent speaking:
    ┌──────┐                     ┌──────┐
    │ 🟣   │  purple             │ 🔵   │  blue
    │ glow │  #8844ff            │ glow │  #4488ff
    └──────┘                     └──────┘
     scale: 1.0 + audioLevel       scale: 1.0 + audioLevel
```

## Transcript Rendering Rules

```
Entry age / type        │ Font size │ Color
──────────────────────────────────────────────
Current (streaming)     │  28px     │ #aaa (dimmed until final)
Current (final)         │  28px     │ #fff
Previous entries        │  20px     │ #555 (faded)
Max visible entries     │  6        │ oldest scroll out
```

## State Machine

```
         connect()
idle ────────────────► connecting
  ▲                         │ room.connect() OK
  │         error           ▼
  │    ◄────────────── connected ──────────────► disconnect()
  │                         │                        │
  │                         │ DataReceived            │
  │                         ▼                        │
  │                   transcript[] update             │
  │                   activeSpeaker update            │
  │                   audioLevel update               │
  └───────────────────────────────────────────────────┘
```

## Key Design Decisions

- **No persistent history**: Only last 6 entries visible — matches Gemini Live behavior
  where older text fades rather than scrolling into a log.
- **Orb over avatar**: Avoids uncanny valley; communicates activity without a face.
- **Single large font**: Readable at arm's length on a phone; no header/footer chrome.
