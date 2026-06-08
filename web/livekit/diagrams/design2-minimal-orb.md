# Design 2 — Minimal Orb

## Visual Layout

```
┌─────────────────────────────────────┐
│ ← designs    Design 2 · Minimal     │  ← minimal nav (low opacity)
│                                     │
│                                     │
│                                     │
│                                     │
│           ╭──────────╮              │
│         ╭╯            ╰╮            │
│        ╱                ╲           │
│       │    morphing      │          │  ← canvas blob, full-screen
│       │     blob         │          │     centered
│        ╲                ╱           │
│         ╰╮            ╭╯            │
│           ╰──────────╯              │
│                                     │
│                                     │
│                                     │
│              🎙    ✕               │  ← fade-in controls
└─────────────────────────────────────┘
```

## Blob Animation Parameters

```
Base radius:     32% of min(canvas width, height)
Wobble layers:   sin(angle×3 + t×1.1) × r×0.12
                 sin(angle×5 + t×0.7) × r×0.06
Audio reactive:  audioLevel × r×0.18 × sin(angle×2 + t×3)
Pulse:           r += audioLevel × baseR × 0.6

Speed:           connecting → t += dt×0.4 (slow idle)
                 connected  → t += dt×1.0 (normal)
```

## Color Mapping

```
activeSpeaker   hue     gradient
──────────────────────────────────────────────
null / user     270     purple   #8844ff family
agent           220     blue     #4488ff family

Glow:  shadowBlur = 60 + audioLevel×40
       shadowColor = hsla(hue, 90%, 60%, 0.6)
```

## Interaction Model

```
State: idle
  → Tap anywhere on screen to connect
  → Shows "tap to begin" hint

State: connecting
  → Slow-pulsing idle blob (t speed 0.4×)
  → No controls visible

State: connected
  → Full-speed reactive blob
  → Mic toggle + end call fade in at bottom

State: error
  → Error text replaces blob
  → No tap-to-connect (avoids retry loop)
```

## Design Rationale

Pure distraction-free voice UI. No text means users focus entirely on
speaking naturally rather than reading or watching a transcript. The
canvas blob provides just enough visual feedback that the session feels
"alive" without dominating attention. Inspired by Apple's Siri orb and
the ambient-computing design aesthetic.
