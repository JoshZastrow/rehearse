# Design System — realtalk

## Product Context
- **What this is:** A terminal-based conversational game where players navigate emotionally charged social scenes with an LLM-powered character
- **Who it's for:** Developers and players who want a text game with real emotional stakes; sessions also serve as LLM training data
- **Space/industry:** Interactive fiction, terminal games, LLM-powered experiences
- **Project type:** Textual TUI application (Python)

## Aesthetic Direction
- **Direction:** Intimate Manuscript
- **Decoration level:** Minimal — ASCII scene illustrations only
- **Mood:** Correspondence, not a terminal. Reading someone's letter by lamplight. The screen should feel like you're in the room with them, not running a program.
- **Key insight:** Every terminal game defaults to black + green (hacker aesthetic). realtalk's content is human emotional dynamics — vulnerability, connection, tension. The visual environment should serve that content, not fight it.

## Color

- **Approach:** Single shifting accent — the one accent color tracks emotional state. Everything else is near-neutral.

### Base palette

| Role | Hex | Notes |
|------|-----|-------|
| Background | `#f5f0e8` | Warm parchment — light mode default |
| Action zone | `#ede8de` | Slightly cooler parchment for status/input area |
| Primary text | `#2a1f15` | Near-black with warm undertone |
| Muted text | `#8a7a65` | Warm taupe — instructions, labels, secondary info |
| Border/separator | `#c8c0b0` | Hairline only — barely there |

### Accent — shifts with mood value

| Mood range | Hex | Label | Notes |
|------------|-----|-------|-------|
| ≥ 65 | `#c4703a` | Warm | Terracotta — connection, closeness |
| 35–64 | `#c4a03a` | Present | Amber — uncertain, holding |
| < 35 | `#5b8fa8` | Distant | Steel blue — cold, withdrawn |

The accent applies to: status bar fill, status label, reaction input highlight, any interactive emphasis.

### Status bar colors

| State | Hex | MOOD label | SECURITY label |
|-------|-----|------------|----------------|
| High (≥ 65) | `#7da85a` | Warm | Grounded |
| Mid (35–64) | `#c4a03a` | Strained | Uneasy |
| Low (< 35) | `#9e4a4a` | Fragile | Exposed |

### Dark mode
Invert to warm dark palette — same hue relationships, increased contrast:

| Role | Hex |
|------|-----|
| Background | `#1a1510` |
| Action zone | `#26200e` |
| Primary text | `#e8d5b0` |
| Muted text | `#8a7a5a` |
| Border | `#3a3020` |

Accent hex values unchanged in dark mode.

## Typography

- **Font:** JetBrains Mono — richer glyph set than system mono, better Unicode block character support for status bars
- **Fallback:** `'Courier New', monospace`
- **Scale:** Single size (13–14px). Weight and color do the hierarchy work, not size.

### Hierarchy via Rich markup

| Role | Markup | Appearance |
|------|--------|-----------|
| Active dialogue | `[bold]text[/bold]` | Full weight, primary text color |
| Scene description | `[italic dim]text[/italic dim]` | Italic, muted text color |
| Instructions / hints | `[dim]text[/dim]` | Dim muted — low visual weight |
| Options | plain | Regular weight, primary text |
| Status labels | plain | Muted color via widget style |

## Status Bars

Replace `[======    ] 35` with block character gradient fills.

**Fill characters:** `█▓▒░` — full to empty
**Formula:** 10-char bar. `filled = round(value / 10)`, remainder uses `▒` then `░`.

**Labels change with state:**

```
MOOD  ████████▒░  Warm      84
MOOD  ███▒░░░░░░  Strained  38
MOOD  █░░░░░░░░░  Fragile   14

SECURITY  █████▒░░░░  Grounded  59
SECURITY  ████░░░░░░  Uneasy    42
SECURITY  ██░░░░░░░░  Exposed   18
```

## Scene Headers (ASCII Illustrations)

Each scene type gets a small ASCII art illustration header before dialogue begins. Centered, muted color, 4–5 lines.

```
Coffee Shop           Late Night Office     Hiking Trail

   ( )                    💡                    /\
  (   )                 ┌───┐                  /  \
   \_/                  │   │                 /────\
  ──────                └───┘                 ────
```

Display format:
```
[centered ASCII art, muted color]
[SCENE NAME · ROLE]  ← uppercase, letter-spaced, muted
────────────────────────────────
```

## Layout

```
┌─────────────────────────────────────────┐
│  [scene header: ASCII + scene/role]      │  bg: #f5f0e8
│  ────────────────────────────────────    │
│                                          │
│  [dialogue area — borderless]            │  bg: #f5f0e8
│  Bold text, generous top/bottom padding  │  padding: 24px 28px
│                                          │
│  ────────────────────────────────────    │
│  MOOD  ████████▒░  Warm          84      │  bg: #ede8de
│  SECURITY  █████▒░░░░  Grounded  59      │
│  ────────────────────────────────────    │
│  Reaction: a2 (attract, intensity 2)     │
│  ────────────────────────────────────    │
│  1.  Option one text                     │
│  2.  Option two text                     │
│  3.  Option three text                   │
└─────────────────────────────────────────┘
```

- **Dialogue zone:** No border. 24px top/bottom padding. Text breathes.
- **Action zone:** Background `#ede8de`. Single hairline top border `─` separates it from dialogue.
- **Options:** Dim numbered prefix (`1.`), plain text for option content.
- **Title bar:** Minimal — `realtalk` left, `^C quit` right, muted.

## Motion

- **Dialogue streaming:** New dialogue text appears character-by-character at ~20ms/char
- **Status bar animation:** Values ease in when they change (Textual `animate()`)
- **Accent crossfade:** When mood crosses a tier boundary, accent color transitions over ~300ms
- Nothing else. Every animation is load-bearing.

## Implementation Notes (Textual)

**CSS location:** `RealTalkApp.CSS` (inline) or `realtalk/tui/app.tcss` (preferred for larger systems)

**Mood tier CSS classes** — apply to screen root, let children inherit:
```css
.mood-warm  { --accent: #c4703a; }
.mood-mid   { --accent: #c4a03a; }
.mood-cold  { --accent: #5b8fa8; }
```

**Block fill function:**
```python
def block_bar(value: int, width: int = 10) -> str:
    filled = round(value / 100 * width)
    bar = "█" * filled
    if filled < width:
        bar += "▒"
        bar += "░" * (width - filled - 1)
    return bar
```

**Status label lookup:**
```python
MOOD_LABELS = {range(65, 101): "Warm", range(35, 65): "Strained", range(0, 35): "Fragile"}
SECURITY_LABELS = {range(65, 101): "Grounded", range(35, 65): "Uneasy", range(0, 35): "Exposed"}
```

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-15 | Light parchment background instead of dark | Breaks hacker aesthetic that fights emotional content. Nobody in terminal game space does light mode. |
| 2026-04-15 | Single shifting accent tracks mood | Color change IS the emotional signal. One accent has more weight than a full palette. |
| 2026-04-15 | ASCII scene illustrations | Grounds the player in physical space before dialogue. 3 scenes = 3 illustrations, manageable. |
| 2026-04-15 | Block character status bars with emotional copy | HUD speaks the game's language ("Warm"/"Fragile") rather than exposing raw numbers. |
| 2026-04-15 | Borderless dialogue area | Text breathes. Intimate, not widget-y. |
