/**
 * Live Speech Dialogue
 *
 * Full-duplex call UI: two side-by-side segmented equalizers — YOU (caller,
 * left) and SYSTEM (provider, right) — each driven by its own audio activity.
 * Below each equalizer an RSVP reader streams that speaker's transcript one
 * word at a time: the previous word is greyed, the focus word is bright with
 * its center (pivot) letter in red, the next word dim.
 *
 * Header shows LIVE / CONNECTED status; bottom bar has Mute, a large mic
 * start/stop ring, and Settings.
 */

import { useEffect, useRef, useState } from 'react'
import { useVoiceSession, type TranscriptEntry } from '../hooks/useVoiceSession'

// ── segmented equalizer ───────────────────────────────────────────────────────

const COLUMNS = 26
const ROWS = 15

/**
 * Lit-cell color by row index (0 = bottom). A warm-digital ramp:
 * jungle green → olive → sandalwood → warm clay.
 */
function cellColor(row: number): string {
  const t = row / (ROWS - 1)
  if (t < 0.5) return '#2f8f68' // jungle green
  if (t < 0.72) return '#6f9b54' // olive
  if (t < 0.88) return '#c2a06a' // sandalwood
  return '#cf6a43' // warm clay
}

/**
 * A column of stacked LED cells. `lit` cells (from the bottom) glow in their
 * row color; the rest sit dark. Renders as a spectrum-analyzer bar.
 */
function EqColumn({ lit }: { lit: number }) {
  const cells = []
  for (let row = ROWS - 1; row >= 0; row--) {
    const on = row < lit
    cells.push(
      <span
        key={row}
        style={{
          height: 'var(--cell-h)',
          borderRadius: 1.5,
          background: on ? cellColor(row) : '#1a160e',
          boxShadow: on ? `0 0 4px ${cellColor(row)}aa` : 'none',
          opacity: on ? 1 : 0.6,
          transition: 'background 70ms linear, opacity 70ms linear',
        }}
      />,
    )
  }
  return <div style={eqColumn}>{cells}</div>
}

/**
 * Segmented equalizer. A smooth hump profile (tallest toward the panel's inner
 * edge) is scaled by live audio level and animated with per-column jitter so
 * the two panels read as a mirrored, breathing spectrum.
 */
function Equalizer({
  side,
  active,
  level,
}: {
  side: 'left' | 'right'
  active: boolean
  level: number
}) {
  const [heights, setHeights] = useState<number[]>(() => Array(COLUMNS).fill(0))
  const frameRef = useRef(0)
  const tRef = useRef(0)
  const lastRef = useRef(performance.now())
  const levelRef = useRef(level)
  const activeRef = useRef(active)
  levelRef.current = level
  activeRef.current = active

  useEffect(() => {
    function draw(now: number) {
      const dt = Math.min((now - lastRef.current) / 1000, 0.05)
      lastRef.current = now
      tRef.current += dt
      const t = tRef.current
      const lvl = levelRef.current
      const on = activeRef.current

      const next = new Array(COLUMNS)
      for (let i = 0; i < COLUMNS; i++) {
        // Hump profile: peak toward the inner edge of each panel.
        const pos = side === 'left' ? i / (COLUMNS - 1) : 1 - i / (COLUMNS - 1)
        const hump = Math.sin(Math.PI * (0.25 + 0.6 * pos))
        const wobble =
          0.5 +
          0.5 * Math.sin(t * 9 + i * 0.7) * Math.sin(t * 3.3 + i * 0.31)
        const amp = on
          ? hump * (0.35 + lvl * 0.9) * (0.55 + 0.45 * wobble)
          : hump * 0.06 * (0.5 + 0.5 * Math.sin(t * 1.6 + i * 0.5))
        next[i] = Math.max(0, Math.min(1, amp))
      }
      setHeights(next)
      frameRef.current = requestAnimationFrame(draw)
    }
    frameRef.current = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(frameRef.current)
  }, [side])

  return (
    <div style={eqGrid}>
      {heights.map((h, i) => (
        <EqColumn key={i} lit={Math.round(h * ROWS)} />
      ))}
    </div>
  )
}

// ── RSVP per-speaker reader ───────────────────────────────────────────────────

interface RsvpFrame {
  prev: string
  current: string
  next: string
}

/**
 * Stream a single speaker's transcript as a paced word sequence. Returns the
 * previous / focus / next words so the reader can grey context around the
 * bright focus word. New tail words are appended as partial entries grow.
 */
function useRsvpStream(
  transcript: TranscriptEntry[],
  speaker: 'user' | 'agent',
  active: boolean,
): RsvpFrame {
  const [frame, setFrame] = useState<RsvpFrame>({ prev: '', current: '', next: '' })
  const wordsRef = useRef<string[]>([])
  const queuedRef = useRef(new Map<string, number>())
  const idxRef = useRef(-1)

  useEffect(() => {
    for (const entry of transcript) {
      if (entry.speaker !== speaker) continue
      const words = entry.text.split(/\s+/).filter(Boolean)
      const queued = queuedRef.current.get(entry.id) ?? 0
      if (words.length > queued) {
        wordsRef.current.push(...words.slice(queued))
        queuedRef.current.set(entry.id, words.length)
      }
    }
  }, [transcript, speaker])

  useEffect(() => {
    if (!active) {
      wordsRef.current = []
      queuedRef.current = new Map()
      idxRef.current = -1
      setFrame({ prev: '', current: '', next: '' })
      return
    }
    let timer = 0
    function tick() {
      const words = wordsRef.current
      const backlog = words.length - 1 - idxRef.current
      if (backlog > 0) {
        idxRef.current += 1
        const i = idxRef.current
        setFrame({
          prev: words[i - 1] ?? '',
          current: words[i] ?? '',
          next: words[i + 1] ?? '',
        })
      }
      // ~240 wpm steady; rush when a long response lands at once.
      let delay = 250
      if (backlog > 12) delay = 130
      else if (backlog <= 0) delay = 90
      timer = window.setTimeout(tick, delay)
    }
    tick()
    return () => clearTimeout(timer)
  }, [active, speaker])

  return frame
}

/** Split a word at its optical-recognition-point (center) letter. */
function pivotSplit(word: string): [string, string, string] {
  if (!word) return ['', '', '']
  const p = Math.floor((word.length - 1) / 2)
  return [word.slice(0, p), word[p], word.slice(p + 1)]
}

function RsvpReader({
  frame,
  accent,
  speaker,
}: {
  frame: RsvpFrame
  accent: string
  speaker: 'user' | 'agent'
}) {
  const [before, pivot, after] = pivotSplit(frame.current)
  const has = Boolean(frame.current)
  return (
    <div style={{ ...rsvpPill, borderColor: `${accent}55` }} data-testid="rsvp-pill">
      {has ? (
        <span
          style={rsvpLine}
          data-testid="transcript-entry"
          data-speaker={speaker}
        >
          {frame.prev && <span style={rsvpContext}>{frame.prev}</span>}
          <span style={rsvpFocus}>
            {before}
            <span style={rsvpPivot} data-testid="rsvp-pivot">{pivot}</span>
            {after}
          </span>
          {frame.next && <span style={rsvpContext}>{frame.next}</span>}
        </span>
      ) : (
        <span style={rsvpDots}>· · ·</span>
      )}
    </div>
  )
}

// ── speaker panel (avatar + status + equalizer + reader) ──────────────────────

function PersonIcon({ color }: { color: string }) {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.6">
      <circle cx="12" cy="8" r="3.4" />
      <path d="M5 20c0-3.6 3.1-6 7-6s7 2.4 7 6" />
    </svg>
  )
}

function RobotIcon({ color }: { color: string }) {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.6">
      <rect x="4.5" y="7.5" width="15" height="11" rx="3" />
      <circle cx="9" cy="13" r="1.3" fill={color} stroke="none" />
      <circle cx="15" cy="13" r="1.3" fill={color} stroke="none" />
      <path d="M12 4.5v3M8 18.5v2M16 18.5v2" />
    </svg>
  )
}

function SpeakerPanel({
  side,
  label,
  accent,
  active,
  level,
  frame,
  speaker,
}: {
  side: 'left' | 'right'
  label: string
  accent: string
  active: boolean
  level: number
  frame: RsvpFrame
  speaker: 'user' | 'agent'
}) {
  const icon = speaker === 'user' ? <PersonIcon color={accent} /> : <RobotIcon color={accent} />
  const meta = (
    <div style={panelMeta}>
      <span style={iconRing(accent)}>{icon}</span>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        <span style={{ ...panelLabel, color: accent }}>{label}</span>
        <span
          style={{ ...statusRow, color: active ? accent : '#3f5563' }}
          data-testid={`status-${speaker}`}
        >
          <span style={{ ...statusDot, background: active ? accent : '#334155' }} />
          {active ? 'SPEAKING' : 'LISTENING'}
        </span>
      </div>
    </div>
  )
  return (
    <div style={panel}>
      <div style={{ ...panelTop, flexDirection: side === 'left' ? 'row' : 'row-reverse' }}>
        {meta}
      </div>
      <Equalizer side={side} active={active} level={level} />
      <RsvpReader frame={frame} accent={accent} speaker={speaker} />
    </div>
  )
}

// ── main component ────────────────────────────────────────────────────────────

const YOU_ACCENT = '#3f9b78' // jungle green
const SYSTEM_ACCENT = '#c2a06a' // sandalwood
const LIVE_GREEN = '#3f9b78'
const WARM_MUTED = '#8a7d6a'
const WARM_TAUPE = '#6b6151'

export default function LiveSpeechDialogue() {
  const session = useVoiceSession()
  const state = session.connectionState
  const connected = state === 'connected'

  const callerActive = connected && session.activeSpeaker === 'user'
  const providerActive = connected && session.activeSpeaker === 'agent'

  const youFrame = useRsvpStream(session.transcript, 'user', connected)
  const systemFrame = useRsvpStream(session.transcript, 'agent', connected)

  // Provider audio level isn't measured locally; approximate from activity.
  const [synthLevel, setSynthLevel] = useState(0)
  useEffect(() => {
    if (!providerActive) return
    let frame = 0
    let t = 0
    const loop = () => {
      t += 0.05
      setSynthLevel(0.45 + 0.4 * Math.abs(Math.sin(t * 2.1)))
      frame = requestAnimationFrame(loop)
    }
    frame = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(frame)
  }, [providerActive])

  const statusText =
    state === 'connected'
      ? 'CONNECTED'
      : state === 'connecting'
        ? 'CONNECTING'
        : state === 'error'
          ? 'CONNECTION FAILED'
          : 'TAP TO START'

  const onMic = () => {
    if (state === 'idle' || state === 'error') void session.connect()
    else if (connected) session.disconnect()
  }

  return (
    <div style={shell}>
      <style>{`
        @keyframes lsd-blink { 0%,100%{opacity:1} 50%{opacity:.3} }
        @keyframes lsd-pulse { 0%,100%{transform:scale(1);opacity:.7} 50%{transform:scale(1.12);opacity:.25} }
      `}</style>

      {/* ── header ──────────────────────────────────────────────────────────── */}
      <header style={header}>
        <div style={headerSide}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={LIVE_GREEN} strokeWidth="2">
            <path d="M3 12h3l2-7 4 14 2-7h3" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span style={{ ...miniLabel, color: connected ? LIVE_GREEN : WARM_TAUPE }}>LIVE</span>
        </div>

        <div style={headerCenter}>
          <h1 style={title}>REHEARSE</h1>
          <div style={statusLine}>
            <span
              style={{
                ...statusDot,
                background: connected ? LIVE_GREEN : WARM_TAUPE,
                animation: connected ? 'lsd-blink 1.4s ease-in-out infinite' : 'none',
              }}
            />
            <span style={{ ...statusText_, color: connected ? LIVE_GREEN : WARM_MUTED }}>
              {statusText}
            </span>
          </div>
        </div>

        <div style={headerSide}>
          <button style={gearBtn} aria-label="Settings">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={WARM_MUTED} strokeWidth="1.6">
              <circle cx="12" cy="12" r="3" />
              <path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M19 5l-2 2M7 17l-2 2" />
            </svg>
          </button>
        </div>
      </header>

      {/* ── dual equalizer + reader ─────────────────────────────────────────── */}
      <main style={stage}>
        <SpeakerPanel
          side="left"
          label="YOU"
          accent={YOU_ACCENT}
          active={callerActive}
          level={callerActive ? session.audioLevel : 0}
          frame={youFrame}
          speaker="user"
        />
        <div style={centerGap} />
        <SpeakerPanel
          side="right"
          label="SYSTEM"
          accent={SYSTEM_ACCENT}
          active={providerActive}
          level={providerActive ? synthLevel : 0}
          frame={systemFrame}
          speaker="agent"
        />
      </main>

      {/* ── bottom control bar ──────────────────────────────────────────────── */}
      <footer style={bottomWrap}>
        <div style={controlBar}>
          <button
            style={micBtn(connected, state === 'connecting')}
            onClick={onMic}
            aria-label={connected ? 'End call' : 'Start call'}
          >
            {connected && <span style={micPulse} />}
            {connected ? (
              <svg width="22" height="22" viewBox="0 0 24 24" fill="#0e1a12" stroke="#0e1a12" strokeWidth="2">
                <rect x="7" y="7" width="10" height="10" rx="2" />
              </svg>
            ) : (
              <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#7fb497" strokeWidth="1.9">
                <path
                  d="M6.6 10.8a15 15 0 0 0 6.6 6.6l2.2-2.2a1 1 0 0 1 1-.24 11.4 11.4 0 0 0 3.6.58 1 1 0 0 1 1 1V20a1 1 0 0 1-1 1A17 17 0 0 1 3 4a1 1 0 0 1 1-1h3.5a1 1 0 0 1 1 1 11.4 11.4 0 0 0 .58 3.6 1 1 0 0 1-.24 1z"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            )}
          </button>
        </div>
      </footer>
    </div>
  )
}

// ── styles ────────────────────────────────────────────────────────────────────

const shell: React.CSSProperties = {
  minHeight: '100dvh',
  background: 'radial-gradient(circle at 50% -10%, #15110a 0%, #0c0a06 58%, #070504 100%)',
  display: 'flex',
  flexDirection: 'column',
  color: '#ece0cd',
  fontFamily: 'system-ui, -apple-system, sans-serif',
  userSelect: 'none',
}

const header: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  padding: '16px 24px 8px',
  gap: 12,
  flexShrink: 0,
}

const headerSide: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 6,
  minWidth: 90,
}

const miniLabel: React.CSSProperties = {
  fontSize: 11,
  fontWeight: 700,
  letterSpacing: '0.18em',
}

const headerCenter: React.CSSProperties = {
  flex: 1,
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  gap: 5,
}

const title: React.CSSProperties = {
  margin: 0,
  fontSize: 17,
  fontWeight: 600,
  letterSpacing: '0.34em',
  color: '#f1e8d6',
  textIndent: '0.34em',
}

const statusLine: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 7,
}

const statusText_: React.CSSProperties = {
  fontSize: 10,
  fontWeight: 600,
  letterSpacing: '0.22em',
}

const statusDot: React.CSSProperties = {
  width: 7,
  height: 7,
  borderRadius: '50%',
  flexShrink: 0,
}

const gearBtn: React.CSSProperties = {
  marginLeft: 'auto',
  background: 'none',
  border: 'none',
  cursor: 'pointer',
  padding: 4,
  display: 'flex',
}

const stage: React.CSSProperties = {
  flex: 1,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  padding: '8px 28px',
  gap: 0,
}

const centerGap: React.CSSProperties = {
  width: 56,
  flexShrink: 0,
}

const panel: React.CSSProperties = {
  flex: 1,
  maxWidth: 460,
  display: 'flex',
  flexDirection: 'column',
  gap: 16,
}

const panelTop: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
}

const panelMeta: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 10,
}

const iconRing = (accent: string): React.CSSProperties => ({
  width: 40,
  height: 40,
  borderRadius: '50%',
  border: `1.5px solid ${accent}66`,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  boxShadow: `0 0 14px ${accent}33`,
  flexShrink: 0,
})

const panelLabel: React.CSSProperties = {
  fontSize: 14,
  fontWeight: 700,
  letterSpacing: '0.2em',
}

const statusRow: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 6,
  fontSize: 9,
  fontWeight: 600,
  letterSpacing: '0.16em',
}

const eqGrid: React.CSSProperties = {
  display: 'flex',
  alignItems: 'flex-end',
  justifyContent: 'space-between',
  gap: 2,
  height: 190,
  // each cell height so ROWS cells + gaps fill the grid
  // (consumed via CSS var on the grid)
  ['--cell-h' as string]: `calc((190px - ${(ROWS - 1) * 2}px) / ${ROWS})`,
  padding: '0 2px',
}

const eqColumn: React.CSSProperties = {
  flex: 1,
  display: 'flex',
  flexDirection: 'column',
  gap: 2,
  justifyContent: 'flex-end',
}

const rsvpPill: React.CSSProperties = {
  height: 56,
  borderRadius: 14,
  border: '1px solid',
  background: 'rgba(14, 11, 6, 0.55)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  padding: '0 18px',
  overflow: 'hidden',
}

const rsvpLine: React.CSSProperties = {
  display: 'flex',
  alignItems: 'baseline',
  gap: 12,
  fontSize: 'clamp(1.4rem, 3vw, 2rem)',
  fontWeight: 600,
  letterSpacing: '0.01em',
  whiteSpace: 'nowrap',
}

const rsvpContext: React.CSSProperties = {
  color: '#6b6151', // warm taupe (previous / next word)
  fontSize: '0.78em',
  fontWeight: 500,
}

const rsvpFocus: React.CSSProperties = {
  color: '#f1e8d6', // warm off-white focus word
}

const rsvpPivot: React.CSSProperties = {
  color: '#e0503a', // warm red center letter
}

const rsvpDots: React.CSSProperties = {
  color: '#3a3328',
  fontSize: 20,
  letterSpacing: '0.4em',
}

const bottomWrap: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'center',
  padding: '12px 24px 28px',
  flexShrink: 0,
}

const controlBar: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 28,
  padding: '10px 22px',
  borderRadius: 999,
  border: '1px solid #2a2318',
  background: 'rgba(16, 12, 7, 0.72)',
}

const micBtn = (connected: boolean, connecting: boolean): React.CSSProperties => ({
  position: 'relative',
  width: 62,
  height: 62,
  borderRadius: '50%',
  border: `2px solid ${connected ? '#4caf86' : '#3f9b7888'}`,
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  background: connected
    ? 'linear-gradient(135deg, #4caf86, #2f8f68)'
    : 'radial-gradient(circle at 50% 40%, #1b2419, #0c130c)',
  boxShadow: connected ? '0 0 26px #3f9b7888' : '0 0 14px #3f9b7833',
  transition: 'background 0.3s, box-shadow 0.3s',
  opacity: connecting ? 0.7 : 1,
})

const micPulse: React.CSSProperties = {
  position: 'absolute',
  inset: -2,
  borderRadius: '50%',
  border: '2px solid #4caf86',
  animation: 'lsd-pulse 1.6s ease-in-out infinite',
  pointerEvents: 'none',
}
