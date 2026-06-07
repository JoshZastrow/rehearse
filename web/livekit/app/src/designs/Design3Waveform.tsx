/**
 * Design 3 — Waveform Split
 *
 * Layout: two panels.
 *   Top (60%): scrollable transcript history with speaker labels.
 *   Bottom (40%): dual real-time waveform bars — user left, agent right.
 * Mirrors a call/interview layout with clear visual separation of voices.
 */

import { useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import { useVoiceSession, type TranscriptEntry } from '../hooks/useVoiceSession'

export default function Design3Waveform() {
  const session = useVoiceSession()
  return (
    <div style={shell}>
      <nav style={nav}>
        <Link to="/" style={backLink}>← designs</Link>
        <span style={navTitle}>Design 3 · Waveform Split</span>
        {session.connectionState === 'connected' && (
          <button style={endCallBtn} onClick={session.disconnect}>End</button>
        )}
      </nav>

      {session.connectionState === 'idle' && (
        <div style={centerContent}>
          <p style={idleHint}>Real-time waveform — split layout</p>
          <button style={startBtn} onClick={session.connect}>Start Session</button>
        </div>
      )}

      {session.connectionState === 'connecting' && (
        <div style={centerContent}><p style={idleHint}>Connecting…</p></div>
      )}

      {session.connectionState === 'error' && (
        <div style={centerContent}>
          <p style={{ color: '#f55', fontSize: 13 }}>Connection failed</p>
          <button style={startBtn} onClick={session.connect}>Retry</button>
        </div>
      )}

      {session.connectionState === 'connected' && (
        <>
          <TranscriptHistory entries={session.transcript} />
          <WaveformPanel
            audioLevel={session.audioLevel}
            activeSpeaker={session.activeSpeaker}
            isMicOn={session.isMicOn}
            onToggleMic={session.toggleMic}
          />
        </>
      )}
    </div>
  )
}

function TranscriptHistory({ entries }: { entries: TranscriptEntry[] }) {
  const bottomRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [entries])

  return (
    <div style={transcriptPanel}>
      {entries.length === 0 && (
        <p style={emptyHint}>Transcript will appear here…</p>
      )}
      {entries.map((e) => (
        <div key={e.id} style={{ ...bubble, alignSelf: e.speaker === 'user' ? 'flex-end' : 'flex-start' }}>
          <span style={{ ...speakerLabel, color: e.speaker === 'user' ? '#8844ff' : '#4488ff' }}>
            {e.speaker === 'user' ? 'You' : 'Coach'}
          </span>
          <p style={{ ...bubbleText, opacity: e.final ? 1 : 0.6 }}>{e.text}</p>
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  )
}

function WaveformPanel({
  audioLevel, activeSpeaker, isMicOn, onToggleMic,
}: {
  audioLevel: number
  activeSpeaker: 'user' | 'agent' | null
  isMicOn: boolean
  onToggleMic: () => void
}) {
  return (
    <div style={wavePanel}>
      <WaveBar label="You" color="#8844ff" level={activeSpeaker === 'user' ? audioLevel : 0} active={activeSpeaker === 'user'} />
      <div style={divider} />
      <WaveBar label="Coach" color="#4488ff" level={activeSpeaker === 'agent' ? audioLevel : 0} active={activeSpeaker === 'agent'} />
      <button
        style={{ ...micBtn, opacity: isMicOn ? 1 : 0.4 }}
        onClick={onToggleMic}
        aria-label="Toggle mic"
      >
        {isMicOn ? '🎙' : '🔇'}
      </button>
    </div>
  )
}

function WaveBar({ label, color, level, active }: {
  label: string; color: string; level: number; active: boolean
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const frameRef = useRef<number>(0)
  const historyRef = useRef<number[]>(Array(80).fill(0))

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!

    function draw() {
      historyRef.current.push(level)
      historyRef.current.shift()

      const W = canvas!.width
      const H = canvas!.height
      ctx.clearRect(0, 0, W, H)

      const barW = W / historyRef.current.length
      historyRef.current.forEach((v, i) => {
        const h = Math.max(2, v * H * 0.85)
        const alpha = 0.3 + v * 0.7
        ctx.fillStyle = `${color}${Math.round(alpha * 255).toString(16).padStart(2, '0')}`
        ctx.fillRect(i * barW, (H - h) / 2, barW - 1, h)
      })

      frameRef.current = requestAnimationFrame(draw)
    }

    frameRef.current = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(frameRef.current)
  }, [level, color])

  return (
    <div style={waveBarContainer}>
      <span style={{ ...waveLabel, color: active ? color : '#444' }}>{label}</span>
      <canvas ref={canvasRef} width={160} height={60} style={{ width: '100%', height: 60 }} />
    </div>
  )
}

// ── styles ────────────────────────────────────────────────────────────────────

const shell: React.CSSProperties = {
  height: '100dvh', background: '#080808', display: 'flex',
  flexDirection: 'column', overflow: 'hidden',
}
const nav: React.CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 12,
  padding: '12px 20px', borderBottom: '1px solid #111', flexShrink: 0,
}
const backLink: React.CSSProperties = { color: '#555', fontSize: 13, textDecoration: 'none' }
const navTitle: React.CSSProperties = { color: '#444', fontSize: 13, flex: 1 }
const endCallBtn: React.CSSProperties = {
  padding: '4px 14px', background: '#1a0808', border: '1px solid #3a1010',
  borderRadius: 6, color: '#f44', fontSize: 13, cursor: 'pointer',
}
const centerContent: React.CSSProperties = {
  flex: 1, display: 'flex', flexDirection: 'column',
  alignItems: 'center', justifyContent: 'center', gap: 20,
}
const idleHint: React.CSSProperties = { color: '#444', fontSize: 14 }
const startBtn: React.CSSProperties = {
  padding: '10px 28px', background: '#111', border: '1px solid #2a2a2a',
  borderRadius: 20, color: '#fff', fontSize: 14, cursor: 'pointer',
}
const transcriptPanel: React.CSSProperties = {
  flex: 1, overflowY: 'auto', padding: '16px 20px',
  display: 'flex', flexDirection: 'column', gap: 10,
}
const emptyHint: React.CSSProperties = {
  color: '#333', fontSize: 13, margin: 'auto', textAlign: 'center',
}
const bubble: React.CSSProperties = {
  maxWidth: '75%', display: 'flex', flexDirection: 'column', gap: 3,
}
const speakerLabel: React.CSSProperties = { fontSize: 10, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase' }
const bubbleText: React.CSSProperties = { fontSize: 14, lineHeight: 1.5, color: '#e0e0e0', margin: 0 }
const wavePanel: React.CSSProperties = {
  height: 120, flexShrink: 0, borderTop: '1px solid #111',
  display: 'flex', alignItems: 'center', gap: 0,
  background: '#050505', padding: '8px 20px',
  position: 'relative',
}
const divider: React.CSSProperties = {
  width: 1, height: 60, background: '#1a1a1a', flexShrink: 0, margin: '0 12px',
}
const waveBarContainer: React.CSSProperties = {
  flex: 1, display: 'flex', flexDirection: 'column', gap: 4,
}
const waveLabel: React.CSSProperties = {
  fontSize: 10, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase',
}
const micBtn: React.CSSProperties = {
  position: 'absolute', right: 20, bottom: 16,
  width: 36, height: 36, borderRadius: '50%',
  background: '#111', border: '1px solid #222',
  color: '#fff', fontSize: 14, cursor: 'pointer',
}
