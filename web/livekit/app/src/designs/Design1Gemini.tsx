/**
 * Design 1 — Gemini Style
 *
 * Layout: full black screen, large streaming transcript text upper-center,
 * bottom control bar with camera/share/animated-orb/mic/end-call icons.
 * The orb pulses and shifts blue↔purple based on who is speaking.
 */

import { useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import { useVoiceSession } from '../hooks/useVoiceSession'

export default function Design1Gemini() {
  const session = useVoiceSession()
  return (
    <div style={shell}>
      <nav style={nav}>
        <Link to="/" style={backLink}>← designs</Link>
        <span style={navTitle}>Design 1 · Gemini Style</span>
      </nav>

      {session.connectionState === 'idle' && (
        <div style={centerContent}>
          <p style={idleHint}>Tap to start your session</p>
          <button style={startBtn} onClick={session.connect}>Start Call</button>
        </div>
      )}

      {session.connectionState === 'connecting' && (
        <div style={centerContent}>
          <p style={idleHint}>Connecting…</p>
        </div>
      )}

      {session.connectionState === 'error' && (
        <div style={centerContent}>
          <p style={{ color: '#f55', fontSize: 14 }}>
            Connection error — check VITE_LIVEKIT_URL and VITE_TOKEN_ENDPOINT
          </p>
          <button style={startBtn} onClick={session.connect}>Retry</button>
        </div>
      )}

      {session.connectionState === 'connected' && (
        <>
          <TranscriptPanel entries={session.transcript} activeSpeaker={session.activeSpeaker} />
          <BottomBar
            isMicOn={session.isMicOn}
            onToggleMic={session.toggleMic}
            onEndCall={session.disconnect}
            audioLevel={session.audioLevel}
            activeSpeaker={session.activeSpeaker}
          />
        </>
      )}
    </div>
  )
}

function TranscriptPanel({ entries, activeSpeaker }: {
  entries: { id: string; speaker: 'user' | 'agent'; text: string; final: boolean }[]
  activeSpeaker: 'user' | 'agent' | null
}) {
  const containerRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    containerRef.current?.scrollTo({ top: 99999, behavior: 'smooth' })
  }, [entries])

  const visible = entries.slice(-6)

  return (
    <div ref={containerRef} style={transcriptArea}>
      {visible.map((entry, i) => {
        const isLast = i === visible.length - 1
        const dim = entry.speaker === 'user' && !isLast
        return (
          <p
            key={entry.id}
            style={{
              ...transcriptLine,
              color: dim ? '#555' : entry.final ? '#fff' : '#aaa',
              fontSize: isLast ? 28 : 20,
              lineHeight: isLast ? 1.35 : 1.5,
            }}
          >
            {entry.text}
          </p>
        )
      })}
      {activeSpeaker === 'agent' && entries.length === 0 && (
        <p style={{ ...transcriptLine, color: '#555', fontSize: 24 }}>Listening…</p>
      )}
    </div>
  )
}

function BottomBar({
  isMicOn, onToggleMic, onEndCall, audioLevel, activeSpeaker,
}: {
  isMicOn: boolean
  onToggleMic: () => void
  onEndCall: () => void
  audioLevel: number
  activeSpeaker: 'user' | 'agent' | null
}) {
  return (
    <div style={bar}>
      <IconBtn label="Camera" icon="□" />
      <IconBtn label="Share" icon="↑" />
      <VoiceOrb audioLevel={audioLevel} activeSpeaker={activeSpeaker} />
      <IconBtn label={isMicOn ? 'Mute' : 'Unmute'} icon="🎙" onClick={onToggleMic} active={isMicOn} />
      <IconBtn label="End" icon="✕" onClick={onEndCall} danger />
    </div>
  )
}

function IconBtn({ label, icon, onClick, active, danger }: {
  label: string; icon: string; onClick?: () => void; active?: boolean; danger?: boolean
}) {
  return (
    <button
      aria-label={label}
      onClick={onClick}
      style={{
        ...iconBtn,
        background: danger ? '#333' : active === false ? '#1a1a1a' : '#222',
        color: danger ? '#f44' : '#fff',
      }}
    >
      {icon}
    </button>
  )
}

function VoiceOrb({ audioLevel, activeSpeaker }: {
  audioLevel: number; activeSpeaker: 'user' | 'agent' | null
}) {
  const scale = 1 + Math.min(audioLevel * 0.8, 0.5)
  const agentColor = '#4488ff'
  const userColor = '#8844ff'
  const color = activeSpeaker === 'agent' ? agentColor : userColor

  return (
    <div style={orbWrapper}>
      <div
        style={{
          ...orb,
          transform: `scale(${scale})`,
          background: `radial-gradient(ellipse at 50% 60%, ${color} 0%, transparent 70%)`,
          boxShadow: `0 0 ${40 + audioLevel * 60}px ${color}88`,
          transition: 'transform 0.05s ease-out, box-shadow 0.05s ease-out',
        }}
      />
    </div>
  )
}

// ── styles ────────────────────────────────────────────────────────────────────

const shell: React.CSSProperties = {
  height: '100dvh', background: '#000', display: 'flex',
  flexDirection: 'column', overflow: 'hidden',
}
const nav: React.CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 12,
  padding: '12px 20px', borderBottom: '1px solid #111',
}
const backLink: React.CSSProperties = { color: '#666', fontSize: 13, textDecoration: 'none' }
const navTitle: React.CSSProperties = { color: '#444', fontSize: 13 }
const centerContent: React.CSSProperties = {
  flex: 1, display: 'flex', flexDirection: 'column',
  alignItems: 'center', justifyContent: 'center', gap: 20,
}
const idleHint: React.CSSProperties = { color: '#555', fontSize: 16 }
const startBtn: React.CSSProperties = {
  padding: '12px 32px', background: '#1a1a1a', border: '1px solid #333',
  borderRadius: 24, color: '#fff', fontSize: 15, cursor: 'pointer',
}
const transcriptArea: React.CSSProperties = {
  flex: 1, overflowY: 'auto', padding: '40px 28px 24px',
  display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', gap: 12,
}
const transcriptLine: React.CSSProperties = {
  margin: 0, fontWeight: 400, letterSpacing: '-0.2px',
}
const bar: React.CSSProperties = {
  display: 'flex', alignItems: 'center', justifyContent: 'space-around',
  padding: '16px 24px 28px', background: '#000',
}
const iconBtn: React.CSSProperties = {
  width: 52, height: 52, borderRadius: '50%', border: 'none',
  fontSize: 18, cursor: 'pointer', display: 'flex',
  alignItems: 'center', justifyContent: 'center',
}
const orbWrapper: React.CSSProperties = {
  width: 72, height: 72, display: 'flex',
  alignItems: 'center', justifyContent: 'center',
}
const orb: React.CSSProperties = {
  width: 60, height: 60, borderRadius: '50%',
  transformOrigin: 'center',
}
