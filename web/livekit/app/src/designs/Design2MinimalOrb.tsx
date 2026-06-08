/**
 * Design 2 — Waveform Split
 *
 * Two stacked scrolling waveform panels (Caller=cyan, Agent=purple).
 * Large concentric-ring button to start. Session timer top right.
 * Mute / Settings bottom bar.
 */

import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { useVoiceSession } from '../hooks/useVoiceSession'

// ── elapsed timer hook ────────────────────────────────────────────────────────

function useElapsed(running: boolean) {
  const [secs, setSecs] = useState(0)
  useEffect(() => {
    if (!running) { setSecs(0); return }
    const id = setInterval(() => setSecs(s => s + 1), 1000)
    return () => clearInterval(id)
  }, [running])
  const h = String(Math.floor(secs / 3600)).padStart(2, '0')
  const m = String(Math.floor((secs % 3600) / 60)).padStart(2, '0')
  const s = String(secs % 60).padStart(2, '0')
  return `${h}:${m}:${s}`
}

// ── scrolling waveform canvas ─────────────────────────────────────────────────

function WaveformCanvas({
  color,
  audioLevel,
  active,
  height = 90,
}: {
  color: string
  audioLevel: number
  active: boolean
  height?: number
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const historyRef = useRef<number[]>(Array(200).fill(0))
  const frameRef = useRef(0)
  const tRef = useRef(0)
  const lastRef = useRef(performance.now())

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const dpr = window.devicePixelRatio || 1
    const W = canvas.offsetWidth || 600
    const H = height
    canvas.width = W * dpr
    canvas.height = H * dpr
    canvas.style.height = `${H}px`
    const ctx = canvas.getContext('2d')!
    ctx.scale(dpr, dpr)

    function draw(now: number) {
      const dt = Math.min((now - lastRef.current) / 1000, 0.05)
      lastRef.current = now
      tRef.current += dt

      // push new sample
      const amplitude = active
        ? audioLevel * 0.7 + Math.sin(tRef.current * 8) * audioLevel * 0.3
        : Math.sin(tRef.current * 1.2) * 0.04 + Math.sin(tRef.current * 3.1) * 0.015
      historyRef.current.push(Math.max(0, Math.min(1, amplitude)))
      if (historyRef.current.length > W) historyRef.current.shift()

      ctx.clearRect(0, 0, W, H)

      const history = historyRef.current
      const mid = H / 2
      const barW = W / history.length

      ctx.beginPath()
      for (let i = 0; i < history.length; i++) {
        const amp = history[i]
        const x = i * barW
        const h2 = Math.max(1, amp * mid * 0.9)
        ctx.moveTo(x, mid - h2)
        ctx.lineTo(x, mid + h2)
      }
      // glow pass
      ctx.strokeStyle = color
      ctx.lineWidth = barW * 0.7
      ctx.shadowBlur = active ? 8 : 4
      ctx.shadowColor = color
      ctx.globalAlpha = active ? 0.9 : 0.35
      ctx.stroke()
      ctx.shadowBlur = 0
      ctx.globalAlpha = 1

      frameRef.current = requestAnimationFrame(draw)
    }

    frameRef.current = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(frameRef.current)
  }, [color, audioLevel, active, height])

  return (
    <canvas
      ref={canvasRef}
      style={{ display: 'block', width: '100%', height }}
    />
  )
}

// ── concentric ring button ────────────────────────────────────────────────────

function RingButton({ state, onClick }: { state: string; onClick: () => void }) {
  const isIdle = state === 'idle' || state === 'error'
  const isConnecting = state === 'connecting'
  const isConnected = state === 'connected'

  const ringColors = ['#6366f1', '#818cf8', '#a5b4fc', '#c7d2fe']

  return (
    <button onClick={isIdle ? onClick : undefined} style={ringBtnShell}>
      {/* concentric rings */}
      {ringColors.map((c, i) => (
        <span
          key={i}
          style={{
            position: 'absolute',
            borderRadius: '50%',
            border: `1.5px solid ${c}`,
            opacity: isConnected ? 0.5 + i * 0.1 : isConnecting ? 0.3 + i * 0.1 : 0.18 + i * 0.08,
            width: 64 + i * 22,
            height: 64 + i * 22,
            boxShadow: isConnected ? `0 0 ${6 + i * 4}px ${c}` : 'none',
            transition: 'opacity 0.4s, box-shadow 0.4s',
            animation: isConnecting ? `pulse-ring ${1.2 + i * 0.15}s ease-in-out infinite` : 'none',
            animationDelay: `${i * 0.1}s`,
          }}
        />
      ))}
      {/* center circle */}
      <span style={{
        position: 'absolute',
        borderRadius: '50%',
        width: 58,
        height: 58,
        background: isConnected
          ? 'linear-gradient(135deg, #6366f1, #a855f7)'
          : isConnecting
          ? 'linear-gradient(135deg, #4f46e5, #7c3aed)'
          : 'linear-gradient(135deg, #1e1b4b, #312e81)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        boxShadow: isConnected ? '0 0 20px #6366f1aa' : 'none',
        transition: 'background 0.4s, box-shadow 0.4s',
      }}>
        <span style={{ fontSize: 22, userSelect: 'none' }}>
          {isConnecting ? '⟳' : '🎙'}
        </span>
      </span>
      {/* label */}
      <span style={{
        position: 'absolute',
        bottom: -30,
        left: '50%',
        transform: 'translateX(-50%)',
        whiteSpace: 'nowrap',
        color: isConnected ? '#a5b4fc' : '#475569',
        fontSize: 11,
        letterSpacing: '0.12em',
        textTransform: 'uppercase',
      }}>
        {isConnected ? 'Session Active' : isConnecting ? 'Connecting…' : 'Tap to Start'}
      </span>
      <span style={{
        position: 'absolute',
        bottom: -46,
        left: '50%',
        transform: 'translateX(-50%)',
        whiteSpace: 'nowrap',
        color: '#334155',
        fontSize: 10,
        letterSpacing: '0.08em',
      }}>
        {isConnected ? '' : isConnecting ? '' : 'Start the conversation'}
      </span>
    </button>
  )
}

// ── main component ────────────────────────────────────────────────────────────

export default function Design2MinimalOrb() {
  const session = useVoiceSession()
  const state = session.connectionState
  const elapsed = useElapsed(state === 'connected')

  const callerActive = state === 'connected' && session.activeSpeaker === 'user'
  const agentActive = state === 'connected' && session.activeSpeaker === 'agent'

  const handleConnect = () => {
    if (state === 'idle' || state === 'error') void session.connect()
  }

  return (
    <div style={shell}>
      <style>{`
        @keyframes pulse-ring {
          0%, 100% { transform: scale(1); opacity: inherit; }
          50% { transform: scale(1.06); opacity: 0.6; }
        }
        @keyframes blink-live {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
      `}</style>

      {/* ── header ─────────────────────────────────────────────────────────── */}
      <header style={header}>
        <div style={logoWrap}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M12 2L4 7v10l8 5 8-5V7L12 2z" fill="#6366f1" opacity="0.9"/>
            <path d="M12 2L4 7l8 5 8-5L12 2z" fill="#818cf8"/>
          </svg>
          <span style={logoText}>REHEARSE</span>
        </div>

        <div style={headerCenter}>
          {state === 'connected' && (
            <>
              <span style={liveDot} />
              <span style={liveLabel}>LIVE</span>
              <span style={sessionLabel}>Session Active</span>
            </>
          )}
          {state === 'connecting' && (
            <span style={sessionLabel}>Connecting…</span>
          )}
          {(state === 'idle' || state === 'error') && (
            <span style={{ ...sessionLabel, color: '#334155' }}>
              {state === 'error' ? 'Connection failed' : 'Ready'}
            </span>
          )}
        </div>

        <div style={headerRight}>
          {state === 'connected' && (
            <span style={timerText}>{elapsed}</span>
          )}
          <nav style={{ marginLeft: 12 }} onClick={e => e.stopPropagation()}>
            <Link to="/" style={backLink}>← designs</Link>
          </nav>
          <button style={menuBtn}>···</button>
        </div>
      </header>

      {/* ── waveform box ───────────────────────────────────────────────────── */}
      <div style={waveBox}>
        {/* caller panel */}
        <div style={wavePanel}>
          <div style={panelHeader}>
            <span style={speakerIcon}>👤</span>
            <span style={{ ...panelLabel, color: '#00dcff' }}>CALLER</span>
            <span style={{
              ...statusChip,
              color: callerActive ? '#00dcff' : '#334155',
              borderColor: callerActive ? '#00dcff44' : '#1e293b',
            }}>
              {callerActive ? '● SPEAKING' : '○ LISTENING'}
            </span>
          </div>
          <WaveformCanvas
            color="#00dcff"
            audioLevel={callerActive ? session.audioLevel : 0}
            active={callerActive}
            height={80}
          />
        </div>

        {/* divider */}
        <div style={panelDivider} />

        {/* agent panel */}
        <div style={wavePanel}>
          <div style={panelHeader}>
            <span style={speakerIcon}>✦</span>
            <span style={{ ...panelLabel, color: '#c084fc' }}>AGENT</span>
            <span style={{
              ...statusChip,
              color: agentActive ? '#c084fc' : '#334155',
              borderColor: agentActive ? '#c084fc44' : '#1e293b',
            }}>
              {agentActive ? '● SPEAKING' : '○ LISTENING'}
            </span>
          </div>
          <WaveformCanvas
            color="#c084fc"
            audioLevel={agentActive ? session.audioLevel : 0}
            active={agentActive}
            height={80}
          />
        </div>

        {/* center cursor line */}
        <div style={cursorLine}>
          <div style={{ ...cursorDot, background: '#00dcff', boxShadow: '0 0 8px #00dcff' }} />
          <div style={{ flex: 1, width: 1, background: 'linear-gradient(to bottom, #00dcff44, #c084fc44)' }} />
          <div style={{ ...cursorDot, background: '#c084fc', boxShadow: '0 0 8px #c084fc' }} />
        </div>
      </div>

      {/* ── side labels ────────────────────────────────────────────────────── */}
      <div style={sideLabelRow}>
        <div style={sideLabel}>
          <span style={{ color: '#00dcff88', fontSize: 10 }}>CALLER</span>
          <span style={{ color: '#475569', fontSize: 9 }}>Listen</span>
        </div>
        <div style={{ flex: 1 }} />
        <div style={{ ...sideLabel, alignItems: 'flex-end' }}>
          <span style={{ color: '#c084fc88', fontSize: 10 }}>AGENT</span>
          <span style={{ color: '#475569', fontSize: 9 }}>Speak</span>
        </div>
      </div>

      {/* ── ring button ────────────────────────────────────────────────────── */}
      <div style={buttonRow}>
        <RingButton state={state} onClick={handleConnect} />
      </div>

      {/* ── bottom bar ─────────────────────────────────────────────────────── */}
      <footer style={bottomBar}>
        <button
          style={{
            ...barBtn,
            opacity: state === 'connected' ? 1 : 0.3,
            color: session.isMicOn ? '#e2e8f0' : '#f87171',
          }}
          onClick={session.toggleMic}
          disabled={state !== 'connected'}
          aria-label={session.isMicOn ? 'Mute' : 'Unmute'}
        >
          <span style={{ fontSize: 18 }}>{session.isMicOn ? '🎧' : '🔇'}</span>
          <span style={barBtnLabel}>Mute</span>
        </button>

        <div style={{ flex: 1 }} />

        {state === 'connected' && (
          <button style={{ ...barBtn, color: '#f87171' }} onClick={session.disconnect} aria-label="End call">
            <span style={{ fontSize: 18 }}>✕</span>
            <span style={barBtnLabel}>End</span>
          </button>
        )}

        <div style={{ flex: 1 }} />

        <button style={{ ...barBtn, opacity: 0.4 }} aria-label="Settings">
          <span style={{ fontSize: 18 }}>⚙</span>
          <span style={barBtnLabel}>Settings</span>
        </button>
      </footer>
    </div>
  )
}

// ── styles ────────────────────────────────────────────────────────────────────

const shell: React.CSSProperties = {
  minHeight: '100dvh',
  background: '#020617',
  display: 'flex',
  flexDirection: 'column',
  color: '#e2e8f0',
  fontFamily: 'system-ui, -apple-system, sans-serif',
  userSelect: 'none',
}

const header: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  padding: '12px 20px',
  borderBottom: '1px solid #0f172a',
  gap: 12,
  flexShrink: 0,
}

const logoWrap: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 7,
}

const logoText: React.CSSProperties = {
  fontSize: 13,
  fontWeight: 600,
  letterSpacing: '0.15em',
  color: '#94a3b8',
}

const headerCenter: React.CSSProperties = {
  flex: 1,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: 7,
}

const liveDot: React.CSSProperties = {
  width: 7,
  height: 7,
  borderRadius: '50%',
  background: '#22c55e',
  animation: 'blink-live 1.4s ease-in-out infinite',
}

const liveLabel: React.CSSProperties = {
  fontSize: 10,
  fontWeight: 700,
  letterSpacing: '0.15em',
  color: '#22c55e',
}

const sessionLabel: React.CSSProperties = {
  fontSize: 12,
  color: '#64748b',
  letterSpacing: '0.05em',
}

const headerRight: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
}

const timerText: React.CSSProperties = {
  fontSize: 13,
  fontFamily: 'monospace',
  color: '#94a3b8',
  letterSpacing: '0.05em',
}

const backLink: React.CSSProperties = {
  color: '#475569',
  fontSize: 11,
  textDecoration: 'none',
}

const menuBtn: React.CSSProperties = {
  background: 'none',
  border: 'none',
  color: '#475569',
  fontSize: 16,
  cursor: 'pointer',
  padding: '0 4px',
  letterSpacing: '0.05em',
}

const waveBox: React.CSSProperties = {
  position: 'relative',
  margin: '20px 24px 0',
  border: '1px solid #1e293b',
  borderRadius: 12,
  overflow: 'hidden',
  background: '#030712',
  flexShrink: 0,
}

const wavePanel: React.CSSProperties = {
  padding: '10px 0 6px',
}

const panelHeader: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 7,
  padding: '0 14px 6px',
}

const speakerIcon: React.CSSProperties = {
  fontSize: 13,
  opacity: 0.7,
}

const panelLabel: React.CSSProperties = {
  fontSize: 11,
  fontWeight: 700,
  letterSpacing: '0.15em',
}

const statusChip: React.CSSProperties = {
  marginLeft: 'auto',
  fontSize: 9,
  letterSpacing: '0.1em',
  border: '1px solid',
  borderRadius: 20,
  padding: '1px 8px',
  transition: 'color 0.3s, border-color 0.3s',
}

const panelDivider: React.CSSProperties = {
  height: 1,
  background: 'linear-gradient(to right, transparent, #1e293b 20%, #1e293b 80%, transparent)',
  margin: '0 14px',
}

const cursorLine: React.CSSProperties = {
  position: 'absolute',
  top: 0,
  left: '50%',
  transform: 'translateX(-50%)',
  width: 1,
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  pointerEvents: 'none',
}

const cursorDot: React.CSSProperties = {
  width: 8,
  height: 8,
  borderRadius: '50%',
  flexShrink: 0,
}

const sideLabelRow: React.CSSProperties = {
  display: 'flex',
  padding: '6px 28px 0',
  flexShrink: 0,
}

const sideLabel: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 1,
  alignItems: 'flex-start',
  letterSpacing: '0.1em',
  textTransform: 'uppercase',
}

const buttonRow: React.CSSProperties = {
  flex: 1,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  paddingTop: 40,
  paddingBottom: 20,
}

const ringBtnShell: React.CSSProperties = {
  position: 'relative',
  width: 160,
  height: 160,
  background: 'none',
  border: 'none',
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
}

const bottomBar: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  padding: '14px 32px 24px',
  borderTop: '1px solid #0f172a',
  flexShrink: 0,
  gap: 0,
}

const barBtn: React.CSSProperties = {
  background: 'none',
  border: 'none',
  cursor: 'pointer',
  color: '#e2e8f0',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  gap: 4,
  padding: '6px 14px',
  borderRadius: 8,
  transition: 'opacity 0.2s',
}

const barBtnLabel: React.CSSProperties = {
  fontSize: 10,
  letterSpacing: '0.08em',
  color: '#475569',
}
