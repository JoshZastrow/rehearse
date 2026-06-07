/**
 * Design 2 — Minimal Orb
 *
 * Layout: pure black screen. A single large morphing blob is centered.
 * No text, no chrome. Blob color shifts and size pulses with audio level.
 * Tap to connect; small end-call button fades in on hover/connected.
 * Inspired by Siri's liquid orb and Notion AI's pulse indicator.
 */

import { useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import { useVoiceSession } from '../hooks/useVoiceSession'

export default function Design2MinimalOrb() {
  const session = useVoiceSession()

  return (
    <div style={shell} onClick={session.connectionState === 'idle' ? session.connect : undefined}>
      <nav style={nav}>
        <Link to="/" style={backLink} onClick={(e) => e.stopPropagation()}>← designs</Link>
        <span style={navTitle}>Design 2 · Minimal Orb</span>
      </nav>

      <div style={stage}>
        {session.connectionState === 'idle' && (
          <p style={tapHint}>tap to begin</p>
        )}
        {session.connectionState === 'connecting' && (
          <MorphBlob audioLevel={0} activeSpeaker={null} state="connecting" />
        )}
        {session.connectionState === 'connected' && (
          <MorphBlob
            audioLevel={session.audioLevel}
            activeSpeaker={session.activeSpeaker}
            state="connected"
          />
        )}
        {session.connectionState === 'error' && (
          <p style={{ color: '#f55', fontSize: 13, textAlign: 'center', padding: 20 }}>
            Connection failed — add VITE_LIVEKIT_URL to .env
          </p>
        )}
      </div>

      {session.connectionState === 'connected' && (
        <div style={footer}>
          <button
            style={{ ...endBtn, opacity: session.isMicOn ? 1 : 0.5 }}
            onClick={session.toggleMic}
          >
            {session.isMicOn ? '🎙' : '🔇'}
          </button>
          <button style={endBtn} onClick={session.disconnect}>✕</button>
        </div>
      )}
    </div>
  )
}

function MorphBlob({
  audioLevel, activeSpeaker, state,
}: {
  audioLevel: number
  activeSpeaker: 'user' | 'agent' | null
  state: 'connecting' | 'connected'
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const frameRef = useRef<number>(0)
  const timeRef = useRef(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!
    let last = performance.now()

    function draw(now: number) {
      const dt = (now - last) / 1000
      last = now
      timeRef.current += dt * (state === 'connecting' ? 0.4 : 1)
      const t = timeRef.current

      const W = canvas!.width
      const H = canvas!.height
      ctx.clearRect(0, 0, W, H)

      const cx = W / 2
      const cy = H / 2
      const baseR = Math.min(W, H) * 0.32
      const pulse = audioLevel * baseR * 0.6
      const r = baseR + pulse

      const agentHue = 220   // blue
      const userHue = 270    // purple
      const hue = activeSpeaker === 'agent' ? agentHue : userHue

      const grad = ctx.createRadialGradient(cx, cy - r * 0.15, r * 0.05, cx, cy, r * 1.1)
      grad.addColorStop(0, `hsla(${hue}, 90%, 75%, 0.95)`)
      grad.addColorStop(0.5, `hsla(${hue + 20}, 80%, 50%, 0.7)`)
      grad.addColorStop(1, `hsla(${hue + 40}, 70%, 30%, 0)`)

      ctx.beginPath()
      const pts = 8
      for (let i = 0; i <= pts; i++) {
        const angle = (i / pts) * Math.PI * 2
        const wobble =
          Math.sin(angle * 3 + t * 1.1) * r * 0.12 +
          Math.sin(angle * 5 + t * 0.7) * r * 0.06 +
          audioLevel * r * 0.18 * Math.sin(angle * 2 + t * 3)
        const x = cx + Math.cos(angle) * (r + wobble)
        const y = cy + Math.sin(angle) * (r + wobble)
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
      }
      ctx.closePath()
      ctx.fillStyle = grad
      ctx.fill()

      // soft glow
      ctx.shadowBlur = 60 + audioLevel * 40
      ctx.shadowColor = `hsla(${hue}, 90%, 60%, 0.6)`
      ctx.fill()
      ctx.shadowBlur = 0

      frameRef.current = requestAnimationFrame(draw)
    }

    frameRef.current = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(frameRef.current)
  }, [audioLevel, activeSpeaker, state])

  return (
    <canvas
      ref={canvasRef}
      width={400}
      height={400}
      style={{ width: '100%', maxWidth: 360, maxHeight: 360 }}
    />
  )
}

// ── styles ────────────────────────────────────────────────────────────────────

const shell: React.CSSProperties = {
  height: '100dvh', background: '#000', display: 'flex',
  flexDirection: 'column', cursor: 'default',
}
const nav: React.CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 12,
  padding: '12px 20px',
}
const backLink: React.CSSProperties = { color: '#333', fontSize: 13, textDecoration: 'none' }
const navTitle: React.CSSProperties = { color: '#333', fontSize: 13 }
const stage: React.CSSProperties = {
  flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
}
const tapHint: React.CSSProperties = {
  color: '#444', fontSize: 14, letterSpacing: '0.1em', userSelect: 'none',
}
const footer: React.CSSProperties = {
  display: 'flex', justifyContent: 'center', gap: 16,
  padding: '0 0 32px',
}
const endBtn: React.CSSProperties = {
  width: 44, height: 44, borderRadius: '50%', border: '1px solid #222',
  background: '#111', color: '#fff', fontSize: 16, cursor: 'pointer',
}
