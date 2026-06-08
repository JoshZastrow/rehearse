/**
 * Design 2 — Minimal Orb
 *
 * Pure black canvas. One morphing blob, full screen. No chrome.
 * Tap anywhere to connect; mic + end buttons slide in when connected.
 *
 * States:
 *   idle       — dim grey breathing orb + "tap to begin" hint
 *   connecting — orb brightens + churns faster
 *   connected  — full audio-reactive orb; blue = agent, purple = user
 *   error      — orb turns red, retry hint
 */

import { useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import { useVoiceSession } from '../hooks/useVoiceSession'

type OrbState = 'idle' | 'connecting' | 'connected' | 'error'

export default function Design2MinimalOrb() {
  const session = useVoiceSession()
  const state = session.connectionState as OrbState

  const handleStageClick = () => {
    if (state === 'idle' || state === 'error') void session.connect()
  }

  return (
    <div style={shell} onClick={handleStageClick}>
      <nav style={nav} onClick={(e) => e.stopPropagation()}>
        <Link to="/" style={backLink}>← designs</Link>
        <span style={navTitle}>Design 2 · Minimal Orb</span>
      </nav>

      <div style={stage}>
        <MorphOrb
          audioLevel={session.audioLevel}
          activeSpeaker={session.activeSpeaker}
          state={state}
        />
        {state === 'idle' && <p style={hint}>tap to begin</p>}
        {state === 'error' && <p style={{ ...hint, color: '#f55' }}>tap to retry</p>}
      </div>

      {/* Mic + End slide up only when connected */}
      <div
        style={{
          ...footer,
          opacity: state === 'connected' ? 1 : 0,
          transform: state === 'connected' ? 'translateY(0)' : 'translateY(14px)',
          pointerEvents: state === 'connected' ? 'auto' : 'none',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <button
          aria-label={session.isMicOn ? 'Mute' : 'Unmute'}
          style={{ ...ctrlBtn, opacity: session.isMicOn ? 1 : 0.4 }}
          onClick={session.toggleMic}
        >
          {session.isMicOn ? '🎙' : '🔇'}
        </button>
        <button aria-label="End call" style={{ ...ctrlBtn, color: '#f44' }} onClick={session.disconnect}>
          ✕
        </button>
      </div>
    </div>
  )
}

// ── MorphOrb ──────────────────────────────────────────────────────────────────

function MorphOrb({
  audioLevel,
  activeSpeaker,
  state,
}: {
  audioLevel: number
  activeSpeaker: 'user' | 'agent' | null
  state: OrbState
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const frameRef = useRef(0)
  const tRef = useRef(0)
  const lastRef = useRef(performance.now())

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    // DPR-aware sizing — crisp on retina
    const dpr = window.devicePixelRatio || 1
    const SIZE = 500
    canvas.width = SIZE * dpr
    canvas.height = SIZE * dpr
    canvas.style.width = `${SIZE}px`
    canvas.style.height = `${SIZE}px`

    const ctx = canvas.getContext('2d')!
    ctx.scale(dpr, dpr)

    function draw(now: number) {
      const dt = Math.min((now - lastRef.current) / 1000, 0.05)
      lastRef.current = now

      const speed = state === 'idle' ? 0.28 : state === 'connecting' ? 1.4 : 1.0
      tRef.current += dt * speed
      const t = tRef.current

      ctx.clearRect(0, 0, SIZE, SIZE)
      const cx = SIZE / 2
      const cy = SIZE / 2

      // ── radius by state ──────────────────────────────────────────────────
      const breathe = Math.sin(t * 1.6) * 0.04
      const baseR =
        state === 'idle'
          ? SIZE * 0.20 * (1 + breathe)
          : state === 'connecting'
          ? SIZE * 0.28 * (1 + Math.sin(t * 3.5) * 0.07)
          : state === 'error'
          ? SIZE * 0.22 * (1 + breathe)
          : SIZE * 0.28 * (1 + audioLevel * 0.28)

      // ── hue / saturation / lightness by state ────────────────────────────
      const hue =
        state === 'idle'
          ? 240
          : state === 'connecting'
          ? (t * 35) % 360
          : state === 'error'
          ? 0
          : activeSpeaker === 'agent'
          ? 220
          : 270

      const sat = state === 'idle' ? 18 : state === 'error' ? 80 : 82
      const lit = state === 'idle' ? 52 : 68
      const alpha = state === 'idle' ? 0.50 : 0.88

      // ── smooth blob via Catmull-Rom splines ──────────────────────────────
      const N = 10
      const pts: { x: number; y: number }[] = []
      for (let i = 0; i < N; i++) {
        const angle = (i / N) * Math.PI * 2
        const churn = state === 'connecting' ? 4 : 3
        const wobble =
          Math.sin(angle * churn + t * 1.1) * baseR * 0.14 +
          Math.sin(angle * (churn + 2) + t * (state === 'connecting' ? 2.4 : 1.3)) * baseR * 0.07 +
          (state === 'connected' ? audioLevel * baseR * 0.22 * Math.sin(angle * 2 + t * 4) : 0)
        pts.push({
          x: cx + Math.cos(angle) * (baseR + wobble),
          y: cy + Math.sin(angle) * (baseR + wobble),
        })
      }

      ctx.beginPath()
      for (let i = 0; i < N; i++) {
        const p0 = pts[(i - 1 + N) % N]
        const p1 = pts[i]
        const p2 = pts[(i + 1) % N]
        const p3 = pts[(i + 2) % N]
        // Catmull-Rom → cubic bezier control points
        const cp1x = p1.x + (p2.x - p0.x) / 6
        const cp1y = p1.y + (p2.y - p0.y) / 6
        const cp2x = p2.x - (p3.x - p1.x) / 6
        const cp2y = p2.y - (p3.y - p1.y) / 6
        if (i === 0) ctx.moveTo(p1.x, p1.y)
        ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p2.x, p2.y)
      }
      ctx.closePath()

      // ── radial gradient fill ──────────────────────────────────────────────
      const grad = ctx.createRadialGradient(
        cx, cy - baseR * 0.18, baseR * 0.04,
        cx, cy, baseR * 1.1,
      )
      grad.addColorStop(0,    `hsla(${hue},     ${sat + 12}%, ${lit + 12}%, ${alpha})`)
      grad.addColorStop(0.45, `hsla(${hue + 15},${sat}%,      ${lit - 5}%,  ${alpha * 0.82})`)
      grad.addColorStop(1,    `hsla(${hue + 30},${sat - 15}%, ${lit - 22}%, 0)`)
      ctx.fillStyle = grad
      ctx.fill()

      // ── outer glow pass ───────────────────────────────────────────────────
      const glowR =
        state === 'idle' ? 20
        : state === 'connecting' ? 35
        : 45 + audioLevel * 55
      ctx.shadowBlur = glowR
      ctx.shadowColor = `hsla(${hue}, ${sat}%, 62%, 0.55)`
      ctx.fill()
      ctx.shadowBlur = 0

      frameRef.current = requestAnimationFrame(draw)
    }

    frameRef.current = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(frameRef.current)
  }, [audioLevel, activeSpeaker, state])

  return <canvas ref={canvasRef} style={{ display: 'block' }} />
}

// ── styles ────────────────────────────────────────────────────────────────────

const shell: React.CSSProperties = {
  height: '100dvh',
  background: '#000',
  display: 'flex',
  flexDirection: 'column',
  userSelect: 'none',
}

const nav: React.CSSProperties = {
  position: 'absolute',
  top: 0,
  left: 0,
  display: 'flex',
  alignItems: 'center',
  gap: 12,
  padding: '12px 20px',
  opacity: 0.3,
  zIndex: 10,
  transition: 'opacity 0.2s',
}

const backLink: React.CSSProperties = { color: '#fff', fontSize: 13, textDecoration: 'none' }
const navTitle: React.CSSProperties = { color: '#fff', fontSize: 13 }

const stage: React.CSSProperties = {
  flex: 1,
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  gap: 20,
  cursor: 'pointer',
}

const hint: React.CSSProperties = {
  color: '#555',
  fontSize: 12,
  letterSpacing: '0.15em',
  textTransform: 'lowercase',
  margin: 0,
}

const footer: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  gap: 24,
  paddingBottom: 36,
  transition: 'opacity 0.3s ease, transform 0.3s ease',
}

const ctrlBtn: React.CSSProperties = {
  width: 48,
  height: 48,
  borderRadius: '50%',
  border: '1px solid #222',
  background: '#111',
  color: '#fff',
  fontSize: 17,
  cursor: 'pointer',
}
