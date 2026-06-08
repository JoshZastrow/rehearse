import { Link } from 'react-router-dom'

const designs = [
  {
    path: '/design/gemini',
    label: 'Design 1 — Gemini Style',
    description: 'Black background · streaming transcript · animated orb · control bar',
  },
  {
    path: '/design/minimal',
    label: 'Design 2 — Minimal Orb',
    description: 'Full-screen morphing orb · audio-reactive · no chrome',
  },
  {
    path: '/design/waveform',
    label: 'Design 3 — Waveform Split',
    description: 'Transcript history top · live waveform visualizer bottom',
  },
]

export default function App() {
  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>Rehearse</h1>
        <p style={styles.subtitle}>WebRTC Voice Agent — Design Prototypes</p>
      </div>
      <div style={styles.grid}>
        {designs.map((d) => (
          <Link key={d.path} to={d.path} style={styles.card}>
            <div style={styles.cardLabel}>{d.label}</div>
            <div style={styles.cardDesc}>{d.description}</div>
          </Link>
        ))}
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    height: '100dvh',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 48,
    padding: 24,
    background: '#0a0a0a',
  },
  header: { textAlign: 'center' },
  title: { fontSize: 32, fontWeight: 600, letterSpacing: '-0.5px' },
  subtitle: { marginTop: 8, color: '#888', fontSize: 14 },
  grid: {
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
    width: '100%',
    maxWidth: 420,
  },
  card: {
    display: 'block',
    padding: '20px 24px',
    background: '#141414',
    border: '1px solid #222',
    borderRadius: 12,
    textDecoration: 'none',
    color: 'inherit',
    transition: 'border-color 0.15s',
  },
  cardLabel: { fontSize: 15, fontWeight: 500, marginBottom: 4 },
  cardDesc: { fontSize: 13, color: '#666' },
}
