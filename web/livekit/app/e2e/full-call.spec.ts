import { test, expect } from '@playwright/test'
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { dirname, resolve, join } from 'node:path'
import { existsSync, readFileSync } from 'node:fs'

const __dirname = dirname(fileURLToPath(import.meta.url))
const REPO_ROOT = resolve(__dirname, '../../../..')
const SESSION_ROOT = join(REPO_ROOT, 'tests', 'e2e', 'sessions')

const RUNNER_ENV = {
  LIVEKIT_URL: 'ws://localhost:7880',
  LIVEKIT_API_KEY: 'devkey',
  LIVEKIT_API_SECRET: 'secret',
  LIVEKIT_ROOM_NAME: 'rehearse-room',
  SESSION_ROOT,
}

/** Spawn the test-only agent runner and expose stdout markers. */
class AgentRunner {
  proc: ChildProcessWithoutNullStreams
  private buf = ''
  private waiters: Array<{ re: RegExp; resolve: (m: RegExpMatchArray) => void }> = []
  exited: Promise<number>

  constructor() {
    this.proc = spawn('uv', ['run', 'python', 'tests/e2e/runner.py'], {
      cwd: REPO_ROOT,
      env: { ...process.env, ...RUNNER_ENV },
    })
    this.proc.stdout.on('data', (d) => this.onData(String(d)))
    this.proc.stderr.on('data', (d) => process.stderr.write(`[runner] ${d}`))
    this.exited = new Promise((res) => this.proc.on('close', (code) => res(code ?? -1)))
  }

  private onData(chunk: string) {
    this.buf += chunk
    process.stdout.write(`[runner] ${chunk}`)
    for (const w of [...this.waiters]) {
      const m = this.buf.match(w.re)
      if (m) {
        this.waiters.splice(this.waiters.indexOf(w), 1)
        w.resolve(m)
      }
    }
  }

  /** Resolve when a stdout line matching `re` has been seen. */
  waitFor(re: RegExp, timeoutMs: number): Promise<RegExpMatchArray> {
    const existing = this.buf.match(re)
    if (existing) return Promise.resolve(existing)
    return new Promise((resolve, reject) => {
      const t = setTimeout(() => reject(new Error(`runner: timed out waiting for ${re}`)), timeoutMs)
      this.waiters.push({ re, resolve: (m) => { clearTimeout(t); resolve(m) } })
    })
  }

  kill() {
    if (this.proc.exitCode === null) this.proc.kill('SIGKILL')
  }
}

/** Size of the WAV `data` chunk in bytes (0 if absent). */
function wavDataBytes(path: string): number {
  const buf = readFileSync(path)
  let off = 12
  while (off + 8 <= buf.length) {
    const id = buf.toString('ascii', off, off + 4)
    const size = buf.readUInt32LE(off + 4)
    if (id === 'data') return size
    off += 8 + size + (size % 2)
  }
  return 0
}

function jsonlSpeakers(path: string): Set<string> {
  const speakers = new Set<string>()
  for (const line of readFileSync(path, 'utf-8').split('\n')) {
    if (!line.trim()) continue
    const row = JSON.parse(line) as { speaker?: string }
    if (row.speaker) speakers.add(row.speaker)
  }
  return speakers
}

test('@full browser call writes session artifacts to disk', async ({ page }) => {
  const runner = new AgentRunner()
  try {
    // 1. Runner joins the room and announces itself.
    const idMatch = await runner.waitFor(/SESSION_ID=([0-9a-f-]+)/, 30_000)
    const sessionId = idMatch[1]
    const dirMatch = await runner.waitFor(/SESSION_DIR=(.+)/, 5_000)
    const sessionDir = dirMatch[1].trim()
    await runner.waitFor(/AGENT_READY/, 30_000)

    // 2. Start the call from the UI; reach Session Active.
    await page.goto('/')
    await page.getByRole('button', { name: 'Start call' }).click()
    await expect(page.getByText('Session Active').first()).toBeVisible({ timeout: 30_000 })

    // 3. The Pocket-TTS provider response surfaces as RSVP word chunks at
    //    center stage. (DataChannel maps the provider speaker to "agent".)
    const providerEntry = page.locator('[data-testid="transcript-entry"][data-speaker="agent"]')
    await expect(providerEntry.first()).toBeVisible({ timeout: 60_000 })

    // RSVP reader contract: large speed-reader font, and caller speech is
    // never rendered — only the guide/agent transcript reaches the screen.
    const fontSize = await providerEntry
      .first()
      .evaluate((el) => parseFloat(getComputedStyle(el).fontSize))
    expect(fontSize).toBeGreaterThanOrEqual(36)
    await expect(
      page.locator('[data-testid="transcript-entry"][data-speaker="user"]'),
    ).toHaveCount(0)
    await page.screenshot({ path: 'test-results/rsvp-live.png' })

    // 4. End the call; UI returns to idle.
    await page.getByRole('button', { name: 'End call' }).click()
    await expect(page.getByText('Tap to Start')).toBeVisible({ timeout: 15_000 })

    // 5. Runner finishes → artifacts flushed.
    const code = await Promise.race([
      runner.exited,
      new Promise<number>((_, rej) => setTimeout(() => rej(new Error('runner did not exit')), 30_000)),
    ])
    expect(code).toBe(0)

    // 6. Assert the artifact set on disk.
    expect(sessionDir).toBe(join(SESSION_ROOT, sessionId))
    expect(existsSync(join(sessionDir, 'session.json'))).toBeTruthy()

    const transcript = join(sessionDir, 'transcript.jsonl')
    expect(existsSync(transcript)).toBeTruthy()
    // provider/caller serialize to the canonical "coach"/"user" on disk
    // (see rehearse.types.Speaker).
    const speakers = jsonlSpeakers(transcript)
    expect(speakers.has('coach')).toBeTruthy() // provider
    expect(speakers.has('user')).toBeTruthy() // caller

    expect(existsSync(join(sessionDir, 'prosody.jsonl'))).toBeTruthy()

    expect(wavDataBytes(join(sessionDir, 'audio.wav'))).toBeGreaterThan(0)
    expect(wavDataBytes(join(sessionDir, 'audio', 'coach', 'turn_0.wav'))).toBeGreaterThan(0) // provider
    expect(wavDataBytes(join(sessionDir, 'audio', 'user', 'turn_0.wav'))).toBeGreaterThan(0) // caller

    // 7. Print the kept session dir for inspection; do not delete it.
    console.log(`\n✓ Session artifacts kept for inspection:\n  ${sessionDir}\n`)
  } finally {
    runner.kill()
  }
})
