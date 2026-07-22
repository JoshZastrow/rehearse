import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { dirname, resolve, join } from 'node:path'

const __dirname = dirname(fileURLToPath(import.meta.url))
export const REPO_ROOT = resolve(__dirname, '../../../..')
export const SESSION_ROOT = join(REPO_ROOT, 'tests', 'e2e', 'sessions')

const RUNNER_ENV = {
  LIVEKIT_URL: 'ws://localhost:7880',
  LIVEKIT_API_KEY: 'devkey',
  LIVEKIT_API_SECRET: 'secret',
  LIVEKIT_ROOM_NAME: 'rehearse-room',
  SESSION_ROOT,
}

/** Spawn the test-only agent runner (tests/e2e/runner.py) and expose stdout markers. */
export class AgentRunner {
  proc: ChildProcessWithoutNullStreams
  private buf = ''
  private waiters: Array<{ re: RegExp; resolve: (m: RegExpMatchArray) => void }> = []
  exited: Promise<number>

  constructor() {
    // detached: own process group, so signals reach the whole tree
    // (uv → python → livekit native threads). Signalling just the `uv` parent
    // orphans the python child, which lingers in the LiveKit room.
    this.proc = spawn('uv', ['run', 'python', 'tests/e2e/runner.py'], {
      cwd: REPO_ROOT,
      env: { ...process.env, ...RUNNER_ENV },
      detached: true,
    })
    // Match markers across BOTH streams: stdout carries the SESSION_* markers,
    // stderr carries the python logging (incl. the graceful-shutdown marker).
    this.proc.stdout.on('data', (d) => this.onData(String(d)))
    this.proc.stderr.on('data', (d) => this.onData(String(d)))
    this.exited = new Promise((res) => this.proc.on('close', (code) => res(code ?? -1)))
  }

  private onData(chunk: string) {
    this.buf += chunk
    process.stderr.write(`[runner] ${chunk}`)
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

  /** Everything the runner has printed so far (stdout + stderr), for assertions. */
  output(): string {
    return this.buf
  }

  /** Send `signal` to the whole process group (uv + python child). No-op if exited. */
  signal(sig: NodeJS.Signals) {
    if (this.proc.exitCode !== null || this.proc.pid === undefined) return
    try {
      process.kill(-this.proc.pid, sig)
    } catch {
      this.proc.kill(sig)
    }
  }

  /** Graceful shutdown: the signal a `make` Ctrl+C delivers to the agent. */
  sigint() {
    this.signal('SIGINT')
  }

  /** Last-resort reap. */
  kill() {
    this.signal('SIGKILL')
  }
}
