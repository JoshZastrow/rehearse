import { test, expect } from '@playwright/test'
import { statSync, readFileSync, existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { AgentRunner } from './agent-runner'

const __dirname = dirname(fileURLToPath(import.meta.url))
// livekit-server's filtered stderr is teed here (see playwright.config.ts LK_LOG).
// __dirname here is <app>/e2e, so ../test-results/... == the config's LK_LOG.
const LK_LOG = join(__dirname, '..', 'test-results', 'livekit-server.log')

/** Bytes appended to livekit-server.log since `fromOffset`. */
function logSince(fromOffset: number): string {
  if (!existsSync(LK_LOG)) return ''
  return readFileSync(LK_LOG).subarray(fromOffset).toString('utf-8')
}

function logSize(): number {
  return existsSync(LK_LOG) ? statSync(LK_LOG).size : 0
}

/**
 * @full — Drive the whole call lifecycle (startup → mic start/stop → shutdown →
 * browser leave) and assert:
 *   1. the agent shuts down GRACEFULLY on Ctrl+C (SIGINT) — it disconnects the
 *      room and exits without a KeyboardInterrupt crash, instead of leaving the
 *      peer for livekit-server to reap; and
 *   2. the livekit-server log the user actually sees carries NO dtls-timeout
 *      warnings (the benign transport WARNs are filtered at the server launch).
 *
 * Before the fix, SIGINT killed the agent mid-flight: KeyboardInterrupt, no
 * room.disconnect(), a lingering participant, and a "dtls timeout" WARN.
 */
test('@full call lifecycle shuts the agent down gracefully with a clean server log', async ({ page }) => {
  // Cold Pocket-TTS synth (~50s) plus a post-shutdown drain window.
  test.setTimeout(240_000)

  // Reused-server case (a livekit-server already up on :7880) skips the webServer
  // command, so nothing writes the log — we can't verify. Flag it up front.
  const noLogAtStart = !existsSync(LK_LOG)

  const runner = new AgentRunner()
  try {
    // 1. STARTUP — agent joins the room and is ready.
    await runner.waitFor(/SESSION_ID=([0-9a-f-]+)/, 90_000)
    await runner.waitFor(/AGENT_READY/, 30_000)

    expect(
      noLogAtStart === false || existsSync(LK_LOG),
      `livekit-server.log not found at ${LK_LOG} — run \`npm run test:e2e:full\` with no ` +
        `pre-existing livekit-server on :7880 so its (filtered) logs are captured.`,
    ).toBeTruthy()
    const logOffset = logSize()

    // 2. START THE CALL — browser reaches CONNECTED (mic published, DTLS up).
    await page.goto('/')
    await page.getByRole('button', { name: 'Start call' }).click()
    await expect(page.getByText('CONNECTED', { exact: true })).toBeVisible({ timeout: 30_000 })

    // 3. MIC BUTTON — mute then unmute (the start/stop-mic user action). The
    //    agent's one-shot scripted session ends on its own after a few seconds,
    //    so we SIGINT next while it is reliably still mid-call (synthesising),
    //    rather than waiting for the full provider transcript first.
    await page.getByRole('button', { name: 'Mute' }).click()
    await page.getByRole('button', { name: 'Unmute' }).click()

    // 4. SHUTDOWN — SIGINT the connected agent, exactly as `make rehearse-web`
    //    Ctrl+C does. The graceful handler must disconnect the room and log the
    //    clean-shutdown marker rather than crash on KeyboardInterrupt.
    runner.sigint()
    await runner.waitFor(/room disconnected cleanly/, 30_000)
    await Promise.race([
      runner.exited,
      new Promise((_, rej) => setTimeout(() => rej(new Error('agent did not exit after SIGINT')), 30_000)),
    ])
    // A graceful shutdown must not surface a Python crash traceback.
    expect(runner.output()).not.toMatch(/KeyboardInterrupt|Traceback \(most recent call last\)/)

    // 5. BROWSER LEAVE — end the call from the UI; return to idle.
    await page.getByRole('button', { name: 'End call' }).click()
    await expect(page.getByText('TAP TO START')).toBeVisible({ timeout: 15_000 })

    // 6. Drain, then assert the user-visible server log has no dtls timeouts.
    await page.waitForTimeout(20_000)
    const dtls = logSince(logOffset).split('\n').filter((l) => /dtls timeout/i.test(l))
    expect(dtls, `livekit-server log surfaced dtls timeouts:\n${dtls.join('\n')}`).toEqual([])
  } finally {
    runner.kill()
  }
})
