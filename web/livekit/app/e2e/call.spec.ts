import { test, expect } from '@playwright/test'
import * as fs from 'fs'
import * as path from 'path'

// Path where the agent writes session artifacts (relative to repo root,
// resolved from SESSION_ROOT env or default ./sessions)
const SESSION_ROOT = process.env.SESSION_ROOT
  ? path.resolve(process.env.SESSION_ROOT)
  : path.resolve(__dirname, '../../../../sessions')

function listSessionDirs(): string[] {
  if (!fs.existsSync(SESSION_ROOT)) return []
  return fs.readdirSync(SESSION_ROOT).filter(f =>
    fs.statSync(path.join(SESSION_ROOT, f)).isDirectory()
  )
}

test.describe('Waveform UI', () => {
  test('idle state renders waveform panels and start button', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByText('CALLER').first()).toBeVisible()
    await expect(page.getByText('AGENT').first()).toBeVisible()
    await expect(page.getByText('Tap to Start')).toBeVisible()
  })

  test('clicking start button transitions out of idle', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('button', { name: 'Start call' }).click()
    await expect(page.getByText('Tap to Start')).not.toBeVisible({ timeout: 5_000 })
  })

  test('connects to LiveKit room and shows session active', async ({ page }) => {
    await page.goto('/')

    // Fail fast if connection errors appear
    page.on('console', msg => {
      if (msg.type() === 'error') console.error('browser error:', msg.text())
    })

    await page.getByRole('button', { name: 'Start call' }).click()

    // Should not show connection failed
    await expect(page.getByText('Connection failed')).not.toBeVisible({ timeout: 2_000 }).catch(() => {
      throw new Error('UI shows "Connection failed" — check token server and LiveKit server are running')
    })

    await expect(page.getByText('Session Active').first()).toBeVisible({ timeout: 10_000 })
    await expect(page.getByText('LIVE')).toBeVisible()
  })

  test('End button returns to idle', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('button', { name: 'Start call' }).click()
    await expect(page.getByText('Session Active').first()).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'End call' }).click()
    await expect(page.getByText('Tap to Start')).toBeVisible({ timeout: 8_000 })
    await expect(page.getByText('Session Active').first()).not.toBeVisible()
  })

  test('session timer counts up when connected', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('button', { name: 'Start call' }).click()
    await expect(page.getByText('Session Active').first()).toBeVisible({ timeout: 10_000 })

    const timer = page.getByText(/^\d{2}:\d{2}:\d{2}$/)
    await expect(timer).toBeVisible()
    const t1 = await timer.textContent()
    await page.waitForTimeout(2_000)
    const t2 = await timer.textContent()
    expect(t1).not.toBe(t2)
  })

  test('full call lifecycle: connect, talk, end — session written to disk', async ({ page }) => {
    const sessionsBefore = new Set(listSessionDirs())

    await page.goto('/')
    await page.getByRole('button', { name: 'Start call' }).click()

    // Must reach connected — if this fails the agent/token server are broken
    await expect(page.getByText('Session Active').first()).toBeVisible({ timeout: 15_000 })

    // Let the session run briefly
    await page.waitForTimeout(3_000)

    await page.getByRole('button', { name: 'End call' }).click()
    await expect(page.getByText('Tap to Start')).toBeVisible({ timeout: 8_000 })

    // Give the agent a moment to flush artifacts
    await page.waitForTimeout(2_000)

    // Verify a new session directory was created
    const sessionsAfter = listSessionDirs()
    const newSessions = sessionsAfter.filter(s => !sessionsBefore.has(s))

    expect(newSessions.length, `Expected a new session dir under ${SESSION_ROOT} but found none.\nExisting sessions: ${sessionsAfter.join(', ') || '(empty)'}`).toBeGreaterThan(0)

    // Verify session.json exists in the new session dir
    const sessionDir = path.join(SESSION_ROOT, newSessions[0])
    const manifestPath = path.join(sessionDir, 'session.json')
    expect(fs.existsSync(manifestPath), `session.json not found in ${sessionDir}`).toBe(true)
  })
})
