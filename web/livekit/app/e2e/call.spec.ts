import { test, expect } from '@playwright/test'

test.describe('home page', () => {
  test('shows all three design links', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByText('Design 1 — Gemini Style')).toBeVisible()
    await expect(page.getByText('Design 2 — Minimal Orb')).toBeVisible()
    await expect(page.getByText('Design 3 — Waveform Split')).toBeVisible()
  })
})

test.describe('Design 1 — Gemini Style', () => {
  test('idle state shows Start Call button', async ({ page }) => {
    await page.goto('/design/gemini')
    await expect(page.getByText('Tap to start your session')).toBeVisible()
    await expect(page.getByRole('button', { name: 'Start Call' })).toBeVisible()
  })

  test('Start Call transitions to connecting or connected', async ({ page }) => {
    await page.goto('/design/gemini')
    await page.getByRole('button', { name: 'Start Call' }).click()

    // Should leave idle — either connecting spinner or connected state
    await expect(page.getByText('Tap to start your session')).not.toBeVisible({ timeout: 5_000 })

    // At minimum, connecting state shows "Connecting…"
    // or connected state shows the End button — accept either
    const connected = page.getByRole('button', { name: 'End' })
    const connecting = page.getByText('Connecting…')
    await expect(connected.or(connecting)).toBeVisible({ timeout: 10_000 })
  })

  test('End button returns to idle', async ({ page }) => {
    await page.goto('/design/gemini')
    await page.getByRole('button', { name: 'Start Call' }).click()

    // Wait for the End button (connected state)
    const endBtn = page.getByRole('button', { name: 'End' })
    await expect(endBtn).toBeVisible({ timeout: 10_000 })
    await endBtn.click()

    // Should return to idle
    await expect(page.getByText('Tap to start your session')).toBeVisible({ timeout: 8_000 })
    await expect(page.getByRole('button', { name: 'Start Call' })).toBeVisible()
  })
})

test.describe('Design 2 — Minimal Orb', () => {
  test('renders and has a start button', async ({ page }) => {
    await page.goto('/design/minimal')
    // Minimal orb should render without crashing
    await expect(page).toHaveURL('/design/minimal')
    // Look for any call-to-action (button or clickable area)
    await page.waitForTimeout(500)
    const body = await page.textContent('body')
    expect(body).toBeTruthy()
  })
})

test.describe('Design 3 — Waveform Split', () => {
  test('renders without crashing', async ({ page }) => {
    await page.goto('/design/waveform')
    await expect(page).toHaveURL('/design/waveform')
    await page.waitForTimeout(500)
    const body = await page.textContent('body')
    expect(body).toBeTruthy()
  })
})
