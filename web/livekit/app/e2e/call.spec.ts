import { test, expect } from '@playwright/test'
import { ROLE_LABELS } from '../src/roles'

test.describe('Waveform UI', () => {
  test('idle state renders waveform panels and start button', async ({ page }) => {
    await page.goto('/')
    // Role labels come from the single source of truth (src/roles.ts); each
    // appears twice (panel + side label). getByText is case-insensitive, so it
    // matches both the uppercased panel chip and the title-case side label.
    await expect(page.getByText(ROLE_LABELS.user).first()).toBeVisible()
    await expect(page.getByText(ROLE_LABELS.agent).first()).toBeVisible()
    await expect(page.getByText('Tap to Start')).toBeVisible()
  })

  test('clicking start button transitions out of idle', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('button', { name: 'Start call' }).click()
    await expect(page.getByText('Tap to Start')).not.toBeVisible({ timeout: 5_000 })
  })

  test('connects to LiveKit room and shows session active', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('button', { name: 'Start call' }).click()
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
})
