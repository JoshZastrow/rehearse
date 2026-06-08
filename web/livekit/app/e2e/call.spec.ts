import { test, expect } from '@playwright/test'

test.describe('Waveform UI', () => {
  test('idle state renders waveform panels and start button', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByText('CALLER')).toBeVisible()
    await expect(page.getByText('AGENT')).toBeVisible()
    await expect(page.getByText('TAP TO START')).toBeVisible()
  })

  test('clicking start button transitions out of idle', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('button').first().click()
    await expect(page.getByText('TAP TO START')).not.toBeVisible({ timeout: 5_000 })
  })

  test('connects to LiveKit room and shows session active', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('button').first().click()
    await expect(page.getByText('Session Active')).toBeVisible({ timeout: 10_000 })
    await expect(page.getByText('LIVE')).toBeVisible()
  })

  test('End button returns to idle', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('button').first().click()
    await expect(page.getByText('Session Active')).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'End call' }).click()
    await expect(page.getByText('TAP TO START')).toBeVisible({ timeout: 8_000 })
    await expect(page.getByText('Session Active')).not.toBeVisible()
  })

  test('session timer counts up when connected', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('button').first().click()
    await expect(page.getByText('Session Active')).toBeVisible({ timeout: 10_000 })

    const timer = page.getByText(/^\d{2}:\d{2}:\d{2}$/)
    await expect(timer).toBeVisible()
    const t1 = await timer.textContent()
    await page.waitForTimeout(2_000)
    const t2 = await timer.textContent()
    expect(t1).not.toBe(t2)
  })
})
