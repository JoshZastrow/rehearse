import { test, expect } from '@playwright/test'

test.describe('Live Speech Dialogue UI', () => {
  test('idle state renders both speaker panels and start button', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByRole('heading', { name: 'LIVE SPEECH DIALOGUE' })).toBeVisible()
    await expect(page.getByText('YOU', { exact: true })).toBeVisible()
    await expect(page.getByText('SYSTEM', { exact: true })).toBeVisible()
    await expect(page.getByText('TAP TO START')).toBeVisible()
    await expect(page.getByRole('button', { name: 'Start call' })).toBeVisible()
  })

  test('clicking start transitions out of idle', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('button', { name: 'Start call' }).click()
    await expect(page.getByText('TAP TO START')).not.toBeVisible({ timeout: 5_000 })
  })

  test('connects to LiveKit room and shows CONNECTED', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('button', { name: 'Start call' }).click()
    await expect(page.getByText('CONNECTED', { exact: true })).toBeVisible({ timeout: 10_000 })
    // The LIVE indicator and the End-call control are present once connected.
    await expect(page.getByText('LIVE', { exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: 'End call' })).toBeVisible()
  })

  test('End returns to idle', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('button', { name: 'Start call' }).click()
    await expect(page.getByText('CONNECTED', { exact: true })).toBeVisible({ timeout: 10_000 })

    await page.getByRole('button', { name: 'End call' }).click()
    await expect(page.getByText('TAP TO START')).toBeVisible({ timeout: 8_000 })
  })

  test('both YOU and SYSTEM RSVP pills are present when connected', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('button', { name: 'Start call' }).click()
    await expect(page.getByText('CONNECTED', { exact: true })).toBeVisible({ timeout: 10_000 })
    await expect(page.locator('[data-testid="rsvp-pill"]')).toHaveCount(2)
  })
})
