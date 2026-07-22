import { test, expect } from '@playwright/test'

// The call UI is now gated behind Clerk auth (see src/main.tsx). These browser
// tests require a running dev server (`npm run dev`) and a configured Clerk
// instance; they are not part of the hermetic Python gate.
//
//   VITE_CLERK_PUBLISHABLE_KEY  — required for ClerkProvider to initialize.
//   E2E_LIVE_STACK=1            — set when the token server + LiveKit + a
//                                 signed-in Clerk session are available, to run
//                                 the full call-flow assertions.
const CLERK_KEY = process.env.VITE_CLERK_PUBLISHABLE_KEY
const LIVE_STACK = process.env.E2E_LIVE_STACK === '1'

test.describe('Auth gate (signed out)', () => {
  test.skip(!CLERK_KEY, 'requires VITE_CLERK_PUBLISHABLE_KEY')

  test('signed-out user sees the sign-in prompt, not the call UI', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByText('Sign in to start a coaching session.')).toBeVisible()
    // The call UI must not be reachable without signing in.
    await expect(page.getByText('TAP TO START')).not.toBeVisible()
  })
})

test.describe('Call flow (signed in)', () => {
  test.skip(
    !CLERK_KEY || !LIVE_STACK,
    'requires a signed-in Clerk session + running token/LiveKit stack (E2E_LIVE_STACK=1)',
  )

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
