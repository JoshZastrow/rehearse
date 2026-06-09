import { defineConfig, devices } from '@playwright/test'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const __dirname = dirname(fileURLToPath(import.meta.url))
const REPO_ROOT = resolve(__dirname, '../../..')

// Shared LiveKit dev credentials. livekit-server --dev ships with these built in.
const LK_ENV = {
  LIVEKIT_API_KEY: 'devkey',
  LIVEKIT_API_SECRET: 'secret',
  LIVEKIT_ROOM_NAME: 'rehearse-room',
}

export default defineConfig({
  testDir: './e2e',
  timeout: 120_000,
  // One room, one one-shot agent per run → never parallelize.
  workers: 1,
  fullyParallel: false,
  use: {
    baseURL: 'http://localhost:3000',
    permissions: ['microphone'],
  },
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          args: [
            '--use-fake-ui-for-media-stream',
            '--use-fake-device-for-media-stream',
          ],
        },
      },
    },
  ],
  webServer: [
    {
      // LiveKit dev server: in-memory, built-in devkey/secret, ws on :7880.
      command: 'livekit-server --dev --bind 0.0.0.0',
      url: 'http://localhost:7880',
      reuseExistingServer: !process.env.CI,
      timeout: 30_000,
      stdout: 'pipe',
      stderr: 'pipe',
    },
    {
      // Token minting server (FastAPI) — health check is the token endpoint.
      command: 'uv run python web/livekit/token_server.py',
      url: 'http://localhost:8765/api/livekit/token',
      cwd: REPO_ROOT,
      env: { ...LK_ENV, TOKEN_SERVER_PORT: '8765' },
      reuseExistingServer: !process.env.CI,
      timeout: 60_000,
      stdout: 'pipe',
      stderr: 'pipe',
    },
    {
      // Vite dev server hosting the React app.
      command: 'npm run dev',
      url: 'http://localhost:3000',
      reuseExistingServer: !process.env.CI,
      timeout: 60_000,
    },
  ],
})
