import { defineConfig } from '@playwright/test';

const port = 48761;

export default defineConfig({
  testDir: './e2e',
  workers: 1,
  retries: 0,
  timeout: 60_000,
  use: {
    baseURL: `http://127.0.0.1:${port}/lockstep/`,
    viewport: { width: 1440, height: 900 },
    launchOptions: { args: ['--disable-gpu', '--renderer-process-limit=1'] },
  },
  webServer: {
    command: `npx vite preview --port ${port} --strictPort --host 127.0.0.1`,
    url: `http://127.0.0.1:${port}/lockstep/`,
    reuseExistingServer: false,
  },
});
