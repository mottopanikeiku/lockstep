import { defineConfig } from '@playwright/test';

// BASE_URL=https://mottopanikeiku.github.io/lockstep/ runs the smoke test against the live site.
const live = process.env.BASE_URL;
const port = 48761;
const local = `http://127.0.0.1:${port}/lockstep/`;

export default defineConfig({
  testDir: './e2e',
  workers: 1,
  retries: 0,
  timeout: 60_000,
  use: {
    baseURL: live ?? local,
    viewport: { width: 1440, height: 900 },
    launchOptions: { args: ['--disable-gpu', '--renderer-process-limit=1'] },
  },
  webServer: live
    ? undefined
    : {
        command: `npx vite preview --port ${port} --strictPort --host 127.0.0.1`,
        url: local,
        reuseExistingServer: false,
      },
});
