import { defineConfig } from 'vitest/config';

export default defineConfig({
  base: '/lockstep/',
  build: { target: 'es2022' },
  test: { include: ['test/**/*.test.ts'] },
});
