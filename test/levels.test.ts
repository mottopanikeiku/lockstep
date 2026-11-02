import { writeFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { attempt } from '../src/game/attempt';
import { LEVELS, PAR } from '../src/levels';
import type { Par } from '../src/levels/types';

const scores: Record<string, Par> = {};

describe('reference solutions', () => {
  it.each(LEVELS.map((l) => [l.id, l] as const))('%s', (_, level) => {
    const a = attempt(level, level.reference);
    expect(a.problem).toBeUndefined();
    expect(a.warnings).toEqual([]);
    expect(a.solved).toBe(true);
    // The breakdown docs/cost-model.md describes adds up to the score.
    const m = a.sim!.metrics;
    expect(m.issued + m.specialCycles + m.shuffleCycles + m.globalCycles + m.sharedCycles + m.barrierWait).toBe(m.cycles);
    expect(a.sim!.lineCycles.reduce((x, y) => x + y, 0)).toBe(m.cycles);
    scores[level.id] = a.score!;
  });

  // UPDATE_PAR=1 npx vitest run test/levels.test.ts rewrites the table after a cost model change.
  it('match the par table', () => {
    if (process.env.UPDATE_PAR) {
      writeFileSync(new URL('../src/levels/par.json', import.meta.url), JSON.stringify(scores, null, 2) + '\n');
    } else {
      expect(scores).toEqual(PAR);
    }
  });
});

describe('starter code', () => {
  it.each(LEVELS.map((l) => [l.id, l] as const))('%s compiles but does not solve the level', (_, level) => {
    const a = attempt(level, level.starter);
    expect(a.compiled.ok).toBe(true);
    expect(a.solved).toBe(false);
  });
});

describe('levels', () => {
  it('have unique ids and par for each', () => {
    expect(new Set(LEVELS.map((l) => l.id)).size).toBe(LEVELS.length);
    expect(LEVELS).toHaveLength(20);
  });

  it('reject a kernel with the wrong signature', () => {
    const a = attempt(LEVELS[0], '__global__ void map(const float* a, int* out) { }');
    expect(a.problem).toMatchObject({ kind: 'signature', message: `this level's kernel takes (const float* a, float* out)` });
  });

  it('report the first wrong value and point at the store', () => {
    const src = LEVELS[0].reference.replace('10.0f', '11.0f');
    const a = attempt(LEVELS[0], src);
    expect(a.problem).toMatchObject({ kind: 'wrong', line: 3, message: 'out[0] should be 10 but is 11 (32 wrong values)' });
    expect(a.outputs[0].wrong).toHaveLength(32);
  });
});
