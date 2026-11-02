import { describe, expect, it } from 'vitest';
import { verdict } from '../src/game/attempt';
import { isUnlocked, loadProgress, recordSolve, saveProgress, type Progress } from '../src/game/progress';
import { decodeShare, encodeShare } from '../src/game/share';
import { INTRO_LEVELS, LEVELS } from '../src/levels';

describe('sharing', () => {
  it('round-trips a solution through the URL hash', async () => {
    const code = LEVELS.at(-1)!.reference;
    const hash = await encodeShare('attention', code);
    expect(hash).toMatch(/^attention\/[A-Za-z0-9_-]+$/);
    expect(hash.length).toBeLessThan(code.length);
    expect(await decodeShare('#' + hash)).toEqual({ levelId: 'attention', code });
  });

  it('ignores hashes that are not solutions', async () => {
    expect(await decodeShare('#about')).toBeNull();
    expect(await decodeShare('#map/not-deflate')).toBeNull();
  });
});

describe('progress', () => {
  it('opens the intro levels one by one, then everything', () => {
    const p: Progress = { levels: {} };
    const score = { cycles: 1, sectors: 1, lines: 1 };
    expect(isUnlocked(p, 0)).toBe(true);
    expect(isUnlocked(p, 1)).toBe(false);
    for (let i = 0; i < INTRO_LEVELS; i++) {
      expect(isUnlocked(p, INTRO_LEVELS)).toBe(false);
      recordSolve(p, LEVELS[i].id, score);
    }
    expect(isUnlocked(p, LEVELS.length - 1)).toBe(true);
  });

  it('keeps the best of each metric separately', () => {
    const p: Progress = { levels: {} };
    recordSolve(p, 'map', { cycles: 50, sectors: 8, lines: 3 });
    const better = recordSolve(p, 'map', { cycles: 40, sectors: 9, lines: 3 });
    expect(better).toEqual({ cycles: true, sectors: false, lines: false });
    expect(p.levels.map.best).toEqual({ cycles: 40, sectors: 8, lines: 3 });
  });

  it('survives missing or corrupt storage', () => {
    const store = new Map<string, string>();
    const storage = { getItem: (k: string) => store.get(k) ?? null, setItem: (k: string, v: string) => void store.set(k, v) };
    expect(loadProgress(storage)).toEqual({ levels: {} });
    store.set('lockstep:v1', '{nope');
    expect(loadProgress(storage)).toEqual({ levels: {} });
    saveProgress({ levels: { map: { code: 'x' } }, current: 'map' }, storage);
    expect(loadProgress(storage).levels.map.code).toBe('x');
  });
});

describe('verdict', () => {
  const par = { cycles: 100, sectors: 10, lines: 5 };

  it('compares a solve with par', () => {
    expect(verdict(par, par)).toBe('All three scores match par.');
    expect(verdict({ cycles: 90, sectors: 10, lines: 4 }, par)).toBe('You beat par on cycles and lines.');
    expect(verdict({ cycles: 120, sectors: 10, lines: 5 }, par)).toBe('Cycles are over par, so there is a better kernel to find.');
    expect(verdict({ cycles: 90, sectors: 12, lines: 6 }, par)).toBe('You beat par on cycles, while sectors and lines are over par.');
  });

  it('mentions a new personal best only when there was an old one', () => {
    expect(verdict({ cycles: 90, sectors: 10, lines: 5 }, par, { cycles: 95, sectors: 9, lines: 5 })).toBe(
      'You beat par on cycles. That beats your previous best on cycles.',
    );
    expect(verdict({ cycles: 90, sectors: 10, lines: 5 }, par, undefined)).toBe('You beat par on cycles.');
  });
});
