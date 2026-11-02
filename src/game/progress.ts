import { INTRO_LEVELS, LEVELS } from '../levels';
import type { Par } from '../levels/types';

const KEY = 'lockstep:v1';

export interface LevelProgress {
  code?: string;
  /** Best score on each metric, kept separately like a Zachtronics histogram. */
  best?: Par;
}

export interface Progress {
  levels: Record<string, LevelProgress>;
  current?: string;
  theme?: 'light' | 'dark';
}

export function loadProgress(storage: Pick<Storage, 'getItem'> = localStorage): Progress {
  try {
    const raw = storage.getItem(KEY);
    const p = raw ? (JSON.parse(raw) as Progress) : undefined;
    if (p && typeof p.levels === 'object') return p;
  } catch {
    // Corrupt or blocked storage: start fresh.
  }
  return { levels: {} };
}

export function saveProgress(p: Progress, storage: Pick<Storage, 'setItem'> = localStorage): void {
  try {
    storage.setItem(KEY, JSON.stringify(p));
  } catch {
    // Private browsing or quota: progress just isn't kept.
  }
}

export const isSolved = (p: Progress, id: string) => Boolean(p.levels[id]?.best);

/** The first few levels open one at a time; after that everything is open. */
export function isUnlocked(p: Progress, index: number): boolean {
  for (let i = 0; i < Math.min(index, INTRO_LEVELS); i++) if (!isSolved(p, LEVELS[i].id)) return false;
  return true;
}

/** Record a solve; returns which metrics improved on the previous best. */
export function recordSolve(p: Progress, id: string, score: Par): Record<keyof Par, boolean> {
  const entry = (p.levels[id] ??= {});
  const prev = entry.best;
  const better = {
    cycles: !prev || score.cycles < prev.cycles,
    sectors: !prev || score.sectors < prev.sectors,
    lines: !prev || score.lines < prev.lines,
  };
  entry.best = {
    cycles: better.cycles ? score.cycles : prev!.cycles,
    sectors: better.sectors ? score.sectors : prev!.sectors,
    lines: better.lines ? score.lines : prev!.lines,
  };
  return better;
}
