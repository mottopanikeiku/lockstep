import { advancedLevels } from './advanced';
import { basics } from './basics';
import par from './par.json';
import { sharedLevels } from './shared';
import type { Level, Par } from './types';

export const LEVELS: Level[] = [...basics, ...sharedLevels, ...advancedLevels];

/** Scores of each level's reference solution; test/levels.test.ts keeps this file in sync. */
export const PAR: Record<string, Par> = par;

/** Levels that must be solved in order before the rest open up. */
export const INTRO_LEVELS = 3;
