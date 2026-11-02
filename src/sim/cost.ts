/**
 * The cost model, in made-up cycles. Every warp has its own clock and each
 * instruction it issues advances that clock. See docs/cost-model.md.
 */
export const COST = {
  /** Any instruction, once per warp, no matter how many lanes are active. */
  issue: 1,
  /** expf, logf, sqrtf, rsqrtf, float division, int division by a variable: total cost. */
  special: 4,
  /** Warp shuffle: total cost. */
  shuffle: 2,
  /** Global memory: per 32-byte sector a warp instruction touches, on top of issue. */
  globalSector: 4,
  /** Global atomics: per extra lane that hits an address another lane already hit. */
  atomicCollision: 4,
  /** Shared memory: per wavefront (1 if conflict-free, n for an n-way bank conflict), on top of issue. */
  sharedWavefront: 1,
} as const;

export const SECTOR_BYTES = 32;
export const BANKS = 32;
