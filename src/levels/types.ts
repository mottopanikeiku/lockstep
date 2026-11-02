import type { ElemType, ValType } from '../ir/ir';
import type { Launch } from '../sim/machine';

export type ArrayParam = {
  kind: 'array';
  name: string;
  type: ElemType;
  /** [n] or [rows, cols]; only used for display. */
  shape: number[];
  /** Input data, or omitted for outputs (which start as zeros). */
  data?: (rng: () => number, i: number) => number;
  output?: boolean;
};

export type ScalarParam = { kind: 'scalar'; name: string; type: ValType; value: number };

export type LevelParam = ArrayParam | ScalarParam;

export type Inputs = Record<string, Float64Array | number>;

/** The array behind an input name (inputs mix arrays and scalars). */
export const arr = (inputs: Inputs, name: string) => inputs[name] as Float64Array;
export const num = (inputs: Inputs, name: string) => inputs[name] as number;

export interface Par {
  cycles: number;
  sectors: number;
  lines: number;
}

export interface Level {
  id: string;
  title: string;
  /** A few plain sentences; `code` spans in backticks. */
  text: string;
  params: LevelParam[];
  launch: Launch;
  starter: string;
  /** The solution par comes from. */
  reference: string;
  /** One nudge toward the technique the level is about. */
  hint?: string;
  /** What the outputs should hold, computed in plain JS. */
  expected: (inputs: Inputs) => Record<string, ArrayLike<number>>;
  /** Allowed error for float outputs, relative to the expected value's size (and absolute near zero). */
  tolerance?: number;
}

/** Deterministic PRNG so every player sees the same inputs. */
export function rng(seed: string): () => number {
  let h = 2166136261;
  for (let i = 0; i < seed.length; i++) h = Math.imul(h ^ seed.charCodeAt(i), 16777619);
  let a = h >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Integer in [lo, hi). */
export const randInt = (lo: number, hi: number) => (r: () => number) => lo + Math.floor(r() * (hi - lo));

/** Float in [lo, hi), rounded to float32. */
export const randFloat = (lo: number, hi: number) => (r: () => number) => Math.fround(lo + r() * (hi - lo));

export const grid = (gx: number, gy: number, bx: number, by: number): Launch => ({
  grid: { x: gx, y: gy },
  block: { x: bx, y: by },
});

export const size = (p: ArrayParam) => p.shape.reduce((a, b) => a * b, 1);

export function makeInputs(level: Level): Inputs {
  const r = rng(level.id);
  const out: Inputs = {};
  for (const p of level.params) {
    if (p.kind === 'scalar') out[p.name] = p.value;
    else {
      const a = new Float64Array(size(p));
      if (p.data) for (let i = 0; i < a.length; i++) a[i] = p.data(r, i);
      out[p.name] = a;
    }
  }
  return out;
}
