import { describe, expect, it } from 'vitest';
import { kernel, launch1d, oneWarp, run, stepsOnLine, zeros } from './helpers';
import { simulate } from '../src/sim/machine';

const lanes = (lo: number, hi: number) => {
  let m = 0;
  for (let l = lo; l < hi; l++) m |= 1 << l;
  return m >>> 0;
};

describe('lockstep execution', () => {
  it('runs both sides of a divergent branch, then reconverges', () => {
    const src = `__global__ void k(int* out) {
  int t = threadIdx.x;
  int v;
  if (t < 8) v = 1;
  else v = 2;
  out[t] = v;
}`;
    const k = kernel(src);
    const r = simulate(k, oneWarp, [zeros(32)], { trace: true });
    const t = r.trace!;
    const maskOn = (line: number) => stepsOnLine(r, k, line).map((s) => t.mask[s]);
    expect(maskOn(4)).toContain(lanes(0, 8));
    expect(maskOn(5)).toContain(lanes(8, 32));
    expect(maskOn(6)).toEqual([lanes(0, 32)]);
    expect(stepsOnLine(r, k, 4).at(-1)!).toBeLessThan(stepsOnLine(r, k, 5)[0]);
    expect(r.metrics.divergentBranches).toBe(1);
    expect([...r.buffers[0]]).toEqual(Array.from({ length: 32 }, (_, i) => (i < 8 ? 1 : 2)));
  });

  it('handles nested divergence', () => {
    const src = `__global__ void k(int* out) {
  int t = threadIdx.x, v;
  if (t < 16) {
    if (t < 4) v = 1;
    else v = 2;
  } else v = 3;
  out[t] = v;
}`;
    const k = kernel(src);
    const r = simulate(k, oneWarp, [zeros(32)], { trace: true });
    const t = r.trace!;
    const maskOn = (line: number) => new Set(stepsOnLine(r, k, line).map((s) => t.mask[s]));
    expect(maskOn(4)).toContain(lanes(0, 4));
    expect(maskOn(5)).toContain(lanes(4, 16));
    expect(maskOn(6)).toContain(lanes(16, 32));
    expect([...maskOn(7)]).toEqual([lanes(0, 32)]);
    // base, outer else, outer then (parked at the inner join), inner else, inner then
    expect(Math.max(...t.depth.subarray(0, t.steps))).toBe(5);
  });

  it('keeps looping with fewer lanes as threads finish their trips', () => {
    const src = `__global__ void k(int* out) {
  int t = threadIdx.x, s = 0;
  for (int j = 0; j < t; j++)
    s += j;
  out[t] = s;
}`;
    const k = kernel(src);
    const r = simulate(k, oneWarp, [zeros(32)], { trace: true });
    const t = r.trace!;
    const body = stepsOnLine(r, k, 4).map((s) => t.mask[s]);
    expect(body).toHaveLength(31);
    expect(body[0]).toBe(lanes(1, 32));
    expect(body[30]).toBe(lanes(31, 32));
    expect(stepsOnLine(r, k, 5).map((s) => t.mask[s])).toEqual([lanes(0, 32)]);
    expect([...r.buffers[0]]).toEqual(Array.from({ length: 32 }, (_, i) => (i * (i - 1)) / 2 + 0));
  });

  it('reconverges lanes that break out of a loop early', () => {
    const src = `__global__ void k(int* out) {
  int t = threadIdx.x, s = 0;
  for (int j = 0; j < 32; j++) {
    if (j == t) break;
    s++;
  }
  out[t] = s;
}`;
    const k = kernel(src);
    const r = simulate(k, oneWarp, [zeros(32)], { trace: true });
    expect([...r.buffers[0]]).toEqual(Array.from({ length: 32 }, (_, i) => i));
    expect(stepsOnLine(r, k, 7).map((s) => r.trace!.mask[s])).toEqual([lanes(0, 32)]);
  });

  it('drops lanes that return and keeps going with the rest', () => {
    const src = `__global__ void k(int* out) {
  int t = threadIdx.x;
  if (t >= 20) return;
  out[t] = 1;
}`;
    const k = kernel(src);
    const r = simulate(k, oneWarp, [zeros(32)], { trace: true });
    expect(stepsOnLine(r, k, 4).map((s) => r.trace!.mask[s])).toEqual([lanes(0, 20)]);
    expect(r.buffers[0][19]).toBe(1);
    expect(r.buffers[0][20]).toBe(0);
  });

  it('only charges divergence when lanes of one warp disagree', () => {
    const uniform = `__global__ void k(float* out) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x < 32) out[t] = expf(1.0f * t);
  else out[t] = sqrtf(1.0f * t);
}`;
    const split = uniform.replace('threadIdx.x < 32', 'threadIdx.x % 2 == 0');
    const a = run(uniform, launch1d(1, 64), [zeros(64)]);
    const b = run(split, launch1d(1, 64), [zeros(64)]);
    expect(a.metrics.divergentBranches).toBe(0);
    expect(b.metrics.divergentBranches).toBe(2);
    expect(b.metrics.issued).toBeGreaterThan(a.metrics.issued);
    expect(b.metrics.cycles).toBeGreaterThan(a.metrics.cycles);
    expect(a.metrics.activeLanes / a.metrics.issued).toBe(32);
  });

  it('computes 32-bit int and float arithmetic like the GPU', () => {
    const src = `__global__ void k(int* o, float* f) {
  int big = 2147483647;
  o[0] = big + 1;
  o[1] = -7 / 2;
  o[2] = -7 % 2;
  o[3] = (int)-2.7f;
  o[4] = 1 << 31;
  f[0] = 0.1f + 0.2f;
  f[1] = (float)16777217;
}`;
    const r = run(src, { grid: { x: 1, y: 1 }, block: { x: 1, y: 1 } }, [zeros(5), zeros(2)]);
    expect([...r.buffers[0]]).toEqual([-2147483648, -3, -1, -2, -2147483648]);
    expect(r.buffers[1][0]).toBe(Math.fround(Math.fround(0.1) + Math.fround(0.2)));
    expect(r.buffers[1][1]).toBe(16777216);
  });

  it('reduces a warp with shuffles', () => {
    const src = `__global__ void k(const int* a, int* down, int* xr) {
  int t = threadIdx.x;
  int v = a[t], w = a[t];
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, off);
    w += __shfl_xor_sync(0xffffffff, w, off);
  }
  down[t] = v;
  xr[t] = w;
}`;
    const a = Float64Array.from({ length: 32 }, (_, i) => i + 1);
    const r = run(src, oneWarp, [a, zeros(32), zeros(32)]);
    expect(r.buffers[1][0]).toBe(528);
    expect([...r.buffers[2]].every((x) => x === 528)).toBe(true);
    expect(r.metrics.shuffleCycles).toBeGreaterThan(0);
  });

  it('reports a shuffle in divergent code', () => {
    const src = `__global__ void k(int* out) {
  int v = threadIdx.x;
  if (threadIdx.x < 16)
    v = __shfl_down_sync(0xffffffff, v, 1);
  out[threadIdx.x] = v;
}`;
    const r = run(src, oneWarp, [zeros(32)]);
    expect(r.fault).toMatchObject({ kind: 'deadlock', line: 4 });
    expect(r.fault!.message).toContain('lanes 16-31');
  });

  it('stops runaway kernels', () => {
    const src = `__global__ void k(int* out) { while (out[0] == 0) { } }`;
    const r = simulate(kernel(src), oneWarp, [zeros(1)], { maxSteps: 1000 });
    expect(r.fault?.kind).toBe('timeout');
  });

  it('reports integer division by zero at the line', () => {
    const src = `__global__ void k(int* out) {
  int z = threadIdx.x - threadIdx.x;
  out[0] = 5 / z;
}`;
    const r = run(src, oneWarp, [zeros(1)]);
    expect(r.fault).toMatchObject({ kind: 'runtime', line: 3 });
  });
});
