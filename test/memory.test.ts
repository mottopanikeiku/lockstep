import { describe, expect, it } from 'vitest';
import { iota, launch1d, oneWarp, run, zeros } from './helpers';

const copy = (index: string) => `__global__ void k(const float* a, float* out) {
  int t = threadIdx.x;
  out[t] = a[${index}];
}`;

describe('coalescing', () => {
  it('counts 32-byte sectors per warp access', () => {
    const sectorsFor = (index: string) => {
      const r = run(copy(index), oneWarp, [iota(1056), zeros(32)]);
      expect(r.fault).toBeUndefined();
      // The store to out[t] always takes 4 sectors.
      return r.metrics.globalSectors - 4;
    };
    expect(sectorsFor('t')).toBe(4);
    expect(sectorsFor('t + 1')).toBe(5);
    expect(sectorsFor('31 - t')).toBe(4);
    expect(sectorsFor('0')).toBe(1);
    expect(sectorsFor('2 * t')).toBe(8);
    expect(sectorsFor('32 * t')).toBe(32);
  });

  it('charges cycles per sector', () => {
    const fast = run(copy('t'), oneWarp, [iota(1024), zeros(32)]);
    const slow = run(copy('32 * t'), oneWarp, [iota(1024), zeros(32)]);
    // 28 more sectors at 4 cycles each, plus the multiply for 32 * t.
    expect(slow.metrics.cycles - fast.metrics.cycles).toBe((32 - 4) * 4 + 1);
    expect([...slow.buffers[1]]).toEqual(Array.from({ length: 32 }, (_, i) => 32 * i));
  });

  it('reports out-of-bounds accesses with the thread', () => {
    const r = run(copy('t + 20'), oneWarp, [iota(40), zeros(32)]);
    expect(r.fault).toMatchObject({ kind: 'oob', line: 3 });
    expect(r.fault!.message).toBe('out-of-bounds read: a[40] but a has 40 elements (thread 20)');
  });
});

const sharedRead = (decl: string, write: string, read: string) => `__global__ void k(float* out) {
  int t = threadIdx.x;
  __shared__ float s${decl};
  ${write} = 1.0f;
  __syncwarp();
  out[t] = ${read};
}`;

describe('bank conflicts', () => {
  const wavefronts = (decl: string, write: string, read: string) => {
    const r = run(sharedRead(decl, write, read), oneWarp, [zeros(32)]);
    expect(r.fault).toBeUndefined();
    const t = r.trace!;
    for (let s = 0; s < t.steps; s++) {
      if (t.units[s] && r.trace!.accStart[s + 1] > r.trace!.accStart[s] && t.aBuf[t.accStart[s]] < 0 && t.aKind[t.accStart[s]] === 0) {
        return t.units[s];
      }
    }
    throw new Error('no shared read');
  };

  it('is conflict-free along a row', () => {
    expect(wavefronts('[32][32]', 's[t][t]', 's[0][t]')).toBe(1);
  });

  it('serializes a column of a 32-wide tile', () => {
    expect(wavefronts('[32][32]', 's[t][t]', 's[t][0]')).toBe(32);
  });

  it('padding the row to 33 removes the conflict', () => {
    expect(wavefronts('[32][33]', 's[t][t]', 's[t][0]')).toBe(1);
  });

  it('broadcasts one word to every lane', () => {
    expect(wavefronts('[32]', 's[t]', 's[0]')).toBe(1);
  });

  it('two-way conflict for stride 2', () => {
    expect(wavefronts('[64]', 's[2 * t]', 's[2 * t]')).toBe(2);
  });

  it('adds one cycle per extra wavefront', () => {
    // Write and read the same column: 32-way in both directions without padding.
    const a = run(sharedRead('[32][33]', 's[t][0]', 's[t][0]'), oneWarp, [zeros(32)]);
    const b = run(sharedRead('[32][32]', 's[t][0]', 's[t][0]'), oneWarp, [zeros(32)]);
    expect(b.metrics.bankConflicts - a.metrics.bankConflicts).toBe(31 + 31);
    expect(b.metrics.cycles - a.metrics.cycles).toBe(31 + 31);
  });
});

describe('atomics', () => {
  it('serializes lanes that hit the same global address', () => {
    const src = `__global__ void k(int* out) { atomicAdd(&out[0], 1); }`;
    const r = run(src, launch1d(2, 64), [zeros(1)]);
    expect(r.buffers[0][0]).toBe(128);
    expect(r.metrics.atomicCollisions).toBe(4 * 31);
    expect(r.fault).toBeUndefined();
  });

  it('serializes shared atomics on one word', () => {
    const src = `__global__ void k(int* out) {
  __shared__ int c;
  if (threadIdx.x == 0) c = 0;
  __syncthreads();
  int old = atomicAdd(&c, 1);
  __syncthreads();
  if (threadIdx.x == 0) out[0] = c;
  out[1 + threadIdx.x] = old;
}`;
    const r = run(src, oneWarp, [zeros(33)]);
    expect(r.fault).toBeUndefined();
    expect(r.buffers[0][0]).toBe(32);
    expect([...r.buffers[0].subarray(1)].sort((x, y) => x - y)).toEqual(Array.from({ length: 32 }, (_, i) => i));
    expect(r.metrics.bankConflicts).toBe(31);
  });
});
