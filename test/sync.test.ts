import { describe, expect, it } from 'vitest';
import { iota, launch1d, oneWarp, run, zeros } from './helpers';

describe('barriers', () => {
  it('reports __syncthreads() inside a divergent branch as a deadlock', () => {
    const src = `__global__ void k(float* out) {
  int t = threadIdx.x;
  if (t < 16) {
    __syncthreads();
  }
  out[t] = 1.0f;
}`;
    const r = run(src, oneWarp, [zeros(32)]);
    expect(r.fault).toMatchObject({ kind: 'deadlock', line: 4 });
    expect(r.fault!.message).toContain('lanes 0-15 of warp 0 while lanes 16-31 took another branch');
  });

  it('reports a barrier after some threads returned', () => {
    const src = `__global__ void k(const float* a, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  __shared__ float s[64];
  s[threadIdx.x] = a[i];
  __syncthreads();
  out[i] = s[63 - threadIdx.x];
}`;
    const r = run(src, launch1d(2, 64), [iota(100), zeros(100), 100]);
    expect(r.fault).toMatchObject({ kind: 'deadlock', line: 6 });
    expect(r.fault!.message).toContain('28 returned early (line 3)');
  });

  it('reports warps waiting at different barriers', () => {
    const src = `__global__ void k(float* out) {
  if (threadIdx.x < 32) {
    __syncthreads();
  } else {
    __syncthreads();
  }
}`;
    const r = run(src, launch1d(1, 64), [zeros(1)]);
    expect(r.fault?.kind).toBe('deadlock');
    expect(r.fault!.message).toContain('warp 0 on line 3; warp 1 on line 5');
  });

  it('accepts a warp-uniform branch around work and charges the wait', () => {
    const src = `__global__ void k(float* out) {
  float x = 1.0f;
  if (threadIdx.x < 32) {
    for (int j = 0; j < 10; j++) x = sqrtf(x + 1.0f);
  }
  __syncthreads();
  out[threadIdx.x] = x;
}`;
    const r = run(src, launch1d(1, 64), [zeros(64)]);
    expect(r.fault).toBeUndefined();
    expect(r.metrics.barriers).toBe(1);
    expect(r.metrics.divergentBranches).toBe(0);
    expect(r.metrics.barrierWait).toBeGreaterThan(40);
    // Every cycle lands on a line, including the time warp 1 waits at the barrier.
    expect(r.lineCycles.reduce((a, b) => a + b, 0)).toBe(r.metrics.cycles);
    expect(r.lineCycles[6]).toBeGreaterThanOrEqual(r.metrics.barrierWait);
  });
});

describe('race detector', () => {
  const rotate = (sync: string) => `__global__ void k(const float* a, float* out) {
  __shared__ float s[64];
  int t = threadIdx.x;
  s[t] = a[t];
  ${sync}
  out[t] = s[(t + 1) % 64];
}`;

  it('catches a missing __syncthreads()', () => {
    const r = run(rotate(''), launch1d(1, 64), [iota(64), zeros(64)]);
    expect(r.fault).toMatchObject({ kind: 'race', line: 6 });
    expect(r.fault!.message).toMatch(/^race on shared s\[1\]: thread 1 wrote it on line 4 and thread 0 read it on line 6/);
    expect(r.races).toBe(64);
  });

  it('is quiet once the barrier is there', () => {
    const r = run(rotate('__syncthreads();'), launch1d(1, 64), [iota(64), zeros(64)]);
    expect(r.fault).toBeUndefined();
    expect(r.buffers[1][63]).toBe(0);
    expect(r.buffers[1][0]).toBe(1);
  });

  it('treats lanes of one warp as unordered without __syncwarp()', () => {
    const src = (sync: string) => `__global__ void k(int* out) {
  __shared__ int s[32];
  s[threadIdx.x] = threadIdx.x;
  ${sync}
  out[threadIdx.x] = s[threadIdx.x ^ 1];
}`;
    expect(run(src(''), oneWarp, [zeros(32)]).fault?.kind).toBe('race');
    expect(run(src('__syncwarp();'), oneWarp, [zeros(32)]).fault).toBeUndefined();
  });

  it('flags many threads writing one shared word', () => {
    const src = `__global__ void k(int* out) {
  __shared__ int s;
  s = threadIdx.x;
  __syncthreads();
  out[0] = s;
}`;
    const r = run(src, oneWarp, [zeros(1)]);
    expect(r.fault?.kind).toBe('race');
    expect(r.fault!.message).toContain('Several threads write the same address');
  });

  it('flags plain read-modify-write across blocks and accepts atomics', () => {
    const plain = `__global__ void k(const int* a, int* out) {
  if (threadIdx.x == 0) out[0] += a[blockIdx.x];
}`;
    const r = run(plain, launch1d(4, 32), [iota(4), zeros(1)]);
    expect(r.fault?.kind).toBe('race');
    expect(r.fault!.message).toContain('thread 0 of block 0 wrote it on line 2 and thread 0 of block 1 read it on line 2');
    expect(r.fault!.message).toContain('use an atomic');
    const atomic = plain.replace('out[0] += a[blockIdx.x]', 'atomicAdd(&out[0], a[blockIdx.x])');
    const ok = run(atomic, launch1d(4, 32), [iota(4), zeros(1)]);
    expect(ok.fault).toBeUndefined();
    expect(ok.buffers[1][0]).toBe(6);
  });

  it('allows one thread to read and write its own element', () => {
    const src = `__global__ void k(float* x) { x[threadIdx.x] = x[threadIdx.x] * 2.0f; }`;
    expect(run(src, oneWarp, [iota(32)]).fault).toBeUndefined();
  });
});
