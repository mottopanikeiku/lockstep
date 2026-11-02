import { describe, expect, it } from 'vitest';
import { attempt, type Attempt } from '../src/game/attempt';
import { LEVELS, PAR } from '../src/levels';

const level = (id: string) => LEVELS.find((l) => l.id === id)!;

/** A solution that passes, as an attempt with a score. */
function passes(id: string, src: string): Attempt {
  const a = attempt(level(id), src);
  expect(a.problem).toBeUndefined();
  expect(a.solved).toBe(true);
  return a;
}

function fails(id: string, src: string): Attempt {
  const a = attempt(level(id), src);
  expect(a.solved).toBe(false);
  return a;
}

/** The reference solution with one piece of text swapped. */
function edit(id: string, from: string | RegExp, to: string): string {
  const ref = level(id).reference;
  const out = ref.replace(from, to);
  if (out === ref) throw new Error(`edit did not apply to ${id}`);
  return out;
}

describe('naive solutions pass with a worse score', () => {
  it('blocks-2d: swapping x and y breaks coalescing', () => {
    const a = passes('blocks-2d', `__global__ void blocks2d(const float* a, float* out, int size) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < size && col < size) out[row * size + col] = a[row * size + col] + 10.0f;
}`);
    expect(a.score!.sectors).toBeGreaterThanOrEqual(3 * PAR['blocks-2d'].sectors);
    expect(a.score!.cycles).toBeGreaterThan(PAR['blocks-2d'].cycles);
  });

  it('shared: reading neighbours from global memory costs more sectors', () => {
    const a = passes('shared', `__global__ void neighbours(const float* a, float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i] = a[i] + a[blockIdx.x * 64 + (threadIdx.x + 1) % 64];
}`);
    expect(a.score!.sectors).toBeGreaterThan(PAR.shared.sectors);
    expect(a.score!.cycles).toBeGreaterThan(PAR.shared.cycles);
  });

  it('pooling: three global reads per thread', () => {
    const a = passes('pooling', `__global__ void pool(const float* a, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float s = a[i];
    if (i >= 1) s += a[i - 1];
    if (i >= 2) s += a[i - 2];
    out[i] = s;
  }
}`);
    expect(a.score!.sectors).toBeGreaterThan(PAR.pooling.sectors);
    expect(a.score!.cycles).toBeGreaterThan(PAR.pooling.cycles);
  });

  it('dot: interleaved addressing diverges inside every warp', () => {
    const a = passes('dot', edit('dot', /for \(int stride[^\n]*\n[^\n]*\n/, `for (int stride = 1; stride < TPB; stride *= 2) {
    if (t % (2 * stride) == 0) s[t] += s[t + stride];
`));
    const ref = attempt(level('dot'), level('dot').reference);
    expect(a.sim!.metrics.divergentBranches).toBeGreaterThan(5 * ref.sim!.metrics.divergentBranches);
    expect(a.sim!.metrics.activeLanes / a.sim!.metrics.issued).toBeLessThan(ref.sim!.metrics.activeLanes / ref.sim!.metrics.issued);
    expect(a.score!.cycles).toBeGreaterThan(PAR.dot.cycles);
  });

  it('axis-sum: contiguous chunks per thread do not coalesce', () => {
    const a = passes('axis-sum', edit('axis-sum', 'for (int c = t; c < cols; c += TPB) acc += a[row * cols + c];', 'for (int c = 2 * t; c < 2 * t + 2; c++) acc += a[row * cols + c];'));
    expect(a.score!.sectors).toBeGreaterThan(PAR['axis-sum'].sectors);
  });

  it('axis-sum: one thread per row is slow', () => {
    const a = passes('axis-sum', `__global__ void rowsum(const float* a, float* out, int cols) {
  if (threadIdx.x == 0) {
    float acc = 0.0f;
    for (int c = 0; c < cols; c++) acc += a[blockIdx.x * cols + c];
    out[blockIdx.x] = acc;
  }
}`);
    expect(a.score!.cycles).toBeGreaterThan(PAR['axis-sum'].cycles);
    expect(a.score!.lines).toBeLessThan(PAR['axis-sum'].lines);
  });

  it('matmul: rows along threadIdx.x read B down columns', () => {
    const a = passes('matmul', edit('matmul', /int row = .*\n.*int col = .*\n/, `int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
`));
    expect(a.score!.sectors).toBeGreaterThan(PAR.matmul.sectors);
  });

  it('tiled-matmul: the untiled kernel reads far more from global memory', () => {
    const a = passes('tiled-matmul', level('matmul').reference);
    expect(a.score!.sectors).toBeGreaterThan(1.5 * PAR['tiled-matmul'].sectors);
    expect(a.score!.cycles).toBeGreaterThan(PAR['tiled-matmul'].cycles);
  });

  it('transpose: writing straight to out does not coalesce', () => {
    const a = passes('transpose', `__global__ void transpose(const float* a, float* out, int rows, int cols) {
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  for (int j = 0; j < 32; j += 8) out[x * rows + y + j] = a[(y + j) * cols + x];
}`);
    expect(a.score!.sectors).toBeGreaterThan(2 * PAR.transpose.sectors);
  });

  it('transpose: an unpadded tile has 32-way bank conflicts', () => {
    const a = passes('transpose', edit('transpose', 'tile[TILE][TILE + 1]', 'tile[TILE][TILE]'));
    const ref = attempt(level('transpose'), level('transpose').reference);
    expect(ref.sim!.metrics.bankConflicts).toBe(0);
    expect(a.sim!.metrics.bankConflicts).toBe(6 * 8 * 4 * 31);
    expect(a.score!.cycles - PAR.transpose.cycles).toBe(6 * 8 * 4 * 31);
  });

  it('warp-reduce: a shared-memory tree is slower than shuffles', () => {
    const a = passes('warp-reduce', `#define TPB 256
__global__ void reduce(const float* a, float* out) {
  __shared__ float s[TPB];
  int t = threadIdx.x;
  s[t] = a[blockIdx.x * TPB + t];
  __syncthreads();
  for (int stride = TPB / 2; stride > 0; stride /= 2) {
    if (t < stride) s[t] += s[t + stride];
    __syncthreads();
  }
  if (t == 0) atomicAdd(out, s[0]);
}`);
    expect(a.score!.cycles).toBeGreaterThan(PAR['warp-reduce'].cycles);
  });

  it('histogram: global atomics collide more than shared ones', () => {
    const a = passes('histogram', `__global__ void histogram(const int* a, int* hist, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&hist[a[i]], 1);
}`);
    expect(a.sim!.metrics.atomicCollisions).toBeGreaterThan(500);
    expect(a.score!.cycles).toBeGreaterThan(PAR.histogram.cycles);
  });

  it('attention: streaming K and V from global memory without tiles', () => {
    const a = passes('attention', `#define D 16
__global__ void attention(const float* Q, const float* K, const float* V, float* O, int nkeys) {
  int d = threadIdx.x;
  int row = blockIdx.x * 16 + threadIdx.y;
  float m = -INFINITY, l = 0.0f, acc = 0.0f;
  for (int j = 0; j < nkeys; j++) {
    float dot = 0.0f;
    for (int c = 0; c < D; c++) dot += Q[row * D + c] * K[j * D + c];
    float s = dot * 0.25f;
    float m2 = fmaxf(m, s);
    float scale = expf(m - m2);
    float p = expf(s - m2);
    l = l * scale + p;
    acc = acc * scale + p * V[j * D + d];
    m = m2;
  }
  O[row * D + d] = acc / l;
}`);
    expect(a.score!.cycles).toBeGreaterThan(PAR.attention.cycles);
    expect(a.score!.sectors).toBeGreaterThan(PAR.attention.sectors);
  });
});

describe('buggy solutions are caught', () => {
  it('guard: no bounds check reads past the end', () => {
    const a = fails('guard', edit('guard', 'if (i < n) ', ''));
    expect(a.problem).toMatchObject({ kind: 'oob', line: 3 });
  });

  it('shared: no barrier is a race', () => {
    const a = fails('shared', edit('shared', '  __syncthreads();\n', ''));
    expect(a.problem).toMatchObject({ kind: 'race', line: 8 });
  });

  it('pooling: returning before the barrier', () => {
    const a = fails('pooling', edit('pooling', '  if (i < n) s[t + 2] = a[i];', '  if (i >= n) return;\n  s[t + 2] = a[i];'));
    expect(a.problem?.kind).toBe('deadlock');
  });

  it('dot: no barrier inside the reduction loop', () => {
    const a = fails('dot', edit('dot', '    __syncthreads();\n  }', '  }'));
    expect(a.problem).toMatchObject({ kind: 'race', line: 10 });
  });

  it('dot: a barrier inside the if', () => {
    const a = fails('dot', edit('dot', /if \(t < stride\) s\[t\] \+= s\[t \+ stride\];\n\s*__syncthreads\(\);/, 'if (t < stride) { s[t] += s[t + stride]; __syncthreads(); }'));
    expect(a.problem).toMatchObject({ kind: 'deadlock', line: 10 });
  });

  it('scan: one barrier per round is a race', () => {
    const a = fails('scan', edit('scan', /int v = [^\n]*\n\s*__syncthreads\(\);\n\s*s\[t\] \+= v;/, 'if (t >= off) s[t] += s[t - off];'));
    expect(a.problem?.kind).toBe('race');
  });

  it('tiled-matmul: no barrier before the next tile is loaded', () => {
    const a = fails('tiled-matmul', edit('tiled-matmul', /(acc \+= [^\n]*\n)\s*__syncthreads\(\);\n/, '$1'));
    expect(a.problem?.kind).toBe('race');
    expect(a.problem!.message).toMatch(/As|Bs/);
  });

  it('histogram: a plain increment races across threads', () => {
    const a = fails('histogram', `__global__ void histogram(const int* a, int* hist, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) hist[a[i]] += 1;
}`);
    expect(a.problem).toMatchObject({ kind: 'race', line: 3 });
  });

  it('softmax: reusing the scratch array without a barrier', () => {
    const a = fails('softmax', edit('softmax', /(m = fmaxf\(fmaxf[^\n]*\n)\s*__syncthreads\(\);\n/, '$1'));
    expect(a.problem?.kind).toBe('race');
  });

  it('warp-reduce: shuffling inside a branch only some lanes take', () => {
    const a = fails('warp-reduce', `__global__ void reduce(const float* a, float* out) {
  float v = a[blockIdx.x * 256 + threadIdx.x];
  if (v > 4.0f) v += __shfl_down_sync(0xffffffff, v, 1);
  atomicAdd(out, v);
}`);
    expect(a.problem).toMatchObject({ kind: 'deadlock', line: 3 });
  });
});
