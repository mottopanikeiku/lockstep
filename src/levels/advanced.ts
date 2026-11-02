import { arr, grid, num, randFloat, randInt, type Inputs, type Level } from './types';

function matmul(x: Inputs): Float64Array {
  const A = arr(x, 'A');
  const B = arr(x, 'B');
  const n = num(x, 'n');
  const C = new Float64Array(n * n);
  for (let r = 0; r < n; r++)
    for (let c = 0; c < n; c++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += A[r * n + k] * B[k * n + c];
      C[r * n + c] = s;
    }
  return C;
}

const small = randInt(0, 5);

export const advancedLevels: Level[] = [
  {
    id: 'matmul',
    title: 'Matrix multiply',
    text: 'Multiply two 32×32 matrices: `C[row][col]` is the dot product of row `row` of `A` and column `col` of `B`. One thread per output element, in 16×16 blocks.',
    params: [
      { kind: 'array', name: 'A', type: 'float', shape: [32, 32], data: small },
      { kind: 'array', name: 'B', type: 'float', shape: [32, 32], data: small },
      { kind: 'array', name: 'C', type: 'float', shape: [32, 32], output: true },
      { kind: 'scalar', name: 'n', type: 'int', value: 32 },
    ],
    launch: grid(2, 2, 16, 16),
    starter: `__global__ void matmul(const float* A, const float* B, float* C, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
}
`,
    reference: `__global__ void matmul(const float* A, const float* B, float* C, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float acc = 0.0f;
  for (int k = 0; k < n; k++) acc += A[row * n + k] * B[k * n + col];
  C[row * n + col] = acc;
}
`,
    expected: (x) => ({ C: matmul(x) }),
  },
  {
    id: 'tiled-matmul',
    title: 'Tiled matmul',
    text: 'The same product for 48×48 matrices, but now make it cheap. Every element of `A` and `B` is read by 48 threads; load each 16×16 tile into shared memory once per block and do the inner loop there.',
    hint: 'Loop over tiles: each thread loads one element of the `A` tile and one of the `B` tile, barrier, 16 multiply-adds from shared memory, barrier again before the next tile overwrites them.',
    params: [
      { kind: 'array', name: 'A', type: 'float', shape: [48, 48], data: small },
      { kind: 'array', name: 'B', type: 'float', shape: [48, 48], data: small },
      { kind: 'array', name: 'C', type: 'float', shape: [48, 48], output: true },
      { kind: 'scalar', name: 'n', type: 'int', value: 48 },
    ],
    launch: grid(3, 3, 16, 16),
    starter: `#define TILE 16

__global__ void matmul(const float* A, const float* B, float* C, int n) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * TILE + ty;
  int col = blockIdx.x * TILE + tx;
}
`,
    reference: `#define TILE 16

__global__ void matmul(const float* A, const float* B, float* C, int n) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * TILE + ty;
  int col = blockIdx.x * TILE + tx;
  float acc = 0.0f;
  for (int t = 0; t < n; t += TILE) {
    As[ty][tx] = A[row * n + t + tx];
    Bs[ty][tx] = B[(t + ty) * n + col];
    __syncthreads();
    for (int k = 0; k < TILE; k++) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  C[row * n + col] = acc;
}
`,
    expected: (x) => ({ C: matmul(x) }),
  },
  {
    id: 'transpose',
    title: 'Transpose',
    text: 'Write the transpose of the 64×96 matrix `a` into `out`, which is 96×64. Each block owns a 32×32 tile and has 32×8 threads, so every thread moves four elements.',
    hint: 'Reading `a` along rows is cheap, but writing `out` down columns costs a sector per element. Stage the tile in shared memory so the write walks along rows too. Then read the tile down a column, and watch the bank conflicts until you pad each row to 33 floats.',
    params: [
      { kind: 'array', name: 'a', type: 'float', shape: [64, 96], data: (_, i) => i },
      { kind: 'array', name: 'out', type: 'float', shape: [96, 64], output: true },
      { kind: 'scalar', name: 'rows', type: 'int', value: 64 },
      { kind: 'scalar', name: 'cols', type: 'int', value: 96 },
    ],
    launch: grid(3, 2, 32, 8),
    starter: `#define TILE 32
#define ROWS 8

__global__ void transpose(const float* a, float* out, int rows, int cols) {
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += ROWS) {
    // this thread handles row y + j, column x of a
  }
}
`,
    reference: `#define TILE 32
#define ROWS 8

__global__ void transpose(const float* a, float* out, int rows, int cols) {
  __shared__ float tile[TILE][TILE + 1];
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += ROWS)
    tile[threadIdx.y + j][threadIdx.x] = a[(y + j) * cols + x];
  __syncthreads();
  x = blockIdx.y * TILE + threadIdx.x;
  y = blockIdx.x * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += ROWS)
    out[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
}
`,
    expected: (x) => {
      const a = arr(x, 'a');
      const rows = num(x, 'rows');
      const cols = num(x, 'cols');
      const out = new Float64Array(rows * cols);
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out[c * rows + r] = a[r * cols + c];
      return { out };
    },
  },
  {
    id: 'warp-reduce',
    title: 'Warp shuffle',
    text: 'Sum 2048 floats into `out[0]` with 8 blocks of 256 threads, this time with warp shuffles. `__shfl_down_sync(0xffffffff, v, k)` hands each lane the `v` of the lane `k` places above it, without touching memory.',
    hint: 'Five shuffle rounds (16, 8, 4, 2, 1) leave each warp\'s total in lane 0. Put the 8 warp totals in shared memory, barrier, and let the first warp shuffle those down to one number.',
    params: [
      { kind: 'array', name: 'a', type: 'float', shape: [2048], data: randInt(0, 10) },
      { kind: 'array', name: 'out', type: 'float', shape: [1], output: true },
    ],
    launch: grid(8, 1, 256, 1),
    starter: `#define TPB 256

__global__ void reduce(const float* a, float* out) {
  int t = threadIdx.x;
  float v = a[blockIdx.x * TPB + t];
}
`,
    reference: `#define TPB 256

__global__ void reduce(const float* a, float* out) {
  __shared__ float warpSums[TPB / 32];
  int t = threadIdx.x;
  float v = a[blockIdx.x * TPB + t];
  for (int off = 16; off > 0; off /= 2) v += __shfl_down_sync(0xffffffff, v, off);
  if (t % 32 == 0) warpSums[t / 32] = v;
  __syncthreads();
  if (t < 32) {
    v = t < TPB / 32 ? warpSums[t] : 0.0f;
    for (int off = 16; off > 0; off /= 2) v += __shfl_down_sync(0xffffffff, v, off);
    if (t == 0) atomicAdd(out, v);
  }
}
`,
    expected: (x) => ({ out: [arr(x, 'a').reduce((s, v) => s + v, 0)] }),
  },
  {
    id: 'histogram',
    title: 'Histogram',
    text: 'Count how often each value 0 to 15 appears among 1024 ints and store the counts in `hist`. Many threads hit the same bin at once, so the increments have to be atomic.',
    hint: 'Every warp fights over 16 global counters. Give each block its own 16 counters in shared memory, count there, and add the block\'s counts to `hist` at the end.',
    params: [
      { kind: 'array', name: 'a', type: 'int', shape: [1024], data: randInt(0, 16) },
      { kind: 'array', name: 'hist', type: 'int', shape: [16], output: true },
      { kind: 'scalar', name: 'n', type: 'int', value: 1024 },
    ],
    launch: grid(4, 1, 256, 1),
    starter: `#define BINS 16

__global__ void histogram(const int* a, int* hist, int n) {
  int t = threadIdx.x;
  int i = blockIdx.x * blockDim.x + t;
}
`,
    reference: `#define BINS 16

__global__ void histogram(const int* a, int* hist, int n) {
  __shared__ int local[BINS];
  int t = threadIdx.x;
  int i = blockIdx.x * blockDim.x + t;
  if (t < BINS) local[t] = 0;
  __syncthreads();
  if (i < n) atomicAdd(&local[a[i]], 1);
  __syncthreads();
  if (t < BINS) atomicAdd(&hist[t], local[t]);
}
`,
    expected: (x) => {
      const hist = new Float64Array(16);
      for (const v of arr(x, 'a')) hist[v]++;
      return { hist };
    },
  },
  {
    id: 'softmax',
    title: 'Softmax',
    text: 'Softmax over each row of a 16×128 matrix: `out[r][c] = exp(x[r][c] - m) / s`, where `m` is the row maximum and `s` the sum of the exponentials. One block of 128 threads per row, one element per thread.',
    hint: 'Two block-wide reductions: a max, then a sum. Shuffles do most of it; four warps need a small shared array to combine. Mind the race when you reuse that array for the second reduction.',
    params: [
      { kind: 'array', name: 'x', type: 'float', shape: [16, 128], data: randFloat(-4, 4) },
      { kind: 'array', name: 'out', type: 'float', shape: [16, 128], output: true },
      { kind: 'scalar', name: 'cols', type: 'int', value: 128 },
    ],
    launch: grid(16, 1, 128, 1),
    tolerance: 1e-5,
    starter: `#define TPB 128

__global__ void softmax(const float* x, float* out, int cols) {
  int t = threadIdx.x;
  float v = x[blockIdx.x * cols + t];
}
`,
    reference: `#define TPB 128

__global__ void softmax(const float* x, float* out, int cols) {
  __shared__ float red[TPB / 32];
  int t = threadIdx.x;
  int lane = t % 32, warp = t / 32;
  float v = x[blockIdx.x * cols + t];

  float m = v;
  for (int off = 16; off > 0; off /= 2) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
  if (lane == 0) red[warp] = m;
  __syncthreads();
  m = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
  __syncthreads();

  float e = expf(v - m);
  float s = e;
  for (int off = 16; off > 0; off /= 2) s += __shfl_xor_sync(0xffffffff, s, off);
  if (lane == 0) red[warp] = s;
  __syncthreads();
  s = red[0] + red[1] + red[2] + red[3];
  out[blockIdx.x * cols + t] = e / s;
}
`,
    expected: (x) => {
      const v = arr(x, 'x');
      const cols = num(x, 'cols');
      const out = new Float64Array(v.length);
      for (let r = 0; r < v.length / cols; r++) {
        const row = v.subarray(r * cols, (r + 1) * cols);
        const m = Math.max(...row);
        const e = Array.from(row, (y) => Math.exp(y - m));
        const s = e.reduce((p, q) => p + q, 0);
        e.forEach((y, c) => (out[r * cols + c] = y / s));
      }
      return { out };
    },
  },
  {
    id: 'attention',
    title: 'Flash attention',
    text: 'Attention for 32 queries over 128 keys, head size 16: `O = softmax(Q Kᵀ / 4) V`, one softmax per query row. Each block takes 16 queries with 16×16 threads (x is the output column, y the query). Never hold all 128 scores at once: walk over the keys in tiles of 32 held in shared memory.',
    hint: 'Keep a running max `m`, running sum `l` and running output `acc` per thread. When a tile raises the max to `m2`, scale `l` and `acc` by `expf(m - m2)` before adding the tile\'s terms. That is the online softmax from FlashAttention.',
    params: [
      { kind: 'array', name: 'Q', type: 'float', shape: [32, 16], data: randFloat(-1, 1) },
      { kind: 'array', name: 'K', type: 'float', shape: [128, 16], data: randFloat(-1, 1) },
      { kind: 'array', name: 'V', type: 'float', shape: [128, 16], data: randFloat(-1, 1) },
      { kind: 'array', name: 'O', type: 'float', shape: [32, 16], output: true },
      { kind: 'scalar', name: 'nkeys', type: 'int', value: 128 },
    ],
    launch: grid(2, 1, 16, 16),
    tolerance: 1e-4,
    starter: `#define D 16   // head size
#define BQ 16  // queries per block
#define BK 32  // keys per tile

__global__ void attention(const float* Q, const float* K, const float* V, float* O, int nkeys) {
  int d = threadIdx.x;
  int row = blockIdx.x * BQ + threadIdx.y;
}
`,
    reference: `#define D 16   // head size
#define BQ 16  // queries per block
#define BK 32  // keys per tile

__global__ void attention(const float* Q, const float* K, const float* V, float* O, int nkeys) {
  __shared__ float q[BQ][D];
  __shared__ float k[BK][D + 1];
  __shared__ float v[BK][D];
  __shared__ float s[BQ][BK + 1];
  int d = threadIdx.x, r = threadIdx.y;
  int row = blockIdx.x * BQ + r;
  q[r][d] = Q[row * D + d];
  float m = -INFINITY, l = 0.0f, acc = 0.0f;
  for (int start = 0; start < nkeys; start += BK) {
    for (int e = r * D + d; e < BK * D; e += BQ * D) {
      k[e / D][e % D] = K[start * D + e];
      v[e / D][e % D] = V[start * D + e];
    }
    __syncthreads();
    for (int j = d; j < BK; j += D) {
      float dot = 0.0f;
      for (int c = 0; c < D; c++) dot += q[r][c] * k[j][c];
      s[r][j] = dot * 0.25f;
    }
    __syncthreads();
    float m2 = m;
    for (int j = 0; j < BK; j++) m2 = fmaxf(m2, s[r][j]);
    float scale = expf(m - m2);
    l *= scale;
    acc *= scale;
    for (int j = 0; j < BK; j++) {
      float p = expf(s[r][j] - m2);
      l += p;
      acc += p * v[j][d];
    }
    m = m2;
    __syncthreads();
  }
  O[row * D + d] = acc / l;
}
`,
    expected: (x) => {
      const Q = arr(x, 'Q');
      const K = arr(x, 'K');
      const V = arr(x, 'V');
      const nk = num(x, 'nkeys');
      const D = 16;
      const nq = Q.length / D;
      const O = new Float64Array(nq * D);
      for (let q = 0; q < nq; q++) {
        const scores = Array.from({ length: nk }, (_, j) => {
          let dot = 0;
          for (let c = 0; c < D; c++) dot += Q[q * D + c] * K[j * D + c];
          return dot / 4;
        });
        const m = Math.max(...scores);
        const p = scores.map((s) => Math.exp(s - m));
        const sum = p.reduce((a, b) => a + b, 0);
        for (let c = 0; c < D; c++) {
          let o = 0;
          for (let j = 0; j < nk; j++) o += p[j] * V[j * D + c];
          O[q * D + c] = o / sum;
        }
      }
      return { O };
    },
  },
];
