import { arr, grid, num, randInt, type Level } from './types';

const digits = randInt(0, 10);

export const sharedLevels: Level[] = [
  {
    id: 'shared',
    title: 'Shared memory',
    text: 'Each thread adds its right-hand neighbour within the block: `out[i] = a[i] + a[next]`, where the last thread of a block wraps around to the first thread of the same block. Load the block\'s slice of `a` into `__shared__` memory once, wait for everyone with `__syncthreads()`, then read the neighbour from shared memory.',
    hint: 'Without the barrier a thread can read its neighbour\'s slot before the neighbour has written it, and the race detector reports exactly that.',
    params: [
      { kind: 'array', name: 'a', type: 'float', shape: [256], data: digits },
      { kind: 'array', name: 'out', type: 'float', shape: [256], output: true },
    ],
    launch: grid(4, 1, 64, 1),
    starter: `#define TPB 64

__global__ void neighbours(const float* a, float* out) {
  __shared__ float s[TPB];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int t = threadIdx.x;
}
`,
    reference: `#define TPB 64

__global__ void neighbours(const float* a, float* out) {
  __shared__ float s[TPB];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int t = threadIdx.x;
  s[t] = a[i];
  __syncthreads();
  out[i] = s[t] + s[(t + 1) % TPB];
}
`,
    expected: (x) => {
      const a = arr(x, 'a');
      return { out: a.map((v, i) => v + a[i - (i % 64) + ((i + 1) % 64)]) };
    },
  },
  {
    id: 'pooling',
    title: 'Pooling',
    text: 'Sum a sliding window of three, `out[i] = a[i - 2] + a[i - 1] + a[i]`, where anything before `a[0]` counts as 0, over `n` = 300 elements with 5 blocks of 64 threads.',
    hint: 'Every element is needed by three threads, so read it from global memory once into shared memory, and let the first two threads of each block also load the two elements just before the block.',
    params: [
      { kind: 'array', name: 'a', type: 'float', shape: [300], data: digits },
      { kind: 'array', name: 'out', type: 'float', shape: [300], output: true },
      { kind: 'scalar', name: 'n', type: 'int', value: 300 },
    ],
    launch: grid(5, 1, 64, 1),
    starter: `#define TPB 64

__global__ void pool(const float* a, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int t = threadIdx.x;
}
`,
    reference: `#define TPB 64

__global__ void pool(const float* a, float* out, int n) {
  __shared__ float s[TPB + 2];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int t = threadIdx.x;
  if (i < n) s[t + 2] = a[i];
  if (t < 2) s[t] = i >= 2 ? a[i - 2] : 0.0f;
  __syncthreads();
  if (i < n) out[i] = s[t] + s[t + 1] + s[t + 2];
}
`,
    expected: (x) => {
      const a = arr(x, 'a');
      return { out: a.map((v, i) => v + (a[i - 1] ?? 0) + (a[i - 2] ?? 0)) };
    },
  },
  {
    id: 'dot',
    title: 'Dot product',
    text: 'Compute `out[0] = a[0]*b[0] + a[1]*b[1] + ...` over `n` = 1024 elements, with 4 blocks of 256 threads. Each block adds up its 256 products in shared memory, then one thread per block adds the block total to `out[0]` with `atomicAdd`.',
    hint: 'Halve the number of active threads each round and keep the active ones packed at the front (`if (t < stride)`), so that whole warps drop out together. Picking every other thread instead (`t % (2 * stride) == 0`) keeps every warp busy with a few lanes, which shows up as divergence.',
    params: [
      { kind: 'array', name: 'a', type: 'float', shape: [1024], data: digits },
      { kind: 'array', name: 'b', type: 'float', shape: [1024], data: digits },
      { kind: 'array', name: 'out', type: 'float', shape: [1], output: true },
      { kind: 'scalar', name: 'n', type: 'int', value: 1024 },
    ],
    launch: grid(4, 1, 256, 1),
    starter: `#define TPB 256

__global__ void dot(const float* a, const float* b, float* out, int n) {
  __shared__ float s[TPB];
  int t = threadIdx.x;
  int i = blockIdx.x * blockDim.x + t;
}
`,
    reference: `#define TPB 256

__global__ void dot(const float* a, const float* b, float* out, int n) {
  __shared__ float s[TPB];
  int t = threadIdx.x;
  int i = blockIdx.x * blockDim.x + t;
  s[t] = a[i] * b[i];
  __syncthreads();
  for (int stride = TPB / 2; stride > 0; stride /= 2) {
    if (t < stride) s[t] += s[t + stride];
    __syncthreads();
  }
  if (t == 0) atomicAdd(out, s[0]);
}
`,
    expected: (x) => {
      const a = arr(x, 'a');
      const b = arr(x, 'b');
      return { out: [a.reduce((s, v, i) => s + v * b[i], 0)] };
    },
  },
  {
    id: 'conv',
    title: '1D convolution',
    text: 'Convolve `a` (300 elements) with the 5-tap kernel `b`, so that `out[i] = a[i]*b[0] + a[i+1]*b[1] + ... + a[i+4]*b[4]` where elements past the end of `a` count as 0, using 3 blocks of 128 threads.',
    hint: 'A block needs its 128 elements of `a` plus 4 more from the next block, so load all 132 into shared memory along with `b`.',
    params: [
      { kind: 'array', name: 'a', type: 'float', shape: [300], data: digits },
      { kind: 'array', name: 'b', type: 'float', shape: [5], data: randInt(0, 5) },
      { kind: 'array', name: 'out', type: 'float', shape: [300], output: true },
      { kind: 'scalar', name: 'n', type: 'int', value: 300 },
    ],
    launch: grid(3, 1, 128, 1),
    starter: `#define TPB 128
#define K 5

__global__ void conv(const float* a, const float* b, float* out, int n) {
  int t = threadIdx.x;
  int i = blockIdx.x * blockDim.x + t;
}
`,
    reference: `#define TPB 128
#define K 5

__global__ void conv(const float* a, const float* b, float* out, int n) {
  __shared__ float sa[TPB + K - 1];
  __shared__ float sb[K];
  int t = threadIdx.x;
  int i = blockIdx.x * blockDim.x + t;
  sa[t] = i < n ? a[i] : 0.0f;
  if (t < K - 1) sa[TPB + t] = i + TPB < n ? a[i + TPB] : 0.0f;
  if (t < K) sb[t] = b[t];
  __syncthreads();
  if (i < n) {
    float acc = 0.0f;
    for (int j = 0; j < K; j++) acc += sa[t + j] * sb[j];
    out[i] = acc;
  }
}
`,
    expected: (x) => {
      const a = arr(x, 'a');
      const b = arr(x, 'b');
      return { out: a.map((_, i) => b.reduce((s, w, j) => s + (a[i + j] ?? 0) * w, 0)) };
    },
  },
  {
    id: 'scan',
    title: 'Prefix sum',
    text: 'Compute the inclusive prefix sum `out[i] = a[0] + a[1] + ... + a[i]` of 256 ints with one block of 256 threads.',
    hint: 'In round k every thread adds the value 2^k places to its left, so after 8 rounds each thread has its whole prefix. Every thread has to read before any thread writes, which takes one barrier between the read and the write and another before the next round.',
    params: [
      { kind: 'array', name: 'a', type: 'int', shape: [256], data: digits },
      { kind: 'array', name: 'out', type: 'int', shape: [256], output: true },
    ],
    launch: grid(1, 1, 256, 1),
    starter: `#define N 256

__global__ void scan(const int* a, int* out) {
  __shared__ int s[N];
  int t = threadIdx.x;
}
`,
    reference: `#define N 256

__global__ void scan(const int* a, int* out) {
  __shared__ int s[N];
  int t = threadIdx.x;
  s[t] = a[t];
  __syncthreads();
  for (int off = 1; off < N; off *= 2) {
    int v = t >= off ? s[t - off] : 0;
    __syncthreads();
    s[t] += v;
    __syncthreads();
  }
  out[t] = s[t];
}
`,
    expected: (x) => {
      let s = 0;
      return { out: arr(x, 'a').map((v) => (s += v)) };
    },
  },
  {
    id: 'axis-sum',
    title: 'Axis sum',
    text: 'Sum each row of the 16×256 matrix `a` into `out[row]`, with one block of 128 threads per row.',
    hint: 'Let thread t add columns t, t + 128, ... so that a warp reads 32 neighbouring floats at a time, then combine the 128 partial sums in shared memory. Giving each thread its own contiguous chunk of columns looks tidier but makes every warp access stride across memory.',
    params: [
      { kind: 'array', name: 'a', type: 'float', shape: [16, 256], data: digits },
      { kind: 'array', name: 'out', type: 'float', shape: [16], output: true },
      { kind: 'scalar', name: 'cols', type: 'int', value: 256 },
    ],
    launch: grid(16, 1, 128, 1),
    starter: `#define TPB 128

__global__ void rowsum(const float* a, float* out, int cols) {
  int t = threadIdx.x;
  int row = blockIdx.x;
}
`,
    reference: `#define TPB 128

__global__ void rowsum(const float* a, float* out, int cols) {
  __shared__ float s[TPB];
  int t = threadIdx.x;
  int row = blockIdx.x;
  float acc = 0.0f;
  for (int c = t; c < cols; c += TPB) acc += a[row * cols + c];
  s[t] = acc;
  __syncthreads();
  for (int stride = TPB / 2; stride > 0; stride /= 2) {
    if (t < stride) s[t] += s[t + stride];
    __syncthreads();
  }
  if (t == 0) out[row] = s[0];
}
`,
    expected: (x) => {
      const a = arr(x, 'a');
      const cols = num(x, 'cols');
      return { out: Array.from({ length: a.length / cols }, (_, r) => a.subarray(r * cols, (r + 1) * cols).reduce((s, v) => s + v, 0)) };
    },
  },
];
