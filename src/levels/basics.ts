import { arr, grid, num, randInt, type Level } from './types';

const index = (_: () => number, i: number) => i;

export const basics: Level[] = [
  {
    id: 'map',
    title: 'Map',
    text: 'Add 10 to every element of `a` and store the result in `out`. There is one thread per element, so thread `i` handles `a[i]`.',
    params: [
      { kind: 'array', name: 'a', type: 'float', shape: [32], data: index },
      { kind: 'array', name: 'out', type: 'float', shape: [32], output: true },
    ],
    launch: grid(1, 1, 32, 1),
    starter: `__global__ void map(const float* a, float* out) {
  int i = threadIdx.x;
  // out[i] = ...
}
`,
    reference: `__global__ void map(const float* a, float* out) {
  int i = threadIdx.x;
  out[i] = a[i] + 10.0f;
}
`,
    expected: (x) => ({ out: arr(x, 'a').map((v) => v + 10) }),
  },
  {
    id: 'zip',
    title: 'Zip',
    text: 'Add `a` and `b` element by element into `out`.',
    params: [
      { kind: 'array', name: 'a', type: 'float', shape: [32], data: index },
      { kind: 'array', name: 'b', type: 'float', shape: [32], data: randInt(0, 10) },
      { kind: 'array', name: 'out', type: 'float', shape: [32], output: true },
    ],
    launch: grid(1, 1, 32, 1),
    starter: `__global__ void zip(const float* a, const float* b, float* out) {
  int i = threadIdx.x;
}
`,
    reference: `__global__ void zip(const float* a, const float* b, float* out) {
  int i = threadIdx.x;
  out[i] = a[i] + b[i];
}
`,
    expected: (x) => ({ out: arr(x, 'a').map((v, i) => v + arr(x, 'b')[i]) }),
  },
  {
    id: 'guard',
    title: 'Guard',
    text: 'Add 10 to every element again, but now the block has 64 threads and `a` has only `n` = 50 elements. Threads past the end must not touch memory.',
    hint: 'Wrap the work in `if (i < n)`. The lanes that fail the test sit idle while the others run, which you can see in the thread view.',
    params: [
      { kind: 'array', name: 'a', type: 'float', shape: [50], data: index },
      { kind: 'array', name: 'out', type: 'float', shape: [50], output: true },
      { kind: 'scalar', name: 'n', type: 'int', value: 50 },
    ],
    launch: grid(1, 1, 64, 1),
    starter: `__global__ void guard(const float* a, float* out, int n) {
  int i = threadIdx.x;
}
`,
    reference: `__global__ void guard(const float* a, float* out, int n) {
  int i = threadIdx.x;
  if (i < n) out[i] = a[i] + 10.0f;
}
`,
    expected: (x) => ({ out: arr(x, 'a').map((v) => v + 10) }),
  },
  {
    id: 'map-2d',
    title: 'Map 2D',
    text: '`a` is a 12×12 matrix stored row by row, so element (row, col) lives at `a[row * size + col]`. The block is 16×16 threads: `threadIdx.x` picks the column and `threadIdx.y` the row. Add 10 to every element.',
    params: [
      { kind: 'array', name: 'a', type: 'float', shape: [12, 12], data: index },
      { kind: 'array', name: 'out', type: 'float', shape: [12, 12], output: true },
      { kind: 'scalar', name: 'size', type: 'int', value: 12 },
    ],
    launch: grid(1, 1, 16, 16),
    starter: `__global__ void map2d(const float* a, float* out, int size) {
  int col = threadIdx.x;
  int row = threadIdx.y;
}
`,
    reference: `__global__ void map2d(const float* a, float* out, int size) {
  int col = threadIdx.x;
  int row = threadIdx.y;
  if (row < size && col < size) out[row * size + col] = a[row * size + col] + 10.0f;
}
`,
    expected: (x) => ({ out: arr(x, 'a').map((v) => v + 10) }),
  },
  {
    id: 'broadcast',
    title: 'Broadcast',
    text: '`a` holds one number per row and `b` one number per column. Fill the 12×12 matrix `out` with `out[row][col] = a[row] + b[col]`.',
    params: [
      { kind: 'array', name: 'a', type: 'float', shape: [12], data: (_, i) => 10 * i },
      { kind: 'array', name: 'b', type: 'float', shape: [12], data: index },
      { kind: 'array', name: 'out', type: 'float', shape: [12, 12], output: true },
      { kind: 'scalar', name: 'size', type: 'int', value: 12 },
    ],
    launch: grid(1, 1, 16, 16),
    starter: `__global__ void broadcast(const float* a, const float* b, float* out, int size) {
  int col = threadIdx.x;
  int row = threadIdx.y;
}
`,
    reference: `__global__ void broadcast(const float* a, const float* b, float* out, int size) {
  int col = threadIdx.x;
  int row = threadIdx.y;
  if (row < size && col < size) out[row * size + col] = a[row] + b[col];
}
`,
    expected: (x) => {
      const a = arr(x, 'a');
      const b = arr(x, 'b');
      const n = num(x, 'size');
      return { out: Array.from({ length: n * n }, (_, i) => a[Math.floor(i / n)] + b[i % n]) };
    },
  },
  {
    id: 'blocks',
    title: 'Blocks',
    text: 'A block holds at most 1024 threads, so big arrays need many blocks. Here 4 blocks of 64 threads cover `n` = 200 elements. Block `blockIdx.x` starts at element `blockIdx.x * blockDim.x`. Add 10 to every element.',
    params: [
      { kind: 'array', name: 'a', type: 'float', shape: [200], data: index },
      { kind: 'array', name: 'out', type: 'float', shape: [200], output: true },
      { kind: 'scalar', name: 'n', type: 'int', value: 200 },
    ],
    launch: grid(4, 1, 64, 1),
    starter: `__global__ void blocks(const float* a, float* out, int n) {
  int i = threadIdx.x;  // only right for block 0
}
`,
    reference: `__global__ void blocks(const float* a, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + 10.0f;
}
`,
    expected: (x) => ({ out: arr(x, 'a').map((v) => v + 10) }),
  },
  {
    id: 'blocks-2d',
    title: 'Blocks 2D',
    text: 'A 40×40 matrix, covered by a 3×3 grid of 16×16 blocks. Add 10 to every element.',
    hint: 'A warp is 32 consecutive threads in x order: here two rows of 16. When `threadIdx.x` walks along a row of the matrix, the warp reads 2 runs of 64 bytes, 4 sectors. Swap x and y and the same warp touches 32 different rows.',
    params: [
      { kind: 'array', name: 'a', type: 'float', shape: [40, 40], data: randInt(0, 100) },
      { kind: 'array', name: 'out', type: 'float', shape: [40, 40], output: true },
      { kind: 'scalar', name: 'size', type: 'int', value: 40 },
    ],
    launch: grid(3, 3, 16, 16),
    starter: `__global__ void blocks2d(const float* a, float* out, int size) {
}
`,
    reference: `__global__ void blocks2d(const float* a, float* out, int size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < size && col < size) out[row * size + col] = a[row * size + col] + 10.0f;
}
`,
    expected: (x) => ({ out: arr(x, 'a').map((v) => v + 10) }),
  },
];
