import { describe, expect, it } from 'vitest';
import { postDominators } from '../src/ir/postdom';
import { kernel } from './helpers';

describe('post-dominators', () => {
  it('finds the merge point of a diamond', () => {
    // 0 -> 1, 2; 1 -> 3; 2 -> 3; 3 -> exit(4)
    const ipdom = postDominators([[1, 2], [3], [3], [4], []], 4);
    expect([...ipdom]).toEqual([3, 3, 3, 4, 4]);
  });

  it('sends a break to the loop exit', () => {
    // 0 -> 1 (header); 1 -> 2 (body), 4 (exit); 2 -> 3, 4 (break); 3 -> 1 (latch); 4 -> exit(5)
    const ipdom = postDominators([[1], [2, 4], [3, 4], [1], [5], []], 5);
    expect(ipdom[1]).toBe(4);
    expect(ipdom[2]).toBe(4);
    expect(ipdom[3]).toBe(1);
  });

  it('handles infinite loops through a virtual exit edge', () => {
    const ipdom = postDominators([[1], [1], []], 2);
    expect(ipdom[1]).toBe(2);
    expect(ipdom[0]).toBe(2);
  });
});

describe('lowering', () => {
  it('records the join point of an if/else as the first instruction after it', () => {
    const k = kernel(`__global__ void k(int* out) {
      int v;
      if (threadIdx.x < 8) v = 1; else v = 2;
      out[threadIdx.x] = v;
    }`);
    const cbr = k.code.find((i) => i.op === 'cbr');
    if (cbr?.op !== 'cbr') throw new Error('no branch');
    const join = k.code[cbr.join];
    expect(join.op).toBe('st');
    expect(join.line).toBe(4);
  });

  it('fuses multiply-add and folds constants', () => {
    const k = kernel(`__global__ void k(const float* a, const float* b, float* out) {
      int i = threadIdx.x;
      int stride = 4 * 8 + 1;
      out[i] = a[i] * b[i] + a[i + stride];
    }`);
    const ops = k.code.map((i) => (i.op === 'bin' ? `bin.${i.fn}` : i.op));
    expect(ops).toEqual(['ld', 'ld', 'bin.add', 'ld', 'mad', 'st', 'ret']);
  });

  it('rotates loops so each iteration pays one compare and one branch', () => {
    const k = kernel(`__global__ void k(int* out) {
      int s = 0;
      for (int j = 0; j < 10; j++) s += j;
      out[threadIdx.x] = s;
    }`);
    const ops = k.code.map((i) => (i.op === 'bin' ? `bin.${i.fn}` : i.op));
    // s = 0; j = 0; (guard folds away) body: s += j; j++; j < 10; branch back
    expect(ops).toEqual(['mov', 'mov', 'bin.add', 'bin.add', 'bin.lt', 'cbr', 'st', 'ret']);
  });

  it('flattens 2D shared indices with one mad', () => {
    const k = kernel(`__global__ void k(float* out) {
      __shared__ float t[8][9];
      t[threadIdx.y][threadIdx.x] = 1.0f;
      out[0] = t[0][1];
    }`);
    expect(k.shared[0]).toMatchObject({ name: 't', dims: [8, 9], size: 72, offset: 0 });
    expect(k.code.filter((i) => i.op === 'mad')).toHaveLength(1);
  });
});
