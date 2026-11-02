import { describe, expect, it } from 'vitest';
import { compile } from '../src/compile';
import { lex } from '../src/lang/lexer';
import { Parser } from '../src/lang/parser';

const parse = (src: string) => new Parser(lex(src).tokens).parseProgram();

function errorOf(src: string) {
  const r = compile(src);
  if (r.ok) throw new Error('expected a compile error');
  return r.error;
}

describe('lexer', () => {
  it('reads float, hex and suffixed literals', () => {
    const toks = lex('1.5f 0xffffffff 2.0 3 1e-3f 7u').tokens;
    expect(toks.map((t) => t.kind)).toEqual(['float', 'int', 'float', 'int', 'float', 'int', 'eof']);
    expect(toks[0].value).toBe(1.5);
    expect(toks[1].value).toBe(-1);
    expect(toks[2].double).toBe(true);
    expect(toks[4].value).toBeCloseTo(0.001);
  });

  it('warns about double literals', () => {
    const r = compile('__global__ void k(float* a) { a[0] = a[0] * 2.0; }');
    expect(r.warnings).toHaveLength(1);
    expect(r.warnings[0]).toMatchObject({ line: 1, col: 45 });
    expect(r.warnings[0].message).toContain('2.0f');
  });

  it('expands #define and skips #include and comments', () => {
    const src = ['#include <cuda_runtime.h>', '#define TILE 16 // tile width', '/* #define NOT 1 */', 'TILE + 1'].join('\n');
    const toks = lex(src).tokens;
    expect(toks.map((t) => t.text)).toEqual(['16', '+', '1', '']);
    expect(toks[0].line).toBe(4);
  });

  it('rejects function-like macros', () => {
    expect(() => lex('#define SQ(x) ((x)*(x))')).toThrow(/function-like macros/);
  });
});

describe('parser', () => {
  it('parses a kernel with every statement form', () => {
    const p = parse(`
      #define N 64
      const int TILE = 16;
      __global__ void k(const float* __restrict__ a, float* out, int n) {
        __shared__ float tile[TILE][TILE + 1];
        int i = blockIdx.x * blockDim.x + threadIdx.x, j = 0;
        if (i < n && j == 0) out[i] = a[i]; else { out[i] = -a[i]; }
        for (int k = 0; k < N; k++) { if (k == 3) continue; if (k > 5) break; j += k; }
        while (j > 0) j--;
        do { j++; } while (j < 2);
        float x = i > 2 ? (float)i : static_cast<float>(j);
        x = fmaxf(x, 0.0f);
        atomicAdd(&out[0], x);
        __syncthreads();
        return;
      }`);
    expect(p.kernel.name).toBe('k');
    expect(p.kernel.params.map((x) => [x.name, x.type, x.pointer, x.isConst])).toEqual([
      ['a', 'float', true, true],
      ['out', 'float', true, false],
      ['n', 'int', false, false],
    ]);
    expect(p.consts).toHaveLength(1);
    expect(p.kernel.body.map((s) => s.k)).toEqual([
      'decl', 'decl', 'if', 'for', 'while', 'do', 'decl', 'expr', 'expr', 'expr', 'return',
    ]);
  });

  it('respects C precedence', () => {
    const p = parse('__global__ void k(int* a) { a[0] = 1 + 2 * 3 << 1 & 7 | 8 ^ 2; }');
    const s = p.kernel.body[0];
    if (s.k !== 'expr' || s.e.k !== 'assign') throw new Error('shape');
    // ((((1 + (2*3)) << 1) & 7) | (8 ^ 2))
    expect(s.e.r).toMatchObject({ k: 'binary', op: '|', l: { op: '&', l: { op: '<<', l: { op: '+' } } }, r: { op: '^' } });
  });

  it('points at the spot of a missing semicolon', () => {
    const e = errorOf('__global__ void k(float* a) {\n  int i = threadIdx.x\n  a[i] = 1.0f;\n}');
    expect(e).toMatchObject({ line: 2, col: 22 });
    expect(e.message).toBe(`expected ';' after the declaration, found 'a'`);
  });

  it('suggests a close name for an undeclared identifier', () => {
    const e = errorOf('__global__ void k(float* out) {\n  out[threadIdx.x] = valu;\n}');
    expect(e).toMatchObject({ line: 2, col: 22 });
    expect(e.message).toContain(`'valu' is not declared`);
    const f = errorOf('__global__ void k(float* out) { out[threadIdz.x] = 1.0f; }');
    expect(f.message).toContain(`did you mean 'threadIdx'?`);
  });

  it('explains unsupported features', () => {
    expect(errorOf('__global__ void k(double* a) {}').message).toBe('double is not supported; use float');
    expect(errorOf('__global__ void k(float* a) { float r[4]; }').message).toMatch(/local arrays/);
    expect(errorOf('__device__ float f() {}').message).toMatch(/__device__/);
    expect(errorOf('__global__ void k(const float* a) { a[0] = 1.0f; }').message).toMatch(/'a' is const/);
    expect(errorOf('__global__ void k(float* a) { int n = 3; __shared__ float s[n]; }').message).toMatch(/compile-time/);
    expect(errorOf('__global__ void k(float* a) { a[0] = 1.0f % 2.0f; }').message).toMatch(/'%' needs int/);
    expect(errorOf('__global__ void k(float* a) { break; }').message).toBe(`'break' outside a loop`);
    expect(errorOf('__global__ void k(float* a) { a[0] = __syncthreads(); }').message).toMatch(/does not return a value/);
    expect(errorOf('__global__ void k(float* a) { atomicAdd(a[0], 1.0f); }').message).toMatch(/address of an array element/);
    expect(errorOf('__global__ void k(float* a) { a[0] = sqrtff(a[1]); }').message).toContain(`did you mean 'sqrtf'?`);
  });

  it('checks indices against declared dimensions', () => {
    const e = errorOf('__global__ void k(float* a) { __shared__ float t[4][4]; t[1] = 0.0f; }');
    expect(e.message).toBe(`'t' has 2 dimensions, got 1 index`);
  });

  it('counts only lines with code', () => {
    const r = compile(`// header comment
__global__ void k(float* a)
{
  /* block
     comment */
  a[threadIdx.x] = 1.0f;   // trailing

}`);
    expect(r.ok && r.kernel.lines).toBe(2);
  });
});
