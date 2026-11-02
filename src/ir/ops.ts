import type { BinFn, CvtFn, MathFn, UnFn } from './ir';

export class DivideByZero extends Error {}

const f = Math.fround;

/** fminf/fmaxf return the other operand when one is NaN. */
function fmin(a: number, b: number): number {
  if (Number.isNaN(a)) return b;
  if (Number.isNaN(b)) return a;
  return Math.min(a, b);
}

function fmax(a: number, b: number): number {
  if (Number.isNaN(a)) return b;
  if (Number.isNaN(b)) return a;
  return Math.max(a, b);
}

/** 32-bit int and float semantics, as nvcc would compile them. */
export function binOp(fn: BinFn, float: boolean, a: number, b: number): number {
  if (float) {
    switch (fn) {
      case 'add': return f(a + b);
      case 'sub': return f(a - b);
      case 'mul': return f(a * b);
      case 'div': return f(a / b);
      case 'min': return fmin(a, b);
      case 'max': return fmax(a, b);
      case 'lt': return a < b ? 1 : 0;
      case 'le': return a <= b ? 1 : 0;
      case 'gt': return a > b ? 1 : 0;
      case 'ge': return a >= b ? 1 : 0;
      case 'eq': return a === b ? 1 : 0;
      case 'ne': return a !== b ? 1 : 0;
      default: throw new Error(`float ${fn}`);
    }
  }
  switch (fn) {
    case 'add': return (a + b) | 0;
    case 'sub': return (a - b) | 0;
    case 'mul': return Math.imul(a, b);
    case 'div':
      if (b === 0) throw new DivideByZero();
      return (a / b) | 0;
    case 'rem':
      if (b === 0) throw new DivideByZero();
      return (a % b) | 0;
    case 'and': return a & b;
    case 'or': return a | b;
    case 'xor': return a ^ b;
    case 'shl': return a << (b & 31);
    case 'shr': return a >> (b & 31);
    case 'min': return Math.min(a, b);
    case 'max': return Math.max(a, b);
    case 'lt': return a < b ? 1 : 0;
    case 'le': return a <= b ? 1 : 0;
    case 'gt': return a > b ? 1 : 0;
    case 'ge': return a >= b ? 1 : 0;
    case 'eq': return a === b ? 1 : 0;
    case 'ne': return a !== b ? 1 : 0;
  }
}

/** a*b + c with a single rounding, like FFMA/IMAD. */
export function madOp(float: boolean, a: number, b: number, c: number): number {
  return float ? f(a * b + c) : (Math.imul(a, b) + c) | 0;
}

export function unOp(fn: UnFn, float: boolean, a: number): number {
  switch (fn) {
    case 'neg': return float ? -a : -a | 0;
    case 'not': return a === 0 ? 1 : 0;
    case 'bnot': return ~a;
    case 'abs': return float ? Math.abs(a) : Math.abs(a) | 0;
    case 'floor': return Math.floor(a);
    case 'ceil': return Math.ceil(a);
  }
}

/** Float to int conversion saturates and maps NaN to 0, like cvt.rzi.s32.f32. */
export function cvtOp(fn: CvtFn, a: number): number {
  switch (fn) {
    case 'i2f': return f(a);
    case 'tobool': return a !== 0 ? 1 : 0;
    case 'f2i':
      if (Number.isNaN(a)) return 0;
      if (a >= 2147483647) return 2147483647;
      if (a <= -2147483648) return -2147483648;
      return Math.trunc(a) | 0;
  }
}

export function mathOp(fn: MathFn, a: number): number {
  switch (fn) {
    case 'exp': return f(Math.exp(a));
    case 'log': return f(Math.log(a));
    case 'sqrt': return f(Math.sqrt(a));
    case 'rsqrt': return f(1 / Math.sqrt(a));
  }
}
