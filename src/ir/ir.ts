import type { Diagnostic } from '../lang/diagnostics';

export type ValType = 'int' | 'float' | 'bool';
export type ElemType = 'int' | 'float';
export type Space = 'global' | 'shared';

export type BinFn =
  | 'add' | 'sub' | 'mul' | 'div' | 'rem'
  | 'and' | 'or' | 'xor' | 'shl' | 'shr'
  | 'lt' | 'le' | 'gt' | 'ge' | 'eq' | 'ne'
  | 'min' | 'max';
export type UnFn = 'neg' | 'not' | 'bnot' | 'abs' | 'floor' | 'ceil';
/** `u2f` converts an unsigned int, which is how CUDA types threadIdx, blockIdx, blockDim and gridDim. */
export type CvtFn = 'i2f' | 'u2f' | 'f2i' | 'tobool';
/** Functions that run on the special function unit and cost extra cycles. */
export type MathFn = 'exp' | 'log' | 'sqrt' | 'rsqrt';
export type AtomFn = 'add' | 'sub' | 'max' | 'min' | 'exch';
export type ShflMode = 'down' | 'up' | 'xor' | 'idx';

export type Special =
  | 'threadIdx.x' | 'threadIdx.y' | 'threadIdx.z'
  | 'blockIdx.x' | 'blockIdx.y' | 'blockIdx.z'
  | 'blockDim.x' | 'blockDim.y' | 'blockDim.z'
  | 'gridDim.x' | 'gridDim.y' | 'gridDim.z';

interface At {
  line: number;
  col: number;
}

/**
 * One warp instruction. Operands are register numbers; constants live in
 * registers that are filled before the kernel starts, so they cost nothing.
 * Branch targets and `join` are instruction indices; `join` is the immediate
 * post-dominator of the branch (-1 when that is the kernel exit).
 */
export type Instr = At &
  (
    | { op: 'mov'; dst: number; a: number }
    /** `unsigned` makes div, rem, shr, min, max and comparisons treat both operands as unsigned ints. */
    | { op: 'bin'; fn: BinFn; float: boolean; unsigned?: boolean; dst: number; a: number; b: number }
    | { op: 'mad'; float: boolean; dst: number; a: number; b: number; c: number }
    | { op: 'un'; fn: UnFn; float: boolean; dst: number; a: number }
    | { op: 'cvt'; fn: CvtFn; dst: number; a: number }
    | { op: 'math'; fn: MathFn; dst: number; a: number }
    | { op: 'ld'; space: Space; buf: number; idx: number; dst: number }
    | { op: 'st'; space: Space; buf: number; idx: number; val: number }
    | { op: 'atom'; fn: AtomFn; space: Space; buf: number; idx: number; val: number; dst: number }
    | { op: 'shfl'; mode: ShflMode; dst: number; val: number; arg: number; mask: number; width: number }
    | { op: 'bar' }
    | { op: 'syncwarp' }
    | { op: 'br'; target: number }
    | { op: 'cbr'; cond: number; t: number; f: number; join: number }
    /** `early` for a return statement, false where the kernel body ends. */
    | { op: 'ret'; early: boolean }
  );

export type Op = Instr['op'];

export interface BufferParam {
  kind: 'buffer';
  name: string;
  type: ElemType;
  isConst: boolean;
  /** Index into the launch's buffer list. */
  buf: number;
}

export interface ScalarParam {
  kind: 'scalar';
  name: string;
  type: ValType;
  reg: number;
}

export type KernelParam = BufferParam | ScalarParam;

export interface SharedArray {
  name: string;
  type: ElemType;
  dims: number[];
  size: number;
  /** Offset in 4-byte words from the start of the block's shared memory. */
  offset: number;
  line: number;
}

export interface KernelIR {
  name: string;
  params: KernelParam[];
  shared: SharedArray[];
  sharedWords: number;
  numRegs: number;
  consts: { reg: number; value: number }[];
  specials: { reg: number; which: Special }[];
  /** Scalar parameters and locals in declaration order, for inspecting a thread. Folded constants carry a value instead of a register. */
  vars: { name: string; type: ValType; reg?: number; value?: number }[];
  code: Instr[];
  /** Lines of the source that contain code, for the "lines" score. */
  lines: number;
  warnings: Diagnostic[];
}
