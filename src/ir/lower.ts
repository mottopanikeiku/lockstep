import type { BinaryOp, Expr, Program, Stmt } from '../lang/ast';
import { CompileError, suggest, type Diagnostic, type Loc } from '../lang/diagnostics';
import type {
  AtomFn,
  BinFn,
  ElemType,
  Instr,
  KernelIR,
  KernelParam,
  MathFn,
  SharedArray,
  ShflMode,
  Special,
  UnFn,
  ValType,
} from './ir';
import { binOp, cvtOp, DivideByZero, unOp } from './ops';
import { postDominators } from './postdom';

/** A value during lowering: either a register or a compile-time constant. */
type V = { t: ValType; reg: number; c?: undefined } | { t: ValType; c: number; reg?: undefined };

type Sym =
  | { k: 'var'; reg: number; t: ValType; isConst: boolean }
  /** `constexpr` is false for values we fold but C++ would not accept as an array size. */
  | { k: 'const'; value: number; t: ValType; constexpr: boolean }
  | { k: 'buffer'; buf: number; t: ElemType; isConst: boolean }
  | { k: 'shared'; arr: number; t: ElemType; dims: number[] };

type LValue =
  | { k: 'reg'; reg: number; t: ValType; name: string }
  | { k: 'mem'; space: 'global' | 'shared'; buf: number; idx: V; t: ElemType; name: string };

type Term = { line: number; col: number } & (
  | { k: 'br'; to: number }
  | { k: 'cbr'; cond: number; t: number; f: number }
  | { k: 'ret' }
);

interface Block {
  id: number;
  instrs: Instr[];
  term?: Term;
}

const BUILTIN_CONSTS: Record<string, { value: number; t: ValType; constexpr: boolean }> = {
  warpSize: { value: 32, t: 'int', constexpr: false },
  INFINITY: { value: Infinity, t: 'float', constexpr: true },
  HUGE_VALF: { value: Infinity, t: 'float', constexpr: true },
  FLT_MAX: { value: 3.4028234663852886e38, t: 'float', constexpr: true },
  INT_MAX: { value: 2147483647, t: 'int', constexpr: true },
  INT_MIN: { value: -2147483648, t: 'int', constexpr: true },
};

const SPECIAL_OBJECTS: Record<string, true> = { threadIdx: true, blockIdx: true, blockDim: true, gridDim: true };

type MathEntry =
  | { k: 'math'; fn: MathFn }
  | { k: 'un'; fn: UnFn; type: 'float' | 'any' }
  | { k: 'bin'; fn: 'min' | 'max'; type: 'float' | 'any' };

const MATH: Record<string, MathEntry> = {
  expf: { k: 'math', fn: 'exp' },
  __expf: { k: 'math', fn: 'exp' },
  exp: { k: 'math', fn: 'exp' },
  logf: { k: 'math', fn: 'log' },
  __logf: { k: 'math', fn: 'log' },
  log: { k: 'math', fn: 'log' },
  sqrtf: { k: 'math', fn: 'sqrt' },
  sqrt: { k: 'math', fn: 'sqrt' },
  rsqrtf: { k: 'math', fn: 'rsqrt' },
  fabsf: { k: 'un', fn: 'abs', type: 'float' },
  fabs: { k: 'un', fn: 'abs', type: 'float' },
  abs: { k: 'un', fn: 'abs', type: 'any' },
  floorf: { k: 'un', fn: 'floor', type: 'float' },
  floor: { k: 'un', fn: 'floor', type: 'float' },
  ceilf: { k: 'un', fn: 'ceil', type: 'float' },
  ceil: { k: 'un', fn: 'ceil', type: 'float' },
  fminf: { k: 'bin', fn: 'min', type: 'float' },
  fmaxf: { k: 'bin', fn: 'max', type: 'float' },
  min: { k: 'bin', fn: 'min', type: 'any' },
  max: { k: 'bin', fn: 'max', type: 'any' },
};

const ATOMICS: Record<string, AtomFn> = {
  atomicAdd: 'add',
  atomicSub: 'sub',
  atomicMax: 'max',
  atomicMin: 'min',
  atomicExch: 'exch',
};

const SHUFFLES: Record<string, ShflMode> = {
  __shfl_down_sync: 'down',
  __shfl_up_sync: 'up',
  __shfl_xor_sync: 'xor',
  __shfl_sync: 'idx',
};

const KNOWN_CALLS = [
  ...Object.keys(MATH),
  ...Object.keys(ATOMICS),
  ...Object.keys(SHUFFLES),
  '__syncthreads',
  '__syncwarp',
];

const ARITH: Partial<Record<BinaryOp, BinFn>> = {
  '+': 'add',
  '-': 'sub',
  '*': 'mul',
  '/': 'div',
  '%': 'rem',
  '&': 'and',
  '|': 'or',
  '^': 'xor',
  '<<': 'shl',
  '>>': 'shr',
  '<': 'lt',
  '<=': 'le',
  '>': 'gt',
  '>=': 'ge',
  '==': 'eq',
  '!=': 'ne',
};

const COMPARISON: Record<string, true> = { '<': true, '<=': true, '>': true, '>=': true, '==': true, '!=': true };
const BITWISE: Record<string, true> = { '&': true, '|': true, '^': true, '<<': true, '>>': true };

const promote = (a: ValType, b: ValType): 'int' | 'float' => (a === 'float' || b === 'float' ? 'float' : 'int');

function isMul(e: Expr): e is Extract<Expr, { k: 'binary' }> {
  return e.k === 'binary' && e.op === '*';
}

class Lowerer {
  private blocks: Block[] = [];
  private layout: number[] = [];
  private cur!: Block;
  private scopes: Map<string, Sym>[] = [];
  private globals = new Map<string, Sym>();
  private nregs = 0;
  private regNames: (string | null)[] = [];
  private constRegs = new Map<string, number>();
  private consts: { reg: number; value: number }[] = [];
  private specialRegs = new Map<Special, number>();
  private shared: SharedArray[] = [];
  private sharedWords = 0;
  private params: KernelParam[] = [];
  private loops: { brk: number; cont: number }[] = [];
  /** Names that some statement assigns after declaration. */
  private assigned = new Set<string>();
  /** Registers of variables that get reassigned, so other variables must copy them. */
  private mutableRegs = new Set<number>();

  constructor(private readonly warnings: Diagnostic[]) {}

  lower(prog: Program, lines: number): KernelIR {
    for (const g of prog.consts) {
      const value = this.tryConst(g.init);
      if (value === undefined) throw new CompileError(`'${g.name}' needs a compile-time value`, g.loc);
      if (this.globals.has(g.name)) throw new CompileError(`'${g.name}' is declared twice`, g.loc);
      this.globals.set(g.name, { k: 'const', value: this.constConvert(value, g.type), t: g.type, constexpr: true });
    }
    const k = prog.kernel;
    this.cur = this.newBlock();
    this.layout.push(this.cur.id);
    this.scopes.push(new Map());
    let nbufs = 0;
    for (const p of k.params) {
      if (this.scopes[0].has(p.name)) throw new CompileError(`parameter '${p.name}' is declared twice`, p.loc);
      if (p.pointer) {
        if (p.type === 'bool') throw new CompileError('bool arrays are not supported; use int', p.loc);
        const buf = nbufs++;
        this.params.push({ kind: 'buffer', name: p.name, type: p.type, isConst: p.isConst, buf });
        this.scopes[0].set(p.name, { k: 'buffer', buf, t: p.type, isConst: p.isConst });
      } else {
        const reg = this.newReg(p.name);
        this.params.push({ kind: 'scalar', name: p.name, type: p.type, reg });
        this.scopes[0].set(p.name, { k: 'var', reg, t: p.type, isConst: p.isConst });
      }
    }
    for (const s of k.body) collectAssigned(s, this.assigned);
    for (const p of this.params) if (p.kind === 'scalar' && this.assigned.has(p.name)) this.mutableRegs.add(p.reg);
    for (const s of k.body) this.stmt(s);
    this.terminate({ k: 'ret', ...k.end });
    this.scopes.pop();
    return {
      name: k.name,
      params: this.params,
      shared: this.shared,
      sharedWords: this.sharedWords,
      numRegs: this.nregs,
      consts: this.consts,
      specials: [...this.specialRegs].map(([which, reg]) => ({ reg, which })),
      regNames: this.regNames,
      code: this.flatten(),
      lines,
      warnings: this.warnings,
    };
  }

  // ---- blocks and registers ----

  private newBlock(): Block {
    const b: Block = { id: this.blocks.length, instrs: [] };
    this.blocks.push(b);
    return b;
  }

  /** Make `b` the current block, falling through from the previous one. */
  private place(b: Block, at: Loc): void {
    if (!this.cur.term) this.cur.term = { k: 'br', to: b.id, line: at.line, col: at.col };
    this.layout.push(b.id);
    this.cur = b;
  }

  private terminate(term: Term): void {
    if (!this.cur.term) this.cur.term = term;
  }

  /** Code after break/continue/return goes into a block nothing jumps to. */
  private startDeadBlock(): void {
    this.cur = this.newBlock();
    this.layout.push(this.cur.id);
  }

  private emit(ins: Instr): void {
    if (!this.cur.term) this.cur.instrs.push(ins);
  }

  private newReg(name: string | null = null): number {
    this.regNames.push(name);
    return this.nregs++;
  }

  private constReg(t: ValType, value: number): number {
    const key = `${t === 'float' ? 'f' : 'i'}${Object.is(value, -0) ? '-0' : value}`;
    let reg = this.constRegs.get(key);
    if (reg === undefined) {
      reg = this.newReg();
      this.constRegs.set(key, reg);
      this.consts.push({ reg, value });
    }
    return reg;
  }

  private reg(v: V): number {
    return v.reg !== undefined ? v.reg : this.constReg(v.t, v.c);
  }

  private into(v: V, dst: number, at: Loc): void {
    if (v.reg === dst) return;
    this.emit({ op: 'mov', dst, a: this.reg(v), ...at });
  }

  // ---- symbols ----

  private lookup(name: string): Sym | undefined {
    for (let i = this.scopes.length - 1; i >= 0; i--) {
      const s = this.scopes[i].get(name);
      if (s) return s;
    }
    const g = this.globals.get(name);
    if (g) return g;
    const b = BUILTIN_CONSTS[name];
    if (b && Object.hasOwn(BUILTIN_CONSTS, name)) return { k: 'const', ...b };
    return undefined;
  }

  private undeclared(name: string, loc: Loc): CompileError {
    if (Object.hasOwn(SPECIAL_OBJECTS, name)) return new CompileError(`${name} has fields .x, .y and .z`, loc);
    const names = new Set<string>([...this.globals.keys(), ...Object.keys(BUILTIN_CONSTS), ...Object.keys(SPECIAL_OBJECTS)]);
    for (const s of this.scopes) for (const n of s.keys()) names.add(n);
    const hint = suggest(name, names);
    return new CompileError(`'${name}' is not declared${hint ? `; did you mean '${hint}'?` : ''}`, loc);
  }

  private declare(name: string, sym: Sym, loc: Loc): void {
    const scope = this.scopes[this.scopes.length - 1];
    if (scope.has(name)) throw new CompileError(`'${name}' is already declared in this scope`, loc);
    if (Object.hasOwn(SPECIAL_OBJECTS, name) || Object.hasOwn(BUILTIN_CONSTS, name)) {
      throw new CompileError(`'${name}' is a built-in name`, loc);
    }
    scope.set(name, sym);
  }

  private special(obj: string, prop: string, loc: Loc): V {
    if (!Object.hasOwn(SPECIAL_OBJECTS, obj)) {
      if (this.lookup(obj)) throw new CompileError(`'${obj}' has no fields; only threadIdx, blockIdx, blockDim and gridDim do`, loc);
      throw this.undeclared(obj, loc);
    }
    if (prop !== 'x' && prop !== 'y' && prop !== 'z') throw new CompileError(`${obj} has fields .x, .y and .z, not .${prop}`, loc);
    const which = `${obj}.${prop}` as Special;
    let reg = this.specialRegs.get(which);
    if (reg === undefined) {
      reg = this.newReg(which);
      this.specialRegs.set(which, reg);
    }
    return { t: 'int', reg };
  }

  // ---- constants ----

  private constConvert(value: number, to: ValType): number {
    if (to === 'float') return Math.fround(value);
    if (to === 'bool') return value !== 0 ? 1 : 0;
    return Number.isInteger(value) ? value | 0 : cvtOp('f2i', value);
  }

  /** Inside tryConst: only accept what C++ calls a constant expression. */
  private strict = false;

  /** Evaluate a C++ constant expression, or undefined if it is not one. */
  private tryConst(e: Expr): number | undefined {
    this.strict = true;
    try {
      return this.constValue(e)?.c;
    } finally {
      this.strict = false;
    }
  }

  private constValue(e: Expr): { t: ValType; c: number } | undefined {
    switch (e.k) {
      case 'num':
        return { t: e.float ? 'float' : 'int', c: e.value };
      case 'bool':
        return { t: 'bool', c: e.value ? 1 : 0 };
      case 'name': {
        const s = this.lookup(e.name);
        return s?.k === 'const' && (s.constexpr || !this.strict) ? { t: s.t, c: s.value } : undefined;
      }
      case 'unary': {
        const v = this.constValue(e.e);
        if (!v) return undefined;
        if (e.op === '-') return { t: v.t === 'bool' ? 'int' : v.t, c: unOp('neg', v.t === 'float', v.c) };
        if (e.op === '+') return { t: v.t === 'bool' ? 'int' : v.t, c: v.c };
        if (e.op === '!') return { t: 'bool', c: v.c === 0 ? 1 : 0 };
        if (e.op === '~' && v.t !== 'float') return { t: 'int', c: ~v.c };
        return undefined;
      }
      case 'binary': {
        const l = this.constValue(e.l);
        const r = this.constValue(e.r);
        if (!l || !r) return undefined;
        if (e.op === '&&') return { t: 'bool', c: l.c !== 0 && r.c !== 0 ? 1 : 0 };
        if (e.op === '||') return { t: 'bool', c: l.c !== 0 || r.c !== 0 ? 1 : 0 };
        const t = promote(l.t, r.t);
        if (t === 'float' && (BITWISE[e.op] || e.op === '%')) return undefined;
        try {
          const c = binOp(ARITH[e.op]!, t === 'float', this.constConvert(l.c, t), this.constConvert(r.c, t));
          return { t: COMPARISON[e.op] ? 'bool' : t, c };
        } catch (err) {
          if (err instanceof DivideByZero) throw new CompileError('division by zero', e.loc);
          throw err;
        }
      }
      case 'cast': {
        const v = this.constValue(e.e);
        return v ? { t: e.to, c: this.constConvert(v.c, e.to) } : undefined;
      }
      case 'cond': {
        const c = this.constValue(e.c);
        if (!c) return undefined;
        return this.constValue(c.c !== 0 ? e.t : e.f);
      }
      default:
        return undefined;
    }
  }

  // ---- conversions ----

  private convert(v: V, to: ValType, at: Loc, dst?: number): V {
    if (v.t === to) return v;
    if (v.c !== undefined) return { t: to, c: this.constConvert(v.c, to) };
    if (to === 'int' && v.t === 'bool') return { t: 'int', reg: v.reg };
    const fn = to === 'float' ? 'i2f' : to === 'int' ? 'f2i' : 'tobool';
    const out = dst ?? this.newReg();
    this.emit({ op: 'cvt', fn, dst: out, a: v.reg, ...at });
    return { t: to, reg: out };
  }

  private intIndex(e: Expr): V {
    const v = this.value(e);
    if (v.t === 'float') throw new CompileError('array index must be an int, not a float', e.loc);
    return v.t === 'bool' ? { ...v, t: 'int' } : v;
  }

  // ---- statements ----

  private stmt(s: Stmt): void {
    switch (s.k) {
      case 'block':
        this.scopes.push(new Map());
        for (const x of s.body) this.stmt(x);
        this.scopes.pop();
        return;
      case 'empty':
        return;
      case 'decl':
        return this.decl(s);
      case 'expr':
        return this.stmtExpr(s.e);
      case 'if': {
        const thenB = this.newBlock();
        const elseB = s.else ? this.newBlock() : undefined;
        const merge = this.newBlock();
        this.cond(s.cond, thenB.id, (elseB ?? merge).id);
        this.place(thenB, s.loc);
        this.scoped(s.then);
        if (elseB) {
          this.terminate({ k: 'br', to: merge.id, ...s.loc });
          this.place(elseB, s.loc);
          this.scoped(s.else!);
        }
        this.place(merge, s.loc);
        return;
      }
      case 'while': {
        const body = this.newBlock();
        const test = this.newBlock();
        const exit = this.newBlock();
        this.cond(s.cond, body.id, exit.id);
        this.place(body, s.loc);
        this.loops.push({ brk: exit.id, cont: test.id });
        this.scoped(s.body);
        this.loops.pop();
        this.place(test, s.cond.loc);
        this.cond(s.cond, body.id, exit.id);
        this.place(exit, s.loc);
        return;
      }
      case 'do': {
        const body = this.newBlock();
        const test = this.newBlock();
        const exit = this.newBlock();
        this.place(body, s.loc);
        this.loops.push({ brk: exit.id, cont: test.id });
        this.scoped(s.body);
        this.loops.pop();
        this.place(test, s.cond.loc);
        this.cond(s.cond, body.id, exit.id);
        this.place(exit, s.loc);
        return;
      }
      case 'for': {
        this.scopes.push(new Map());
        if (s.init) this.stmt(s.init);
        const body = this.newBlock();
        const step = this.newBlock();
        const exit = this.newBlock();
        const entry = this.entryCondition(s);
        if (entry === undefined && s.cond) this.cond(s.cond, body.id, exit.id);
        else if (entry === false) this.terminate({ k: 'br', to: exit.id, ...s.loc });
        this.place(body, s.loc);
        this.loops.push({ brk: exit.id, cont: step.id });
        this.scoped(s.body);
        this.loops.pop();
        this.place(step, s.step?.loc ?? s.loc);
        if (s.step) this.stmtExpr(s.step);
        if (s.cond) this.cond(s.cond, body.id, exit.id);
        else this.terminate({ k: 'br', to: body.id, ...s.loc });
        this.place(exit, s.loc);
        this.scopes.pop();
        return;
      }
      case 'break':
      case 'continue': {
        const loop = this.loops[this.loops.length - 1];
        if (!loop) throw new CompileError(`'${s.k}' outside a loop`, s.loc);
        this.terminate({ k: 'br', to: s.k === 'break' ? loop.brk : loop.cont, ...s.loc });
        this.startDeadBlock();
        return;
      }
      case 'return':
        this.terminate({ k: 'ret', ...s.loc });
        this.startDeadBlock();
        return;
    }
  }

  /** A statement that forms its own scope, like the body of an if. */
  private scoped(s: Stmt): void {
    this.scopes.push(new Map());
    this.stmt(s);
    this.scopes.pop();
  }

  private decl(s: Extract<Stmt, { k: 'decl' }>): void {
    for (const d of s.decls) {
      if (s.shared) {
        if (s.type === 'bool') throw new CompileError('__shared__ bool is not supported; use int', d.loc);
        if (d.init) throw new CompileError('a __shared__ variable cannot have an initializer', d.loc);
        const dims = d.dims.map((e) => {
          const n = this.tryConst(e);
          if (n === undefined) throw new CompileError('__shared__ array sizes must be compile-time constants', e.loc);
          if (!Number.isInteger(n) || n <= 0) throw new CompileError(`array size must be a positive integer, got ${n}`, e.loc);
          return n;
        });
        const size = dims.reduce((a, b) => a * b, 1);
        const arr = this.shared.length;
        this.shared.push({ name: d.name, type: s.type, dims, size, offset: this.sharedWords, line: d.loc.line });
        this.sharedWords += size;
        if (this.sharedWords > 12288) throw new CompileError('more than 48 KB of shared memory per block', d.loc);
        this.declare(d.name, { k: 'shared', arr, t: s.type, dims }, d.loc);
        continue;
      }
      if (d.dims.length) {
        throw new CompileError('local arrays are not supported; use __shared__ memory or separate variables', d.loc);
      }
      if (s.isConst) {
        if (!d.init) throw new CompileError(`const '${d.name}' needs a value`, d.loc);
        const c = this.tryConst(d.init);
        if (c !== undefined) {
          this.declare(d.name, { k: 'const', value: this.constConvert(c, s.type), t: s.type, constexpr: true }, d.loc);
          continue;
        }
      }
      if (d.init && !this.assigned.has(d.name)) {
        // Never reassigned, so the name can refer to the initializer's value without a copy.
        const v = this.convert(this.value(d.init), s.type, d.loc);
        if (v.c !== undefined) {
          this.declare(d.name, { k: 'const', value: v.c, t: s.type, constexpr: false }, d.loc);
          continue;
        }
        if (!this.mutableRegs.has(v.reg)) {
          this.regNames[v.reg] ??= d.name;
          this.declare(d.name, { k: 'var', reg: v.reg, t: s.type, isConst: s.isConst }, d.loc);
          continue;
        }
        const reg = this.newReg(d.name);
        this.into(v, reg, d.loc);
        this.declare(d.name, { k: 'var', reg, t: s.type, isConst: s.isConst }, d.loc);
        continue;
      }
      const reg = this.newReg(d.name);
      if (this.assigned.has(d.name)) this.mutableRegs.add(reg);
      if (d.init) {
        const v = this.convert(this.value(d.init, reg), s.type, d.loc, reg);
        this.into(v, reg, d.loc);
      }
      this.declare(d.name, { k: 'var', reg, t: s.type, isConst: s.isConst }, d.loc);
    }
  }

  /**
   * The loop condition's value on entry when it is known at compile time,
   * as in `for (int i = 0; i < 16; i++)`, so the entry check can be skipped.
   */
  private entryCondition(s: Extract<Stmt, { k: 'for' }>): boolean | undefined {
    if (!s.cond) return true;
    const init = s.init;
    if (init?.k !== 'decl' || init.shared || init.decls.length !== 1) return undefined;
    const d = init.decls[0];
    const c = d.init ? this.constValue(d.init)?.c : undefined;
    if (c === undefined) return undefined;
    const value = this.constConvert(c, init.type);
    const probe = new Map<string, Sym>([[d.name, { k: 'const', value, t: init.type, constexpr: false }]]);
    this.scopes.push(probe);
    const v = this.constValue(s.cond);
    this.scopes.pop();
    return v ? v.c !== 0 : undefined;
  }

  /** An expression whose value is discarded. */
  private stmtExpr(e: Expr): void {
    if (e.k === 'incdec') {
      this.incdec({ ...e, prefix: true });
      return;
    }
    if (e.k === 'call') {
      this.call(e);
      return;
    }
    this.value(e);
  }

  /** Branch to `t` if `e` is true, else to `f`. */
  private cond(e: Expr, t: number, f: number): void {
    if (e.k === 'binary' && (e.op === '&&' || e.op === '||') && !this.isSimple(e.r)) {
      const mid = this.newBlock();
      if (e.op === '&&') this.cond(e.l, mid.id, f);
      else this.cond(e.l, t, mid.id);
      this.place(mid, e.r.loc);
      this.cond(e.r, t, f);
      return;
    }
    if (e.k === 'unary' && e.op === '!') {
      this.cond(e.e, f, t);
      return;
    }
    const v = this.value(e);
    if (v.c !== undefined) {
      this.terminate({ k: 'br', to: v.c !== 0 ? t : f, ...e.loc });
      return;
    }
    this.terminate({ k: 'cbr', cond: v.reg, t, f, ...e.loc });
  }

  /** Cheap and side-effect free, so `a && b` can evaluate both sides without a branch. */
  private isSimple(e: Expr): boolean {
    switch (e.k) {
      case 'num':
      case 'bool':
      case 'member':
        return true;
      case 'name': {
        const s = this.lookup(e.name);
        return s?.k === 'var' || s?.k === 'const';
      }
      case 'unary':
        return e.op !== '&' && this.isSimple(e.e);
      case 'cast':
        return this.isSimple(e.e);
      case 'binary':
        return e.op !== '/' && e.op !== '%' && this.isSimple(e.l) && this.isSimple(e.r);
      default:
        return false;
    }
  }

  // ---- expressions ----

  /** Lower an expression that must produce a value. `dst` is where to put a freshly computed result. */
  private value(e: Expr, dst?: number): V {
    const v = this.expr(e, dst);
    if (!v) throw new CompileError(`${e.k === 'call' ? e.callee + '()' : 'this expression'} does not return a value`, e.loc);
    return v;
  }

  private expr(e: Expr, dst?: number): V | null {
    switch (e.k) {
      case 'num':
        return { t: e.float ? 'float' : 'int', c: e.value };
      case 'bool':
        return { t: 'bool', c: e.value ? 1 : 0 };
      case 'name': {
        const s = this.lookup(e.name);
        if (!s) throw this.undeclared(e.name, e.loc);
        switch (s.k) {
          case 'var':
            return { t: s.t, reg: s.reg };
          case 'const':
            return { t: s.t, c: s.value };
          case 'buffer':
            throw new CompileError(`'${e.name}' is a pointer; read an element with ${e.name}[i]`, e.loc);
          case 'shared':
            if (s.dims.length) throw new CompileError(`'${e.name}' is an array; index it like ${e.name}[i]`, e.loc);
            return this.load(this.lvalue(e), e.loc, dst);
        }
        break;
      }
      case 'member':
        return this.special(e.obj, e.prop, e.loc);
      case 'index':
        return this.load(this.lvalue(e), e.loc, dst);
      case 'call':
        return this.call(e, dst);
      case 'unary':
        return this.unary(e, dst);
      case 'incdec':
        return this.incdec(e);
      case 'binary':
        return this.binary(e, dst);
      case 'assign':
        return this.assign(e);
      case 'cond':
        return this.ternary(e, dst);
      case 'cast':
        return this.convert(this.value(e.e), e.to, e.loc, dst);
    }
    return null;
  }

  private unary(e: Extract<Expr, { k: 'unary' }>, dst?: number): V {
    if (e.op === '&') {
      throw new CompileError(`'&' is only allowed for atomic targets, like atomicAdd(&out[i], v)`, e.loc);
    }
    const v = this.value(e.e);
    const t: ValType = v.t === 'bool' ? 'int' : v.t;
    let fn: UnFn;
    let rt: ValType = t;
    switch (e.op) {
      case '+':
        return { ...v, t } as V;
      case '-':
        fn = 'neg';
        break;
      case '!':
        fn = 'not';
        rt = 'bool';
        break;
      case '~':
        if (t === 'float') throw new CompileError(`'~' needs an int operand`, e.loc);
        fn = 'bnot';
        break;
    }
    if (v.c !== undefined) return { t: rt, c: unOp(fn, t === 'float', v.c) };
    const out = dst ?? this.newReg();
    this.emit({ op: 'un', fn, float: t === 'float', dst: out, a: v.reg, ...e.loc });
    return { t: rt, reg: out };
  }

  private binary(e: Extract<Expr, { k: 'binary' }>, dst?: number): V {
    if (e.op === '&&' || e.op === '||') return this.logical(e);
    if (e.op === '+' && (isMul(e.l) || isMul(e.r))) {
      const prod = isMul(e.l) ? e.l : (e.r as Extract<Expr, { k: 'binary' }>);
      const other = prod === e.l ? e.r : e.l;
      return this.mulAdd(prod, () => this.value(other), e.loc, dst);
    }
    const l = this.value(e.l);
    const r = this.value(e.r);
    return this.arith(e.op, l, r, e.loc, dst);
  }

  /** `a*b + c`, fused into one mad instruction when the types allow it. */
  private mulAdd(prod: Extract<Expr, { k: 'binary' }>, addend: () => V, at: Loc, dst?: number): V {
    const a = this.value(prod.l);
    const b = this.value(prod.r);
    const c = addend();
    const pt = promote(a.t, b.t);
    const t = promote(pt, c.t);
    if (pt !== t || (a.c !== undefined && b.c !== undefined)) {
      return this.arith('+', this.arith('*', a, b, prod.loc), c, at, dst);
    }
    const ac = this.convert(a, t, at);
    const bc = this.convert(b, t, at);
    const cc = this.convert(c, t, at);
    if (ac.c === 0 || bc.c === 0) return cc;
    if (cc.c === 0 && t === 'int') return this.arith('*', ac, bc, at, dst);
    const out = dst ?? this.newReg();
    this.emit({ op: 'mad', float: t === 'float', dst: out, a: this.reg(ac), b: this.reg(bc), c: this.reg(cc), ...at });
    return { t, reg: out };
  }

  private arith(op: BinaryOp, l: V, r: V, at: Loc, dst?: number): V {
    const fn = ARITH[op]!;
    const t = promote(l.t, r.t);
    if (t === 'float' && BITWISE[op]) throw new CompileError(`'${op}' needs int operands, not float`, at);
    if (t === 'float' && op === '%') throw new CompileError(`'%' needs int operands; fmodf is not supported`, at);
    const a = this.convert(l, t, at);
    const b = this.convert(r, t, at);
    const rt: ValType = COMPARISON[op] ? 'bool' : t;
    if (a.c !== undefined && b.c !== undefined) {
      try {
        return { t: rt, c: binOp(fn, t === 'float', a.c, b.c) };
      } catch (err) {
        if (err instanceof DivideByZero) throw new CompileError('division by zero', at);
        throw err;
      }
    }
    if (t === 'int' && (fn === 'div' || fn === 'rem') && b.c === 0) throw new CompileError('division by zero', at);
    if (b.c === 0 && (fn === 'add' || fn === 'sub' || fn === 'or' || fn === 'xor' || fn === 'shl' || fn === 'shr')) return a;
    if (a.c === 0 && (fn === 'add' || fn === 'or' || fn === 'xor')) return b;
    if (b.c === 1 && (fn === 'mul' || fn === 'div')) return a;
    if (a.c === 1 && fn === 'mul') return b;
    const out = dst ?? this.newReg();
    this.emit({ op: 'bin', fn, float: t === 'float', dst: out, a: this.reg(a), b: this.reg(b), ...at });
    return { t: rt, reg: out };
  }

  private toBool(v: V, at: Loc, dst?: number): V {
    return this.convert(v, 'bool', at, dst);
  }

  private logical(e: Extract<Expr, { k: 'binary' }>): V {
    if (this.isSimple(e.r)) {
      const l = this.toBool(this.value(e.l), e.loc);
      const r = this.toBool(this.value(e.r), e.loc);
      if (l.c !== undefined && r.c !== undefined) {
        return { t: 'bool', c: e.op === '&&' ? l.c & r.c : l.c | r.c };
      }
      if (l.c !== undefined) {
        if (e.op === '&&') return l.c ? r : l;
        return l.c ? l : r;
      }
      if (r.c !== undefined) {
        if (e.op === '&&') return r.c ? l : r;
        return r.c ? r : l;
      }
      const out = this.newReg();
      this.emit({ op: 'bin', fn: e.op === '&&' ? 'and' : 'or', float: false, dst: out, a: l.reg, b: r.reg, ...e.loc });
      return { t: 'bool', reg: out };
    }
    const out = this.newReg();
    const rhs = this.newBlock();
    const end = this.newBlock();
    this.into(this.toBool(this.value(e.l), e.loc, out), out, e.loc);
    if (e.op === '&&') this.terminate({ k: 'cbr', cond: out, t: rhs.id, f: end.id, ...e.loc });
    else this.terminate({ k: 'cbr', cond: out, t: end.id, f: rhs.id, ...e.loc });
    this.place(rhs, e.r.loc);
    this.into(this.toBool(this.value(e.r), e.loc, out), out, e.loc);
    this.place(end, e.loc);
    return { t: 'bool', reg: out };
  }

  private ternary(e: Extract<Expr, { k: 'cond' }>, dst?: number): V {
    const c = this.constValue(e.c);
    if (c) return this.value(c.c !== 0 ? e.t : e.f, dst);
    const t = this.typeOf(e);
    const out = dst ?? this.newReg();
    const tb = this.newBlock();
    const fb = this.newBlock();
    const end = this.newBlock();
    this.cond(e.c, tb.id, fb.id);
    this.place(tb, e.t.loc);
    this.into(this.convert(this.value(e.t, out), t, e.t.loc, out), out, e.t.loc);
    this.terminate({ k: 'br', to: end.id, ...e.loc });
    this.place(fb, e.f.loc);
    this.into(this.convert(this.value(e.f, out), t, e.f.loc, out), out, e.f.loc);
    this.place(end, e.loc);
    return { t, reg: out };
  }

  /** Static type of an expression, without emitting code. */
  private typeOf(e: Expr): ValType {
    switch (e.k) {
      case 'num':
        return e.float ? 'float' : 'int';
      case 'bool':
        return 'bool';
      case 'name': {
        const s = this.lookup(e.name);
        if (!s) throw this.undeclared(e.name, e.loc);
        return s.t;
      }
      case 'member':
        return 'int';
      case 'index': {
        let b: Expr = e;
        while (b.k === 'index') b = b.base;
        if (b.k !== 'name') throw new CompileError('only arrays can be indexed', e.loc);
        const s = this.lookup(b.name);
        if (!s) throw this.undeclared(b.name, b.loc);
        return s.t;
      }
      case 'call': {
        const m = MATH[e.callee];
        if (m && Object.hasOwn(MATH, e.callee)) {
          if (m.k === 'math' || m.type === 'float') return 'float';
          return e.args.reduce<ValType>((t, a) => promote(t, this.typeOf(a)), 'int');
        }
        if (Object.hasOwn(SHUFFLES, e.callee) && e.args[1]) return promote('int', this.typeOf(e.args[1]));
        if (Object.hasOwn(ATOMICS, e.callee) && e.args[0]) {
          const a = e.args[0];
          return this.typeOf(a.k === 'unary' && a.op === '&' ? a.e : a);
        }
        return 'int';
      }
      case 'unary':
        if (e.op === '!') return 'bool';
        return promote('int', this.typeOf(e.e));
      case 'incdec':
        return this.typeOf(e.e);
      case 'binary':
        if (e.op === '&&' || e.op === '||' || COMPARISON[e.op]) return 'bool';
        return promote(this.typeOf(e.l), this.typeOf(e.r));
      case 'assign':
        return this.typeOf(e.l);
      case 'cond': {
        const a = this.typeOf(e.t);
        const b = this.typeOf(e.f);
        return a === 'bool' && b === 'bool' ? 'bool' : promote(a, b);
      }
      case 'cast':
        return e.to;
    }
  }

  private lvalue(e: Expr): LValue {
    if (e.k === 'name') {
      const s = this.lookup(e.name);
      if (!s) throw this.undeclared(e.name, e.loc);
      if (s.k === 'var') {
        if (s.isConst) throw new CompileError(`'${e.name}' is const`, e.loc);
        return { k: 'reg', reg: s.reg, t: s.t, name: e.name };
      }
      if (s.k === 'shared' && s.dims.length === 0) {
        return { k: 'mem', space: 'shared', buf: s.arr, idx: { t: 'int', c: 0 }, t: s.t, name: e.name };
      }
      if (s.k === 'const') throw new CompileError(`'${e.name}' is a constant`, e.loc);
      if (s.k === 'buffer') throw new CompileError(`cannot assign to the pointer '${e.name}'; write ${e.name}[i] = ...`, e.loc);
      throw new CompileError(`'${e.name}' is an array; index it like ${e.name}[i]`, e.loc);
    }
    if (e.k === 'index') {
      const indices: Expr[] = [];
      let b: Expr = e;
      while (b.k === 'index') {
        indices.unshift(b.index);
        b = b.base;
      }
      if (b.k !== 'name') throw new CompileError('only arrays can be indexed', e.loc);
      const s = this.lookup(b.name);
      if (!s) throw this.undeclared(b.name, b.loc);
      if (s.k === 'buffer') {
        if (indices.length !== 1) throw new CompileError(`'${b.name}' is a flat array; use one index like ${b.name}[row * width + col]`, e.loc);
        return { k: 'mem', space: 'global', buf: s.buf, idx: this.intIndex(indices[0]), t: s.t, name: b.name };
      }
      if (s.k === 'shared') {
        if (indices.length !== s.dims.length) {
          throw new CompileError(`'${b.name}' has ${s.dims.length} dimension${s.dims.length === 1 ? '' : 's'}, got ${indices.length} index${indices.length === 1 ? '' : 'es'}`, e.loc);
        }
        let flat = this.intIndex(indices[0]);
        for (let j = 1; j < indices.length; j++) {
          const next = this.intIndex(indices[j]);
          flat = this.imad(flat, s.dims[j], next, indices[j].loc);
        }
        return { k: 'mem', space: 'shared', buf: s.arr, idx: flat, t: s.t, name: b.name };
      }
      throw new CompileError(`'${b.name}' is not an array`, b.loc);
    }
    if (e.k === 'member') throw new CompileError(`${e.obj}.${e.prop} is read-only`, e.loc);
    throw new CompileError('this expression cannot be assigned to', e.loc);
  }

  /** a * k + b for a row-major index, folding constants. */
  private imad(a: V, k: number, b: V, at: Loc): V {
    if (a.c !== undefined) return this.arith('+', { t: 'int', c: Math.imul(a.c, k) }, b, at);
    if (b.c === 0) return this.arith('*', a, { t: 'int', c: k }, at);
    const out = this.newReg();
    this.emit({ op: 'mad', float: false, dst: out, a: a.reg, b: this.constReg('int', k), c: this.reg(b), ...at });
    return { t: 'int', reg: out };
  }

  private load(lv: LValue, at: Loc, dst?: number): V {
    if (lv.k === 'reg') return { t: lv.t, reg: lv.reg };
    const out = dst ?? this.newReg();
    this.emit({ op: 'ld', space: lv.space, buf: lv.buf, idx: this.reg(lv.idx), dst: out, ...at });
    return { t: lv.t, reg: out };
  }

  private store(lv: LValue, v: V, at: Loc): V {
    const cv = this.convert(v, lv.t, at, lv.k === 'reg' ? lv.reg : undefined);
    if (lv.k === 'reg') {
      this.into(cv, lv.reg, at);
      return { t: lv.t, reg: lv.reg };
    }
    this.checkWritable(lv, at);
    this.emit({ op: 'st', space: lv.space, buf: lv.buf, idx: this.reg(lv.idx), val: this.reg(cv), ...at });
    return cv;
  }

  private checkWritable(lv: LValue, at: Loc): void {
    if (lv.k === 'mem' && lv.space === 'global') {
      const p = this.params.find((x) => x.kind === 'buffer' && x.buf === lv.buf);
      if (p && p.kind === 'buffer' && p.isConst) throw new CompileError(`'${lv.name}' is const and cannot be written`, at);
    }
  }

  private assign(e: Extract<Expr, { k: 'assign' }>): V {
    const lv = this.lvalue(e.l);
    if (e.op === '=') {
      if (lv.k === 'reg') {
        const v = this.value(e.r, lv.t === this.typeOf(e.r) ? lv.reg : undefined);
        return this.store(lv, v, e.loc);
      }
      return this.store(lv, this.value(e.r), e.loc);
    }
    this.checkWritable(lv, e.loc);
    const op = e.op.slice(0, -1) as BinaryOp;
    const old = this.load(lv, e.loc);
    const resultType = promote(old.t, this.typeOf(e.r));
    const dst = lv.k === 'reg' && resultType === lv.t ? lv.reg : undefined;
    const res =
      op === '+' && isMul(e.r)
        ? this.mulAdd(e.r, () => old, e.loc, dst)
        : this.arith(op, old, this.value(e.r), e.loc, dst);
    return this.store(lv, res, e.loc);
  }

  private incdec(e: Extract<Expr, { k: 'incdec' }>): V {
    const lv = this.lvalue(e.e);
    if (lv.t === 'bool') throw new CompileError(`'${e.op}' does not work on bool`, e.loc);
    this.checkWritable(lv, e.loc);
    const one: V = { t: lv.t, c: 1 };
    const fn = e.op === '++' ? '+' : '-';
    const old = this.load(lv, e.loc);
    let saved = old;
    if (!e.prefix && lv.k === 'reg') {
      saved = { t: lv.t, reg: this.newReg() };
      this.into(old, saved.reg!, e.loc);
    }
    const res = this.arith(fn, old, one, e.loc, lv.k === 'reg' ? lv.reg : undefined);
    const stored = this.store(lv, res, e.loc);
    return e.prefix ? stored : saved;
  }

  private call(e: Extract<Expr, { k: 'call' }>, dst?: number): V | null {
    const name = e.callee;
    const args = e.args;
    const arity = (n: number, m = n) => {
      if (args.length < n || args.length > m) {
        const want = n === m ? `${n}` : `${n} or ${m}`;
        throw new CompileError(`${name}() takes ${want} argument${want === '1' ? '' : 's'}, got ${args.length}`, e.loc);
      }
    };
    if (name === '__syncthreads') {
      arity(0);
      this.emit({ op: 'bar', ...e.loc });
      return null;
    }
    if (name === '__syncwarp') {
      arity(0, 1);
      if (args[0]) this.value(args[0]);
      this.emit({ op: 'syncwarp', ...e.loc });
      return null;
    }
    if (Object.hasOwn(ATOMICS, name)) {
      arity(2);
      const fn = ATOMICS[name];
      const target = this.atomicTarget(args[0], name);
      if (target.t === 'float' && (fn === 'max' || fn === 'min')) {
        throw new CompileError(`${name} works on int here, not float`, e.loc);
      }
      this.checkWritable(target, e.loc);
      const val = this.convert(this.value(args[1]), target.t, e.loc);
      const out = dst ?? this.newReg();
      this.emit({ op: 'atom', fn, space: target.space, buf: target.buf, idx: this.reg(target.idx), val: this.reg(val), dst: out, ...e.loc });
      return { t: target.t, reg: out };
    }
    if (Object.hasOwn(SHUFFLES, name)) {
      arity(3, 4);
      const mask = this.convert(this.value(args[0]), 'int', e.loc);
      let val = this.value(args[1]);
      if (val.t === 'bool') val = { ...val, t: 'int' } as V;
      const arg = this.convert(this.value(args[2]), 'int', e.loc);
      let width = 32;
      if (args[3]) {
        const w = this.tryConst(args[3]);
        if (w === undefined || ![1, 2, 4, 8, 16, 32].includes(w)) {
          throw new CompileError('shuffle width must be a constant power of two up to 32', args[3].loc);
        }
        width = w;
      }
      const out = dst ?? this.newReg();
      this.emit({ op: 'shfl', mode: SHUFFLES[name], dst: out, val: this.reg(val), arg: this.reg(arg), mask: this.reg(mask), width, ...e.loc });
      return { t: val.t, reg: out };
    }
    if (Object.hasOwn(MATH, name)) {
      const m = MATH[name];
      if (m.k === 'bin') {
        arity(2);
        let a = this.value(args[0]);
        let b = this.value(args[1]);
        const t = m.type === 'float' ? 'float' : promote(a.t, b.t);
        a = this.convert(a, t, e.loc);
        b = this.convert(b, t, e.loc);
        if (a.c !== undefined && b.c !== undefined) return { t, c: binOp(m.fn, t === 'float', a.c, b.c) };
        const out = dst ?? this.newReg();
        this.emit({ op: 'bin', fn: m.fn, float: t === 'float', dst: out, a: this.reg(a), b: this.reg(b), ...e.loc });
        return { t, reg: out };
      }
      arity(1);
      let a = this.value(args[0]);
      const t: 'int' | 'float' = m.k === 'math' || m.type === 'float' ? 'float' : promote('int', a.t);
      a = this.convert(a, t, e.loc);
      const out = dst ?? this.newReg();
      if (m.k === 'math') this.emit({ op: 'math', fn: m.fn, dst: out, a: this.reg(a), ...e.loc });
      else this.emit({ op: 'un', fn: m.fn, float: t === 'float', dst: out, a: this.reg(a), ...e.loc });
      return { t, reg: out };
    }
    if (this.lookup(name)) throw new CompileError(`'${name}' is not a function`, e.loc);
    const hint = suggest(name, KNOWN_CALLS);
    throw new CompileError(`unknown function '${name}'${hint ? `; did you mean '${hint}'?` : ''}`, e.loc);
  }

  /** The address argument of an atomic: &a[i], &s, a (element 0) or a + i. */
  private atomicTarget(arg: Expr, name: string): Extract<LValue, { k: 'mem' }> {
    let target: LValue | undefined;
    if (arg.k === 'unary' && arg.op === '&') target = this.lvalue(arg.e);
    else if (arg.k === 'name') {
      const s = this.lookup(arg.name);
      if (s?.k === 'buffer') target = { k: 'mem', space: 'global', buf: s.buf, idx: { t: 'int', c: 0 }, t: s.t, name: arg.name };
    } else if (arg.k === 'binary' && arg.op === '+' && arg.l.k === 'name') {
      const s = this.lookup(arg.l.name);
      if (s?.k === 'buffer') {
        target = { k: 'mem', space: 'global', buf: s.buf, idx: this.intIndex(arg.r), t: s.t, name: arg.l.name };
      }
    }
    if (!target || target.k !== 'mem') {
      throw new CompileError(`${name} needs the address of an array element, like ${name}(&out[i], v)`, arg.loc);
    }
    return target;
  }

  // ---- layout ----

  private flatten(): Instr[] {
    const reachable = new Uint8Array(this.blocks.length);
    const stack = [this.layout[0]];
    reachable[this.layout[0]] = 1;
    const succOf = (b: Block): number[] => {
      const t = b.term!;
      if (t.k === 'br') return [t.to];
      if (t.k === 'cbr') return [t.t, t.f];
      return [];
    };
    for (const b of this.blocks) {
      if (!b.term) b.term = { k: 'ret', line: 0, col: 0 };
    }
    while (stack.length) {
      const b = this.blocks[stack.pop()!];
      for (const s of succOf(b)) {
        if (!reachable[s]) {
          reachable[s] = 1;
          stack.push(s);
        }
      }
    }
    const order = this.layout.filter((id) => reachable[id]);
    // Post-dominators over the reachable blocks plus a virtual exit node.
    const index = new Map<number, number>();
    order.forEach((id, i) => index.set(id, i));
    const exit = order.length;
    const succ = order.map((id) => {
      const b = this.blocks[id];
      return b.term!.k === 'ret' ? [exit] : succOf(b).map((s) => index.get(s)!);
    });
    succ.push([]);
    const ipdom = postDominators(succ, exit);

    const code: Instr[] = [];
    const start = new Int32Array(order.length + 1);
    const branches: { ins: Extract<Instr, { op: 'br' }>; to: number }[] = [];
    const conds: { ins: Extract<Instr, { op: 'cbr' }>; t: number; f: number; join: number }[] = [];
    for (let i = 0; i < order.length; i++) {
      const b = this.blocks[order[i]];
      start[i] = code.length;
      code.push(...b.instrs);
      const t = b.term!;
      const at = { line: t.line, col: t.col };
      if (t.k === 'br') {
        const to = index.get(t.to)!;
        if (to !== i + 1) {
          const ins: Extract<Instr, { op: 'br' }> = { op: 'br', target: -1, ...at };
          branches.push({ ins, to });
          code.push(ins);
        }
      } else if (t.k === 'cbr') {
        const ins: Extract<Instr, { op: 'cbr' }> = { op: 'cbr', cond: t.cond, t: -1, f: -1, join: -1, ...at };
        conds.push({ ins, t: index.get(t.t)!, f: index.get(t.f)!, join: ipdom[i] });
        code.push(ins);
      } else {
        code.push({ op: 'ret', ...at });
      }
    }
    start[order.length] = -1;
    for (const b of branches) b.ins.target = start[b.to];
    for (const c of conds) {
      c.ins.t = start[c.t];
      c.ins.f = start[c.f];
      c.ins.join = start[c.join];
    }
    return code;
  }
}

/** Collect the names of scalar variables that are assigned or incremented anywhere in `s`. */
function collectAssigned(s: Stmt, out: Set<string>): void {
  const expr = (e: Expr | undefined): void => {
    if (!e) return;
    switch (e.k) {
      case 'assign':
        if (e.l.k === 'name') out.add(e.l.name);
        expr(e.l);
        expr(e.r);
        return;
      case 'incdec':
        if (e.e.k === 'name') out.add(e.e.name);
        expr(e.e);
        return;
      case 'index':
        expr(e.base);
        expr(e.index);
        return;
      case 'call':
        e.args.forEach(expr);
        return;
      case 'unary':
      case 'cast':
        expr(e.e);
        return;
      case 'binary':
        expr(e.l);
        expr(e.r);
        return;
      case 'cond':
        expr(e.c);
        expr(e.t);
        expr(e.f);
        return;
    }
  };
  switch (s.k) {
    case 'block':
      s.body.forEach((x) => collectAssigned(x, out));
      return;
    case 'decl':
      s.decls.forEach((d) => {
        d.dims.forEach(expr);
        expr(d.init);
      });
      return;
    case 'expr':
      expr(s.e);
      return;
    case 'if':
      expr(s.cond);
      collectAssigned(s.then, out);
      if (s.else) collectAssigned(s.else, out);
      return;
    case 'for':
      if (s.init) collectAssigned(s.init, out);
      expr(s.cond);
      expr(s.step);
      collectAssigned(s.body, out);
      return;
    case 'while':
    case 'do':
      expr(s.cond);
      collectAssigned(s.body, out);
      return;
  }
}

/** Count lines that hold code: not blank, not only comments, braces or preprocessor includes. */
export function countCodeLines(src: string): number {
  const stripped = src.replace(/\/\*[\s\S]*?\*\//g, (m) => m.replace(/[^\n]/g, ' ')).replace(/\/\/.*$/gm, '');
  let n = 0;
  for (const line of stripped.split('\n')) {
    const t = line.trim();
    if (!t || /^[{};\s]*$/.test(t) || /^#\s*(include|pragma)/.test(t)) continue;
    n++;
  }
  return n;
}

export function lowerProgram(prog: Program, src: string, warnings: Diagnostic[]): KernelIR {
  return new Lowerer(warnings).lower(prog, countCodeLines(src));
}
