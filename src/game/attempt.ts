import { compile, type CompileResult } from '../compile';
import type { KernelIR } from '../ir/ir';
import type { Diagnostic } from '../lang/diagnostics';
import { makeInputs, type Inputs, type Level, type Par } from '../levels/types';
import { simulate, type Arg, type FaultKind, type SimResult } from '../sim/machine';

export type ProblemKind = 'compile' | 'signature' | FaultKind | 'wrong';

export interface Problem {
  kind: ProblemKind;
  message: string;
  line: number;
  col: number;
  /** Trace step to jump to, for runtime faults. */
  step?: number;
}

export interface OutputCheck {
  name: string;
  expected: Float64Array;
  actual: Float64Array;
  /** Indices that differ beyond the tolerance. */
  wrong: number[];
}

export interface Attempt {
  compiled: CompileResult;
  inputs: Inputs;
  sim?: SimResult;
  outputs: OutputCheck[];
  problem?: Problem;
  warnings: Diagnostic[];
  solved: boolean;
  score?: Par;
}

const describeParam = (p: Level['params'][number]) =>
  p.kind === 'scalar' ? `${p.type} ${p.name}` : `${p.output ? '' : 'const '}${p.type}* ${p.name}`;

export function signature(level: Level): string {
  return level.params.map(describeParam).join(', ');
}

function checkSignature(level: Level, k: KernelIR): string | undefined {
  const want = level.params;
  const got = k.params;
  const ok =
    want.length === got.length &&
    want.every((p, i) => {
      const g = got[i];
      return p.kind === 'scalar' ? g.kind === 'scalar' && g.type === p.type : g.kind === 'buffer' && g.type === p.type;
    });
  return ok ? undefined : `this level's kernel takes (${signature(level)})`;
}

function close(a: number, b: number, tol: number): boolean {
  if (a === b) return true;
  if (Number.isNaN(a) || Number.isNaN(b)) return false;
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));
}

/** Compile and run a player's kernel on a level, and check the outputs. */
export function attempt(level: Level, src: string, opts: { trace?: boolean } = {}): Attempt {
  const compiled = compile(src);
  const inputs = makeInputs(level);
  const base = { compiled, inputs, outputs: [], warnings: compiled.warnings, solved: false };
  if (!compiled.ok) {
    return { ...base, problem: { kind: 'compile', ...compiled.error } };
  }
  const k = compiled.kernel;
  const sigError = checkSignature(level, k);
  if (sigError) return { ...base, problem: { kind: 'signature', message: sigError, line: 1, col: 1 } };

  const args: Arg[] = level.params.map((p) => inputs[p.name]);
  const sim = simulate(k, level.launch, args, { trace: opts.trace });
  const warnings = [...compiled.warnings, ...sim.notes];
  if (sim.fault) {
    const f = sim.fault;
    return { ...base, sim, warnings, problem: { kind: f.kind, message: f.message, line: f.line, col: f.col, step: f.step } };
  }
  const expected = level.expected(inputs);
  const tol = level.tolerance ?? 0;
  const outputs: OutputCheck[] = [];
  level.params.forEach((p, i) => {
    if (p.kind !== 'array' || !p.output) return;
    const param = k.params[i];
    if (param.kind !== 'buffer') return;
    const actual = sim.buffers[param.buf];
    const exp = Float64Array.from(expected[p.name]);
    const wrong: number[] = [];
    for (let j = 0; j < exp.length; j++) if (!close(actual[j], exp[j], tol)) wrong.push(j);
    outputs.push({ name: p.name, expected: exp, actual, wrong });
  });
  const firstWrong = outputs.find((o) => o.wrong.length);
  if (firstWrong) {
    const j = firstWrong.wrong[0];
    const n = outputs.reduce((s, o) => s + o.wrong.length, 0);
    const message =
      `${firstWrong.name}[${j}] should be ${fmt(firstWrong.expected[j])} but is ${fmt(firstWrong.actual[j])}` +
      (n > 1 ? ` (${n} wrong values)` : '');
    const line = lastStoreLine(k, bufferOf(level, k, firstWrong.name));
    return { ...base, sim, warnings, outputs, problem: { kind: 'wrong', message, line, col: 1 } };
  }
  return {
    ...base,
    sim,
    warnings,
    outputs,
    solved: true,
    score: { cycles: sim.metrics.cycles, sectors: sim.metrics.globalSectors, lines: k.lines },
  };
}

function bufferOf(level: Level, k: KernelIR, name: string): number {
  const i = level.params.findIndex((p) => p.name === name);
  const p = k.params[i];
  return p.kind === 'buffer' ? p.buf : -1;
}

/** The line of the last store to a buffer, a decent place to point at for wrong output. */
function lastStoreLine(k: KernelIR, buf: number): number {
  for (let i = k.code.length - 1; i >= 0; i--) {
    const ins = k.code[i];
    if ((ins.op === 'st' || ins.op === 'atom') && ins.space === 'global' && ins.buf === buf) return ins.line;
  }
  return 0;
}

export function fmt(v: number): string {
  if (Number.isNaN(v)) return 'NaN';
  if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf';
  if (Number.isInteger(v)) return String(v);
  const s = v.toPrecision(5);
  return s.includes('e') ? s : String(parseFloat(s));
}
