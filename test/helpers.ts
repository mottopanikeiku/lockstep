import { compile } from '../src/compile';
import type { KernelIR } from '../src/ir/ir';
import { simulate, type Arg, type Launch, type SimResult } from '../src/sim/machine';

export function kernel(src: string): KernelIR {
  const r = compile(src);
  if (!r.ok) throw new Error(`line ${r.error.line}:${r.error.col}: ${r.error.message}`);
  return r.kernel;
}

export function run(src: string, launch: Launch, args: Arg[], trace = true): SimResult {
  return simulate(kernel(src), launch, args, { trace });
}

export const oneWarp: Launch = { grid: { x: 1, y: 1 }, block: { x: 32, y: 1 } };

export function launch1d(blocks: number, threads: number): Launch {
  return { grid: { x: blocks, y: 1 }, block: { x: threads, y: 1 } };
}

export const zeros = (n: number) => new Float64Array(n);
export const iota = (n: number) => Float64Array.from({ length: n }, (_, i) => i);

/** The steps of a trace whose instruction sits on `line`. */
export function stepsOnLine(r: SimResult, k: KernelIR, line: number): number[] {
  const t = r.trace!;
  const out: number[] = [];
  for (let s = 0; s < t.steps; s++) if (k.code[t.pc[s]].line === line) out.push(s);
  return out;
}
