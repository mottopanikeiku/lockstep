import type { Instr, KernelIR, SharedArray } from '../ir/ir';
import { binOp, cvtOp, DivideByZero, madOp, mathOp, unOp } from '../ir/ops';
import type { Diagnostic } from '../lang/diagnostics';
import { BANKS, COST, SECTOR_BYTES } from './cost';
import { ATOMIC, READ, Trace, WRITE } from './trace';

export interface Dim {
  x: number;
  y: number;
}

export interface Launch {
  grid: Dim;
  block: Dim;
}

/** Kernel arguments in parameter order: array contents for pointers, numbers for scalars. */
export type Arg = Float64Array | number;

export type FaultKind = 'race' | 'deadlock' | 'oob' | 'runtime' | 'timeout';

export interface Fault {
  kind: FaultKind;
  message: string;
  line: number;
  col: number;
  /** Trace step where the fault happened. */
  step: number;
}

/**
 * cycles = issued + specialCycles + shuffleCycles + globalCycles + sharedCycles + barrierWait.
 * Each *Cycles field is what that kind of instruction costs on top of its one issue cycle.
 */
export interface Metrics {
  cycles: number;
  issued: number;
  /** Active lanes summed over every issued instruction. */
  activeLanes: number;
  divergentBranches: number;
  specialCycles: number;
  shuffleCycles: number;
  globalRequests: number;
  globalSectors: number;
  /** Sectors the requests would need if every byte read were packed together. */
  globalIdealSectors: number;
  globalCycles: number;
  sharedRequests: number;
  sharedWavefronts: number;
  /** Extra wavefronts caused by bank conflicts (and serialized shared atomics). */
  bankConflicts: number;
  sharedCycles: number;
  atomicCollisions: number;
  barriers: number;
  barrierWait: number;
}

export interface SimResult {
  /** Final contents of every pointer argument, by buffer number. */
  buffers: Float64Array[];
  fault?: Fault;
  /** Number of racing access pairs seen (the first one is the fault). */
  races: number;
  metrics: Metrics;
  /** Cycles spent on each source line, indexed by line number. */
  lineCycles: Float64Array;
  notes: Diagnostic[];
  warpsPerBlock: number;
  trace?: Trace;
  /** With `inspect`: the watched warp's registers right after the chosen step. */
  inspected?: { regs: Float64Array; live: number };
}

export interface SimOptions {
  trace?: boolean;
  maxSteps?: number;
  /** Stop after this step and capture the registers of one warp (global warp index). */
  inspect?: { step: number; warp: number };
}

/** Thrown to stop a run early once the inspected step is reached. */
class Stop extends Error {}

const RUN = 0;
const BAR = 1;
const DONE = 2;

interface StackEntry {
  pc: number;
  /** Reconvergence point: pop this entry when pc reaches it. -1 is the kernel exit. */
  rpc: number;
  mask: number;
}

interface Warp {
  /** Index across the whole grid. */
  id: number;
  index: number;
  regs: Float64Array;
  stack: StackEntry[];
  /** Lanes that exist (a partial last warp has fewer). */
  exists: number;
  /** Lanes that have not returned. */
  live: number;
  clock: number;
  state: number;
  barPc: number;
  barStep: number;
  /** Bumped at every __syncwarp and __syncthreads, for race ordering. */
  sync: number;
  /** Lanes that left through a return statement, and the line of the first one. */
  early: number;
  earlyLine: number;
}

class Fail extends Error {
  constructor(
    readonly kind: FaultKind,
    message: string,
  ) {
    super(message);
  }
}

const popcount = (m: number) => {
  m = m - ((m >>> 1) & 0x55555555);
  m = (m & 0x33333333) + ((m >>> 2) & 0x33333333);
  return (((m + (m >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
};

/** "lanes 0-15" style description of a lane mask. */
export function laneList(mask: number): string {
  const parts: string[] = [];
  let l = 0;
  while (l < 32) {
    if ((mask >>> l) & 1) {
      let r = l;
      while (r + 1 < 32 && (mask >>> (r + 1)) & 1) r++;
      parts.push(r === l ? `${l}` : `${l}-${r}`);
      l = r + 1;
    } else l++;
  }
  return parts.join(', ');
}

/** Per-word record of the last write and up to two reads, for the race detector. */
class Shadow {
  wT: Int32Array;
  wE: Int32Array;
  wS: Int32Array;
  /** Instruction of the last write. */
  wPc: Int32Array;
  wAtomic: Uint8Array;
  rT: Int32Array;
  rE: Int32Array;
  rS: Int32Array;
  rPc: Int32Array;

  constructor(n: number) {
    this.wT = new Int32Array(n).fill(-1);
    this.wE = new Int32Array(n);
    this.wS = new Int32Array(n);
    this.wPc = new Int32Array(n);
    this.wAtomic = new Uint8Array(n);
    this.rT = new Int32Array(2 * n).fill(-1);
    this.rE = new Int32Array(2 * n);
    this.rS = new Int32Array(2 * n);
    this.rPc = new Int32Array(2 * n);
  }

  reset(): void {
    this.wT.fill(-1);
    this.rT.fill(-1);
  }
}

interface Hazard {
  kind: 'write' | 'read' | 'atomic';
  thread: number;
  pc: number;
}

export class Machine {
  private readonly code: Instr[];
  private readonly baseCost: Int32Array;
  private readonly threads: number;
  private readonly warpsPerBlock: number;
  private readonly buffers: Float64Array[];
  private readonly bufferNames: string[];
  private readonly bufferInt: boolean[];
  private readonly shared: SharedArray[];
  private readonly sharedMem: Float64Array;
  private readonly sharedInit: Uint8Array;
  private readonly globalShadow: Shadow[];
  private readonly sharedShadow: Shadow;
  private readonly trace?: Trace;
  private readonly maxSteps: number;
  private readonly metrics: Metrics = {
    cycles: 0,
    issued: 0,
    activeLanes: 0,
    divergentBranches: 0,
    specialCycles: 0,
    shuffleCycles: 0,
    globalRequests: 0,
    globalSectors: 0,
    globalIdealSectors: 0,
    globalCycles: 0,
    sharedRequests: 0,
    sharedWavefronts: 0,
    bankConflicts: 0,
    sharedCycles: 0,
    atomicCollisions: 0,
    barriers: 0,
    barrierWait: 0,
  };
  private readonly lineCycles: Float64Array;
  private readonly notes: Diagnostic[] = [];
  private readonly notedLines = new Set<number>();
  private races = 0;
  private firstRace?: string;
  private steps = 0;
  private block = 0;
  private blockEpoch = 0;
  private blockIdx: Dim = { x: 0, y: 0 };
  private step = -1;
  private pc = 0;
  /** Scratch space for coalescing and bank conflict counting. */
  private readonly scratch = new Int32Array(32);
  private readonly args: Arg[];
  private raceFault?: Fault;
  /** Warps of the block being run. */
  private warps: Warp[] = [];
  private readonly bankWords = new Int32Array(BANKS * 32);
  private readonly bankCount = new Int32Array(BANKS);
  private readonly shflTmp = new Float64Array(32);

  constructor(
    private readonly k: KernelIR,
    private readonly launch: Launch,
    args: Arg[],
    private readonly opts: SimOptions = {},
  ) {
    if (args.length !== k.params.length) {
      throw new Error(`kernel ${k.name} takes ${k.params.length} arguments, got ${args.length}`);
    }
    this.code = k.code;
    this.threads = launch.block.x * launch.block.y;
    this.warpsPerBlock = Math.ceil(this.threads / 32);
    this.buffers = [];
    this.bufferNames = [];
    this.bufferInt = [];
    k.params.forEach((p, i) => {
      if (p.kind !== 'buffer') return;
      const a = args[i];
      if (typeof a === 'number') throw new Error(`argument ${p.name} must be an array`);
      this.buffers[p.buf] = Float64Array.from(a);
      this.bufferNames[p.buf] = p.name;
      this.bufferInt[p.buf] = p.type === 'int';
    });
    this.shared = k.shared;
    this.sharedMem = new Float64Array(k.sharedWords);
    this.sharedInit = new Uint8Array(k.sharedWords);
    this.globalShadow = this.buffers.map((b) => new Shadow(b.length));
    this.sharedShadow = new Shadow(k.sharedWords);
    this.trace = opts.trace ? new Trace() : undefined;
    this.maxSteps = opts.maxSteps ?? 2_000_000;
    let maxLine = 0;
    for (const ins of this.code) maxLine = Math.max(maxLine, ins.line);
    this.lineCycles = new Float64Array(maxLine + 2);
    const isConst = new Uint8Array(k.numRegs);
    for (const c of k.consts) isConst[c.reg] = 1;
    this.baseCost = Int32Array.from(this.code, (ins) => {
      if (ins.op === 'math') return COST.special;
      if (ins.op === 'shfl') return COST.shuffle;
      if (ins.op === 'bin' && ins.fn === 'div' && ins.float) return COST.special;
      if (ins.op === 'bin' && !ins.float && (ins.fn === 'div' || ins.fn === 'rem') && !isConst[ins.b]) return COST.special;
      return COST.issue;
    });
    this.args = args;
  }

  run(): SimResult {
    let fault: Fault | undefined;
    let inspected: SimResult['inspected'];
    try {
      for (let by = 0; by < this.launch.grid.y; by++) {
        for (let bx = 0; bx < this.launch.grid.x; bx++) {
          this.runBlock(bx, by);
        }
      }
    } catch (err) {
      if (err instanceof Stop) {
        const w = this.warps.find((x) => x.id === this.opts.inspect!.warp);
        if (w) inspected = { regs: w.regs.slice(), live: w.live };
      } else {
        if (!(err instanceof Fail)) throw err;
        const ins = this.code[this.pc];
        fault = { kind: err.kind, message: err.message, line: ins?.line ?? 0, col: ins?.col ?? 0, step: this.step };
      }
    }
    if (!fault && this.firstRace) {
      fault = this.raceFault;
    }
    return {
      buffers: this.buffers,
      fault,
      races: this.races,
      metrics: this.metrics,
      lineCycles: this.lineCycles,
      notes: this.notes,
      warpsPerBlock: this.warpsPerBlock,
      trace: this.trace,
      inspected,
    };
  }

  private runBlock(bx: number, by: number): void {
    this.blockIdx = { x: bx, y: by };
    this.block = by * this.launch.grid.x + bx;
    this.blockEpoch = 0;
    this.sharedShadow.reset();
    this.sharedInit.fill(0);
    for (const arr of this.shared) {
      this.sharedMem.fill(arr.type === 'float' ? NaN : -559038737, arr.offset, arr.offset + arr.size);
    }
    const warps: Warp[] = (this.warps = []);
    for (let w = 0; w < this.warpsPerBlock; w++) warps.push(this.makeWarp(w));
    for (;;) {
      let running = false;
      for (const w of warps) {
        if (w.state !== RUN) continue;
        running = true;
        this.stepWarp(w);
      }
      if (running) continue;
      const waiting = warps.filter((w) => w.state === BAR);
      if (waiting.length === 0) return;
      this.releaseBarrier(warps, waiting);
    }
  }

  private makeWarp(index: number): Warp {
    const regs = new Float64Array(this.k.numRegs * 32);
    let exists = 0;
    for (let l = 0; l < 32; l++) {
      const tid = index * 32 + l;
      if (tid >= this.threads) continue;
      exists |= 1 << l;
      const tx = tid % this.launch.block.x;
      const ty = Math.floor(tid / this.launch.block.x);
      for (const s of this.k.specials) {
        let v = 0;
        switch (s.which) {
          case 'threadIdx.x': v = tx; break;
          case 'threadIdx.y': v = ty; break;
          case 'blockIdx.x': v = this.blockIdx.x; break;
          case 'blockIdx.y': v = this.blockIdx.y; break;
          case 'blockDim.x': v = this.launch.block.x; break;
          case 'blockDim.y': v = this.launch.block.y; break;
          case 'gridDim.x': v = this.launch.grid.x; break;
          case 'gridDim.y': v = this.launch.grid.y; break;
          case 'blockDim.z':
          case 'gridDim.z': v = 1; break;
        }
        regs[s.reg * 32 + l] = v;
      }
    }
    for (const c of this.k.consts) regs.fill(c.value, c.reg * 32, c.reg * 32 + 32);
    this.k.params.forEach((p, i) => {
      if (p.kind === 'scalar') {
        const a = this.args[i] as number;
        const v = p.type === 'float' ? Math.fround(a) : p.type === 'bool' ? (a ? 1 : 0) : a | 0;
        regs.fill(v, p.reg * 32, p.reg * 32 + 32);
      }
    });
    return {
      id: this.block * this.warpsPerBlock + index,
      index,
      regs,
      stack: [{ pc: 0, rpc: -1, mask: exists }],
      exists,
      live: exists,
      clock: 0,
      state: RUN,
      barPc: -1,
      barStep: -1,
      sync: 0,
      early: 0,
      earlyLine: 0,
    };
  }

  /** Name a thread the way CUDA tools do. */
  private threadName(warp: number, lane: number): string {
    const tid = warp * 32 + lane;
    const { block, grid } = this.launch;
    const t = block.y > 1 ? `(${tid % block.x},${Math.floor(tid / block.x)})` : `${tid}`;
    if (grid.x * grid.y === 1) return `thread ${t}`;
    const b = grid.y > 1 ? `(${this.blockIdx.x},${this.blockIdx.y})` : `${this.blockIdx.x}`;
    return `thread ${t} of block ${b}`;
  }

  private globalThreadName(gt: number): string {
    const { block, grid } = this.launch;
    const b = Math.floor(gt / this.threads);
    const tid = gt % this.threads;
    const t = block.y > 1 ? `(${tid % block.x},${Math.floor(tid / block.x)})` : `${tid}`;
    if (grid.x * grid.y === 1) return `thread ${t}`;
    const bn = grid.y > 1 ? `(${b % grid.x},${Math.floor(b / grid.x)})` : `${b}`;
    return `thread ${t} of block ${bn}`;
  }

  private releaseBarrier(warps: Warp[], waiting: Warp[]): void {
    const first = waiting[0];
    this.pc = first.barPc;
    this.step = first.barStep;
    const line = this.code[first.barPc].line;
    let early = 0;
    let earlyLine = 0;
    let finished = 0;
    let finishedWarps = 0;
    for (const w of warps) {
      const gone = w.exists & ~w.live;
      if (!gone) continue;
      early += popcount(gone & w.early);
      earlyLine ||= w.earlyLine;
      if (gone & ~w.early) {
        finished += popcount(gone & ~w.early);
        finishedWarps |= 1 << w.index;
      }
    }
    const expects = `__syncthreads() on line ${line} expects all ${this.threads} threads of the block`;
    if (early > 0) {
      throw new Fail(
        'deadlock',
        `${expects}, but ${early} of them returned early on line ${earlyLine}. ` +
          `CUDA requires every thread to reach the barrier, so guard the work with an if instead of returning.`,
      );
    }
    if (finished > 0) {
      const which = `warp${popcount(finishedWarps) > 1 ? 's' : ''} ${laneList(finishedWarps)}`;
      throw new Fail(
        'deadlock',
        `${expects}, but ${finished} of them (${which}) finished the kernel without reaching it. ` +
          `CUDA requires every thread to reach the barrier, which fails when it sits inside an if or a loop that not every thread runs.`,
      );
    }
    const lines = new Map<number, number[]>();
    for (const w of waiting) {
      const l = this.code[w.barPc].line;
      if (!lines.has(l)) lines.set(l, []);
      lines.get(l)!.push(w.index);
    }
    if (lines.size > 1) {
      const where = [...lines].map(([l, ws]) => `warp${ws.length > 1 ? 's' : ''} ${ws.join(', ')} on line ${l}`);
      throw new Fail(
        'deadlock',
        `the warps of this block wait at different __syncthreads() calls (${where.join(', ')}), ` +
          `and since CUDA leaves that undefined, the kernel may hang or misbehave on a real GPU.`,
      );
    }
    let maxClock = 0;
    for (const w of waiting) maxClock = Math.max(maxClock, w.clock);
    const release = this.trace ? this.trace.steps : 0;
    for (const w of waiting) {
      const wait = maxClock - w.clock;
      this.metrics.barrierWait += wait;
      this.metrics.cycles += wait;
      this.lineCycles[line] += wait;
      w.clock = maxClock;
      w.state = RUN;
      w.sync++;
      if (this.trace) this.trace.release[w.barStep] = release;
    }
    this.metrics.barriers++;
    this.blockEpoch++;
  }

  private stepWarp(w: Warp): void {
    if (++this.steps > this.maxSteps) {
      throw new Fail('timeout', `the kernel ran for more than ${this.maxSteps.toLocaleString('en-US')} warp instructions, which usually means a loop that never ends`);
    }
    const stack = w.stack;
    const top = stack[stack.length - 1];
    const pc = top.pc;
    const m = top.mask;
    const ins = this.code[pc];
    this.pc = pc;
    const tr = this.trace;
    const s = tr ? tr.beginStep(w.id, pc, m, stack.length) : this.steps - 1;
    this.step = s;
    const R = w.regs;
    let cost = this.baseCost[pc];
    if (cost > COST.issue) {
      if (ins.op === 'shfl') this.metrics.shuffleCycles += cost - COST.issue;
      else this.metrics.specialCycles += cost - COST.issue;
    }
    let units = 0;
    top.pc = pc + 1;
    try {
      switch (ins.op) {
        case 'mov': {
          const d = ins.dst * 32;
          const a = ins.a * 32;
          for (let l = 0; l < 32; l++) if ((m >>> l) & 1) R[d + l] = R[a + l];
          break;
        }
        case 'bin': {
          const { fn, float } = ins;
          const uns = ins.unsigned === true;
          const d = ins.dst * 32;
          const a = ins.a * 32;
          const b = ins.b * 32;
          for (let l = 0; l < 32; l++) if ((m >>> l) & 1) R[d + l] = binOp(fn, float, R[a + l], R[b + l], uns);
          break;
        }
        case 'mad': {
          const d = ins.dst * 32;
          const a = ins.a * 32;
          const b = ins.b * 32;
          const c = ins.c * 32;
          for (let l = 0; l < 32; l++) if ((m >>> l) & 1) R[d + l] = madOp(ins.float, R[a + l], R[b + l], R[c + l]);
          break;
        }
        case 'un': {
          const d = ins.dst * 32;
          const a = ins.a * 32;
          for (let l = 0; l < 32; l++) if ((m >>> l) & 1) R[d + l] = unOp(ins.fn, ins.float, R[a + l]);
          break;
        }
        case 'cvt': {
          const d = ins.dst * 32;
          const a = ins.a * 32;
          for (let l = 0; l < 32; l++) if ((m >>> l) & 1) R[d + l] = cvtOp(ins.fn, R[a + l]);
          break;
        }
        case 'math': {
          const d = ins.dst * 32;
          const a = ins.a * 32;
          for (let l = 0; l < 32; l++) if ((m >>> l) & 1) R[d + l] = mathOp(ins.fn, R[a + l]);
          break;
        }
        case 'ld':
        case 'st':
        case 'atom':
          if (ins.space === 'global') {
            const extra = this.globalAccess(w, ins, m);
            units = extra.sectors;
            cost += extra.cycles;
          } else {
            units = this.sharedAccess(w, ins, m);
            cost += units * COST.sharedWavefront;
            this.metrics.sharedCycles += units * COST.sharedWavefront;
          }
          break;
        case 'shfl':
          this.shuffle(w, ins, m);
          break;
        case 'bar':
          if (m !== w.live) this.divergedAt(w, m, '__syncthreads()');
          w.state = BAR;
          w.barPc = pc;
          w.barStep = s;
          break;
        case 'syncwarp':
          if (m !== w.live) this.divergedAt(w, m, '__syncwarp()');
          w.sync++;
          break;
        case 'br':
          top.pc = ins.target;
          break;
        case 'cbr': {
          const c = ins.cond * 32;
          let t = 0;
          for (let l = 0; l < 32; l++) if ((m >>> l) & 1 && R[c + l] !== 0) t |= 1 << l;
          const f = m & ~t;
          if (f === 0) top.pc = ins.t;
          else if (t === 0) top.pc = ins.f;
          else {
            this.metrics.divergentBranches++;
            tr?.divergences.push(s);
            const rpc = ins.join === -1 ? top.rpc : ins.join;
            if (rpc === top.rpc) stack.pop();
            else top.pc = rpc;
            if (ins.f !== rpc) stack.push({ pc: ins.f, rpc, mask: f });
            if (ins.t !== rpc) stack.push({ pc: ins.t, rpc, mask: t });
          }
          break;
        }
        case 'ret':
          for (const e of stack) e.mask &= ~m;
          w.live &= ~m;
          if (ins.early) {
            w.early |= m;
            w.earlyLine ||= ins.line;
          }
          break;
      }
    } catch (err) {
      if (err instanceof DivideByZero) {
        throw new Fail('runtime', 'integer division by zero');
      }
      throw err;
    }
    this.metrics.issued++;
    this.metrics.activeLanes += popcount(m);
    this.metrics.cycles += cost;
    w.clock += cost;
    this.lineCycles[ins.line] += cost;
    tr?.endStep(s, cost, w.clock, units);
    // Reconverge: pop entries that reached their join point or ran out of lanes.
    while (stack.length) {
      const t = stack[stack.length - 1];
      if (t.mask !== 0 && t.pc !== t.rpc) break;
      stack.pop();
    }
    if (stack.length === 0) w.state = DONE;
    if (this.opts.inspect?.step === s) throw new Stop();
  }

  private divergedAt(w: Warp, m: number, what: string): never {
    const others = w.live & ~m;
    let where = '';
    for (let i = w.stack.length - 2; i >= 0; i--) {
      const e = w.stack[i];
      if (e.mask & others && e.pc >= 0 && e.pc < this.code.length) {
        where = ` (their next line is ${this.code[e.pc].line})`;
        break;
      }
    }
    const rule =
      what === '__syncthreads()'
        ? 'CUDA requires every thread of the block to reach the same __syncthreads(), so move it out of the branch.'
        : what === '__syncwarp()'
          ? 'Every lane of the warp has to reach the same __syncwarp(), so move it out of the branch.'
          : `Every lane named in the mask has to make the same ${what} call, so move it out of the branch or leave those lanes out of the mask.`;
    throw new Fail(
      'deadlock',
      `${what} on line ${this.code[this.pc].line} was reached by lanes ${laneList(m)} of warp ${w.index} while lanes ${laneList(others)} took another branch${where}. ${rule}`,
    );
  }

  private globalAccess(w: Warp, ins: Extract<Instr, { op: 'ld' | 'st' | 'atom' }>, m: number): { sectors: number; cycles: number } {
    const R = w.regs;
    const buf = this.buffers[ins.buf];
    const name = this.bufferNames[ins.buf];
    const isInt = this.bufferInt[ins.buf];
    const shadow = this.globalShadow[ins.buf];
    const idxBase = ins.idx * 32;
    const tr = this.trace;
    const kind = ins.op === 'ld' ? READ : ins.op === 'st' ? WRITE : ATOMIC;
    let nsect = 0;
    let lanes = 0;
    let collisions = 0;
    const seen = this.scratch;
    const addrs = this.bankWords;
    let naddr = 0;
    for (let l = 0; l < 32; l++) {
      if (!((m >>> l) & 1)) continue;
      const idx = R[idxBase + l];
      if (idx < 0 || idx >= buf.length) {
        const verb = kind === READ ? 'read' : 'write';
        throw new Fail(
          'oob',
          `out-of-bounds ${verb}: ${name}[${idx}] but ${name} has ${buf.length} elements (${this.threadName(w.index, l)})`,
        );
      }
      lanes++;
      const sector = Math.floor((idx * 4) / SECTOR_BYTES);
      let found = false;
      for (let j = 0; j < nsect; j++) {
        if (seen[j] === sector) {
          found = true;
          break;
        }
      }
      if (!found) seen[nsect++] = sector;
      if (kind === ATOMIC) {
        let dup = false;
        for (let j = 0; j < naddr; j++) {
          if (addrs[j] === idx) {
            dup = true;
            break;
          }
        }
        if (dup) collisions++;
        else addrs[naddr++] = idx;
      }
      const gt = this.block * this.threads + w.index * 32 + l;
      this.checkRace(shadow, idx, gt, w, kind, () => `${name}[${idx}]`);
      const old = buf[idx];
      if (ins.op === 'ld') {
        R[ins.dst * 32 + l] = old;
        tr?.access(l, ins.buf, idx, READ, old, old);
      } else if (ins.op === 'st') {
        buf[idx] = R[ins.val * 32 + l];
        tr?.access(l, ins.buf, idx, WRITE, old, buf[idx]);
      } else {
        buf[idx] = atomApply(ins.fn, !isInt, old, R[ins.val * 32 + l]);
        R[ins.dst * 32 + l] = old;
        tr?.access(l, ins.buf, idx, ATOMIC, old, buf[idx]);
      }
    }
    const metrics = this.metrics;
    metrics.globalRequests++;
    metrics.globalSectors += nsect;
    metrics.globalIdealSectors += Math.ceil((lanes * 4) / SECTOR_BYTES);
    metrics.atomicCollisions += collisions;
    const cycles = nsect * COST.globalSector + collisions * COST.atomicCollision;
    metrics.globalCycles += cycles;
    return { sectors: nsect, cycles };
  }

  /** Returns the number of wavefronts the access needs. */
  private sharedAccess(w: Warp, ins: Extract<Instr, { op: 'ld' | 'st' | 'atom' }>, m: number): number {
    const R = w.regs;
    const arr = this.shared[ins.buf];
    const mem = this.sharedMem;
    const idxBase = ins.idx * 32;
    const tr = this.trace;
    const kind = ins.op === 'ld' ? READ : ins.op === 'st' ? WRITE : ATOMIC;
    const counts = this.bankCount;
    const words = this.bankWords;
    counts.fill(0);
    let wavefronts = 0;
    for (let l = 0; l < 32; l++) {
      if (!((m >>> l) & 1)) continue;
      const idx = R[idxBase + l];
      if (idx < 0 || idx >= arr.size) {
        const verb = kind === READ ? 'read' : 'write';
        throw new Fail(
          'oob',
          `out-of-bounds shared ${verb}: ${sharedName(arr, idx)} is outside ${arr.name}, which has ${arr.size} elements (${this.threadName(w.index, l)})`,
        );
      }
      const word = arr.offset + idx;
      const bank = word % BANKS;
      // Same word in a bank is a broadcast for loads and stores, but atomics serialize.
      let dup = false;
      if (kind !== ATOMIC) {
        for (let j = 0; j < counts[bank]; j++) {
          if (words[bank * 32 + j] === word) {
            dup = true;
            break;
          }
        }
      }
      if (!dup) {
        words[bank * 32 + counts[bank]] = word;
        counts[bank]++;
        if (counts[bank] > wavefronts) wavefronts = counts[bank];
      }
      const gt = this.block * this.threads + w.index * 32 + l;
      this.checkRace(this.sharedShadow, word, gt, w, kind, () => sharedName(arr, idx));
      const old = mem[word];
      const enc = -(ins.buf + 1);
      if (ins.op === 'ld') {
        if (!this.sharedInit[word]) this.noteUninitialized(ins.line, arr, idx);
        R[ins.dst * 32 + l] = old;
        tr?.access(l, enc, idx, READ, old, old);
      } else if (ins.op === 'st') {
        mem[word] = R[ins.val * 32 + l];
        this.sharedInit[word] = 1;
        tr?.access(l, enc, idx, WRITE, old, mem[word]);
      } else {
        if (!this.sharedInit[word]) this.noteUninitialized(ins.line, arr, idx);
        mem[word] = atomApply(ins.fn, arr.type === 'float', old, R[ins.val * 32 + l]);
        this.sharedInit[word] = 1;
        R[ins.dst * 32 + l] = old;
        tr?.access(l, enc, idx, ATOMIC, old, mem[word]);
      }
    }
    this.metrics.sharedRequests++;
    this.metrics.sharedWavefronts += wavefronts;
    this.metrics.bankConflicts += Math.max(0, wavefronts - 1);
    return wavefronts;
  }

  private noteUninitialized(line: number, arr: SharedArray, idx: number): void {
    if (this.notedLines.has(line) || this.notes.length >= 3) return;
    this.notedLines.add(line);
    this.notes.push({
      severity: 'warning',
      message: `reads ${sharedName(arr, idx)} before any thread wrote it, and shared memory starts out as garbage`,
      line,
      col: 1,
    });
  }

  private checkRace(sh: Shadow, word: number, gt: number, w: Warp, kind: number, name: () => string): void {
    const pc = this.pc;
    const e = this.blockEpoch;
    const sy = w.sync;
    const ordered = (pT: number, pE: number, pS: number) => {
      if (pT === gt) return true;
      const pb = Math.floor(pT / this.threads);
      if (pb !== this.block) return false;
      if (pE < e) return true;
      return Math.floor((pT % this.threads) / 32) === w.index && pS < sy;
    };
    let other: Hazard | undefined;
    const wT = sh.wT[word];
    if (wT >= 0 && !ordered(wT, sh.wE[word], sh.wS[word])) {
      const atomic = sh.wAtomic[word] === 1;
      if (kind === WRITE || (!atomic && (kind === READ || kind === ATOMIC))) {
        other = { kind: atomic ? 'atomic' : 'write', thread: wT, pc: sh.wPc[word] };
      }
    }
    if (!other && kind === WRITE) {
      for (let r = 0; r < 2; r++) {
        const rt = sh.rT[2 * word + r];
        if (rt >= 0 && !ordered(rt, sh.rE[2 * word + r], sh.rS[2 * word + r])) {
          other = { kind: 'read', thread: rt, pc: sh.rPc[2 * word + r] };
          break;
        }
      }
    }
    if (other) this.reportRace(other, { kind: kind === READ ? 'read' : kind === WRITE ? 'write' : 'atomic', thread: gt, pc }, name, sh === this.sharedShadow);
    if (kind === READ) {
      let slot = -1;
      for (let r = 0; r < 2; r++) {
        const rt = sh.rT[2 * word + r];
        if (rt < 0 || rt === gt || ordered(rt, sh.rE[2 * word + r], sh.rS[2 * word + r])) {
          slot = r;
          break;
        }
      }
      if (slot >= 0) {
        const i = 2 * word + slot;
        sh.rT[i] = gt;
        sh.rE[i] = e;
        sh.rS[i] = sy;
        sh.rPc[i] = pc;
      }
    } else {
      sh.wT[word] = gt;
      sh.wE[word] = e;
      sh.wS[word] = sy;
      sh.wPc[word] = pc;
      sh.wAtomic[word] = kind === ATOMIC ? 1 : 0;
      if (kind === WRITE) {
        sh.rT[2 * word] = -1;
        sh.rT[2 * word + 1] = -1;
      }
    }
  }

  private reportRace(prev: Hazard, cur: Hazard, name: () => string, shared: boolean): void {
    this.races++;
    if (this.firstRace) return;
    const verb = (h: Hazard) => (h.kind === 'read' ? 'read' : h.kind === 'write' ? 'wrote' : 'atomically updated');
    const prevName = this.globalThreadName(prev.thread);
    const curName = this.globalThreadName(cur.thread);
    const sameBlock = Math.floor(prev.thread / this.threads) === Math.floor(cur.thread / this.threads);
    const p = this.code[prev.pc];
    const c = this.code[cur.pc];
    const sameLine = p.line === c.line;
    const writeWrite = sameLine && prev.kind === 'write' && cur.kind === 'write';
    // A load and a store through the same index register on one line: `x[k] += v` or `x[k]++`.
    const inPlace = sameLine && prev.kind !== cur.kind && 'idx' in p && 'idx' in c && p.idx === c.idx;
    const hint =
      prev.kind === 'atomic' || cur.kind === 'atomic'
        ? 'Mixing atomic and plain accesses to one address is a race, so make them all atomic.'
        : inPlace
          ? 'Both threads update it in place at the same time, so the update has to be an atomic such as atomicAdd.'
          : !sameBlock
            ? writeWrite
              ? 'Blocks run in no fixed order, so if each block should write its own elements the index needs blockIdx, and if they all add into one place the update has to be an atomic.'
              : 'Blocks run in no fixed order and cannot wait for each other inside a kernel, so the update has to be an atomic.'
            : writeWrite
              ? 'Both threads wrote it from the same line, so if each thread should have its own element, check the index math and the bounds, and if they share it, let one thread write it or use an atomic.'
              : 'Put a __syncthreads() between them.';
    const message =
      `race on ${shared ? 'shared ' : ''}${name()}: ${prevName} ${verb(prev)} it on line ${p.line} and ` +
      `${curName} ${verb(cur)} it on line ${c.line}, with nothing ordering the two. ${hint}`;
    this.firstRace = message;
    this.raceFault = { kind: 'race', message, line: c.line, col: c.col, step: this.step };
  }

  private shuffle(w: Warp, ins: Extract<Instr, { op: 'shfl' }>, m: number): void {
    const R = w.regs;
    let first = 0;
    while (!((m >>> first) & 1)) first++;
    const mask = R[ins.mask * 32 + first] | 0;
    const name = `__shfl_${ins.mode === 'idx' ? '' : ins.mode + '_'}sync`;
    if ((m & ~mask) !== 0) {
      throw new Fail('runtime', `lanes ${laneList(m & ~mask)} call ${name} but are not in its mask 0x${(mask >>> 0).toString(16)}`);
    }
    const missing = mask & w.exists & ~m;
    if (missing) {
      const exited = missing & ~w.live;
      if (exited) {
        throw new Fail(
          'deadlock',
          `${name} names lanes ${laneList(exited)} in its mask, but they already returned, which CUDA leaves undefined, so a real GPU may hang or return garbage.`,
        );
      }
      this.divergedAt(w, m, name);
    }
    const tmp = this.shflTmp;
    const width = ins.width;
    for (let l = 0; l < 32; l++) {
      if (!((m >>> l) & 1)) continue;
      const arg = R[ins.arg * 32 + l] | 0;
      const base = l & ~(width - 1);
      let src: number;
      switch (ins.mode) {
        case 'idx': src = base + (arg & (width - 1)); break;
        case 'up': src = (l & (width - 1)) - arg < 0 ? l : l - arg; break;
        case 'down': src = (l & (width - 1)) + arg >= width ? l : l + arg; break;
        case 'xor': src = l ^ arg; if (src >= base + width || src < 0 || src > 31) src = l; break;
      }
      if (!((w.exists >>> src) & 1)) src = l;
      tmp[l] = R[ins.val * 32 + src];
    }
    for (let l = 0; l < 32; l++) if ((m >>> l) & 1) R[ins.dst * 32 + l] = tmp[l];
  }
}

function atomApply(fn: string, float: boolean, old: number, v: number): number {
  switch (fn) {
    case 'add': return float ? Math.fround(old + v) : (old + v) | 0;
    case 'sub': return float ? Math.fround(old - v) : (old - v) | 0;
    case 'max': return Math.max(old, v);
    case 'min': return Math.min(old, v);
    default: return v;
  }
}

/** tile[3][4] for 2D arrays, buf[7] for flat ones. */
export function sharedName(arr: SharedArray, idx: number): string {
  if (arr.dims.length === 0) return arr.name;
  if (arr.dims.length === 1 || idx < 0 || idx >= arr.size) return `${arr.name}[${idx}]`;
  const parts: number[] = [];
  let rest = idx;
  for (let d = arr.dims.length - 1; d >= 0; d--) {
    parts.unshift(rest % arr.dims[d]);
    rest = Math.floor(rest / arr.dims[d]);
  }
  return arr.name + parts.map((p) => `[${p}]`).join('');
}

export function simulate(k: KernelIR, launch: Launch, args: Arg[], opts?: SimOptions): SimResult {
  return new Machine(k, launch, args, opts).run();
}
