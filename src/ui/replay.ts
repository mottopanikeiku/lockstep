import type { KernelIR } from '../ir/ir';
import type { Launch } from '../sim/machine';
import { READ, type Trace } from '../sim/trace';

export type WarpStatus = 'idle' | 'run' | 'bar' | 'done';

export interface WarpView {
  index: number;
  status: WarpStatus;
  /** Lanes active in the warp's latest instruction. */
  mask: number;
  /** Lanes that have not returned. */
  live: number;
  exists: number;
  /** Source line of the warp's latest instruction (0 before it starts). */
  line: number;
  /** The warp issued the current step. */
  issuing: boolean;
}

export interface Access {
  lane: number;
  /** Global buffer index, or -(shared array + 1). */
  buf: number;
  idx: number;
  kind: number;
  value: number;
}

/** Shared memory starts as garbage; the simulator fills it with these. */
export const isGarbage = (v: number) => Number.isNaN(v) || v === -559038737;

/**
 * Rebuilds memory contents and warp states at any step of a trace, moving
 * forward or backward through the write log.
 */
export class Replay {
  readonly warpsPerBlock: number;
  readonly threads: number;
  readonly blocks: number;
  readonly global: Float64Array[];
  readonly shared: Float64Array[];
  private cursor = -1;
  private readonly warpSteps: Int32Array[];
  private readonly retSteps: { step: number; live: number }[][];

  constructor(
    readonly kernel: KernelIR,
    readonly launch: Launch,
    initial: Float64Array[],
    readonly trace: Trace,
  ) {
    this.threads = launch.block.x * launch.block.y;
    this.warpsPerBlock = Math.ceil(this.threads / 32);
    this.blocks = launch.grid.x * launch.grid.y;
    this.global = initial.map((a) => Float64Array.from(a));
    this.shared = Array.from({ length: this.blocks }, () => new Float64Array(kernel.sharedWords).fill(NaN));
    const nwarps = this.blocks * this.warpsPerBlock;
    const counts = new Int32Array(nwarps);
    for (let s = 0; s < trace.steps; s++) counts[trace.warp[s]]++;
    this.warpSteps = Array.from(counts, (n) => new Int32Array(n));
    counts.fill(0);
    this.retSteps = Array.from({ length: nwarps }, () => []);
    const live = Array.from({ length: nwarps }, (_, w) => this.existsMask(w % this.warpsPerBlock));
    for (let s = 0; s < trace.steps; s++) {
      const w = trace.warp[s];
      this.warpSteps[w][counts[w]++] = s;
      if (kernel.code[trace.pc[s]].op === 'ret') {
        live[w] &= ~trace.mask[s];
        this.retSteps[w].push({ step: s, live: live[w] >>> 0 });
      }
    }
  }

  existsMask(warpIndex: number): number {
    const n = Math.min(32, this.threads - warpIndex * 32);
    return n >= 32 ? 0xffffffff : ((1 << n) - 1) >>> 0;
  }

  blockOfStep(s: number): number {
    return Math.floor(this.trace.warp[s] / this.warpsPerBlock);
  }

  /** First step issued by any warp of a block, or -1. */
  firstStepOfBlock(block: number): number {
    let best = -1;
    for (let w = 0; w < this.warpsPerBlock; w++) {
      const steps = this.warpSteps[block * this.warpsPerBlock + w];
      if (steps.length && (best < 0 || steps[0] < best)) best = steps[0];
    }
    return best;
  }

  /** Bring memory to the state right after step `s` (-1 for the initial state). */
  seek(s: number): void {
    const t = this.trace;
    while (this.cursor < s) {
      const k = ++this.cursor;
      for (let a = t.accStart[k]; a < t.accStart[k + 1]; a++) {
        if (t.aKind[a] !== READ) this.target(k, a)[this.slot(a)] = t.aNew[a];
      }
    }
    while (this.cursor > s) {
      const k = this.cursor--;
      for (let a = t.accStart[k + 1] - 1; a >= t.accStart[k]; a--) {
        if (t.aKind[a] === READ) continue;
        const old = t.aOld[a];
        this.target(k, a)[this.slot(a)] = t.aBuf[a] < 0 && isGarbage(old) ? NaN : old;
      }
    }
  }

  private target(step: number, a: number): Float64Array {
    const buf = this.trace.aBuf[a];
    return buf >= 0 ? this.global[buf] : this.shared[this.blockOfStep(step)];
  }

  private slot(a: number): number {
    const buf = this.trace.aBuf[a];
    const idx = this.trace.aIdx[a];
    return buf >= 0 ? idx : this.kernel.shared[-buf - 1].offset + idx;
  }

  accesses(s: number): Access[] {
    const t = this.trace;
    const out: Access[] = [];
    if (s < 0 || s >= t.steps) return out;
    for (let a = t.accStart[s]; a < t.accStart[s + 1]; a++) {
      out.push({ lane: t.aLane[a], buf: t.aBuf[a], idx: t.aIdx[a], kind: t.aKind[a], value: t.aNew[a] });
    }
    return out;
  }

  /** The latest step ≤ s issued by a warp, or -1. */
  lastStep(warp: number, s: number): number {
    const steps = this.warpSteps[warp];
    let lo = 0;
    let hi = steps.length - 1;
    let found = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (steps[mid] <= s) {
        found = steps[mid];
        lo = mid + 1;
      } else hi = mid - 1;
    }
    return found;
  }

  warpsAt(s: number, block: number): WarpView[] {
    const t = this.trace;
    const out: WarpView[] = [];
    for (let w = 0; w < this.warpsPerBlock; w++) {
      const id = block * this.warpsPerBlock + w;
      const exists = this.existsMask(w);
      const last = this.lastStep(id, s);
      let live = exists;
      for (const r of this.retSteps[id]) if (r.step <= s) live = r.live;
      if (last < 0) {
        out.push({ index: w, status: 'idle', mask: exists, live, exists, line: 0, issuing: false });
        continue;
      }
      const ins = this.kernel.code[t.pc[last]];
      const release = t.release[last];
      let status: WarpStatus = 'run';
      if (ins.op === 'bar' && (release < 0 || release > s)) status = 'bar';
      else if (live === 0) status = 'done';
      out.push({ index: w, status, mask: t.mask[last], live, exists, line: ins.line, issuing: last === s });
    }
    return out;
  }
}
