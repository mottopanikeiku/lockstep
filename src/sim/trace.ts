/** Kinds of memory access recorded in a trace. */
export const READ = 0;
export const WRITE = 1;
export const ATOMIC = 2;

function grow<T extends Int32Array | Uint32Array | Float64Array | Uint8Array | Int16Array>(a: T, need: number): T {
  if (need <= a.length) return a;
  const next = new (a.constructor as new (n: number) => T)(Math.max(need, a.length * 2));
  next.set(a);
  return next;
}

/**
 * Everything the UI needs to replay a run: one entry per warp instruction
 * issued (a "step") and one per lane memory access. Stored as typed arrays
 * because runs reach a few hundred thousand steps.
 */
export class Trace {
  steps = 0;
  warp = new Int32Array(1024);
  pc = new Int32Array(1024);
  mask = new Uint32Array(1024);
  cost = new Int32Array(1024);
  /** The issuing warp's clock after this step. */
  clock = new Float64Array(1024);
  /** SIMT stack depth before the step; above 1 means the warp is diverged. */
  depth = new Uint8Array(1024);
  /** Sectors for global accesses, wavefronts for shared ones, 0 otherwise. */
  units = new Int32Array(1024);
  /** For a __syncthreads step, the step index at which the barrier released (-1 if never). */
  release = new Int32Array(1024);
  /** First access of each step; accesses of step s are accStart[s]..accStart[s+1]. */
  accStart = new Int32Array(1025);

  accesses = 0;
  aLane = new Uint8Array(4096);
  /** Global buffer index, or -(shared array index + 1). */
  aBuf = new Int16Array(4096);
  aIdx = new Int32Array(4096);
  aKind = new Uint8Array(4096);
  /** Value before the access (the value read, for reads). */
  aOld = new Float64Array(4096);
  /** Value after the access. */
  aNew = new Float64Array(4096);

  /** Step indices where a branch split a warp. */
  divergences: number[] = [];

  beginStep(warp: number, pc: number, mask: number, depth: number): number {
    const s = this.steps++;
    if (s + 1 >= this.warp.length) {
      const n = s * 2 + 2;
      this.warp = grow(this.warp, n);
      this.pc = grow(this.pc, n);
      this.mask = grow(this.mask, n);
      this.cost = grow(this.cost, n);
      this.clock = grow(this.clock, n);
      this.depth = grow(this.depth, n);
      this.units = grow(this.units, n);
      this.release = grow(this.release, n);
      this.accStart = grow(this.accStart, n + 1);
    }
    this.warp[s] = warp;
    this.pc[s] = pc;
    this.mask[s] = mask >>> 0;
    this.depth[s] = depth;
    this.release[s] = -1;
    this.accStart[s] = this.accesses;
    this.accStart[s + 1] = this.accesses;
    return s;
  }

  access(lane: number, buf: number, idx: number, kind: number, oldV: number, newV: number): void {
    const a = this.accesses++;
    if (a >= this.aLane.length) {
      const n = a * 2;
      this.aLane = grow(this.aLane, n);
      this.aBuf = grow(this.aBuf, n);
      this.aIdx = grow(this.aIdx, n);
      this.aKind = grow(this.aKind, n);
      this.aOld = grow(this.aOld, n);
      this.aNew = grow(this.aNew, n);
    }
    this.aLane[a] = lane;
    this.aBuf[a] = buf;
    this.aIdx[a] = idx;
    this.aKind[a] = kind;
    this.aOld[a] = oldV;
    this.aNew[a] = newV;
    this.accStart[this.steps] = this.accesses;
  }

  endStep(s: number, cost: number, clock: number, units: number): void {
    this.cost[s] = cost;
    this.clock[s] = clock;
    this.units[s] = units;
  }
}
