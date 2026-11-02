import { fmt } from '../game/attempt';
import type { SharedArray } from '../ir/ir';
import { sharedName } from '../sim/machine';
import { ATOMIC, READ, WRITE } from '../sim/trace';
import { fitCanvas, palette, type Palette, type Tooltip } from './canvas';
import { isGarbage, type Access, type Replay } from './replay';

export interface GlobalArrayInfo {
  name: string;
  type: 'int' | 'float';
  shape: number[];
  output: boolean;
  buf: number;
}

interface Grid {
  rows: number;
  cols: number;
  cw: number;
  ch: number;
  /** Cell position of element i. */
  at: (i: number) => [number, number];
  /** Element index at a cell, or -1. */
  index: (row: number, col: number) => number;
}

const MAX_HEIGHT = 140;

function makeGrid(n: number, cols: number, first: number, width: number): Grid {
  const rows = Math.ceil((first + n) / cols);
  const cw = Math.max(3, Math.min(30, Math.floor(width / cols)));
  const ch = rows * cw <= MAX_HEIGHT ? cw : Math.max(3, Math.floor(MAX_HEIGHT / rows));
  return {
    rows,
    cols,
    cw,
    ch,
    at: (i) => [Math.floor((first + i) / cols), (first + i) % cols],
    index: (r, c) => {
      const i = r * cols + c - first;
      return i >= 0 && i < n ? i : -1;
    },
  };
}

const KIND_COLOR = (p: Palette, kind: number) => (kind === READ ? p.read : kind === WRITE ? p.accent : p.atomic);

/**
 * Global arrays and the current block's shared memory. Cells touched by the
 * current step are colored (blue read, orange write, violet atomic); global
 * accesses also shade the whole 32-byte sectors they pulled in, and shared
 * memory is drawn 32 words wide so each column is one bank.
 */
export class MemoryView {
  private replay: Replay | null = null;
  private arrays: GlobalArrayInfo[] = [];
  private step = -1;
  private block = 0;
  private canvases: { canvas: HTMLCanvasElement; caption: HTMLElement; kind: 'global' | 'shared'; index: number }[] = [];
  private grids = new Map<HTMLCanvasElement, Grid>();

  constructor(
    private readonly root: HTMLElement,
    private readonly tooltip: Tooltip,
  ) {}

  setRun(replay: Replay | null, arrays: GlobalArrayInfo[]): void {
    this.replay = replay;
    this.arrays = arrays;
    this.root.replaceChildren();
    this.canvases = [];
    if (!replay) {
      const p = document.createElement('p');
      p.className = 'muted';
      p.textContent = 'Memory shows up after a run.';
      this.root.append(p);
      return;
    }
    const add = (kind: 'global' | 'shared', index: number, title: string, note: string) => {
      const fig = document.createElement('figure');
      fig.className = 'mem';
      const cap = document.createElement('figcaption');
      const name = document.createElement('code');
      name.textContent = title;
      const meta = document.createElement('span');
      meta.className = 'muted';
      meta.textContent = note;
      const live = document.createElement('span');
      live.className = 'mem-live';
      cap.append(name, meta, live);
      const canvas = document.createElement('canvas');
      canvas.addEventListener('pointermove', (e) => this.hover(e, canvas, kind, index));
      canvas.addEventListener('pointerleave', () => this.tooltip.hide());
      fig.append(cap, canvas);
      this.root.append(fig);
      this.canvases.push({ canvas, caption: live, kind, index });
    };
    // Shared memory first: it is the block's working set and changes from step to step.
    replay.kernel.shared.forEach((s, i) => {
      add('shared', i, s.name, `__shared__ ${s.type} ${s.dims.length ? s.dims.join('×') : 'scalar'} · 32 banks across`);
    });
    for (const [i, a] of arrays.entries()) {
      add('global', i, a.name, `${a.type} ${a.shape.join('×')} · ${a.output ? 'output' : 'input'}`);
    }
  }

  setStep(step: number, block: number): void {
    this.step = step;
    this.block = block;
    this.render();
  }

  render(): void {
    const r = this.replay;
    if (!r) return;
    r.seek(this.step);
    const accesses = r.accesses(this.step);
    const p = palette();
    for (const c of this.canvases) {
      const width = (c.canvas.parentElement?.clientWidth ?? 600) - 2;
      if (c.kind === 'global') this.drawGlobal(c.canvas, c.caption, this.arrays[c.index], accesses, width, p);
      else this.drawShared(c.canvas, c.caption, r.kernel.shared[c.index], c.index, accesses, width, p);
    }
  }

  private drawGlobal(canvas: HTMLCanvasElement, caption: HTMLElement, a: GlobalArrayInfo, acc: Access[], width: number, p: Palette): void {
    const values = this.replay!.global[a.buf];
    const n = values.length;
    const cols = a.shape.length === 2 ? a.shape[1] : Math.min(32, n);
    const g = makeGrid(n, cols, 0, width);
    this.grids.set(canvas, g);
    const ctx = this.cells(canvas, g, p, (i) => values[i]);
    const mine = acc.filter((x) => x.buf === a.buf);
    // Whole sectors the warp pulled in, then the elements it asked for.
    const sectors = new Set(mine.map((x) => Math.floor(x.idx / 8)));
    if (mine.length) {
      ctx.fillStyle = KIND_COLOR(p, mine[0].kind);
      ctx.globalAlpha = 0.14;
      for (const s of sectors) {
        for (let i = s * 8; i < Math.min(n, s * 8 + 8); i++) {
          const [row, col] = g.at(i);
          ctx.fillRect(col * g.cw, row * g.ch, g.cw, g.ch);
        }
      }
      ctx.globalAlpha = 1;
    }
    this.marks(ctx, g, mine, p);
    if (mine.length) {
      const used = new Set(mine.map((x) => x.idx)).size * 4;
      caption.textContent = `${sectors.size} sector${sectors.size === 1 ? '' : 's'} · ${used} of ${sectors.size * 32} bytes used`;
    } else caption.textContent = '';
  }

  private drawShared(
    canvas: HTMLCanvasElement,
    caption: HTMLElement,
    s: SharedArray,
    index: number,
    acc: Access[],
    width: number,
    p: Palette,
  ): void {
    const words = this.replay!.shared[this.block];
    const g = makeGrid(s.size, 32, s.offset % 32, width);
    this.grids.set(canvas, g);
    const ctx = this.cells(canvas, g, p, (i) => words[s.offset + i]);
    const mine = acc.filter((x) => x.buf === -(index + 1));
    this.marks(ctx, g, mine, p);
    if (!mine.length) {
      caption.textContent = '';
      return;
    }
    // Distinct words per bank; atomics count every lane.
    const perBank = new Map<number, Set<number>>();
    mine.forEach((x, k) => {
      const w = s.offset + x.idx;
      const bank = w % 32;
      if (!perBank.has(bank)) perBank.set(bank, new Set());
      perBank.get(bank)!.add(x.kind === ATOMIC ? w * 64 + k : w);
    });
    let worst = 1;
    for (const [bank, set] of perBank) {
      if (set.size < 2) continue;
      worst = Math.max(worst, set.size);
      ctx.fillStyle = p.bad;
      ctx.fillRect(bank * g.cw, 0, g.cw, 2);
    }
    caption.textContent = worst > 1 ? `${worst}-way bank conflict` : 'no bank conflict';
    caption.classList.toggle('bad', worst > 1);
  }

  /**
   * Draw the cell grid: numbers when cells are big enough, otherwise a
   * shade per cell scaled to the array's range, so big arrays still read
   * as an image of their contents. Returns the context for overlays.
   */
  private cells(canvas: HTMLCanvasElement, g: Grid, p: Palette, value: (i: number) => number): CanvasRenderingContext2D {
    const ctx = fitCanvas(canvas, g.cols * g.cw + 1, g.rows * g.ch + 1);
    const showText = g.cw >= 18 && g.ch >= 13;
    const maxChars = Math.max(2, Math.floor((g.cw - 2) / 5.5));
    let lo = Infinity;
    let hi = -Infinity;
    if (!showText) {
      for (let r = 0; r < g.rows; r++) {
        for (let c = 0; c < g.cols; c++) {
          const i = g.index(r, c);
          const v = i < 0 ? NaN : value(i);
          if (i < 0 || isGarbage(v)) continue;
          lo = Math.min(lo, v);
          hi = Math.max(hi, v);
        }
      }
    }
    ctx.font = `9px ${p.mono}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = p.rule;
    ctx.fillRect(0, 0, g.cols * g.cw + 1, g.rows * g.ch + 1);
    for (let r = 0; r < g.rows; r++) {
      for (let c = 0; c < g.cols; c++) {
        const i = g.index(r, c);
        if (i < 0) {
          ctx.clearRect(c * g.cw, r * g.ch, g.cw + 1, g.ch + 1);
          continue;
        }
        const v = value(i);
        const gap = g.cw >= 6 && g.ch >= 6 ? 1 : 0;
        ctx.fillStyle = p.panel;
        ctx.fillRect(c * g.cw + gap, r * g.ch + gap, g.cw - gap, g.ch - gap);
        if (showText) {
          ctx.fillStyle = p.ink2;
          const text = isGarbage(v) ? '·' : fmt(v);
          ctx.fillText(text.length > maxChars ? text.slice(0, maxChars) : text, c * g.cw + g.cw / 2, r * g.ch + g.ch / 2 + 0.5);
        } else if (!isGarbage(v) && hi > lo) {
          ctx.fillStyle = p.ink3;
          ctx.globalAlpha = 0.08 + (0.6 * (v - lo)) / (hi - lo);
          ctx.fillRect(c * g.cw + gap, r * g.ch + gap, g.cw - gap, g.ch - gap);
          ctx.globalAlpha = 1;
        }
      }
    }
    ctx.strokeStyle = p.rule;
    ctx.strokeRect(0.5, 0.5, g.cols * g.cw, g.rows * g.ch);
    return ctx;
  }

  private marks(ctx: CanvasRenderingContext2D, g: Grid, mine: Access[], p: Palette): void {
    for (const x of mine) {
      const [row, col] = g.at(x.idx);
      ctx.fillStyle = KIND_COLOR(p, x.kind);
      ctx.globalAlpha = 0.5;
      ctx.fillRect(col * g.cw, row * g.ch, g.cw, g.ch);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = KIND_COLOR(p, x.kind);
      ctx.strokeRect(col * g.cw + 0.5, row * g.ch + 0.5, g.cw - 1, g.ch - 1);
    }
  }

  private hover(e: PointerEvent, canvas: HTMLCanvasElement, kind: 'global' | 'shared', index: number): void {
    const g = this.grids.get(canvas);
    const r = this.replay;
    if (!g || !r) return;
    const rect = canvas.getBoundingClientRect();
    const col = Math.floor((e.clientX - rect.left) / g.cw);
    const row = Math.floor((e.clientY - rect.top) / g.ch);
    const i = col < g.cols ? g.index(row, col) : -1;
    if (i < 0) {
      this.tooltip.hide();
      return;
    }
    if (kind === 'global') {
      const a = this.arrays[index];
      const v = r.global[a.buf][i];
      const where = a.shape.length === 2 ? ` (row ${Math.floor(i / a.shape[1])}, col ${i % a.shape[1]})` : '';
      this.tooltip.show(`${a.name}[${i}]${where} = ${fmt(v)} · sector ${Math.floor(i / 8)}`, e.clientX, e.clientY);
    } else {
      const s = r.kernel.shared[index];
      const v = r.shared[this.block][s.offset + i];
      const text = isGarbage(v) ? 'never written' : fmt(v);
      this.tooltip.show(`${sharedName(s, i)} = ${text} · bank ${(s.offset + i) % 32}`, e.clientX, e.clientY);
    }
  }
}
