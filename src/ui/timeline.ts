import { contentWidth, fitCanvas, palette, type Tooltip } from './canvas';
import type { Replay } from './replay';

const LEFT = 34;
const TOP = 10;

/**
 * Warps down the side, issued instructions left to right. A cell's strength
 * is how many lanes were active; amber cells are a diverged warp.
 */
export class Timeline {
  private replay: Replay | null = null;
  private faultStep = -1;
  private step = 0;
  private width = 0;
  private lanes = new Uint8Array(0);
  private diverged = new Uint8Array(0);
  private dragging = false;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly tooltip: Tooltip,
    private readonly onSeek: (step: number) => void,
  ) {
    canvas.addEventListener('pointerdown', (e) => {
      if (!this.replay) return;
      this.dragging = true;
      canvas.setPointerCapture(e.pointerId);
      this.seekTo(e);
    });
    canvas.addEventListener('pointermove', (e) => {
      if (this.dragging) this.seekTo(e);
      this.hover(e);
    });
    canvas.addEventListener('pointerup', () => (this.dragging = false));
    canvas.addEventListener('pointerleave', () => this.tooltip.hide());
    canvas.addEventListener('keydown', (e) => {
      if (!this.replay) return;
      const n = this.replay.trace.steps;
      const big = Math.max(1, Math.round(n / 20));
      const moves: Record<string, number> = {
        ArrowRight: e.shiftKey ? 10 : 1,
        ArrowLeft: e.shiftKey ? -10 : -1,
        PageDown: big,
        PageUp: -big,
        Home: -n,
        End: n,
      };
      if (!(e.key in moves)) return;
      e.preventDefault();
      this.onSeek(Math.max(0, Math.min(n - 1, this.step + moves[e.key])));
    });
  }

  setRun(replay: Replay | null, faultStep: number): void {
    this.replay = replay;
    this.faultStep = faultStep;
    this.width = 0;
  }

  setStep(step: number): void {
    this.step = step;
    this.render();
  }

  private get rows(): number {
    const r = this.replay;
    return r ? r.blocks * r.warpsPerBlock : 0;
  }

  private rowHeight(): number {
    return Math.max(2, Math.min(12, Math.floor(150 / Math.max(1, this.rows))));
  }

  private plotWidth(): number {
    return Math.max(50, contentWidth(this.canvas.parentElement) - LEFT - 4);
  }

  /**
   * Bucket steps into pixel columns. A warp counts as busy from one of its
   * instructions until its next one (it is waiting for its turn), except
   * while it sits at a barrier, which shows as a gap.
   */
  private bucket(width: number): void {
    const r = this.replay!;
    const t = r.trace;
    this.width = width;
    this.lanes = new Uint8Array(width * this.rows);
    this.diverged = new Uint8Array(width * this.rows);
    const next = new Int32Array(t.steps).fill(-1);
    const last = new Int32Array(this.rows).fill(-1);
    for (let s = 0; s < t.steps; s++) {
      const w = t.warp[s];
      if (last[w] >= 0) next[last[w]] = s;
      last[w] = s;
    }
    const xOf = (s: number) => Math.floor((s * width) / t.steps);
    for (let s = 0; s < t.steps; s++) {
      let m = t.mask[s];
      let n = 0;
      while (m) {
        m &= m - 1;
        n++;
      }
      const x0 = xOf(s);
      const waits = r.kernel.code[t.pc[s]].op === 'bar';
      const end = next[s] < 0 ? xOf(s + 1) : waits ? x0 + 1 : xOf(next[s]);
      const x1 = Math.max(x0 + 1, end);
      const row = t.warp[s] * width;
      for (let x = x0; x < Math.min(x1, width); x++) {
        if (n > this.lanes[row + x]) this.lanes[row + x] = n;
        if (t.depth[s] > 1) this.diverged[row + x] = 1;
      }
    }
  }

  /** Drop the cached drawing, e.g. after a theme change. */
  invalidate(): void {
    this.cache = null;
  }

  private cache: HTMLCanvasElement | null = null;

  render(): void {
    const width = this.plotWidth();
    const rh = this.rowHeight();
    const height = TOP + this.rows * rh + 18;
    const cssWidth = LEFT + width + 4;
    const ctx = fitCanvas(this.canvas, cssWidth, this.replay ? height : 60);
    const p = palette();
    ctx.font = `10px ${p.mono}`;
    const r = this.replay;
    if (!r) {
      ctx.fillStyle = p.ink3;
      ctx.fillText('Run the kernel to see its warps here.', LEFT, 34);
      return;
    }
    if (this.width !== width || !this.cache) {
      if (this.width !== width) this.bucket(width);
      this.cache = document.createElement('canvas');
      this.drawStatic(fitCanvas(this.cache, cssWidth, height), width, rh);
    }
    ctx.drawImage(this.cache, 0, 0, cssWidth, height);
    const t = r.trace;
    const cx = LEFT + Math.floor((this.step * width) / t.steps) + 0.5;
    // Steps after the cursor have not happened yet, so fade them and let a played run fill the chart in.
    ctx.fillStyle = p.panel;
    ctx.globalAlpha = 0.7;
    ctx.fillRect(cx + 0.5, TOP - 1, LEFT + width - cx, this.rows * rh + 1);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = p.ink;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, TOP - 4);
    ctx.lineTo(cx, TOP + this.rows * rh + 2);
    ctx.stroke();
    ctx.fillStyle = p.ink;
    ctx.fillRect(LEFT - 6, TOP + t.warp[this.step] * rh, 4, Math.max(2, rh - 1));
  }

  private drawStatic(ctx: CanvasRenderingContext2D, width: number, rh: number): void {
    const r = this.replay!;
    const t = r.trace;
    const p = palette();
    ctx.font = `10px ${p.mono}`;
    // Block bands and labels.
    for (let b = 0; b < r.blocks; b++) {
      const y = TOP + b * r.warpsPerBlock * rh;
      if (b % 2 === 1) {
        ctx.fillStyle = p.rule;
        ctx.globalAlpha = 0.35;
        ctx.fillRect(LEFT, y, width, r.warpsPerBlock * rh);
        ctx.globalAlpha = 1;
      }
      if (r.warpsPerBlock * rh >= 10) {
        ctx.fillStyle = p.ink3;
        ctx.fillText(`b${b}`, 4, y + Math.min(r.warpsPerBlock * rh, 20) / 2 + 3);
      }
    }
    // Issued instructions.
    for (let row = 0; row < this.rows; row++) {
      const y = TOP + row * rh;
      for (let x = 0; x < width; x++) {
        const n = this.lanes[row * width + x];
        if (!n) continue;
        ctx.fillStyle = this.diverged[row * width + x] ? p.diverge : p.accent;
        ctx.globalAlpha = 0.25 + (0.75 * n) / 32;
        ctx.fillRect(LEFT + x, y, 1, Math.max(1, rh - 1));
      }
    }
    // Barrier releases: a hairline across the block that waited.
    ctx.strokeStyle = p.ink2;
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.5;
    const seen = new Set<number>();
    for (let s = 0; s < t.steps; s++) {
      const rel = t.release[s];
      if (rel < 0 || seen.has(rel)) continue;
      seen.add(rel);
      const b = Math.floor(t.warp[s] / r.warpsPerBlock);
      const x = LEFT + Math.floor((rel * width) / t.steps) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, TOP + b * r.warpsPerBlock * rh);
      ctx.lineTo(x, TOP + (b + 1) * r.warpsPerBlock * rh);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    if (this.faultStep >= 0) {
      const x = LEFT + Math.floor((this.faultStep * width) / t.steps);
      ctx.fillStyle = p.bad;
      ctx.beginPath();
      ctx.moveTo(x - 5, 0);
      ctx.lineTo(x + 5, 0);
      ctx.lineTo(x, 8);
      ctx.fill();
    }
    ctx.fillStyle = p.ink3;
    const axisY = TOP + this.rows * rh + 13;
    ctx.fillText('0', LEFT, axisY);
    const end = `${t.steps.toLocaleString('en-US')} warp instructions`;
    ctx.fillText(end, LEFT + width - ctx.measureText(end).width, axisY);
  }

  private stepAt(e: PointerEvent): number {
    const r = this.replay!;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left - LEFT;
    return Math.max(0, Math.min(r.trace.steps - 1, Math.round((x / this.width) * r.trace.steps)));
  }

  private seekTo(e: PointerEvent): void {
    this.onSeek(this.stepAt(e));
  }

  private hover(e: PointerEvent): void {
    const r = this.replay;
    if (!r) return;
    const rect = this.canvas.getBoundingClientRect();
    const row = Math.floor((e.clientY - rect.top - TOP) / this.rowHeight());
    if (row < 0 || row >= this.rows || e.clientX - rect.left < LEFT) {
      this.tooltip.hide();
      return;
    }
    const block = Math.floor(row / r.warpsPerBlock);
    this.tooltip.show(`block ${block} · warp ${row % r.warpsPerBlock} · step ${this.stepAt(e).toLocaleString('en-US')}`, e.clientX, e.clientY);
  }
}
