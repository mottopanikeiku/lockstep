import { contentWidth, fitCanvas, palette, type Tooltip } from './canvas';
import type { WarpView } from './replay';

const LABEL = 30;
const STATUS_WIDTH = 170;
const GAP = 2;
const ROW = 20;

export interface ThreadsModel {
  warps: WarpView[];
  block: { x: number; y: number };
  blockDim: { x: number; y: number };
}

const STATUS_TEXT: Record<WarpView['status'], string> = {
  idle: 'not started',
  run: '',
  bar: 'waiting at __syncthreads',
  done: 'finished',
};

/**
 * One row per warp of the current block, one cell per lane. Active lanes of
 * the issuing warp are filled; lanes that are alive but masked off by a
 * branch are hatched; lanes that returned are empty.
 */
export class ThreadsView {
  private model: ThreadsModel | null = null;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly tooltip: Tooltip,
    private readonly onLane: (warp: number, lane: number) => void,
  ) {
    canvas.addEventListener('pointermove', (e) => this.hover(e));
    canvas.addEventListener('pointerleave', () => this.tooltip.hide());
    canvas.addEventListener('click', (e) => {
      const hit = this.hit(e);
      if (hit) this.onLane(hit.warp, hit.lane);
    });
  }

  set(model: ThreadsModel | null): void {
    this.model = model;
    this.render();
  }

  /** Lane cell size, fitted to the panel width. */
  private cell = 13;

  render(): void {
    const m = this.model;
    const avail = contentWidth(this.canvas.parentElement, 640);
    const cell = (this.cell = Math.max(7, Math.min(14, Math.floor((avail - LABEL - STATUS_WIDTH) / 32) - GAP)));
    const pitch = cell + GAP;
    const rows = m ? m.warps.length : 1;
    const ctx = fitCanvas(this.canvas, LABEL + 32 * pitch + STATUS_WIDTH, rows * ROW + 4);
    const p = palette();
    ctx.font = `11px ${p.mono}`;
    ctx.textBaseline = 'middle';
    if (!m) {
      ctx.fillStyle = p.ink3;
      ctx.fillText('No run yet.', 0, 12);
      return;
    }
    m.warps.forEach((w, row) => {
      const y = row * ROW + 2;
      ctx.fillStyle = w.issuing ? p.ink : p.ink3;
      ctx.fillText(`w${w.index}`, 0, y + cell / 2);
      for (let l = 0; l < 32; l++) {
        if (!((w.exists >>> l) & 1)) continue;
        const x = LABEL + l * pitch;
        const alive = (w.live >>> l) & 1;
        const active = (w.mask >>> l) & 1;
        ctx.lineWidth = 1;
        if (!alive || w.status === 'done') {
          ctx.strokeStyle = p.rule;
          ctx.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
        } else if (w.status === 'idle') {
          ctx.fillStyle = p.rule;
          ctx.fillRect(x, y, cell, cell);
        } else if (active) {
          ctx.fillStyle = w.issuing ? p.accent : p.ink3;
          ctx.globalAlpha = w.issuing ? 1 : w.status === 'bar' ? 0.35 : 0.55;
          ctx.fillRect(x, y, cell, cell);
          ctx.globalAlpha = 1;
        } else {
          // Alive but masked off by divergence: hatched.
          ctx.strokeStyle = p.diverge;
          ctx.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
          ctx.save();
          ctx.beginPath();
          ctx.rect(x, y, cell, cell);
          ctx.clip();
          ctx.beginPath();
          for (let k = -cell; k < cell; k += 4) {
            ctx.moveTo(x + k, y + cell);
            ctx.lineTo(x + k + cell, y);
          }
          ctx.stroke();
          ctx.restore();
        }
        if (l % 8 === 7 && l < 31) {
          ctx.fillStyle = p.rule;
          ctx.fillRect(x + cell + GAP / 2 - 0.5, y - 1, 1, cell + 2);
        }
      }
      const status = STATUS_TEXT[w.status] || (w.line ? `line ${w.line}` : '');
      ctx.fillStyle = w.issuing ? p.ink : p.ink3;
      ctx.fillText(w.issuing ? `▸ ${status}` : status, LABEL + 32 * pitch + 8, y + cell / 2);
    });
  }

  private hit(e: MouseEvent): { warp: number; lane: number } | null {
    const m = this.model;
    if (!m) return null;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left - LABEL;
    const row = Math.floor((e.clientY - rect.top - 2) / ROW);
    const lane = Math.floor(x / (this.cell + GAP));
    if (row < 0 || row >= m.warps.length || lane < 0 || lane > 31 || x < 0) return null;
    if (!((m.warps[row].exists >>> lane) & 1)) return null;
    return { warp: row, lane };
  }

  private hover(e: PointerEvent): void {
    const m = this.model;
    const hit = this.hit(e);
    if (!m || !hit) {
      this.tooltip.hide();
      this.canvas.style.cursor = '';
      return;
    }
    this.canvas.style.cursor = 'pointer';
    const tid = hit.warp * 32 + hit.lane;
    const tx = tid % m.blockDim.x;
    const ty = Math.floor(tid / m.blockDim.x);
    const t = m.blockDim.y > 1 ? `threadIdx (${tx}, ${ty})` : `threadIdx.x ${tx}`;
    this.tooltip.show(`lane ${hit.lane} · ${t} · click to inspect`, e.clientX, e.clientY);
  }
}
