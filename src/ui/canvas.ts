/** Size a canvas for the device pixel ratio and return a context in CSS pixels. */
export function fitCanvas(canvas: HTMLCanvasElement, width: number, height: number): CanvasRenderingContext2D {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(width * dpr));
  const h = Math.max(1, Math.round(height * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return ctx;
}

/** Width inside an element's padding, for sizing a canvas to its panel. */
export function contentWidth(el: HTMLElement | null, fallback = 600): number {
  if (!el) return fallback;
  const s = getComputedStyle(el);
  return el.clientWidth - parseFloat(s.paddingLeft) - parseFloat(s.paddingRight);
}

export interface Palette {
  paper: string;
  panel: string;
  ink: string;
  ink2: string;
  ink3: string;
  rule: string;
  accent: string;
  read: string;
  diverge: string;
  atomic: string;
  ok: string;
  bad: string;
  mono: string;
}

/** Colors come from CSS custom properties so the canvases follow the theme. */
export function palette(): Palette {
  const s = getComputedStyle(document.documentElement);
  const v = (n: string) => s.getPropertyValue(n).trim();
  return {
    paper: v('--paper'),
    panel: v('--panel'),
    ink: v('--ink'),
    ink2: v('--ink-2'),
    ink3: v('--ink-3'),
    rule: v('--rule'),
    accent: v('--accent'),
    read: v('--read'),
    diverge: v('--diverge'),
    atomic: v('--atomic'),
    ok: v('--ok'),
    bad: v('--bad'),
    mono: v('--mono'),
  };
}

/** A floating label that follows the pointer over a canvas. */
export class Tooltip {
  private readonly el: HTMLDivElement;
  constructor() {
    this.el = document.createElement('div');
    this.el.className = 'tooltip';
    this.el.hidden = true;
    document.body.append(this.el);
  }
  show(text: string, x: number, y: number): void {
    this.el.textContent = text;
    this.el.hidden = false;
    const r = this.el.getBoundingClientRect();
    const left = Math.min(x + 14, window.innerWidth - r.width - 8);
    this.el.style.transform = `translate(${left}px, ${y + 16}px)`;
  }
  hide(): void {
    this.el.hidden = true;
  }
}
