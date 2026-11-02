export interface Loc {
  line: number;
  col: number;
}

export interface Diagnostic {
  severity: 'error' | 'warning';
  message: string;
  line: number;
  col: number;
}

export class CompileError extends Error {
  readonly line: number;
  readonly col: number;
  constructor(message: string, loc: Loc) {
    super(message);
    this.line = loc.line;
    this.col = loc.col;
  }
}

/** Closest candidate within a small edit distance, for "did you mean" hints. */
export function suggest(name: string, candidates: Iterable<string>): string | undefined {
  let best: string | undefined;
  let bestDist = Math.max(1, Math.floor(name.length / 3)) + 1;
  for (const c of candidates) {
    const d = editDistance(name.toLowerCase(), c.toLowerCase());
    if (d < bestDist) {
      bestDist = d;
      best = c;
    }
  }
  return best;
}

function editDistance(a: string, b: string): number {
  const prev = new Array<number>(b.length + 1);
  for (let j = 0; j <= b.length; j++) prev[j] = j;
  for (let i = 1; i <= a.length; i++) {
    let diag = prev[0];
    prev[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const up = prev[j];
      prev[j] = Math.min(up + 1, prev[j - 1] + 1, diag + (a[i - 1] === b[j - 1] ? 0 : 1));
      diag = up;
    }
  }
  return prev[b.length];
}
