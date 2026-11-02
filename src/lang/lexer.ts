import { CompileError, type Diagnostic } from './diagnostics';

export type TokenKind = 'ident' | 'int' | 'float' | 'punct' | 'eof';

export interface Token {
  kind: TokenKind;
  text: string;
  /** Numeric value for int and float literals. */
  value: number;
  /** A float literal without an `f` suffix, which CUDA treats as double. */
  double: boolean;
  line: number;
  col: number;
}

const PUNCTS = [
  '<<=', '>>=',
  '++', '--', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<', '>>', '<=', '>=', '==', '!=', '&&', '||', '->', '::',
  '+', '-', '*', '/', '%', '&', '|', '^', '~', '!', '<', '>', '=', '?', ':', ';', ',', '.', '(', ')', '[', ']', '{', '}',
];

const isIdent = (c: string) => /[A-Za-z0-9_]/.test(c);
const isDigit = (c: string) => c >= '0' && c <= '9';

/** Tokenize one run of source text. Preprocessor lines are handled by `lex`. */
function scan(src: string, line0: number, col0: number, out: Token[], warnings: Diagnostic[]): void {
  let i = 0;
  let line = line0;
  let col = col0;
  const n = src.length;
  const advance = (k: number) => {
    for (let j = 0; j < k; j++) {
      if (src[i] === '\n') {
        line++;
        col = 1;
      } else {
        col++;
      }
      i++;
    }
  };
  while (i < n) {
    const c = src[i];
    if (c === ' ' || c === '\t' || c === '\r' || c === '\n') {
      advance(1);
      continue;
    }
    if (c === '/' && src[i + 1] === '/') {
      while (i < n && src[i] !== '\n') advance(1);
      continue;
    }
    if (c === '/' && src[i + 1] === '*') {
      const start = { line, col };
      advance(2);
      while (i < n && !(src[i] === '*' && src[i + 1] === '/')) advance(1);
      if (i >= n) throw new CompileError('unterminated /* comment', start);
      advance(2);
      continue;
    }
    const tokLine = line;
    const tokCol = col;
    if (/[A-Za-z_]/.test(c)) {
      let j = i;
      while (j < n && isIdent(src[j])) j++;
      const text = src.slice(i, j);
      out.push({ kind: 'ident', text, value: 0, double: false, line: tokLine, col: tokCol });
      advance(j - i);
      continue;
    }
    if (isDigit(c) || (c === '.' && isDigit(src[i + 1] ?? ''))) {
      const tok = scanNumber(src, i, tokLine, tokCol, warnings);
      out.push(tok);
      advance(tok.text.length);
      continue;
    }
    if (c === '"' || c === "'") {
      throw new CompileError('string and character literals are not supported', { line, col });
    }
    const p = PUNCTS.find((s) => src.startsWith(s, i));
    if (!p) throw new CompileError(`unexpected character '${c}'`, { line, col });
    out.push({ kind: 'punct', text: p, value: 0, double: false, line: tokLine, col: tokCol });
    advance(p.length);
  }
}

function scanNumber(src: string, start: number, line: number, col: number, warnings: Diagnostic[]): Token {
  let j = start;
  const loc = { line, col };
  if (src[j] === '0' && (src[j + 1] === 'x' || src[j + 1] === 'X')) {
    j += 2;
    const digitsStart = j;
    while (j < src.length && /[0-9a-fA-F]/.test(src[j])) j++;
    if (j === digitsStart) throw new CompileError('hex literal needs digits after 0x', loc);
    const value = parseInt(src.slice(digitsStart, j), 16);
    while (j < src.length && /[uUlL]/.test(src[j])) j++;
    if (value > 0xffffffff) throw new CompileError('integer literal does not fit in 32 bits', loc);
    if (isIdent(src[j] ?? '')) throw new CompileError(`invalid suffix '${src[j]}' on integer literal`, loc);
    return { kind: 'int', text: src.slice(start, j), value: value | 0, double: false, line, col };
  }
  let isFloat = false;
  while (j < src.length && isDigit(src[j])) j++;
  if (src[j] === '.') {
    isFloat = true;
    j++;
    while (j < src.length && isDigit(src[j])) j++;
  }
  if (src[j] === 'e' || src[j] === 'E') {
    let k = j + 1;
    if (src[k] === '+' || src[k] === '-') k++;
    if (!isDigit(src[k] ?? '')) throw new CompileError('exponent has no digits', loc);
    while (k < src.length && isDigit(src[k])) k++;
    j = k;
    isFloat = true;
  }
  const body = src.slice(start, j);
  if (src[j] === 'f' || src[j] === 'F') {
    if (!isFloat) throw new CompileError(`'${body}f' is not a valid literal; write ${body}.0f`, loc);
    j++;
    if (isIdent(src[j] ?? '')) throw new CompileError(`invalid suffix on float literal`, loc);
    return { kind: 'float', text: src.slice(start, j), value: Math.fround(parseFloat(body)), double: false, line, col };
  }
  if (isFloat) {
    if (isIdent(src[j] ?? '')) throw new CompileError(`invalid suffix '${src[j]}' on float literal`, loc);
    warnings.push({
      severity: 'warning',
      message: `${body} is a double literal in CUDA; write ${body}f to keep the math in float`,
      line,
      col,
    });
    return { kind: 'float', text: body, value: Math.fround(parseFloat(body)), double: true, line, col };
  }
  while (j < src.length && /[uUlL]/.test(src[j])) j++;
  if (isIdent(src[j] ?? '')) throw new CompileError(`invalid suffix '${src[j]}' on integer literal`, loc);
  const value = parseInt(body, 10);
  if (value > 0xffffffff) throw new CompileError('integer literal does not fit in 32 bits', loc);
  return { kind: 'int', text: src.slice(start, j), value: value | 0, double: false, line, col };
}

export interface LexResult {
  tokens: Token[];
  warnings: Diagnostic[];
}

/**
 * Tokenize a CUDA source file. Handles comments, `#include` and `#pragma`
 * (ignored) and object-like `#define`, which is expanded in place.
 */
export function lex(src: string): LexResult {
  const warnings: Diagnostic[] = [];
  const macros = new Map<string, Token[]>();
  const raw: Token[] = [];
  const lines = src.split('\n');
  let chunk: string[] = [];
  let chunkLine = 1;
  const flush = (nextLine: number) => {
    if (chunk.length) scan(chunk.join('\n'), chunkLine, 1, raw, warnings);
    chunk = [];
    chunkLine = nextLine;
  };
  let inBlockComment = false;
  for (let li = 0; li < lines.length; li++) {
    const text = lines[li];
    const trimmed = text.trimStart();
    if (!inBlockComment && trimmed.startsWith('#')) {
      flush(li + 2);
      const col = text.length - trimmed.length + 1;
      directive(trimmed, li + 1, col, macros, warnings);
      continue;
    }
    chunk.push(text);
    inBlockComment = tracksBlockComment(text, inBlockComment);
  }
  flush(lines.length + 1);
  const last = raw[raw.length - 1];
  const tokens = expand(raw, macros);
  tokens.push({
    kind: 'eof',
    text: '',
    value: 0,
    double: false,
    line: last ? last.line : 1,
    col: last ? last.col + last.text.length : 1,
  });
  return { tokens, warnings };
}

/** Whether a line leaves us inside a block comment, so `#` lines inside comments are not directives. */
function tracksBlockComment(line: string, inside: boolean): boolean {
  let i = 0;
  while (i < line.length) {
    if (inside) {
      const end = line.indexOf('*/', i);
      if (end < 0) return true;
      inside = false;
      i = end + 2;
    } else {
      const lc = line.indexOf('//', i);
      const bc = line.indexOf('/*', i);
      if (bc < 0 || (lc >= 0 && lc < bc)) return false;
      inside = true;
      i = bc + 2;
    }
  }
  return inside;
}

function directive(
  text: string,
  line: number,
  col: number,
  macros: Map<string, Token[]>,
  warnings: Diagnostic[],
): void {
  const m = /^#\s*([A-Za-z_]*)/.exec(text)!;
  const name = m[1];
  const rest = text.slice(m[0].length);
  if (name === 'include' || name === 'pragma' || name === '') return;
  if (name === 'define') {
    const dm = /^\s+([A-Za-z_][A-Za-z0-9_]*)/.exec(rest);
    if (!dm) throw new CompileError('#define needs a name', { line, col });
    const after = rest.slice(dm[0].length);
    if (after.startsWith('(')) {
      throw new CompileError('function-like macros are not supported; use a const int or write the expression out', {
        line,
        col,
      });
    }
    const body: Token[] = [];
    const bodyCol = col + m[0].length + dm[0].length;
    scan(after.replace(/\/\/.*$/, ''), line, bodyCol, body, warnings);
    macros.set(dm[1], body);
    return;
  }
  if (name === 'undef') {
    const um = /^\s+([A-Za-z_][A-Za-z0-9_]*)/.exec(rest);
    if (um) macros.delete(um[1]);
    return;
  }
  throw new CompileError(`#${name} is not supported (only #define, #include and #pragma)`, { line, col });
}

function expand(tokens: Token[], macros: Map<string, Token[]>): Token[] {
  if (macros.size === 0) return tokens;
  const out: Token[] = [];
  const emit = (tok: Token, at: Token, hidden: Set<string>) => {
    const body = tok.kind === 'ident' ? macros.get(tok.text) : undefined;
    if (body && !hidden.has(tok.text)) {
      const inner = new Set(hidden).add(tok.text);
      for (const b of body) emit(b, at, inner);
      return;
    }
    out.push({ ...tok, line: at.line, col: at.col });
  };
  for (const tok of tokens) {
    if (tok.kind === 'ident' && macros.has(tok.text)) emit(tok, tok, new Set());
    else out.push(tok);
  }
  return out;
}
