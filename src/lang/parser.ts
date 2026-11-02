import type { AssignOp, BinaryOp, Declarator, Expr, GlobalConst, Kernel, Param, Program, ScalarType, Stmt } from './ast';
import { CompileError, type Loc } from './diagnostics';
import type { Token } from './lexer';

const BINARY_PREC: Record<string, number> = {
  '||': 1,
  '&&': 2,
  '|': 3,
  '^': 4,
  '&': 5,
  '==': 6,
  '!=': 6,
  '<': 7,
  '<=': 7,
  '>': 7,
  '>=': 7,
  '<<': 8,
  '>>': 8,
  '+': 9,
  '-': 9,
  '*': 10,
  '/': 10,
  '%': 10,
};

const ASSIGN_OPS: Record<string, true> = {
  '=': true, '+=': true, '-=': true, '*=': true, '/=': true, '%=': true,
  '&=': true, '|=': true, '^=': true, '<<=': true, '>>=': true,
};

const SCALAR_TYPES: Record<string, true> = { int: true, float: true, bool: true };

const UNSUPPORTED_TYPES: Record<string, string> = {
  unsigned: 'unsigned types are not supported; use int',
  double: 'double is not supported; use float',
  long: 'long is not supported; use int',
  short: 'short is not supported; use int',
  char: 'char is not supported; use int',
  size_t: 'size_t is not supported; use int',
  half: 'half is not supported; use float',
  __half: 'half is not supported; use float',
  auto: 'auto is not supported; write the type (int, float or bool)',
  int2: 'vector types are not supported; use separate int variables',
  float2: 'vector types are not supported; use separate float variables',
  float4: 'vector types are not supported; use separate float variables',
  dim3: 'dim3 is not supported inside kernels; use threadIdx.x and friends directly',
};

/** Keywords that start a declaration inside a function body. */
const DECL_START: Record<string, true> = {
  int: true, float: true, bool: true, const: true, constexpr: true,
  static: true, volatile: true, __shared__: true, extern: true,
};

export class Parser {
  private pos = 0;

  constructor(private readonly toks: Token[]) {}

  private get tok(): Token {
    return this.toks[this.pos];
  }

  private peek(k = 1): Token {
    return this.toks[Math.min(this.pos + k, this.toks.length - 1)];
  }

  private is(text: string): boolean {
    const t = this.tok;
    return (t.kind === 'punct' || t.kind === 'ident') && t.text === text;
  }

  private next(): Token {
    const t = this.tok;
    if (t.kind !== 'eof') this.pos++;
    return t;
  }

  private accept(text: string): boolean {
    if (this.is(text)) {
      this.pos++;
      return true;
    }
    return false;
  }

  /** Location just after the previous token, where a missing `;` or `)` belongs. */
  private afterPrev(): Loc {
    const p = this.toks[this.pos - 1];
    if (!p) return { line: 1, col: 1 };
    return { line: p.line, col: p.col + p.text.length };
  }

  private describe(t: Token): string {
    if (t.kind === 'eof') return 'end of file';
    return `'${t.text}'`;
  }

  private expect(text: string, context: string): Token {
    if (this.is(text)) return this.next();
    const closing = text === ';' || text === ')' || text === ']';
    const loc = closing ? this.afterPrev() : this.tok;
    throw new CompileError(`expected '${text}' ${context}, found ${this.describe(this.tok)}`, loc);
  }

  private ident(context: string): Token {
    const t = this.tok;
    if (t.kind !== 'ident') throw new CompileError(`expected ${context}, found ${this.describe(t)}`, t);
    return this.next();
  }

  private loc(t: Token = this.tok): Loc {
    return { line: t.line, col: t.col };
  }

  parseProgram(): Program {
    const consts: GlobalConst[] = [];
    let kernel: Kernel | undefined;
    while (this.tok.kind !== 'eof') {
      const t = this.tok;
      if (this.is('__global__')) {
        if (kernel) throw new CompileError('only one __global__ kernel per level', t);
        kernel = this.parseKernel();
        continue;
      }
      if (this.is('const') || this.is('constexpr') || this.is('static')) {
        consts.push(...this.parseGlobalConst());
        continue;
      }
      if (this.is('__device__')) {
        throw new CompileError('__device__ functions are not supported; write the code inside the kernel', t);
      }
      if (this.is('__constant__')) throw new CompileError('__constant__ memory is not supported', t);
      if (this.is('template')) throw new CompileError('templates are not supported', t);
      if (this.is('extern')) throw new CompileError(`'extern' is not supported here`, t);
      if (this.is(';')) {
        this.next();
        continue;
      }
      throw new CompileError(`expected a __global__ kernel, found ${this.describe(t)}`, t);
    }
    if (!kernel) throw new CompileError('no __global__ kernel found', this.tok);
    return { consts, kernel };
  }

  private parseGlobalConst(): GlobalConst[] {
    while (this.accept('static') || this.accept('const') || this.accept('constexpr'));
    const type = this.parseType();
    const out: GlobalConst[] = [];
    do {
      const name = this.ident('a constant name');
      if (this.is('[')) throw new CompileError('global arrays are not supported', this.tok);
      this.expect('=', `after '${name.text}' (global constants need a value)`);
      const init = this.parseAssign();
      out.push({ name: name.text, type, init, loc: this.loc(name) });
    } while (this.accept(','));
    this.expect(';', 'after the constant');
    return out;
  }

  private parseType(): ScalarType {
    const t = this.tok;
    if (t.kind === 'ident' && Object.hasOwn(SCALAR_TYPES, t.text)) {
      this.next();
      return t.text as ScalarType;
    }
    if (t.kind === 'ident' && t.text === 'void') throw new CompileError(`'void' is only valid as the kernel return type`, t);
    if (t.kind === 'ident' && Object.hasOwn(UNSUPPORTED_TYPES, t.text)) throw new CompileError(UNSUPPORTED_TYPES[t.text], t);
    throw new CompileError(`expected a type (int, float or bool), found ${this.describe(t)}`, t);
  }

  private parseKernel(): Kernel {
    const start = this.next();
    if (this.accept('__launch_bounds__')) {
      this.expect('(', `after __launch_bounds__`);
      let depth = 1;
      while (depth > 0 && this.tok.kind !== 'eof') {
        const t = this.next();
        if (t.text === '(') depth++;
        if (t.text === ')') depth--;
      }
    }
    if (!this.accept('void')) throw new CompileError('a __global__ kernel must return void', this.tok);
    const name = this.ident('the kernel name');
    this.expect('(', 'after the kernel name');
    const params: Param[] = [];
    if (!this.is(')')) {
      do params.push(this.parseParam());
      while (this.accept(','));
    }
    this.expect(')', 'to close the parameter list');
    const body = this.parseBlockBody();
    const close = this.toks[this.pos - 1];
    return { name: name.text, params, body, loc: this.loc(start), end: this.loc(close) };
  }

  private parseParam(): Param {
    const start = this.tok;
    let isConst = false;
    if (this.accept('const')) isConst = true;
    const type = this.parseType();
    if (this.accept('const')) isConst = true;
    let pointer = false;
    if (this.accept('*')) {
      pointer = true;
      if (this.is('*')) throw new CompileError('pointers to pointers are not supported', this.tok);
      while (this.accept('const') || this.accept('__restrict__') || this.accept('__restrict'));
    }
    if (this.is('&')) throw new CompileError('reference parameters are not supported', this.tok);
    const name = this.ident('a parameter name');
    if (this.accept('[')) {
      this.expect(']', `after '[' (array parameters take no size)`);
      pointer = true;
    }
    return { name: name.text, type, pointer, isConst, loc: this.loc(start) };
  }

  private parseBlockBody(): Stmt[] {
    this.expect('{', 'to open the block');
    const body: Stmt[] = [];
    while (!this.is('}')) {
      if (this.tok.kind === 'eof') throw new CompileError(`missing '}' at end of file`, this.tok);
      body.push(this.parseStmt());
    }
    this.next();
    return body;
  }

  parseStmt(): Stmt {
    const t = this.tok;
    const loc = this.loc(t);
    if (this.is('{')) return { k: 'block', body: this.parseBlockBody(), loc };
    if (this.is(';')) {
      this.next();
      return { k: 'empty', loc };
    }
    if (t.kind === 'ident') {
      switch (t.text) {
        case 'if': {
          this.next();
          this.expect('(', `after 'if'`);
          const cond = this.parseExpr();
          this.expect(')', 'after the if condition');
          const then = this.parseStmt();
          const els = this.accept('else') ? this.parseStmt() : undefined;
          return { k: 'if', cond, then, else: els, loc };
        }
        case 'for': {
          this.next();
          this.expect('(', `after 'for'`);
          let init: Stmt | undefined;
          if (!this.accept(';')) {
            if (this.startsDecl()) init = this.parseDecl();
            else {
              init = { k: 'expr', e: this.parseExpr(), loc: this.loc() };
              this.expect(';', 'after the loop initializer');
            }
          }
          const cond = this.is(';') ? undefined : this.parseExpr();
          this.expect(';', 'after the loop condition');
          const step = this.is(')') ? undefined : this.parseExpr();
          this.expect(')', 'to close the for header');
          const body = this.parseStmt();
          return { k: 'for', init, cond, step, body, loc };
        }
        case 'while': {
          this.next();
          this.expect('(', `after 'while'`);
          const cond = this.parseExpr();
          this.expect(')', 'after the while condition');
          return { k: 'while', cond, body: this.parseStmt(), loc };
        }
        case 'do': {
          this.next();
          const body = this.parseStmt();
          if (!this.accept('while')) throw new CompileError(`expected 'while' after the do body`, this.tok);
          this.expect('(', `after 'while'`);
          const cond = this.parseExpr();
          this.expect(')', 'after the condition');
          this.expect(';', 'after do-while');
          return { k: 'do', body, cond, loc };
        }
        case 'break':
          this.next();
          this.expect(';', `after 'break'`);
          return { k: 'break', loc };
        case 'continue':
          this.next();
          this.expect(';', `after 'continue'`);
          return { k: 'continue', loc };
        case 'return':
          this.next();
          if (!this.is(';')) throw new CompileError('a kernel returns void; write plain `return;`', this.tok);
          this.next();
          return { k: 'return', loc };
        case 'else':
          throw new CompileError(`'else' without a matching 'if'`, t);
        case 'goto':
        case 'switch':
          throw new CompileError(`'${t.text}' is not supported`, t);
      }
      if (this.startsDecl()) return this.parseDecl();
      if (Object.hasOwn(UNSUPPORTED_TYPES, t.text)) throw new CompileError(UNSUPPORTED_TYPES[t.text], t);
    }
    const e = this.parseExpr();
    this.expect(';', 'after the expression');
    return { k: 'expr', e, loc };
  }

  private startsDecl(): boolean {
    const t = this.tok;
    return (
      t.kind === 'ident' &&
      Object.hasOwn(DECL_START, t.text) &&
      !(Object.hasOwn(SCALAR_TYPES, t.text) && this.peek().text === '(')
    );
  }

  private parseDecl(): Stmt {
    const start = this.tok;
    let shared = false;
    let isConst = false;
    for (;;) {
      if (this.accept('__shared__')) shared = true;
      else if (this.accept('const') || this.accept('constexpr')) isConst = true;
      else if (this.accept('static') || this.accept('volatile')) continue;
      else if (this.is('extern')) {
        throw new CompileError('extern (dynamic) shared memory is not supported; give the array a fixed size', this.tok);
      } else break;
    }
    const type = this.parseType();
    if (this.accept('const')) isConst = true;
    const decls: Declarator[] = [];
    do {
      if (this.is('*')) {
        throw new CompileError('pointer variables are not supported; index the kernel parameter directly', this.tok);
      }
      const name = this.ident('a variable name');
      const dims: Expr[] = [];
      while (this.accept('[')) {
        if (this.is(']')) throw new CompileError('array size is required', this.tok);
        dims.push(this.parseExpr());
        this.expect(']', 'after the array size');
      }
      let init: Expr | undefined;
      if (this.accept('=')) {
        if (this.is('{')) throw new CompileError('brace initializers are not supported', this.tok);
        init = this.parseAssign();
      }
      decls.push({ name: name.text, dims, init, loc: this.loc(name) });
    } while (this.accept(','));
    this.expect(';', 'after the declaration');
    return { k: 'decl', type, isConst, shared, decls, loc: this.loc(start) };
  }

  parseExpr(): Expr {
    const e = this.parseAssign();
    if (this.is(',')) throw new CompileError('the comma operator is not supported; use separate statements', this.tok);
    return e;
  }

  private parseAssign(): Expr {
    const l = this.parseCond();
    const t = this.tok;
    if (t.kind === 'punct' && Object.hasOwn(ASSIGN_OPS, t.text)) {
      this.next();
      const r = this.parseAssign();
      return { k: 'assign', op: t.text as AssignOp, l, r, loc: this.loc(t) };
    }
    return l;
  }

  private parseCond(): Expr {
    const c = this.parseBinary(1);
    if (this.is('?')) {
      const q = this.next();
      const t = this.parseAssign();
      this.expect(':', 'in the ?: expression');
      const f = this.parseCond();
      return { k: 'cond', c, t, f, loc: this.loc(q) };
    }
    return c;
  }

  private parseBinary(minPrec: number): Expr {
    let l = this.parseUnary();
    for (;;) {
      const t = this.tok;
      const prec = t.kind === 'punct' ? BINARY_PREC[t.text] : undefined;
      if (prec === undefined || prec < minPrec) return l;
      this.next();
      const r = this.parseBinary(prec + 1);
      l = { k: 'binary', op: t.text as BinaryOp, l, r, loc: this.loc(t) };
    }
  }

  private parseUnary(): Expr {
    const t = this.tok;
    const loc = this.loc(t);
    if (t.kind === 'punct') {
      if (t.text === '-' || t.text === '+' || t.text === '!' || t.text === '~' || t.text === '&') {
        this.next();
        return { k: 'unary', op: t.text, e: this.parseUnary(), loc };
      }
      if (t.text === '++' || t.text === '--') {
        this.next();
        return { k: 'incdec', op: t.text, prefix: true, e: this.parseUnary(), loc };
      }
      if (t.text === '*') throw new CompileError('pointer dereference is not supported; use a[i]', t);
      if (t.text === '(' && this.peek().kind === 'ident' && this.isTypeWord(this.peek().text)) {
        this.next();
        this.accept('const');
        const to = this.parseType();
        if (this.is('*')) throw new CompileError('pointer casts are not supported', this.tok);
        this.expect(')', 'to close the cast');
        return { k: 'cast', to, e: this.parseUnary(), loc };
      }
    }
    return this.parsePostfix();
  }

  private isTypeWord(text: string): boolean {
    return Object.hasOwn(SCALAR_TYPES, text) || text === 'const' || Object.hasOwn(UNSUPPORTED_TYPES, text);
  }

  private parsePostfix(): Expr {
    let e = this.parsePrimary();
    for (;;) {
      const t = this.tok;
      if (this.is('[')) {
        this.next();
        const index = this.parseExpr();
        this.expect(']', 'after the index');
        e = { k: 'index', base: e, index, loc: this.loc(t) };
      } else if (this.is('(')) {
        if (e.k !== 'name') throw new CompileError('only named functions can be called', t);
        this.next();
        const args: Expr[] = [];
        if (!this.is(')')) {
          do args.push(this.parseAssign());
          while (this.accept(','));
        }
        this.expect(')', `to close the call to ${e.name}`);
        e = { k: 'call', callee: e.name, args, loc: e.loc };
      } else if (this.is('.')) {
        if (e.k !== 'name') throw new CompileError(`'.' only works on threadIdx, blockIdx, blockDim and gridDim`, t);
        this.next();
        const prop = this.ident(`a field name after '.'`);
        e = { k: 'member', obj: e.name, prop: prop.text, loc: e.loc };
      } else if (this.is('++') || this.is('--')) {
        this.next();
        e = { k: 'incdec', op: t.text as '++' | '--', prefix: false, e, loc: this.loc(t) };
      } else if (this.is('->')) {
        throw new CompileError(`'->' is not supported`, t);
      } else {
        return e;
      }
    }
  }

  private parsePrimary(): Expr {
    const t = this.tok;
    const loc = this.loc(t);
    if (t.kind === 'int') {
      this.next();
      return { k: 'num', value: t.value, float: false, loc };
    }
    if (t.kind === 'float') {
      this.next();
      return { k: 'num', value: t.value, float: true, loc };
    }
    if (t.kind === 'ident') {
      if (t.text === 'true' || t.text === 'false') {
        this.next();
        return { k: 'bool', value: t.text === 'true', loc };
      }
      if (t.text === 'static_cast') {
        this.next();
        this.expect('<', 'after static_cast');
        const to = this.parseType();
        this.expect('>', 'after the cast type');
        this.expect('(', 'after static_cast<...>');
        const e = this.parseExpr();
        this.expect(')', 'to close the cast');
        return { k: 'cast', to, e, loc };
      }
      if (Object.hasOwn(SCALAR_TYPES, t.text) && this.peek().text === '(') {
        this.next();
        this.next();
        const e = this.parseExpr();
        this.expect(')', 'to close the cast');
        return { k: 'cast', to: t.text as ScalarType, e, loc };
      }
      this.next();
      return { k: 'name', name: t.text, loc };
    }
    if (this.is('(')) {
      this.next();
      const e = this.parseExpr();
      this.expect(')', 'to close the parenthesis');
      return e;
    }
    throw new CompileError(`expected an expression, found ${this.describe(t)}`, t);
  }
}
