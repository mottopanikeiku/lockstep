import type { Loc } from './diagnostics';

export type ScalarType = 'int' | 'float' | 'bool';

export interface Param {
  name: string;
  type: ScalarType;
  pointer: boolean;
  isConst: boolean;
  loc: Loc;
}

export interface Kernel {
  name: string;
  params: Param[];
  body: Stmt[];
  loc: Loc;
  /** The closing brace, where threads that run off the end exit. */
  end: Loc;
}

export interface GlobalConst {
  name: string;
  type: ScalarType;
  init: Expr;
  loc: Loc;
}

export interface Program {
  consts: GlobalConst[];
  kernel: Kernel;
}

export interface Declarator {
  name: string;
  dims: Expr[];
  init?: Expr;
  loc: Loc;
}

export type Stmt =
  | { k: 'block'; body: Stmt[]; loc: Loc }
  | { k: 'decl'; type: ScalarType; isConst: boolean; shared: boolean; decls: Declarator[]; loc: Loc }
  | { k: 'expr'; e: Expr; loc: Loc }
  | { k: 'if'; cond: Expr; then: Stmt; else?: Stmt; loc: Loc }
  | { k: 'for'; init?: Stmt; cond?: Expr; step?: Expr; body: Stmt; loc: Loc }
  | { k: 'while'; cond: Expr; body: Stmt; loc: Loc }
  | { k: 'do'; body: Stmt; cond: Expr; loc: Loc }
  | { k: 'break'; loc: Loc }
  | { k: 'continue'; loc: Loc }
  | { k: 'return'; loc: Loc }
  | { k: 'empty'; loc: Loc };

export type AssignOp = '=' | '+=' | '-=' | '*=' | '/=' | '%=' | '&=' | '|=' | '^=' | '<<=' | '>>=';

export type BinaryOp =
  | '+' | '-' | '*' | '/' | '%'
  | '<<' | '>>' | '&' | '|' | '^'
  | '<' | '<=' | '>' | '>=' | '==' | '!='
  | '&&' | '||';

export type Expr =
  | { k: 'num'; value: number; float: boolean; loc: Loc }
  | { k: 'bool'; value: boolean; loc: Loc }
  | { k: 'name'; name: string; loc: Loc }
  | { k: 'member'; obj: string; prop: string; loc: Loc }
  | { k: 'index'; base: Expr; index: Expr; loc: Loc }
  | { k: 'call'; callee: string; args: Expr[]; loc: Loc }
  | { k: 'unary'; op: '-' | '+' | '!' | '~' | '&'; e: Expr; loc: Loc }
  | { k: 'incdec'; op: '++' | '--'; prefix: boolean; e: Expr; loc: Loc }
  | { k: 'binary'; op: BinaryOp; l: Expr; r: Expr; loc: Loc }
  | { k: 'assign'; op: AssignOp; l: Expr; r: Expr; loc: Loc }
  | { k: 'cond'; c: Expr; t: Expr; f: Expr; loc: Loc }
  | { k: 'cast'; to: ScalarType; e: Expr; loc: Loc };
