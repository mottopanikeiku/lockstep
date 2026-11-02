import type { KernelIR } from './ir/ir';
import { lowerProgram } from './ir/lower';
import { CompileError, type Diagnostic } from './lang/diagnostics';
import { lex } from './lang/lexer';
import { Parser } from './lang/parser';

export type CompileResult =
  | { ok: true; kernel: KernelIR; warnings: Diagnostic[] }
  | { ok: false; error: Diagnostic; warnings: Diagnostic[] };

export function compile(src: string): CompileResult {
  const warnings: Diagnostic[] = [];
  try {
    const lexed = lex(src);
    warnings.push(...lexed.warnings);
    const program = new Parser(lexed.tokens).parseProgram();
    return { ok: true, kernel: lowerProgram(program, src, warnings), warnings };
  } catch (err) {
    if (err instanceof CompileError) {
      return { ok: false, error: { severity: 'error', message: err.message, line: err.line, col: err.col }, warnings };
    }
    throw err;
  }
}
