import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands';
import {
  bracketMatching,
  HighlightStyle,
  indentOnInput,
  StreamLanguage,
  syntaxHighlighting,
  type StringStream,
} from '@codemirror/language';
import { EditorState, RangeSetBuilder, StateEffect, StateField, type Extension } from '@codemirror/state';
import {
  Decoration,
  drawSelection,
  EditorView,
  gutter,
  GutterMarker,
  highlightActiveLine,
  keymap,
  lineNumbers,
  type DecorationSet,
} from '@codemirror/view';
import { tags } from '@lezer/highlight';

const words = (style: string, list: string) => list.split(' ').map((w) => [w, style] as const);

/** Token style for each word the highlighter knows about. */
const WORD_STYLE: Record<string, string> = Object.fromEntries([
  ...words('keyword', 'if else for while do return break continue const constexpr static volatile true false static_cast extern'),
  ...words('typeName', 'int float bool void unsigned double long char short size_t'),
  ...words('cudaQualifier', '__global__ __shared__ __device__ __restrict__ __launch_bounds__ __constant__'),
  ...words('cudaBuiltin', 'threadIdx blockIdx blockDim gridDim warpSize INFINITY FLT_MAX INT_MAX INT_MIN'),
  ...words(
    'cudaIntrinsic',
    '__syncthreads __syncwarp atomicAdd atomicSub atomicMax atomicMin atomicExch __shfl_down_sync __shfl_up_sync ' +
      '__shfl_xor_sync __shfl_sync min max fminf fmaxf expf __expf exp logf __logf log sqrtf sqrt rsqrtf fabsf fabs ' +
      'abs floorf floor ceilf ceil',
  ),
]);

interface CudaState {
  inComment: boolean;
}

const cuda = StreamLanguage.define<CudaState>({
  name: 'cuda',
  startState: () => ({ inComment: false }),
  token(stream: StringStream, state: CudaState): string | null {
    if (state.inComment) {
      if (stream.skipTo('*/')) {
        stream.next();
        stream.next();
        state.inComment = false;
      } else stream.skipToEnd();
      return 'comment';
    }
    if (stream.eatSpace()) return null;
    if (stream.sol() && stream.peek() === '#') {
      stream.skipToEnd();
      return 'meta';
    }
    if (stream.match('//')) {
      stream.skipToEnd();
      return 'comment';
    }
    if (stream.match('/*')) {
      state.inComment = true;
      return 'comment';
    }
    if (stream.match(/^0x[0-9a-fA-F]+[uUlL]*/) || stream.match(/^(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?[fFuUlL]?/)) return 'number';
    if (stream.match(/^[A-Za-z_]\w*/)) {
      const w = stream.current();
      if (Object.hasOwn(WORD_STYLE, w)) return WORD_STYLE[w];
      return 'variableName';
    }
    if (stream.match(/^[-+*/%&|^~!<>=?:]+/)) return 'operator';
    stream.next();
    return 'punctuation';
  },
  tokenTable: {
    cudaQualifier: tags.modifier,
    cudaBuiltin: tags.special(tags.variableName),
    cudaIntrinsic: tags.function(tags.variableName),
  },
});

const highlight = HighlightStyle.define([
  { tag: tags.keyword, color: 'var(--syn-keyword)', fontWeight: '600' },
  { tag: tags.typeName, color: 'var(--syn-type)' },
  { tag: tags.modifier, color: 'var(--syn-cuda)', fontWeight: '600' },
  { tag: tags.special(tags.variableName), color: 'var(--syn-builtin)' },
  { tag: tags.function(tags.variableName), color: 'var(--syn-fn)' },
  { tag: tags.number, color: 'var(--syn-number)' },
  { tag: tags.comment, color: 'var(--syn-comment)', fontStyle: 'italic' },
  { tag: tags.meta, color: 'var(--syn-meta)' },
  { tag: tags.operator, color: 'var(--syn-op)' },
]);

const theme = EditorView.theme({
  '&': { height: '100%', backgroundColor: 'var(--code-bg)', color: 'var(--ink)' },
  '.cm-scroller': { fontFamily: 'var(--mono)', fontSize: '13.5px', lineHeight: '1.6' },
  '.cm-content': { caretColor: 'var(--accent)', padding: '10px 0' },
  '.cm-cursor': { borderLeftColor: 'var(--accent)', borderLeftWidth: '2px' },
  '&.cm-focused': { outline: 'none' },
  '.cm-gutters': { backgroundColor: 'var(--code-bg)', color: 'var(--ink-3)', border: 'none' },
  '.cm-activeLine': { backgroundColor: 'var(--active-line)' },
  '.cm-activeLineGutter': { backgroundColor: 'transparent', color: 'var(--ink-2)' },
  '&.cm-focused .cm-selectionBackground, .cm-selectionBackground, ::selection': { backgroundColor: 'var(--selection) !important' },
  '.cm-matchingBracket': { backgroundColor: 'var(--selection)', outline: 'none' },
});

// ---- per-line cycle gutter ----

const setLineCycles = StateEffect.define<Float64Array | null>();

const lineCyclesField = StateField.define<Float64Array | null>({
  create: () => null,
  update(value, tr) {
    for (const e of tr.effects) if (e.is(setLineCycles)) return e.value;
    return tr.docChanged ? null : value;
  },
});

class CycleMarker extends GutterMarker {
  constructor(
    readonly cycles: number,
    readonly share: number,
  ) {
    super();
  }
  eq(other: CycleMarker) {
    return other.cycles === this.cycles && other.share === this.share;
  }
  toDOM() {
    const el = document.createElement('span');
    el.className = 'cycles';
    el.textContent = compact(this.cycles);
    el.style.setProperty('--share', String(this.share));
    el.title = `${this.cycles.toLocaleString('en-US')} cycles on this line`;
    return el;
  }
}

export function compact(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e4) return `${Math.round(n / 1e3)}k`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(Math.round(n));
}

const cycleGutter = gutter({
  class: 'cm-cycle-gutter',
  lineMarker(view, line) {
    const cycles = view.state.field(lineCyclesField);
    if (!cycles) return null;
    const n = view.state.doc.lineAt(line.from).number;
    const c = cycles[n] ?? 0;
    if (!c) return null;
    let max = 0;
    for (const v of cycles) max = Math.max(max, v);
    return new CycleMarker(c, c / max);
  },
  lineMarkerChange: (update) => update.transactions.some((t) => t.effects.some((e) => e.is(setLineCycles))) || update.docChanged,
  initialSpacer: () => new CycleMarker(88888, 0),
});

// ---- line highlights for the current step and for problems ----

export interface LineMarks {
  step?: number;
  error?: number;
  warnings?: number[];
}

const setMarks = StateEffect.define<LineMarks>();

const marksField = StateField.define<DecorationSet>({
  create: () => Decoration.none,
  update(deco, tr) {
    for (const e of tr.effects) {
      if (e.is(setMarks)) return buildMarks(tr.state, e.value);
    }
    return tr.docChanged ? Decoration.none : deco;
  },
  provide: (f) => EditorView.decorations.from(f),
});

function buildMarks(state: EditorState, marks: LineMarks): DecorationSet {
  const byLine = new Map<number, string[]>();
  const add = (line: number | undefined, cls: string) => {
    if (!line || line < 1 || line > state.doc.lines) return;
    byLine.set(line, [...(byLine.get(line) ?? []), cls]);
  };
  marks.warnings?.forEach((l) => add(l, 'cm-warn-line'));
  add(marks.error, 'cm-error-line');
  add(marks.step, 'cm-step-line');
  const b = new RangeSetBuilder<Decoration>();
  for (const line of [...byLine.keys()].sort((x, y) => x - y)) {
    b.add(state.doc.line(line).from, state.doc.line(line).from, Decoration.line({ class: byLine.get(line)!.join(' ') }));
  }
  return b.finish();
}

export interface Editor {
  view: EditorView;
  getCode(): string;
  setCode(code: string): void;
  setLineCycles(cycles: Float64Array | null): void;
  setMarks(marks: LineMarks): void;
  scrollToLine(line: number): void;
}

export function createEditor(
  parent: HTMLElement,
  opts: { onChange: () => void; onRun: () => void; onStep: (dir: number) => void },
): Editor {
  const extensions: Extension[] = [
    lineNumbers(),
    cycleGutter,
    history(),
    drawSelection(),
    indentOnInput(),
    bracketMatching(),
    highlightActiveLine(),
    cuda,
    syntaxHighlighting(highlight),
    theme,
    lineCyclesField,
    marksField,
    EditorState.tabSize.of(2),
    keymap.of([
      { key: 'Mod-Enter', run: () => (opts.onRun(), true), preventDefault: true },
      { key: 'Alt-ArrowRight', run: () => (opts.onStep(1), true), preventDefault: true },
      { key: 'Alt-ArrowLeft', run: () => (opts.onStep(-1), true), preventDefault: true },
      ...defaultKeymap,
      ...historyKeymap,
      indentWithTab,
    ]),
    EditorView.updateListener.of((u) => {
      if (u.docChanged) opts.onChange();
    }),
    EditorView.contentAttributes.of({ 'aria-label': 'Kernel source code' }),
  ];
  const view = new EditorView({ parent, state: EditorState.create({ doc: '', extensions }) });
  return {
    view,
    getCode: () => view.state.doc.toString(),
    setCode(code) {
      view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: code } });
    },
    setLineCycles(cycles) {
      view.dispatch({ effects: setLineCycles.of(cycles) });
    },
    setMarks(marks) {
      view.dispatch({ effects: setMarks.of(marks) });
    },
    scrollToLine(line) {
      if (line < 1 || line > view.state.doc.lines) return;
      view.dispatch({ effects: EditorView.scrollIntoView(view.state.doc.line(line).from, { y: 'nearest' }) });
    },
  };
}
