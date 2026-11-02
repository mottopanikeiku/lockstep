import './style.css';
import { fmt, signature, verdict, type Attempt } from './game/attempt';
import { isSolved, isUnlocked, loadProgress, recordSolve, saveProgress } from './game/progress';
import { decodeShare, encodeShare } from './game/share';
import type { RunReply, RunRequest } from './game/worker';
import type { Instr } from './ir/ir';
import { LEVELS, PAR } from './levels';
import type { Level, Par } from './levels/types';
import { simulate } from './sim/machine';
import { contentWidth, fitCanvas, palette, Tooltip } from './ui/canvas';
import { compact, createEditor } from './ui/editor';
import { MemoryView, type GlobalArrayInfo } from './ui/memory';
import { Replay } from './ui/replay';
import { ThreadsView } from './ui/threads';
import { Timeline } from './ui/timeline';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const progress = loadProgress();
const tooltip = new Tooltip();

const state = {
  index: 0,
  run: null as Attempt | null,
  replay: null as Replay | null,
  step: 0,
  block: 0,
  stale: false,
  inspect: null as { warp: number; lane: number } | null,
};

const level = () => LEVELS[state.index];

// ---- editor ----

let saveTimer = 0;
const editor = createEditor($('editor'), {
  onChange() {
    stopPlay();
    if (state.run && !state.stale) {
      state.stale = true;
      $('right').classList.add('stale');
    }
    clearTimeout(saveTimer);
    saveTimer = window.setTimeout(saveCode, 400);
  },
  onRun: () => void run(),
  onStep: (dir) => void step(dir),
});

function saveCode(): void {
  const code = editor.getCode();
  const entry = (progress.levels[level().id] ??= {});
  entry.code = code === level().starter ? undefined : code;
  saveProgress(progress);
}

// ---- views ----

const timeline = new Timeline($<HTMLCanvasElement>('timeline'), tooltip, (s) => {
  stopPlay();
  seek(s);
});
const threads = new ThreadsView($<HTMLCanvasElement>('threads'), tooltip, (warp, lane) => {
  if (!state.replay) return;
  state.inspect = { warp: state.block * state.replay.warpsPerBlock + warp, lane };
  renderLane();
});
const memory = new MemoryView($('memory'), tooltip);

// ---- level brief ----

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/** Level text uses `backticks` for code. */
function prose(s: string): string {
  return escapeHtml(s).replace(/`([^`]+)`/g, '<code>$1</code>');
}

function launchText(l: Level): string {
  const { grid, block } = l.launch;
  const dim = (d: { x: number; y: number }) => (d.y === 1 ? `${d.x}` : `dim3(${d.x}, ${d.y})`);
  return `<<<${dim(grid)}, ${dim(block)}>>>`;
}

/** Shown above the first level, for people who have never written CUDA. */
const INTRO =
  'A CUDA kernel is a function that many GPU threads run at once, each on its own piece of data. ' +
  'This game runs your kernels on a simulated GPU, where threads move in warps of 32, and shows every step they take ' +
  'and what it costs against par, the score of a reference solution. Finish the kernel below and press Run.';

function renderBrief(note = ''): void {
  const l = level();
  const par = PAR[l.id];
  const solved = isSolved(progress, l.id);
  const arrays = l.params
    .map((p) => (p.kind === 'array' ? `${p.name}: ${p.type}[${p.shape.join('×')}]${p.output ? ' (output)' : ''}` : `${p.name} = ${p.value}`))
    .join(', ');
  const intro = state.index === 0 ? `<p class="intro">${prose(INTRO)}</p>` : '';
  $('brief').innerHTML = `
    ${intro}
    <p class="kicker">Level ${state.index + 1} of ${LEVELS.length}</p>
    <h1>${escapeHtml(l.title)}</h1>
    <p class="text">${prose(l.text)}</p>
    <dl class="facts">
      <dt>launch</dt><dd>${escapeHtml(launchText(l))}</dd>
      <dt>data</dt><dd>${escapeHtml(arrays)}</dd>
      <dt>par</dt><dd>${par.cycles.toLocaleString('en-US')} cycles · ${par.sectors.toLocaleString('en-US')} sectors · ${par.lines} lines</dd>
    </dl>
    ${l.hint ? `<details><summary>Hint</summary><p>${prose(l.hint)}</p></details>` : ''}
    ${solved ? `<details><summary>Reference solution (the par score)</summary><pre>${escapeHtml(l.reference)}</pre></details>` : ''}
    ${note ? `<p class="shared-note">${note}</p>` : ''}`;
  $('level-button').textContent = `${String(state.index + 1).padStart(2, '0')} · ${l.title}  ▾`;
}

// ---- level list ----

function renderLevelList(): void {
  const list = $('level-list');
  list.replaceChildren(
    ...LEVELS.map((l, i) => {
      const li = document.createElement('li');
      const b = document.createElement('button');
      const open = isUnlocked(progress, i);
      const best = progress.levels[l.id]?.best;
      const par = PAR[l.id];
      const atPar = best && best.cycles <= par.cycles && best.sectors <= par.sectors && best.lines <= par.lines;
      b.disabled = !open;
      b.setAttribute('aria-current', String(i === state.index));
      b.innerHTML = `<span class="no">${i + 1}</span><span>${escapeHtml(l.title)}</span><span class="badge ${atPar ? 'par' : best ? 'solved' : ''}">${
        !open ? 'locked' : atPar ? 'at par' : best ? 'solved' : ''
      }</span>`;
      b.addEventListener('click', () => {
        ($('levels') as HTMLDialogElement).close();
        loadLevel(i);
      });
      li.append(b);
      return li;
    }),
  );
}

function loadLevel(index: number, code?: string, note = ''): void {
  cancelRun();
  clearTimeout(saveTimer);
  state.index = index;
  progress.current = level().id;
  saveProgress(progress);
  renderBrief(note);
  editor.setCode(code ?? progress.levels[level().id]?.code ?? level().starter);
  // Loading is not editing: don't let the change listener save this code.
  clearTimeout(saveTimer);
  clearRun();
}

function clearRun(): void {
  stopPlay();
  state.run = null;
  state.replay = null;
  state.stale = false;
  state.inspect = null;
  $('right').classList.remove('stale');
  $('problem').replaceChildren();
  $('problem').className = 'problem';
  editor.setLineCycles(null);
  editor.setMarks({});
  timeline.setRun(null, -1);
  timeline.render();
  threads.set(null);
  memory.setRun(null, []);
  $('blocks').replaceChildren();
  $('block-label').textContent = '';
  $('step-info').textContent = '';
  $('lane-info').replaceChildren();
  renderScore();
  renderOutputs();
}

// ---- running ----

// The simulator runs in a worker. A new run, or a change of level, drops the one in flight.
let simWorker = startWorker();
let pending: { id: number; code: string; index: number; resolve: (shown: boolean) => void } | null = null;
let runSeq = 0;
let busyTimer = 0;

function startWorker(): Worker {
  const w = new Worker(new URL('./game/worker.ts', import.meta.url), { type: 'module' });
  w.onmessage = (e: MessageEvent<RunReply>) => finishRun(e.data);
  return w;
}

function setBusy(busy: boolean): void {
  clearTimeout(busyTimer);
  // Only say so when a run takes long enough to notice.
  if (busy) busyTimer = window.setTimeout(() => ($('run-label').textContent = 'Running…'), 150);
  else $('run-label').textContent = 'Run';
}

function cancelRun(): void {
  if (!pending) return;
  simWorker.terminate();
  simWorker = startWorker();
  pending.resolve(false);
  pending = null;
  setBusy(false);
}

/** Run the editor's code; resolves to whether this run's result made it to the screen. */
function run(): Promise<boolean> {
  stopPlay();
  cancelRun();
  saveCode();
  const id = ++runSeq;
  const code = editor.getCode();
  setBusy(true);
  return new Promise((resolve) => {
    pending = { id, code, index: state.index, resolve };
    const req: RunRequest = { id, level: state.index, src: code };
    simWorker.postMessage(req);
  });
}

function finishRun(reply: RunReply): void {
  const p = pending;
  if (!p || reply.id !== p.id) return;
  pending = null;
  setBusy(false);
  if ('error' in reply) {
    toast(`The simulator failed: ${reply.error}`);
    p.resolve(false);
    return;
  }
  showAttempt(reply.attempt);
  // Edits made while the kernel ran make this result stale straight away.
  if (editor.getCode() !== p.code) {
    state.stale = true;
    $('right').classList.add('stale');
  }
  p.resolve(true);
}

function showAttempt(a: Attempt): void {
  const l = level();
  state.run = a;
  state.stale = false;
  state.inspect = null;
  $('right').classList.remove('stale');
  let note = '';
  if (a.solved && a.score) {
    const prev = progress.levels[l.id]?.best;
    note = verdict(a.score, PAR[l.id], prev && { ...prev });
    recordSolve(progress, l.id, a.score);
    saveProgress(progress);
    renderBrief();
  }
  const k = a.compiled.ok ? a.compiled.kernel : null;
  const trace = a.sim?.trace;
  if (k && trace && trace.steps > 0) {
    const initial: Float64Array[] = [];
    l.params.forEach((p, i) => {
      const kp = k.params[i];
      if (p.kind === 'array' && kp.kind === 'buffer') initial[kp.buf] = a.inputs[p.name] as Float64Array;
    });
    state.replay = new Replay(k, l.launch, initial, trace);
    const arrays: GlobalArrayInfo[] = [];
    l.params.forEach((p, i) => {
      const kp = k.params[i];
      if (p.kind === 'array' && kp.kind === 'buffer') arrays.push({ name: kp.name, type: p.type, shape: p.shape, output: !!p.output, buf: kp.buf });
    });
    memory.setRun(state.replay, arrays);
    timeline.setRun(state.replay, a.problem?.step ?? -1);
    editor.setLineCycles(a.sim!.lineCycles);
    renderBlockButtons();
    seek(a.problem?.step ?? trace.steps - 1);
  } else {
    state.replay = null;
    timeline.setRun(null, -1);
    timeline.render();
    threads.set(null);
    memory.setRun(null, []);
    editor.setMarks({ error: a.problem?.line, warnings: a.warnings.map((w) => w.line) });
  }
  renderProblem(a, note);
  renderScore();
  renderOutputs();
}

async function step(dir: number): Promise<void> {
  stopPlay();
  if (!state.replay || state.stale) {
    if ((await run()) && state.replay) seek(0);
    return;
  }
  seek(Math.max(0, Math.min(state.replay.trace.steps - 1, state.step + dir)));
}

// ---- playback ----

/** A whole run plays in about this many seconds, however many steps it has. */
const PLAY_SECONDS = 8;
let playFrame = 0;
let playStarting = false;

function setPlaying(on: boolean): void {
  const b = $('play');
  b.textContent = on ? 'Pause' : 'Play';
  b.setAttribute('aria-pressed', String(on));
}

async function startPlay(): Promise<void> {
  if (playFrame || playStarting) return;
  if (!state.replay || state.stale) {
    playStarting = true;
    const shown = await run();
    playStarting = false;
    if (!shown || !state.replay) return;
    seek(0);
  }
  const last = state.replay!.trace.steps - 1;
  if (state.step >= last) seek(0);
  const rate = Math.max(12, last / PLAY_SECONDS);
  let pos = state.step;
  let prev = performance.now();
  const tick = (now: number) => {
    // A frame's timestamp can be a little older than the performance.now() taken when play started.
    pos = Math.min(last, pos + (Math.max(0, now - prev) / 1000) * rate);
    prev = Math.max(prev, now);
    seek(Math.floor(pos));
    if (pos >= last) stopPlay();
    else playFrame = requestAnimationFrame(tick);
  };
  setPlaying(true);
  playFrame = requestAnimationFrame(tick);
}

function stopPlay(): void {
  if (!playFrame) return;
  cancelAnimationFrame(playFrame);
  playFrame = 0;
  setPlaying(false);
}

let laneTimer = 0;

function seek(s: number): void {
  const r = state.replay;
  if (!r) return;
  state.step = s;
  const block = r.blockOfStep(s);
  if (block !== state.block) {
    state.block = block;
    state.inspect = null;
    renderBlockButtons();
  }
  timeline.setStep(s);
  const { launch } = level();
  threads.set({ warps: r.warpsAt(s, block), block: { x: block % launch.grid.x, y: Math.floor(block / launch.grid.x) }, blockDim: launch.block });
  memory.setStep(s, block);
  const ins = r.kernel.code[r.trace.pc[s]];
  editor.setMarks({ step: ins.line, error: state.run?.problem?.line, warnings: state.run?.warnings.map((w) => w.line) });
  editor.scrollToLine(ins.line);
  renderStepInfo();
  clearTimeout(laneTimer);
  if (state.inspect) laneTimer = window.setTimeout(renderLane, 120);
}

function renderBlockButtons(): void {
  const r = state.replay;
  const wrap = $('blocks');
  if (!r) return wrap.replaceChildren();
  const { grid } = level().launch;
  $('block-label').textContent = r.blocks > 1 ? `block ${blockName(state.block)}` : '';
  if (r.blocks === 1) return wrap.replaceChildren();
  wrap.replaceChildren(
    ...Array.from({ length: r.blocks }, (_, b) => {
      const btn = document.createElement('button');
      btn.textContent = grid.y > 1 ? `(${b % grid.x},${Math.floor(b / grid.x)})` : `${b}`;
      btn.setAttribute('aria-pressed', String(b === state.block));
      btn.setAttribute('aria-label', `Jump to block ${btn.textContent}`);
      btn.addEventListener('click', () => {
        const first = r.firstStepOfBlock(b);
        stopPlay();
        if (first >= 0) seek(first);
      });
      return btn;
    }),
  );
}

function blockName(b: number): string {
  const { grid } = level().launch;
  return grid.y > 1 ? `(${b % grid.x}, ${Math.floor(b / grid.x)})` : `${b}`;
}

function describe(ins: Instr, units: number, s: number): string {
  const r = state.replay!;
  const name = (space: string, buf: number) =>
    space === 'global' ? (r.kernel.params.find((p) => p.kind === 'buffer' && p.buf === buf)?.name ?? '?') : r.kernel.shared[buf].name;
  switch (ins.op) {
    case 'ld':
    case 'st':
    case 'atom': {
      const verb = ins.op === 'ld' ? 'load' : ins.op === 'st' ? 'store' : `atomic ${ins.fn}`;
      if (ins.space === 'global') return `${verb} <b>${name('global', ins.buf)}</b> · ${units} sector${units === 1 ? '' : 's'}`;
      return `${verb} <b>${name('shared', ins.buf)}</b> (shared) · ${units <= 1 ? 'no bank conflict' : `${units}-way bank conflict`}`;
    }
    case 'cbr':
      return r.trace.divergences.includes(s) ? 'branch · <b>the warp splits</b>' : 'branch · all lanes agree';
    case 'br':
      return 'jump';
    case 'bar': {
      const rel = r.trace.release[s];
      return rel >= 0 ? `__syncthreads · released at step ${rel.toLocaleString('en-US')}` : '__syncthreads';
    }
    case 'syncwarp':
      return '__syncwarp';
    case 'ret':
      return 'return';
    case 'shfl':
      return `warp shuffle (${ins.mode})`;
    case 'math':
      return `${ins.fn} on the special function unit`;
    default:
      return 'arithmetic';
  }
}

function renderStepInfo(): void {
  const r = state.replay!;
  const t = r.trace;
  const s = state.step;
  const ins = r.kernel.code[t.pc[s]];
  const lanes = popcount(t.mask[s]);
  const warp = t.warp[s] % r.warpsPerBlock;
  const src = editor.view.state.doc.lines >= ins.line ? editor.view.state.doc.line(ins.line).text.trim() : '';
  $('step-info').innerHTML =
    `step <b>${(s + 1).toLocaleString('en-US')}</b> of ${t.steps.toLocaleString('en-US')}` +
    (r.blocks > 1 ? ` · block ${blockName(r.blockOfStep(s))}` : '') +
    ` · warp ${warp} · line ${ins.line}: <code>${escapeHtml(src.length > 60 ? src.slice(0, 57) + '...' : src)}</code><br>` +
    `${describe(ins, t.units[s], s)} · ${lanes} of 32 lanes · ${t.cost[s]} cycle${t.cost[s] === 1 ? '' : 's'}` +
    (t.depth[s] > 1 ? ' · <b>diverged</b>' : '');
}

function popcount(m: number): number {
  let n = 0;
  while (m) {
    m &= m - 1;
    n++;
  }
  return n;
}

function renderLane(): void {
  const box = $('lane-info');
  const sel = state.inspect;
  const a = state.run;
  if (!sel || !a?.compiled.ok || !state.replay) return box.replaceChildren();
  const k = a.compiled.kernel;
  const l = level();
  const args = l.params.map((p) => a.inputs[p.name]);
  const res = simulate(k, l.launch, args, { inspect: { step: state.step, warp: sel.warp } });
  const r = state.replay;
  const tid = (sel.warp % r.warpsPerBlock) * 32 + sel.lane;
  const bx = l.launch.block.x;
  const who = l.launch.block.y > 1 ? `thread (${tid % bx}, ${Math.floor(tid / bx)})` : `thread ${tid}`;
  const head = `${who}${r.blocks > 1 ? ` of block ${blockName(Math.floor(sel.warp / r.warpsPerBlock))}` : ''}, lane ${sel.lane} of warp ${sel.warp % r.warpsPerBlock}, after step ${(state.step + 1).toLocaleString('en-US')}`;
  if (!res.inspected) {
    box.textContent = `${head}: not started yet.`;
    return;
  }
  if (!((res.inspected.live >>> sel.lane) & 1)) {
    box.textContent = `${head}: this thread has returned.`;
    return;
  }
  const regs = res.inspected.regs;
  const show = (type: string, v: number) => (type === 'bool' ? (v ? 'true' : 'false') : fmt(v));
  const vars = k.vars.map((v) => `${v.name} = ${show(v.type, v.reg === undefined ? v.value! : regs[v.reg * 32 + sel.lane])}`);
  box.innerHTML = `${escapeHtml(head)}<div class="vars">${vars.map((v) => `<span>${escapeHtml(v)}</span>`).join('') || '<span class="muted">no variables declared</span>'}</div>`;
}

// ---- problem, score, outputs ----

const KIND_LABEL: Record<string, string> = {
  compile: 'Compile error',
  signature: 'Wrong signature',
  race: 'Race',
  deadlock: 'Deadlock',
  oob: 'Out of bounds',
  runtime: 'Runtime error',
  timeout: 'Timeout',
  wrong: 'Wrong answer',
};

function renderProblem(a: Attempt, verdictText: string): void {
  const box = $('problem');
  box.replaceChildren();
  const p = a.problem;
  if (p) {
    box.className = 'problem';
    const where = p.line ? `line ${p.line}${p.kind === 'compile' && p.col > 1 ? `, column ${p.col}` : ''}` : '';
    box.innerHTML = `<span class="kind">${KIND_LABEL[p.kind]}</span>${where}<p>${escapeHtml(p.message)}</p>`;
    if (p.line) {
      const go = document.createElement('button');
      go.textContent = p.step !== undefined ? `Show step ${(p.step + 1).toLocaleString('en-US')}` : `Go to line ${p.line}`;
      go.addEventListener('click', () => {
        stopPlay();
        if (p.step !== undefined) seek(p.step);
        editor.scrollToLine(p.line);
        editor.view.focus();
      });
      box.append(go);
    }
  } else if (a.solved) {
    box.className = 'problem good';
    box.innerHTML = `<span class="kind">Solved</span>${escapeHtml(verdictText)}`;
    const next = LEVELS.findIndex((_, i) => i > state.index && isUnlocked(progress, i) && !isSolved(progress, LEVELS[i].id));
    if (next >= 0) {
      const btn = document.createElement('button');
      btn.textContent = `Next: ${LEVELS[next].title} →`;
      btn.addEventListener('click', () => loadLevel(next));
      box.append(document.createElement('br'), btn);
    }
    renderLevelList();
  }
  if (a.warnings.length) {
    const ul = document.createElement('p');
    ul.className = 'muted';
    ul.textContent = a.warnings.map((w) => `line ${w.line}: ${w.message}`).join(' · ');
    box.append(ul);
  }
}

function renderScore(): void {
  const l = level();
  const par = PAR[l.id];
  const a = state.run;
  const score: Par | undefined = a?.score ?? (a?.sim && a.compiled.ok ? { cycles: a.sim.metrics.cycles, sectors: a.sim.metrics.globalSectors, lines: a.compiled.kernel.lines } : undefined);
  const best = progress.levels[l.id]?.best;
  const metric = (key: keyof Par, label: string) => {
    const v = score?.[key];
    const ratio = v === undefined ? 0 : Math.min(1, v / (2 * par[key]));
    const cls = v === undefined || !a?.solved ? '' : v <= par[key] ? 'under' : 'over';
    return `<div class="metric">
      <div class="name">${label}</div>
      <div class="value ${cls}">${v === undefined ? '—' : v.toLocaleString('en-US')}</div>
      <div class="par">par ${par[key].toLocaleString('en-US')}${best ? ` · best ${best[key].toLocaleString('en-US')}` : ''}</div>
      <div class="bar"><i class="${cls}" style="width:${(ratio * 100).toFixed(1)}%"></i></div>
    </div>`;
  };
  let status = '';
  if (a?.solved) status = '<div class="status ok">solved</div>';
  else if (a?.sim) status = '<div class="status muted">not solved, so this run does not count</div>';
  let breakdown = '';
  if (a?.sim) {
    const m = a.sim.metrics;
    const part = (label: string, v: number) => `<span>${label} <b>${compact(v)}</b></span>`;
    breakdown = `<div class="breakdown">
      ${part('issue', m.issued)}${part('global memory', m.globalCycles)}${part('shared memory', m.sharedCycles)}
      ${part('special functions', m.specialCycles)}${part('shuffles', m.shuffleCycles)}${part('barrier wait', m.barrierWait)}
      <span>lanes busy <b>${m.issued ? Math.round((100 * m.activeLanes) / (32 * m.issued)) : 0}%</b></span>
      ${part('divergent branches', m.divergentBranches)}${part('bank conflicts', m.bankConflicts)}${part('atomic collisions', m.atomicCollisions)}
    </div>`;
  }
  $('score').innerHTML = status + metric('cycles', 'cycles') + metric('sectors', 'sectors') + metric('lines', 'lines') + breakdown;
}

function renderOutputs(): void {
  const box = $('outputs');
  box.replaceChildren();
  const a = state.run;
  if (!a) {
    box.innerHTML = `<p class="muted">Expected and actual outputs appear here after a run. The kernel takes (${escapeHtml(signature(level()))}).</p>`;
    return;
  }
  if (!a.outputs.length) {
    box.innerHTML = `<p class="muted">${a.sim ? 'The run stopped with an error, so the output was not checked.' : 'Nothing ran.'}</p>`;
    return;
  }
  const p = palette();
  for (const o of a.outputs) {
    const spec = level().params.find((x) => x.name === o.name);
    const shape = spec?.kind === 'array' ? spec.shape : [o.expected.length];
    const row = document.createElement('div');
    const wrong = new Set(o.wrong);
    row.innerHTML = `<div><code>${escapeHtml(o.name)}</code> <span class="${o.wrong.length ? 'mem-live bad' : 'muted'}">${
      o.wrong.length ? `${o.wrong.length} of ${o.expected.length} wrong` : `all ${o.expected.length} values match`
    }</span></div>`;
    const grids = document.createElement('div');
    grids.className = 'out-row';
    for (const [label, values] of [['expected', o.expected], ['actual', o.actual]] as const) {
      const fig = document.createElement('figure');
      const cap = document.createElement('figcaption');
      cap.textContent = label;
      const canvas = document.createElement('canvas');
      fig.append(cap, canvas);
      grids.append(fig);
      const cols = shape.length === 2 ? shape[1] : Math.min(32, values.length);
      const rows = Math.ceil(values.length / cols);
      const avail = contentWidth(box, 600);
      const cw = Math.max(3, Math.min(26, Math.floor(avail / cols)));
      const ch = Math.max(2, Math.min(cw, Math.floor(160 / rows)));
      const ctx = fitCanvas(canvas, cols * cw + 1, rows * ch + 1);
      ctx.font = `9px ${p.mono}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      for (let i = 0; i < values.length; i++) {
        const x = (i % cols) * cw;
        const y = Math.floor(i / cols) * ch;
        const bad = label === 'actual' && wrong.has(i);
        ctx.fillStyle = bad ? p.bad : p.panel;
        ctx.globalAlpha = bad ? 0.75 : 1;
        ctx.fillRect(x, y, cw, ch);
        ctx.globalAlpha = 1;
        if (cw >= 6) {
          ctx.strokeStyle = p.rule;
          ctx.strokeRect(x + 0.5, y + 0.5, cw, ch);
        }
        if (cw >= 18 && ch >= 13) {
          ctx.fillStyle = bad ? '#fff' : p.ink2;
          ctx.fillText(fmt(values[i]).slice(0, Math.floor((cw - 2) / 5.5)), x + cw / 2, y + ch / 2 + 0.5);
        }
      }
    }
    row.append(grids);
    if (o.wrong.length) {
      const ul = document.createElement('ul');
      for (const j of o.wrong.slice(0, 5)) {
        const li = document.createElement('li');
        li.textContent = `${o.name}[${j}]: expected ${fmt(o.expected[j])}, got ${fmt(o.actual[j])}`;
        ul.append(li);
      }
      row.append(ul);
    }
    box.append(row);
  }
}

// ---- chrome: theme, share, help, keys ----

function setTheme(theme: 'light' | 'dark'): void {
  document.documentElement.dataset.theme = theme;
  progress.theme = theme;
  saveProgress(progress);
  rerender();
}

function rerender(): void {
  timeline.invalidate();
  timeline.render();
  if (state.replay) seek(state.step);
  else threads.render();
  renderOutputs();
}

function toast(text: string): void {
  const t = $('toast');
  t.textContent = text;
  t.hidden = false;
  clearTimeout(Number(t.dataset.timer));
  t.dataset.timer = String(window.setTimeout(() => (t.hidden = true), 2200));
}

$('theme').addEventListener('click', () => setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));
$('run').addEventListener('click', () => void run());
$('step').addEventListener('click', () => void step(1));
$('step-back').addEventListener('click', () => void step(-1));
$('play').addEventListener('click', () => (playFrame ? stopPlay() : void startPlay()));
$('reset').addEventListener('click', () => {
  if (editor.getCode() !== level().starter && !confirm('Replace your code with the starter code?')) return;
  editor.setCode(level().starter);
  saveCode();
  clearRun();
});
$('level-button').addEventListener('click', () => {
  renderLevelList();
  ($('levels') as HTMLDialogElement).showModal();
});
$('help-button').addEventListener('click', () => ($('help') as HTMLDialogElement).showModal());
$('share').addEventListener('click', async () => {
  const hash = await encodeShare(level().id, editor.getCode());
  const url = `${location.origin}${location.pathname}#${hash}`;
  history.replaceState(null, '', `#${hash}`);
  try {
    await navigator.clipboard.writeText(url);
    toast('Link to this solution copied');
  } catch {
    toast('Link is in the address bar');
  }
});

document.addEventListener('keydown', (e) => {
  const inEditor = (e.target as HTMLElement).closest?.('.cm-editor');
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !inEditor) {
    e.preventDefault();
    void run();
  } else if (e.altKey && (e.key === 'ArrowRight' || e.key === 'ArrowLeft') && !inEditor) {
    e.preventDefault();
    void step(e.key === 'ArrowRight' ? 1 : -1);
  } else if (e.key === '?' && !inEditor && !(e.target instanceof HTMLInputElement)) {
    ($('help') as HTMLDialogElement).showModal();
  }
});

let resizeFrame = 0;
window.addEventListener('resize', () => {
  cancelAnimationFrame(resizeFrame);
  resizeFrame = requestAnimationFrame(rerender);
});

// ---- start ----

async function start(): Promise<void> {
  const shared = location.hash ? await decodeShare(location.hash) : null;
  const sharedIndex = shared ? LEVELS.findIndex((l) => l.id === shared.levelId) : -1;
  if (shared && sharedIndex >= 0) {
    history.replaceState(null, '', location.pathname);
    loadLevel(
      sharedIndex,
      shared.code,
      'Someone shared this solution with you, so press Run to see how it does, but note that running or editing it replaces your saved code for this level.',
    );
    return;
  }
  const saved = LEVELS.findIndex((l) => l.id === progress.current);
  const firstOpen = LEVELS.findIndex((l, i) => isUnlocked(progress, i) && !isSolved(progress, l.id));
  loadLevel(saved >= 0 && isUnlocked(progress, saved) ? saved : Math.max(0, firstOpen));
}

void start();
