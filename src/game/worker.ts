// Runs attempts off the main thread, so a slow or endless kernel never freezes the page.
import { LEVELS } from '../levels';
import { attempt, type Attempt } from './attempt';

export interface RunRequest {
  id: number;
  level: number;
  src: string;
}

export type RunReply = { id: number; attempt: Attempt } | { id: number; error: string };

/** The buffers behind every typed array in an attempt, so postMessage moves them instead of copying. */
function transferList(a: Attempt): ArrayBuffer[] {
  const out = new Set<ArrayBuffer>();
  const add = (x: unknown) => {
    if (ArrayBuffer.isView(x)) out.add(x.buffer as ArrayBuffer);
  };
  Object.values(a.inputs).forEach(add);
  for (const o of a.outputs) {
    add(o.expected);
    add(o.actual);
  }
  if (a.sim) {
    a.sim.buffers.forEach(add);
    add(a.sim.lineCycles);
    if (a.sim.trace) Object.values(a.sim.trace).forEach(add);
  }
  return [...out];
}

self.onmessage = (e: MessageEvent<RunRequest>) => {
  const { id, level, src } = e.data;
  let reply: RunReply;
  try {
    const a = attempt(LEVELS[level], src, { trace: true });
    reply = { id, attempt: a };
    self.postMessage(reply, { transfer: transferList(a) });
  } catch (err) {
    reply = { id, error: err instanceof Error ? err.message : String(err) };
    self.postMessage(reply);
  }
};
