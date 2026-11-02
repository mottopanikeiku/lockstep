/**
 * Immediate post-dominators for a CFG given as successor lists, using the
 * Cooper-Harvey-Kennedy iterative algorithm on the reversed graph. `exit` is
 * a virtual node with no successors. Nodes that cannot reach `exit` (infinite
 * loops) get a virtual edge to it so every node has an answer.
 */
export function postDominators(succ: number[][], exit: number): Int32Array {
  const n = succ.length;
  const edges = succ.map((s) => s.slice());
  const preds: number[][] = Array.from({ length: n }, () => []);
  for (let v = 0; v < n; v++) for (const s of edges[v]) preds[s].push(v);

  const reaches = new Uint8Array(n);
  const walk = [exit];
  reaches[exit] = 1;
  while (walk.length) {
    const v = walk.pop()!;
    for (const p of preds[v]) {
      if (!reaches[p]) {
        reaches[p] = 1;
        walk.push(p);
      }
    }
  }
  for (let v = 0; v < n; v++) {
    if (!reaches[v]) {
      edges[v].push(exit);
      preds[exit].push(v);
    }
  }

  // Postorder of a DFS from exit along predecessor edges.
  const order = new Int32Array(n).fill(-1);
  const postorder: number[] = [];
  const seen = new Uint8Array(n);
  const stack: [number, number][] = [[exit, 0]];
  seen[exit] = 1;
  while (stack.length) {
    const top = stack[stack.length - 1];
    const [v, i] = top;
    if (i < preds[v].length) {
      top[1]++;
      const p = preds[v][i];
      if (!seen[p]) {
        seen[p] = 1;
        stack.push([p, 0]);
      }
    } else {
      order[v] = postorder.length;
      postorder.push(v);
      stack.pop();
    }
  }

  const ipdom = new Int32Array(n).fill(-1);
  ipdom[exit] = exit;
  const intersect = (a: number, b: number) => {
    while (a !== b) {
      while (order[a] < order[b]) a = ipdom[a];
      while (order[b] < order[a]) b = ipdom[b];
    }
    return a;
  };
  let changed = true;
  while (changed) {
    changed = false;
    for (let k = postorder.length - 1; k >= 0; k--) {
      const v = postorder[k];
      if (v === exit) continue;
      let next = -1;
      for (const s of edges[v]) {
        if (ipdom[s] === -1) continue;
        next = next === -1 ? s : intersect(s, next);
      }
      if (next !== ipdom[v]) {
        ipdom[v] = next;
        changed = true;
      }
    }
  }
  return ipdom;
}
