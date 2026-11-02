// Compiles every level's starter and reference kernel with nvcc, exactly as the
// game shows them, each in a .cu file with a host function that launches it the
// way the game does. Reference solutions must also compile without warnings.
// Needs nvcc on PATH (or NVCC=/path/to/nvcc); no GPU.
//
//   npm run nvcc
import { spawnSync } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { runnerImport } from 'vite';

const nvcc = process.env.NVCC ?? 'nvcc';
const out = 'build/nvcc';
const load = async (path) => (await runnerImport(fileURLToPath(new URL(path, import.meta.url)))).module;
const { LEVELS } = await load('../src/levels/index.ts');
const { compile } = await load('../src/compile.ts');
const { signature } = await load('../src/game/attempt.ts');

rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });

const dim = (d) => `dim3(${d.x}, ${d.y})`;
let failed = 0;

for (const [i, level] of LEVELS.entries()) {
  for (const which of ['starter', 'reference']) {
    const src = level[which];
    const compiled = compile(src);
    if (!compiled.ok) throw new Error(`${level.id} ${which}: lockstep rejects it: ${compiled.error.message}`);
    const file = `${out}/${String(i + 1).padStart(2, '0')}-${level.id}.${which}.cu`;
    const args = level.params.map((p) => p.name).join(', ');
    writeFileSync(
      file,
      `${src}\n// lockstep launches it like this:\n` +
        `void launch(${signature(level)}) {\n` +
        `  ${compiled.kernel.name}<<<${dim(level.launch.grid)}, ${dim(level.launch.block)}>>>(${args});\n` +
        `}\n`,
    );
    const flags = which === 'reference' ? ['--Werror', 'all-warnings'] : [];
    const r = spawnSync(nvcc, ['-c', ...flags, file, '-o', '/dev/null'], { encoding: 'utf8' });
    if (r.error) throw r.error;
    const ok = r.status === 0;
    if (!ok) failed++;
    console.log(`${ok ? 'ok  ' : 'FAIL'} ${file}`);
    if (!ok) console.log(r.stdout + r.stderr);
  }
}

console.log(`${LEVELS.length * 2 - failed} of ${LEVELS.length * 2} kernels compile`);
process.exit(failed ? 1 : 0);
