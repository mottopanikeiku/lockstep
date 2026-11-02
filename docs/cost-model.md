# How lockstep runs and scores a kernel

The simulator is a teaching model. It charges for the things a real GPU punishes (warps that split, scattered global loads, shared memory bank conflicts, threads waiting at barriers) with small round numbers you can add up by hand. It does not predict run time on any real GPU: there is no latency hiding, no cache, no occupancy limit, and blocks run one after another.

## Execution

Blocks run one at a time, in order. Inside a block the warps take turns, one instruction each.

A warp runs its 32 lanes in lockstep under an active mask. When the lanes disagree at a branch, the warp runs the taken side with those lanes, then the other side with the rest, and the two groups merge again at the branch's immediate post-dominator: the first instruction that every path from the branch has to pass through. This is the reconvergence stack of pre-Volta NVIDIA GPUs. It is why divergence costs time: both sides are issued, each with some lanes switched off. Lanes that `return` drop out of the warp for good.

`threadIdx`, `blockIdx`, `blockDim` and `gridDim` are `int` here. In CUDA they are `unsigned`, which only matters if you subtract below zero.

## Cycles

Every warp keeps its own clock, and each instruction it issues moves the clock forward:

| Instruction | Cycles |
|---|---|
| anything | 1 to issue, whatever the number of active lanes |
| `expf`, `logf`, `sqrtf`, `rsqrtf`, float `/`, int `/` or `%` by a non-constant | 4 in total |
| `__shfl_*_sync` | 2 in total |
| global load, store or atomic | 1 + 4 for each 32-byte sector the warp touches |
| global atomic | 4 more for each lane that hits an address another lane already hit |
| shared load or store | 1 + 1 per wavefront |
| shared atomic | 1 + the most lanes that land in one bank |
| `__syncthreads()` | 1, then the warp waits until the slowest warp of its block arrives |

Shared memory has 32 banks of 4-byte words; word `w` lives in bank `w % 32`. One wavefront serves at most one word per bank, so an access where some bank is asked for n different words takes n wavefronts (an n-way bank conflict). Lanes reading the same word share it. Atomics on the same word do not share, so they serialize.

The cycles score is the sum of all warps' clocks at the end, which is the same as: instructions issued, plus the extra cycles in the table, plus the time warps spent waiting at barriers. The editor margin shows that total split by source line.

The sectors score counts the 32-byte global memory sectors moved, over the whole run. A warp reading 32 consecutive floats moves 4; reading every 32nd float moves 32.

The lines score counts lines that contain code, so blank lines, comments and lines holding only braces are free.

Par for each level is the score of a reference solution, saved in `src/levels/par.json`. The test suite checks that the file matches.

## Errors

A run stops with an error, pointed at a source line, for:

- Out of bounds: an index outside a global array, or outside a whole shared array.
- Deadlock: `__syncthreads()` reached by only some lanes of a warp, warps of one block waiting at different `__syncthreads()` calls, or a barrier after some threads of the block have returned. CUDA needs every thread of the block at the same barrier. Shuffles check that every lane named in the mask takes part.
- Race: two different threads touch the same 4-byte word, at least one of them with a plain (non-atomic) write, and nothing orders them. Threads of one block are ordered by a `__syncthreads()` between the two accesses, and lanes of one warp also by `__syncwarp()`. Threads in different blocks are never ordered. This is the rule compute-sanitizer's racecheck applies to shared memory; lockstep applies it to global memory too.
- Wrong answer: an output differs from the expected values (exactly, except for a small relative tolerance on the softmax and attention levels).

Shared memory starts out as garbage (NaN for floats). Reading a word nobody wrote gives a warning.
