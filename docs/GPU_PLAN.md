# The WebGPU migration

Goal: **~40k fluid particles at a solid 60 Hz on integrated-GPU-class hardware**,
a ~10x raise of the current ceiling, without raising the hardware floor. The
lever is WebGPU compute for the particle pipeline. The prerequisite is moving
the whole game core off the main thread, behind a message protocol.

This document is the architecture and the execution plan. It supersedes the
"main thread to start" line of `docs/PLAN.md` §5.1; nothing else in PLAN.md
changes.

## Status: SHIPPED (2026-08-28) - P1-P4 landed on this branch

Measured on the Intel UHD (gen-12lp) adapter, the hardware floor the goal
was stated against: **40.5k fluid particles at ~20 ms/frame (~50 fps), with
60 Hz holding to ~33k**; the game's own beach scene (22.6k
fluid) runs comfortably inside the frame budget. The CPU reference needs
~156 ms for the 40k scene, so the floor-adapter speedup is ~7x - and this
adapter is the worst case of the worst case (shared with the desktop
compositor); anything above it, discrete GPUs included, clears 40k @ 60 Hz
outright. An earlier ~16.7 ms median was measured before the adversarial
review caught that the gathers lacked the CPU's mid-frame support margin;
correctness cost ~6 ms (the margin is substep-scaled, paying only for the
binning drift actually accrued). The wall number includes host passes, state
upload, encoding and submission. The readback is PIPELINED up to TWO frames
deep, consumed non-blockingly (`reap()`): a hard lesson twice over, not an
optimisation - the mapAsync FENCE costs ~17-21 ms in interactive browsers
regardless of workload (1-3 ms headless, which is why benches missed it),
so any loop that waits for a fence younger than that caps below 60 Hz even
on an empty scene. Pipelining that deep is only sound because positions and
velocities are DEVICE-RESIDENT, exactly as this plan prescribed: each frame
chains from the GPU's own previous output, and sync() uploads only slots
the host wrote (write-stamped creates/recycles). The first implementation
took a stateless full-reupload shortcut instead, and under fence slip it
re-simulated stale host state - the world forked into two interleaved
half-rate timelines (on screen: the whole fluid blinking between two
states, spawn guards admitting overlaps that discharged at the speed cap).
test/gpu/pipeline.test.ts pins both the advancement rate and the recycled-
slot stamp guard, mutation-checked. Host logic sees positions up to two
frames old, the lag this plan always budgeted for.

**The open ceiling, precisely (2026-08-29): host-side coupling forces.**
Buoyancy, hydrostatic wall load and water drag are computed on the host
from read-back state, so pipeline depth IS force lag - and force lag is
dynamically unstable: at depth 4 (66 ms, tried for fence headroom) the
phase error resonance-pumped a floating crate out of the water and into
the sky; even at depth 2, catch-up bursts recompute forces from the same
un-landed state, so the lag doubles exactly when the sim is struggling
(measured: bob amplitude 0.27 m -> 2.57 m in 12 s at 52 Hz). Two
mitigations shipped: rendering is capped at 60 fps (uncapped rAF on
high-refresh displays burned the same iGPU the solver needs - fences went
~20 ms -> ~46 ms median under load), and every catch-up step after the
first in a tick FLUSHES before stepping (test-pinned), which keeps forces
fresh at the price of honest, deeper slow-mo under load (~44 Hz at 7.8k
on the floor adapter). The real fix, next: move the water column field
and the coupling forces INTO kernels (the wetness census already is).
Fresh forces every frame from device state ends the depth ceiling, the
burst flushes, and most of the readback entirely - readback then feeds
only damage, spawn guards, probes and snapshots, all lag-tolerant, and
the fence stops being on the critical path at any depth.

What shipped matches this plan with three notable learnings:

- The dense ROW-MAJOR grid (not the CPU's hashed table) was the decisive
  optimisation: sorted entries become spatially ordered, a cell ring is
  three contiguous runs, and the gather kernels go from latency-bound to
  under budget. Optimisation history with per-step bisect numbers lives in
  the P3 commit message.
- Chromium's headless SHELL has no GPU adapter, and Playwright injects
  --disable-gpu into headless launches; the test rig runs full Chromium
  with that flag stripped (vitest.config.ts documents it).
- A HIDDEN tab runs ~3.5x slower - Chrome deprioritises background GPU
  work. That is the environment, not the solver; visible tabs run at bench
  speed, hidden ones slow-mo gracefully under the stepper's drop-debt
  contract.

Deferred, deliberately: GPU-direct rendering (Pixi stays on WebGL reading
snapshots); a member-segment grid for capsule contacts if member counts
ever reach the thousands (the whole-structure AABB early-out covers current
content); the spawnDisc jitter-order fix (pre-existing, tracked
separately).

## Where the time goes today (measured, 2026-08-28, i9-14900HX)

Per-pass profile of `SimWorld.step` (fluid only, spacing 0.25, 12 substeps),
linear in particle count at ~3.9 µs/particle/frame:

| fluid | step | project | neighbours | velocities | terrain | rest |
|---|---|---|---|---|---|---|
| 4k | 16.0 ms | 58% | 17% | 14% | 6% | 5% |
| 16k | 60.6 ms | 62% | 16% | 11% | 5% | 6% |

Structure passes are <1% even with four walls standing in flood water. >95% of
the step is per-particle/per-pair work. The 60 Hz budget is 16.7 ms: the
real-time ceiling is ~4k particles, and render+sim share one thread on top.

At 40k particles the GPU arithmetic is trivial: ~240k pairs x 2 passes x 12
substeps ~ a few hundred MFLOP/frame ~ 20 GFLOP/s at 60 Hz, about 2% of an
Intel UHD-class iGPU. The cost of this migration is engineering and
revalidation, not player hardware.

## End state

Three layers, two boundaries:

```
main thread            |  sim worker                      |  GPU
-----------------------+----------------------------------+------------------
Pixi render (views)    |  SimHost: fixed-timestep loop    |  particle pipeline
DOM UI, HUD            |  Session (doc+solution+indices)  |  (hash, PBF,
pointer tools, camera  |  Conditions, WaterEmitter        |   contacts, XPBD
editor picking         |  Field/Terrain, level ids        |   joints, velo-
save/load dialogs      |  SimWorld: topology + forces +   |   cities, XSPH)
                       |    damage/breakage + events      |
        <- snapshots   |  SolverBackend: cpu | webgpu     |  <- 1-frame-lagged
        commands ->    |                                  |     readback
```

Two invariants define the boundaries:

1. **Raw slot indices never cross the worker boundary.** The particle and
   constraint stores recycle freed indices on the very next create; the only
   safe place to hold an index is next to the code that frees it. Session -
   the sole keeper of id->index maps - moves into the worker. The main thread
   speaks document ids and positions, nothing else.
2. **The CPU solver is the reference implementation.** Every existing test
   keeps constructing `SimWorld` directly and stepping it synchronously in
   Node. The GPU backend is validated against the CPU backend by
   tolerance-banded physics probes in a browser test rig, never by bitwise
   comparison (f32 kernels, atomics, and neighbour-order differences make
   bitwise parity impossible by construction).

### Main <-> worker protocol

Transport: plain `postMessage`. Commands are structured-clone objects (a typed
discriminated union, FIFO-ordered). Snapshots are binary `ArrayBuffer`s,
transferred (zero copy), pooled - the main thread posts consumed buffers back.
No SharedArrayBuffer anywhere: nothing needs COOP/COEP headers, hosting stays
static-file simple, and at ~400 KB/frame the transfer cost is noise.

**Commands (main -> worker)** - the union covers exactly today's call surface:

- editor: `addNode/addMember/removeMember/addAnchor/removeAnchor/
  addWorldObject/removeObject/clearBuild/clearAll/undo/redo`
- session: `play/reset/setPaused`, `loadDoc` (doc+solution), `setFieldWidth`
- conditions: `setWind/setFlood/setWaves`
- water tool: `splash(x,y)`, `stream(x,y)|null` (held-cursor position, set
  each frame while the tool is held), `setFlow`, `clearFluid`
- solver tuning: `setSubsteps/setLinearDamping/setFluidIterations/
  setFluidSpacing`, `setBackend('cpu'|'webgpu')`
- probes/debug: `loadProbe(name)`, `pump(n)` (advance n steps synchronously,
  for headless verification), `requestSave`

The worker applies commands immediately on arrival - including while paused
(paused building and paused splashes are product features). Splash-while-paused
keeps working because the recent-spawn dedupe lives inside `SimWorld.spawnDisc`
and moves with it.

**Snapshot (worker -> main)**, one per completed sim frame (and one after any
command batch applied while paused). Binary layout, slot-indexed, sparse (dead
slots carry a flag; indices are stable so views can pool sprites):

- header: frame no, simTime, loop stats (stepped/starved), scalar block
  (particle/fluid/object counts, fluid spacing, substeps, wind gust, severity,
  peakLoad, maxDamage, breakCount, budget-relevant echoes)
- particles: `n`, `posX f32[n]`, `posY f32[n]`, `flags u8[n]`
  (alive | kind | pinned)
- structure view-model (id-based, for editor picking and overlay): anchors
  `{id, x, y}`, nodes `{id, x, y}`, members `{id, segments: (x0,y0,x1,y1)[],
  strain, damage, material}` - structure counts are 10^2-10^3, this is small
- clusters: `{alive, cx, cy, angle}` + one-time `{hw, hh, tint}`
- events since last snapshot: `BreakEvent[]` (position, material, strain - the
  future splinter/sound feed, carried in full from day one)
- doc/solution echo: only in reply to `requestSave` and after mutations that
  change them (undo/redo/load), as JSON - not per frame

The HUD reductions (`peakLoad`, `maxDamage`) are computed worker-side from
arrays it already owns; the per-rAF full-table scans in `app.ts` disappear.

**Loop ownership.** The fixed-timestep accumulator (60 Hz, max 3 catch-up
steps, drop-debt slow-mo contract) moves into the worker verbatim, driven by a
self-posting timer. The main thread renders on rAF from the latest snapshot,
keeping the previous snapshot for interpolation: sim frames and rAF are not
phase-locked, and at high particle counts the sim deliberately runs slower
than render, which is exactly when interpolated draw matters. `Game.pump()`
becomes the `pump` command (await-able) so headless verification keeps working.

**Editor picking** runs on the main thread against the structure view-model in
the latest snapshot (node/anchor/member positions by id). Same-frame accuracy
is not required - picking against a <=1-frame-old position was already the
behaviour under the old synchronous code whenever the sim was paused, and a
16 ms stale pick radius is invisible at editor zoom levels.

### The solver seam (inside the worker)

`SimWorld` splits into what it already is in the profile:

- **SimWorld (host side, stays TypeScript/CPU):** topology (particle/
  constraint/cluster create/destroy, spawn guards, terrain boundary resample),
  the once-per-frame force passes (buoyancy, hydrostatic wall load, water
  drag, wind - they write `accX/accY` and touch only structure-scale data),
  WaterField (column heights, built from positions), wind field advance,
  damage/plasticity/breakage (reads strain/angle, mutates topology, emits
  break events and destroyed-index drains), and the conditions-facing spawn
  admission queries.
- **SolverBackend (swappable):** everything per-substep plus the neighbour
  build - `beginFrame` (hash), then 12x (predict, cluster shape match, bend
  solve, distance solve, PBF project, solid contacts, member capsule
  contacts, terrain/bounds contacts, velocity derivation + damping, hull
  viscosity), then XSPH. Interface (all array views on the CPU path;
  upload/dispatch/readback on the GPU path):

  ```ts
  interface SolverBackend {
    // Sync topology/param changes into backend-owned state. Dirty-range
    // based; called once per frame before step.
    sync(world: SimWorldState): void
    // Run one full frame (all substeps). CPU: synchronous, mutates the
    // SoA arrays in place. GPU: submits the frame's command buffer.
    step(dt: number): void
    // Resolve the frame's outputs into the SoA arrays the host logic
    // reads (positions, velocities, strain, angle, density, wet counts).
    // CPU: no-op. GPU: await staging-buffer map; on the pipelined path
    // this is frame N-1's data.
    readback(): Promise<void>
  }
  ```

Two behaviour-preserving refactors are folded into this extraction, because
both currently poke particle arrays from outside and would otherwise need
GPU-side special cases:

- `Conditions.driveWaves` (direct `velX` blending in the wave zone) becomes a
  sim parameter (`waveDrive: {x0, push, blend} | null`) applied inside the
  solver's velocity pass. Conditions sets the parameter; the solver applies
  it on whichever device owns velocities.
- `WaterEmitter`/`Conditions` particle admission and drain already go through
  `SimWorld` create/destroy - they stay host-side as topology commands.

**GPU frame pipelining.** The GPU backend records the entire frame (hash build
+ 12 substeps + readback copy) into one command submission per frame and never
stalls mid-frame. Host logic consumes the *previous* frame's readback:
damage/breakage, water field, force passes, snapshots, and spawn-admission
queries all see positions one frame (16.7 ms) old. Spawn admission already
queries last frame's hash by design on the CPU path today; breakage moving one
frame late is imperceptible; force passes are frame-rate quantities. This lag
is a documented property of the GPU backend, not a bug.

### GPU backend design

Data: the particle SoA maps 1:1 to storage buffers (f32/u32 arrays, slot-
indexed, sized to capacity, grown by reallocate-and-copy between frames).
Constraint tables, cluster tables, material table (2 rows), terrain heights,
and per-frame uniforms (dt, substeps, gravity, caps, wave drive, bounds) ride
along the same way. Host-side dirty tracking uploads changed rows via
`queue.writeBuffer` each frame; `accX/accY` upload in full (they are rewritten
every frame anyway).

Kernels (WGSL), in dispatch order:

| kernel | freq | notes |
|---|---|---|
| `grid_count`, `grid_scan`, `grid_scatter` | frame | counting-sort hash, same cell size (2.75 x spacing) and support margin as CPU so neighbour semantics carry over |
| `census` | frame | solid wet counts + summed fluid velocity at solids (gather per solid) |
| `predict` | substep | integrate acc+gravity, save prev |
| `cluster_reduce` + `cluster_apply` | substep | COM + atan2 shape match; one workgroup per cluster, parallel reduction, matches CPU math exactly |
| `joints_solve` | substep | distance + bend XPBD, **graph-coloured**: constraints pre-coloured host-side so no two in a colour share a particle; colours dispatched sequentially with the pass order preserved (clusters -> bend -> distance). Within a colour, parallel application is exact Gauss-Seidel (disjoint particles); only the within-pass visiting order changes vs CPU |
| `fluid_density` + `fluid_correct` | substep | PBF projection re-formulated as **gather-per-particle** (each particle walks its 3x3 grid ring; kernels evaluated from both sides rather than cached per pair - GPU has flops to burn and this removes every scatter from the hot path). Fluid-solid coupling: fluid side gathers solid neighbours; solid reaction displacement accumulated via fixed-point atomics, applied in `fluid_correct`'s tail with the same clamps as CPU |
| `contacts_solid` | substep | object-object, gather per particle over grid |
| `contacts_member` | substep | capsule contacts: one thread per member sample point walking the grid (structure-scale dispatch); particle pushes and member-endpoint reactions accumulate via fixed-point atomics (i32, 2^-20 m quantum) + u32 hit counters, preserving the sum-then-average order-independence the CPU pass was designed for |
| `contacts_resolve` | substep | apply averaged contact + summed member reactions + terrain/bounds pushout (heightAt/normalAt inlined from the heights buffer), prev-carry semantics identical to CPU |
| `velocities` | substep | (pos-prev)/h, speed cap, per-kind damping, ground friction/restitution, wave drive, then joint damping (colour-sequenced) and hull viscosity (gather both sides, wet-count normalised) |
| `xsph` | frame | XSPH viscosity, gather form |
| `pack_readback` | frame | positions, velocities, strain, angle, density, wet counts into one staging copy |

WGSL has no f32 atomics: every scatter accumulator (contact gathers, member
reactions, solid pressure reactions) uses i32 fixed-point at 2^-20 m (~1 µm)
quantum - orders of magnitude below the solver's own correction caps.

**What deliberately does NOT port:** the CPU's unique-pair lists and cached
pair geometry (a scatter optimisation that is an anti-pattern on GPU), the
`MAX_NEIGHBOURS` truncation order (GPU caps at the same count but in grid
order - a different pair set in pathological crowding, covered by parity
bands), bucket entry ordering, and the exact within-pass constraint visiting
order (colour order instead). Each is listed in the parity suite's "expected
divergence" notes.

**Backend selection:** probe `navigator.gpu` in the worker at startup; use
WebGPU when an adapter with the required limits exists, else CPU. The solver
panel shows the active backend and offers a manual override (`cpu` forced is
also the escape hatch for any GPU-specific bug a player hits). If WebGPU
exists on the main thread but not in the worker (older Firefox), we still run
CPU-in-worker - no main-thread GPU mode; that niche shrinks monthly and the
complexity is not worth carrying.

### Validation strategy

Layered, per the testing policy (every layer red-first or mutation-proven):

1. **Existing suite, untouched semantics** (Node, CPU): the P1/P2 refactors
   must keep all 114 tests green *unchanged*, including the exact-position
   determinism fingerprints - the CPU solve path is a code motion, not a
   rewrite. Any test edit in these phases is a red flag in review.
2. **Protocol layer** (Node): command codec, snapshot encode/decode
   round-trips, SimHost applied-command semantics (paused splash, undo/redo
   rebuild, save echo), buffer-pool recycling, view-model picking maths.
   SimHost is a plain class; the actual `Worker` shell is a thin exempt file.
3. **GPU kernels** (browser, WebGPU): fixture tests with hand-computable
   answers - lattice density vs the CPU calibration constant, single-pair
   lambda, capsule pushout geometry, colour-partition validity (no colour
   shares a particle), fixed-point round-trip error bounds.
4. **CPU/GPU parity** (browser): harness scenes (rest settle, dam break,
   flood-vs-wall, floating crate, breakage under load) run on both backends,
   compared on the probe metrics the suite already trusts - settle speeds,
   energy trends, surface flatness, volume conservation, wall load fraction,
   break counts - inside explicit tolerance bands. Determinism fingerprints
   stay CPU-only by policy.
5. **Perf** (browser): GPU throughput guard at 40k particles on both this
   machine's adapters - the 4090 and, via `powerPreference: 'low-power'`, the
   Intel UHD, which is the hardware-floor proxy the 10x claim is made on.
6. **The screen.** Every phase ends with the real app run and looked at,
   through the existing launch configs.

Browser tests are a new vitest project (`test/gpu/**`, `@vitest/browser` +
playwright/chromium) with a separate npm script; they skip loudly (not
silently) when WebGPU is unavailable.

## Phases

Each phase lands green (`npm run test:run` + typecheck) and is verified on
screen before the next begins.

- **P1 - the worker boundary.** `src/runtime/` (protocol types, snapshot
  codec, SimHost, worker shell, main-thread SimClient); Session, Conditions,
  WaterEmitter, Field/Terrain, level-id counter move worker-side; app.ts
  rewired to SimClient (views draw from snapshots, editor picks from the
  view-model, HUD reads the scalar block); interpolated draw from the last
  two snapshots. Exemption table gains the worker shell + client wiring;
  protocol/host/codec/picking are tested in Node. Acceptance: full suite
  green; app plays identically (build, run, break, water tool, save/load,
  undo, probes); sim overload no longer janks the camera or UI.
- **P2 - the solver seam.** Extract the substep pipeline into
  `CpuSolver implements SolverBackend` (code motion); `waveDrive`
  parameterised; `SolverBackend.sync/step/readback` lifecycle in place with
  CPU no-ops. Acceptance: full suite green with **zero test edits**;
  determinism fingerprints byte-identical to pre-refactor; throughput test
  unchanged.
- **P3 - the WebGPU backend.** Buffers, kernels, colouring, pipelined
  readback, backend probe + selection + panel override; browser test project
  with kernel fixtures and the parity suite; perf guard on both adapters.
  Acceptance: parity suite green inside bands on the 4090 *and* the UHD;
  40k @ 60 Hz sustained on the UHD adapter; CPU fallback still bit-identical
  on the Node suite.
- **P4 - integration polish.** Resolution slider re-ranged per backend
  (spacing floor when GPU active), HUD backend/perf readouts, docs
  (PLAN.md supersession note, CLAUDE.md test-command additions), README run
  instructions for the GPU test project.

## Risks

- **Physics divergence on GPU** - the tuned coupling is the crown jewels.
  Mitigated by: preserving pass order and per-substep structure exactly,
  porting formulas not approximations, the parity bands, and keeping every
  clamp/cap semantically identical. Where divergence is structural
  (neighbour sets, colour order, f32, atomic sums) it is enumerated and
  tested-around, not discovered later.
- **WGSL surface area** (~15 kernels) - the largest new-code mass in the
  project. Mitigated by kernel fixture tests and by the CPU reference living
  forever next to it.
- **Browser test flakiness** (headless WebGPU) - mitigated by running against
  real adapters on this machine now, loud skips elsewhere, and keeping the
  Node suite authoritative for everything non-GPU.
- **Toolchain edges** - TS 7 + Vite 8 (rolldown) + vitest 4 browser mode +
  `@webgpu/types` are all current-generation; each gets verified at install
  time in P3 before any kernel work sits on top.
- **Worker-boundary regressions in feel** (input latency, judder) -
  mitigated by snapshot interpolation in P1 and by on-screen verification
  being a phase gate, not an afterthought.
