# typhoonbuffoon — architecture & execution plan

A bridge-builder where you don't build bridges. You build disaster defenses — flood walls,
stilts, domes, seawalls — then press Play and a storm ramps up against them with simulated
wind and water.

This document is the working plan. It records the decisions already settled, the
architecture those decisions imply, and an ordered path to a playable sandbox.

> **Status:** The deep genre survey (Poly Bridge / Bridge Constructor / Pontifex teardown)
> is deliberately deferred — we're prioritising getting something running. The genre notes
> below are from general knowledge, not a research pass, and should be revisited before
> level design starts in earnest.

---

## 1. Settled decisions

These came out of design discussion and are treated as fixed. They're expensive to reverse,
which is exactly why they were settled first.

| # | Decision | Consequence |
|---|---|---|
| 1 | **Water is particles**, not a heightfield | Water flows through breaches, fills cavities, splashes, two-way couples. Costs the bulk of the perf budget. |
| 2 | **Structures are simulated soft constraints (XPBD)**, not rigid bodies with joints | Squash/stretch is real physical state. Nothing to reconcile between render and sim. |
| 3 | **Stiff axially, compliant in bending** | This is what separates "bends like a palm tree" from "jiggles like jello". Jelly comes from compliant *distance* constraints; bending comes from compliant *angular* ones. |
| 4 | **No high-frequency wobble** | Damping expressed as a ratio of critical (ζ ≈ 0.8–1.0), plus Rayleigh stiffness-proportional damping to preferentially kill high-frequency modes. |
| 5 | **Plastic yield is per-material, defaults off** | Wood is near-linear-elastic to fracture (yield = ∞). Steel yields and stays bent. Gives the two materials different failure personalities for free. |
| 6 | **Damage is a separate scalar from plasticity** | Damage accumulates with stress, lowers the break threshold, changes no geometry, and later drives cracked/splintered board art. Universal to all materials. |
| 7 | **The level document is authoritative; the sim world is a projection of it** | Reset is exact. Sim state never writes back into the document. |
| 8 | **Level width is author-set**, typed in metres, unclamped | Most levels land around 50–150 m, but the field is free. Particle spacing still sets a *minimum structural thickness* — see §8. |
| 9 | **Building is allowed during the sim** | Removes the genre's hard build/sim split. The sim world must support incremental add/remove of nodes and members at runtime. This is a solver data-structure requirement, not a UI feature, and it is far cheaper to design in than to retrofit. |
| 10 | **No storm/event system yet** | Conditions are driven by direct sandbox sliders — wind, flood, wave — plus a simulation-resolution control. Authored storm timelines are deferred. |

### The unifying insight

Decisions 1 and 2 are the *same solver*. Position Based Fluids is PBD applied to a density
constraint; XPBD structures are PBD applied to distance and bending constraints. One
position-based solver runs both, and structure↔fluid coupling falls out of the formulation
instead of being bolted on across an engine boundary.

This is why we write the solver rather than adopting one — see §5.1.

---

## 2. What we inherit from the genre

Bridge builders are, mechanically, all the same game:

- A **node-and-member graph**. The player places nodes and connects them with members. That
  graph is the entire solution.
- A **budget**. Members cost money by length and material. The puzzle is the constraint.
- A **hard build/sim mode split**. You design in a paused world, then hit Play and watch.
- **Stress visualisation**. Members colour green→red by load. This is the genre's single most
  important legibility device — it's how the player learns *why* it failed.
- **Member breakage** past a threshold, usually cascading into total collapse.
- **Instant reset** back to the exact build state.

Five of these six carry over unchanged. **The build/sim split does not** — see §3. That is the
single largest structural departure from the genre and it shapes the solver, not just the UI.

## 3. Where this game diverges

| Divergence | Design risk |
|---|---|
| **Persistent environmental forces** (wind, water) rather than a single load event (a truck crossing) | Failure becomes gradual and time-based rather than instantaneous. Legibility is harder — the player must understand a slow siege, not one snap. |
| **A ramping storm timeline** | The difficulty curve is now temporal. Needs authoring and telegraphing. |
| **Real deformation** | Beautiful, but tuning-heavy. A structure that looks rubbery destroys the fantasy. |
| **Particle water** | The dominant perf cost, and impulsive/noisy as a force source. |
| **Four archetypes** (flood wall, stilts, dome, tidal defense) | Each wants different level geometry and win conditions. Only build one for now. |
| **Building during the sim** — edit while it runs, rather than build-then-watch | The biggest departure. Forces runtime mutation of a live solver. |

The genre's core risk is "why did it fail" legibility. Ours is worse, because the answer
unfolds over ninety seconds. Stress colouring and visible deformation are not polish here —
they are the primary communication channel, and they should exist early.

---

## 4. Verdict on the proposed milestone

**Right destination, wrong path.**

The proposed milestone — pannable parallax field, anchor points, wood/metal construction,
water/wind/flood/tidal tools, clear button, "proper physics for the above" — is a good
definition of a first playable sandbox. It's the right thing to be building toward.

But as an ordering it has two problems.

**The last bullet hides most of the work.** "Proper physics for the above, meaning wood
floats, joints break, etc." is roughly 60% of total effort — the XPBD solver, the fluid
solver, the coupling between them, and the tuning that makes it feel like timber instead of
rubber. Everything above it is comparatively cheap. Sequencing it last means the riskiest
work lands after the most has already been built on assumptions about it.

**Editor-first means authoring content for a game whose feel is unknown.** Build tools encode
assumptions — what a member is, how anchors attach, what snapping feels like. If the solver
turns out to want segmented members with a per-material segment count (it does), an editor
built before that has to be reworked.

**The resequence:** take the cheap, motivating visual layer early (steps 1–2 — you get your
pannable beach on day one), then drive straight through the solver to the *signature moment*
on one hardcoded scene (steps 3–9), then build the editor out of a system you already trust
(steps 10–13). Same destination, risk retired in a sane order, something worth looking at at
every point.

**What's missing from the milestone as stated:** stress visualisation. It isn't in the list,
but it's the genre's core legibility device and doubly important here. It's nearly free once
the solver reports strain, and it should land with the solver, not after.

---

## 5. Architecture

### 5.1 Stack

| Layer | Choice | Why |
|---|---|---|
| Build | **Vite + TypeScript** | Fast HMR, near-zero config, TS throughout. |
| Render | **PixiJS v8** | WebGL/WebGPU batching. Canvas2D will not survive thousands of particles. Flat-shape art means we lean on cheap geometry, not textures. |
| Physics | **Custom XPBD solver** | No JS engine gives XPBD structures + PBF fluid + coupling in one solver. Rapier/planck/matter are rigid-body families; LiquidFun bolts particles onto Box2D and is unmaintained. The core loop is a few hundred lines and both halves share it. |
| UI | **Plain DOM overlay** | Editor panels are a palette, a budget readout, a play bar. A framework is friction at this stage. |
| Data | **Struct-of-arrays typed arrays** for particles and nodes, from day one | Cheap now, expensive to retrofit, and the precondition for a worker/WASM/GPU move later. |

Main thread to start. Move the solver behind a worker boundary when profiling demands it —
the SoA layout makes that a transfer, not a rewrite.

### 5.2 The document/projection split

```ts
// The authoritative, serialisable document. The sim NEVER writes to this.
interface LevelDoc {
  version: number
  bounds: Rect
  terrain: Polyline[]
  worldObjects: WorldObject[]      // houses, restaurant hull, city blocks
  anchors: Anchor[]
  budget: number
  materials: MaterialId[]          // what's available on this level
  storm?: StormTimeline            // deferred - driven by a manual slider for now
}

interface Solution {               // the player's build, also serialisable
  nodes: Node[]
  members: Member[]
}

interface Anchor {
  id: AnchorId
  pos: Vec2
  attachedTo: WorldObjectId | null // null = welded to static terrain
}

interface Member {
  id: MemberId
  a: NodeId; b: NodeId
  material: MaterialId
  segments: number                 // 1 for short braces, 4-6 for trunks/long spans
}

interface WorldObject {            // houses, boats, branches, tables, debris - see 5.8
  id: WorldObjectId
  shape: Rect                      // editor drags out rectangles; richer shapes later
  material: MaterialId             // density drives float vs sink for free
}
```

On Play we **snapshot** the `Solution` and build a `SimWorld` from `LevelDoc + Solution`. On
Reset we discard the `SimWorld` and restore the snapshot. `LevelDoc` is never touched at all,
so reset is exact by construction and there is no drift to debug.

Each sim entity keeps a back-reference to its document id, so stress and breakage can be
reported back to the editor for colouring and for the post-mortem.

**Because building is allowed during the sim (decision 9), the projection is not one-shot.**
Edits made while the sim runs mutate the live `SimWorld` *and* the working `Solution`
together; the snapshot is what makes Reset still mean something.

One architectural requirement follows: **the solver needs runtime insert/remove** — stable
ids, a free list, tolerance for holes in the SoA arrays, and periodic compaction rather than
reindexing on every edit. Cheap to design in now, painful to retrofit later.

Everything else about mid-sim editing — how a new member's rest state is chosen, how you click
a node that's moving under load, whether time dilates while you build — is open, and
deliberately not decided here.

### 5.3 Materials

```ts
interface Material {
  id: MaterialId
  density: number          // kg/m3 - wood ~500 (floats), steel ~7850 (sinks)
  costPerMetre: number

  axialCompliance: number  // small = stiff. NOT where softness lives.
  bendCompliance: number   // large = bendy. This IS where softness lives.
  dampingRatio: number     // zeta, fraction of critical
  rayleighBeta: number     // stiffness-proportional - kills high-frequency modes

  breakStrain: number      // fracture threshold
  yieldStrain: number      // Infinity for wood - never yields, just breaks
  plasticRate: number      // 0 for wood

  maxSegments: number
  dragCoefficient: number
}
```

Wood and steel differ only in this table. That's the whole material system.

Buoyancy is *not* a material flag — it falls out of density against water's 1000 kg/m³.

### 5.4 The solver

Substepped XPBD. Per frame: N substeps, **one** constraint iteration each — substepping
converges far better than iteration count at equal cost (Macklin et al., *Small Steps in
Physics Simulation*, 2019).

```
for substep in 0..N:
    predict positions (gravity, wind, buoyancy, drag)
    solve distance constraints      (axial, stiff)
    solve bending constraints       (angular, compliant)
    solve fluid density constraints (PBF)
    solve collisions                (particles vs member capsules, vs terrain)
    update velocities from position delta
    apply damping (zeta + Rayleigh beta)
    accumulate damage; apply plastic set where the material allows it
```

Compliance in XPBD enters as `alpha_tilde = alpha / dt^2`, which is precisely why stiffness
stays consistent when the substep count changes. That property is the reason for XPBD over
hand-rolled springs: stiff steel doesn't explode, and retuning substeps doesn't retune the
whole game.

> **Superseded (Aug 2026):** the shipped solve order differs from the sketch above:
> clusters solve FIRST, then bending, then distance, then the fluid projection, then
> contacts — the Gauss-Seidel ordering rationale (the last solve wins locally) lives as
> comments in `SimWorld.step` in `src/sim/world.ts`, which is authoritative.

**Stress** is `strain = (liveLength - restLength) / restLength`, available for free. It drives
the colour ramp, the damage accumulation, and the break test. One number, three jobs.

### 5.5 Bending needs segments

An angular constraint needs three nodes. A member that should visibly bow — a stilt, a palm
trunk, a long span — is a short chain of sub-nodes with bending constraints between them.
Short braces stay at `segments: 1` and behave as rigid links.

This is not general soft-body. It's a per-member integer, set by the designer or defaulted
from length.

### 5.6 Forces

- **Gravity** — constant.
- **Wind** — band-limited gusts (see §8). Drag `F = ½·rho·Cd·A_rest·v_rel²` applied per
  segment, which produces torque about the member naturally because segments are separate nodes.
- **Buoyancy** — from **rest** volume, never deformed volume.
- **Water drag** — from **rest** area, never deformed area.

The rest-volume rule is not an optimisation. Computing buoyancy from deformed geometry creates
a genuine runaway: stretch → more submerged volume → more lift → more stretch. Mass is
conserved; a stretched beam did not gain material.

### 5.7 Camera & parallax

World-space simulation, screen-space presentation. One `Camera { pos, zoom }`; each render
layer carries a `parallaxFactor`. Layer transform is

```
screen = (world - camera.pos * factor) * zoom + viewportCentre
```

Applying zoom *after* the parallax offset keeps distant layers correctly anchored while
zooming, which is the usual place parallax implementations go wrong.

Layers, back to front: sky gradient → distant clouds → ocean horizon → mid dunes → terrain →
world objects → structure → fluid → FX.

### 5.8 Physics objects

One entity type covering four uses: the house that sits on stilts, debris blown in from
off-field, floating things (boats, branches), and loose contents — the restaurant's tables and
chairs sloshing around like a snow globe when the dome cracks. In the editor they're rectangles
you drag out; richer shapes can come later.

**They are shape-matched particle clusters, not a separate rigid-body subsystem.** An object is
a small set of particles bound by a shape-matching constraint (Müller et al., *Meshless
Deformations Based on Shape Matching*), which holds the cluster rigid while leaving it a
first-class citizen of the same position-based solver. Buoyancy, spin, wind drag and getting
shoved by a wave all fall out of the per-particle forces already described in §5.6 — there's no
coupling code between a rigid-body world and a fluid world, because there aren't two worlds.

**Why not true rigid bodies?** They'd win on exact rigidity, O(1) cost per object, true flat
edges and sharp corners, and — most of all — stacking and resting contact, which clusters are
bad at. What rules them out is that they create a *second world*: every interaction pair
(fluid↔body, body↔body, body↔member, body↔terrain) needs explicit contact code, where today
everything is particle↔particle and already written.

The decisive factor is that our objects spend most of their lives in water, and fluid↔rigid
coupling is the hard part. Floating a body properly against particle water means sampling
boundary particles on its surface (Akinci et al.) — which is a particle cluster, plus the
machinery to map forces to centre-of-mass and torque and feed velocity back. The cheap
alternative, analytic buoyancy by clipping against a waterline, cannot express water *inside* a
cracked dome, which is a scenario we want.

Clusters' weak spot — resting contact — mostly misses us, because a house is *anchored* to its
stilts rather than resting on them, and anchors are constraints rather than contacts.

**Escape hatch:** XPBD rigid bodies (Müller et al., 2020) are rigid-body dynamics expressed in
the same position-based framework, so they drop into our substep loop and coexist with PBF. If
stacking or sharp corners ever become the thing that's visibly wrong, that's the upgrade, and
it isn't a rewrite.

**Object particle spacing is independent of fluid particle spacing.** A house resolved at fluid
resolution would be hundreds of wasted particles, and shape matching doesn't need density to
stay rigid.

**Anchors bind to an object's frame, not to an individual particle** — a point in object-local
space transformed by the cluster's current position and orientation.

> **Superseded in part (Aug 2026):** object buoyancy does NOT emerge purely from the
> per-particle pressure field as argued above. Measured: PBF's per-substep lambda carries no
> depth gradient in the bulk, so pressure-derived lift is surface-only — a wood crate
> released 5 m down hovered indefinitely. The shipped design is analytic rest-volume
> buoyancy gated by a per-particle fluid-neighbour wetness census, plus pairwise hull
> viscosity for drag; the pressure field still handles displacement, waves, and slams. A
> consequence the "water inside a cracked dome" argument above did not anticipate: the
> one-column height field cannot represent different water levels inside and outside a
> sealed vessel, so the dome archetype needs its own pressure model when it lands. Object
> anchors are frame mounts (a dedicated node with triangulated links to the nearest cluster
> particles). Rationale and measurements live in `src/sim/fluid.ts` and `src/sim/world.ts`.

Objects spawn and despawn at runtime, which is the same solver requirement already established
in §5.2.

---

## 6. Repo layout

```
src/
  core/        loop.ts (fixed timestep + accumulator), math, seeded rng
  sim/
    solver.ts        substep loop
    constraints/     distance.ts, bend.ts, density.ts, collision.ts
    fluid.ts         PBF particles + spatial hash
    forces.ts        gravity, wind, buoyancy, drag
    materials.ts     the material table
    damage.ts        damage accumulation, plastic set, break tests
  model/
    level.ts         LevelDoc / Solution types
    build.ts         graph ops, cost calculation
    serialize.ts     versioned load/save
  render/
    app.ts           pixi bootstrap
    camera.ts        pan/zoom/parallax
    layers/          sky, sea, terrain, structure, fluid, fx
  editor/
    tools/           anchor, member, delete, storm tools
    ui/              palette, budget, playbar
  game/
    storm.ts         timeline & intensity curve
    session.ts       build <-> sim mode, build/reset of SimWorld
  main.ts
levels/        *.json
docs/          this file
```

---

## 7. Execution plan

Every step ends in something you can look at. Steps marked **[M]** are part of the stated
milestone.

### Stage 0 — Something on screen

**1. Skeleton** — *S* — Vite + TS + Pixi, fixed-timestep loop with accumulator, seeded RNG,
debug overlay (frame time, substeps, body and particle counts).
*Demo:* a rectangle moving deterministically. *Retires:* nothing, but everything sits on it.

**2. Field & camera [M]** — *S* — Pan/zoom camera, parallax layers, flat-shape beach, ocean,
clouds, and a **typed field-width box in metres** that resizes the world and refits the
overview zoom.
*Demo:* drag around a beach, zoom, retype the width and watch the field resize.
*Retires:* parallax-under-zoom math.

### Stage 1 — The solver (biggest risk)

**3. XPBD core** — *M* — Nodes with mass, substep loop, distance constraints with compliance,
live-tunable from a debug panel.
*Demo:* a chain hanging from a fixed point, sagging correctly, compliance on a slider.
*Retires:* does our solver work at all.

**4. Bending & damping** — *M* — Bending constraints, segmented members, ζ damping, Rayleigh β.
*Demo:* a cantilever/palm trunk that bows under gravity and **settles without ringing**.
*Retires:* **the jello risk.** This step proves the entire feel is achievable. If it wobbles
here, stop and fix it before anything else gets built on top.

**5. Materials, stress, breakage [M]** — *M* — Material table, strain→colour ramp, damage
accumulation, break tests, per-material plastic set.
*Demo:* load a beam until it fails — wood snaps abruptly, steel yields, sags, then goes.
*Retires:* legibility, and the wood/steel personality split.

### Stage 2 — Water (second biggest risk)

**6. PBF fluid** — *L* — Particles, spatial hash, density constraints, terrain collision, and a
**simulation resolution control** (particle spacing, substep count) in the debug panel. No
structure coupling yet.
*Demo:* dump water into a terrain bowl; it settles flat and sloshes, and you can trade
resolution against framerate live.
*Retires:* can we afford particle water at 60fps, and at what count.

**7. Coupling & physics objects [M]** — *M* — Member capsule colliders tracking node pairs,
buoyancy from rest volume, water drag, force smoothing — plus **physics objects** as
shape-matched clusters (§5.8).
*Demo:* a wooden crate floats and bobs, an identical steel one sinks, a wave shoves both.
*Retires:* the hardest integration in the project.

### Stage 3 — The signature moment

**8. Wind [M]** — *M* — Band-limited gust field, per-segment drag.
*Demo:* the palm tree from step 4 sways in gusts without buzzing.
*Retires:* high-frequency forcing exciting structural modes.

**9. Sandbox conditions** — *M* — The three condition sliders, driving one hardcoded
stilt-house scene. **No timeline, no event system.**

| Control | Range | Notes |
|---|---|---|
| **Wind** | −250–250 kph | UI in kph, sim in m/s. 250 kph ≈ 69 m/s, roughly Cat 5. Sign is the heading (+x is rightward); detent at calm, ticks every 50. |
| **Flood** | 0–20 m target level | Water rolls in from the level edges. |
| **Wave strength** | None → Extreme | |

Sky colour tracks overall severity.
*Demo:* **drag the sliders up and watch the structure groan, lean, and fail.**
*Retires:* is this actually fun. ← *This is the real milestone.*

### Stage 4 — The editor (the stated milestone)

**10. Document, session & live editing [M]** — *M* — `LevelDoc`/`Solution` types, versioned
serialization, snapshot-on-play / restore-on-reset, and runtime insert/remove in the solver
(stable ids, free list, compaction).
*Demo:* save a level, reload it, play it, **add a member while it's running**, reset it, get
identical results.

**11. Build tools [M]** — *L* — Place anchors, drag members with snapping, delete, undo/redo,
material palette, live cost against budget.
*Demo:* build a structure from scratch, then keep building on it while the storm runs.

**12. Object tool & attached anchors [M]** — *M* — Editor support for the physics objects that
already exist from step 7: drag out rectangles, set material, drop anchors along their edges.
Anchors are either welded to terrain or bound to an object's frame.
*Demo:* drag out a house, put anchors under it, build no stilts, press Play, watch it fall.

**13. Sandbox panel [M]** — *M* — The step-9 sliders promoted from debug into real UI, plus
dump-water, clear, and the resolution control from step 6.
*Demo:* the full sandbox from the original milestone.

**The stated milestone is complete at step 13.** Steps 3–9 are the substrate it stands on;
step 9 is the one that tells us whether any of this is worth continuing.

---

## 8. Physics tuning notes

Starting points, not gospel. Expect to tune steps 4, 5 and 7 by feel.

**Units & scale.** 1 world unit = 1 metre. **Field width is typed in by the author**, in metres,
with no clamp; most levels will land around 50–150 m. Base render scale ~24 px/m, with zoom
roughly 0.4× (whole-field overview) to 1.5× (placing individual boards) — the overview zoom is
derived from the field width rather than fixed. Gravity 9.81 m/s².

**Timestep.** Render at 1/60. **8–16 substeps per frame, 1 iteration each.** Never grow Δt to
fix instability — add substeps.

**Densities.** Wood 500, water 1000, steel 7850 kg/m³. Flotation falls out; don't special-case it.

**Compliance.** Axial very stiff (order 1e-8 steel, 1e-7 wood). Bending several orders looser
(order 1e-5 steel, 1e-3…1e-2 wood). The gap between these two numbers *is* the "bends but
doesn't stretch" feel — if it starts looking rubbery, the axial value is too loose, not the
bending value.

**Damping.** ζ ≈ 0.8–1.0 — near-critical: deflects, settles, no overshoot. Rayleigh β generous,
α ≈ 0. β is the explicit "kill the high-frequency wobble, keep the slow lean" knob.

**Mass.** Frequency goes as √(k/m), so heavier nodes give slower, more ominous motion for free.
Derive node mass from segment volume × density and resist the urge to lighten it.

**Failure.** Real wood fractures around 1% strain and steel yields near 0.2% — both far too
small to read on screen, so exaggerate deliberately. Start: wood `breakStrain` 3–4%,
`yieldStrain` ∞. Steel `yieldStrain` ~1%, `breakStrain` ~8%.

**Damage.** Accumulate above ~60% of break strain, at a rate proportional to
(strain/breakStrain)². Effective break threshold = `breakStrain × (1 − damage)`. Irreversible.
This is what makes a long siege feel like it's being lost.

**Wind.** Gusts from summed noise octaves with periods around 8s / 3s / 1.2s. **Clamp the
shortest period near 0.5s.** Per-frame noise will excite exactly the modes the damping in §4
was built to suppress, and the result reads as buzzing no matter how good the solver is.

**Water forces.** Particle impacts are impulsive and noisy in aggregate. Smooth the per-member
force with an EMA over ~3–5 frames before applying it.

**Particle budget.** Count is `area / spacing²` — at **0.25 m** spacing, 16 particles per m². A
typical flood-wall level, say 100 m wide and 4 m deep, is ~400 m²: about **6,400 particles**,
comfortably inside a 3–10k main-thread budget with headroom for spray. **Start at 0.25 m.**

**Width and depth trade off by archetype, not by accident.** A flood wall is wide and shallow;
an undersea dome is narrow and deep. The pathological case — 150 m wide *and* flooded to 20 m,
~3,000 m², roughly **48,000 particles** — isn't a level shape anyone would actually author.

**No clamp.** Field width is typed in and the resolution control is free-range. A configuration
that tanks the framerate is a legitimate experiment, not a bug. The debug overlay carries live
particle count and frame time so the cause is always visible, and the resolution control is the
lever for trading one against the other. Whether deep bulk water eventually earns a cheaper
representation than the active surface layer is left open.

**Minimum structural thickness — a direct consequence of the above.** With 0.25 m particles, a
member thinner than roughly one particle spacing is not watertight; water tunnels through it.
Enforce a **minimum collision thickness of ~1.5× particle spacing (≈0.4 m)** on every member,
independent of its rendered thickness. This bites hardest on the flood-wall archetype, where a
leaking wall reads as a bug rather than as a near miss.

### Stability rules

1. Add substeps, never grow Δt.
2. Buoyancy from **rest** volume. Drag from **rest** area. Non-negotiable — see §5.6.
3. Band-limit every force that drives the structure.
4. Clamp per-particle max velocity.
5. Seeded RNG everywhere, so runs are reproducible and replay stays cheap to add later.

---

## 9. Deferred

| Deferred | Why it's safe | Seam left for it |
|---|---|---|
| Authored storm timelines & event system | A manual intensity slider covers the whole prototype | `LevelDoc.storm` is optional; step 9's `intensity` scalar is the seam |
| Undersea dome pressure archetype | Different force model, not different architecture | `forces.ts` gains a pressure field |
| Cracked/splintered board art | Purely visual | Damage scalar already exists from step 5 |
| Sound (creaks, snaps, groaning) | Additive | Solver emits strain-rate and break events |
| Campaign, progression, win conditions | Sandbox first | `LevelDoc` has room |
| Replay / slow-mo | Cheap later | Fixed timestep + seeded RNG make it nearly free |
| Worker / WASM / GPU solver | Only if profiling demands it | SoA typed arrays from day one |
| Terrain erosion & destruction | Large scope | Terrain is already a polyline, not a bitmap |
| Broken members turning into debris | Body-count cost | Physics objects exist from step 7; a break event just spawns one |
| Deep genre teardown | Not blocking any code | Revisit before level design |

---

## 10. Open questions

**Answered:**

- ~~Level scale~~ → **50–150 m.** Particle spacing and minimum member thickness follow (§8).
- ~~Build during the storm?~~ → **Yes.** Decision 9; consequences in §5.2.
- ~~Storm/event system~~ → **Not yet.** Manual intensity slider only. Decision 10.

**Still open:**

1. **Art direction.** Flat shapes confirmed — crisp vector polygons, or chunky/pixel? Minor
   renderer implications.
2. **Perf target.** Desktop browsers only, or must it hold 60fps on integrated graphics? This
   decides how much of the 3–10k particle headroom is actually real.
3. **Debris.** Do broken members become physical debris that can dam a river or smash a
   window, or do they despawn?

None of these block steps 1–9. Game-balance questions are explicitly out of scope for this
milestone — this is a level editor and a sandbox, not a game yet.
