# Physics test harness

The foundation for validating this simulation. See `CLAUDE.md` for the testing
policy that governs it.

## Why it is built this way

Physics failures are almost never visible in a final value. Energy creeps up
over ten seconds; a surface drifts; a structure never quite settles; water
detonates two seconds after the interesting moment. So the harness records a
**trace over simulated time** and assertions are made against the shape of the
run, not against one number at the end.

Measurements are taken **from outside the sim** wherever an independent
calculation exists. The water surface is recomputed from particle positions
rather than read from `WaterField`, because asserting that a module agrees with
itself proves nothing. Density is the one exception - it has no external
definition, so it is read from the solver.

Assertions compare against **external references**: Euler-Bernoulli deflection,
Archimedes, conservation of volume and energy, shallow-water dam-break speed.
Not against previous outputs of this code.

## Layout

| File | Contents |
| --- | --- |
| `harness/probes.ts` | Measurements: energy, speeds, surface profile, density, nearest-neighbour spacing, volume, escapes, fingerprints |
| `harness/trace.ts` | `run()` steps a world and records a `Trace`; `settle()` steps without recording |
| `harness/scenes.ts` | Terrain and scenario builders, plus the analytic formulas to check against |
| `harness/assertions.ts` | Domain assertions that fail with the shape of the run attached |

Scenario terrain is flat, a basin, or a ramp - never the game's generated beach.
A test whose failure requires reasoning about a procedural dune profile is a
test nobody will diagnose.

## Writing a new physics test

1. Build a world with `makeWorld({ terrain: basinTerrain(...) })`.
2. Put matter in it with `fillWater`, `buildWall`, `buildSimplySupported`,
   `sim.addObject`.
3. `settle()` if you need a quiescent starting state - but **not** before an
   energy assertion, or a blow-up happens off-camera and inflates the baseline.
   That mistake hid a pool reaching 44 m/s.
4. `run()` to get a trace, driving wind/waves/inflow from the `each` callback.
5. Assert with the domain assertions so failures are diagnosable.

Prefer an assertion that names a physical law. `expectNoEnergyGain` says
something true about the universe; `expect(x).toBe(3.14)` says something true
about last Tuesday's build.

## Coverage

- `fluid/rest.test.ts` - still water: energy, settling, level surface, volume,
  level vs volume, clumping, resolution independence
- `fluid/dynamics.test.ts` - hydrostatics, communicating vessels, flow downhill,
  dam break front speed, splash decay
- `fluid/production.test.ts` - the paths the game actually takes: flood inflow
  from the field edges, spawning into occupied space, waves, determinism
- `structure/beam.test.ts` - cantilever and simply-supported deflection against
  textbook formulas, load capacity, cable-tension failure mode, stability
- `interaction/coupling.test.ts` - buoyancy against Archimedes, containment and
  wall loading, wind response, and all three systems running together

## Running

```
npm test           # watch
npm run test:run   # single pass
```
