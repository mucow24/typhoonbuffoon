import { describe, it, expect } from 'vitest'
import type { SolverBackend } from '../../src/sim/solver'
import { fillWater, flatTerrain, makeWorld } from '../harness'

/**
 * The coupling-force ownership seam. Buoyancy, hydrostatic load and water
 * drag move into the GPU backend (computed from device-fresh state - the
 * fix for the force-lag resonance); the CPU reference keeps them in the
 * host frame-head. A backend declares which regime it is in with
 * `ownsCouplingForces`, and the world must apply the host passes EXACTLY
 * when the backend does not own them - both double-application and
 * no-application are wrong physics.
 */

function submergedNodeWorld() {
  const w = makeWorld({ widthM: 40, terrain: flatTerrain(40, 0) })
  fillWater(w, { x0: -8, x1: 8, yTop: 3 })
  // A wood-volume node two metres under the surface: analytic lift is
  // g * (rhoWater * vol / mass) * frac = 9.81 * (1000 * 0.1 / 10) * 1 = 98.1.
  const i = w.particles.create({ x: 0, y: 1, invMass: 1 / 10, radius: 0.12, volume: 0.1 })
  return { w, i }
}

/** Capture accY at sync() time - after the frame-head forces, before the
 *  frame-tail clears the accumulators. */
function captureSolver(
  capture: () => void,
  owns: boolean,
): SolverBackend {
  const s: SolverBackend = {
    sync: () => capture(),
    step: () => {},
    readback: () => Promise.resolve(),
  }
  if (owns) (s as { ownsCouplingForces?: boolean }).ownsCouplingForces = true
  return s
}

describe('coupling-force ownership', () => {
  it('host applies buoyancy for a backend that does not own coupling forces', () => {
    const { w, i } = submergedNodeWorld()
    let acc = NaN
    w.solver = captureSolver(() => {
      acc = w.particles.accY[i]!
    }, false)
    w.step(1 / 60)
    expect(acc).toBeGreaterThan(50) // the analytic lift arrived
  })

  it('host skips buoyancy/hydro/drag for a backend that owns them', () => {
    const { w, i } = submergedNodeWorld()
    let acc = NaN
    w.solver = captureSolver(() => {
      acc = w.particles.accY[i]!
    }, true)
    w.step(1 / 60)
    // No wind is blowing, so a backend that owns the coupling forces must
    // see a clean accumulator - anything here would be applied TWICE.
    expect(acc).toBe(0)
  })
})
