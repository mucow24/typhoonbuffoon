import { clamp, kphToMs } from '../core/math'
import { Rng } from '../core/rng'
import { KIND_FLUID } from '../sim/particles'
import type { SimWorld } from '../sim/world'
import type { Field } from '../world/field'

export const WIND_MAX_KPH = 250
export const FLOOD_MAX_M = 20

export type WaveStrength = 'none' | 'light' | 'moderate' | 'heavy' | 'extreme'

export const WAVE_LEVELS: Record<WaveStrength, number> = {
  none: 0,
  light: 0.25,
  moderate: 0.5,
  heavy: 0.75,
  extreme: 1,
}

/**
 * The sandbox conditions: three direct sliders, no timeline and no event
 * system. Authored storm timelines are deferred; this is the whole driver.
 */
export class Conditions {
  /** 0-250 kph. 250 is roughly a category 5. */
  windKph = 0
  /**
   * Target water level in metres. Water rolls in from the field edges.
   *
   * 0 means DRY, not sea level. The beach runs well below y=0, so treating 0 as
   * sea level meant the map silently flooded itself with thousands of particles
   * the moment it loaded. The distant ocean is scenery; actual simulated water
   * is something you ask for.
   */
  floodLevelM = 0
  waveStrength: WaveStrength = 'none'

  /** Particles admitted per frame, so a flood arrives rather than appearing. */
  inflowRate = 90
  private wavePhase = 0
  /** Seeded: admission jitter must not break run-to-run determinism. */
  private readonly rng = new Rng(0x51de5)

  constructor(
    private readonly sim: SimWorld,
    private readonly field: Field,
  ) {}

  reset(): void {
    this.windKph = 0
    this.floodLevelM = 0
    this.waveStrength = 'none'
    this.wavePhase = 0
    this.sim.wind.baseSpeed = 0
    this.sim.wind.reset()
    // update() only runs while the sim does; a paused reset must not leave a
    // stale paddle armed in the solver.
    this.sim.waveDrive = null
  }

  /** 0..1 overall severity, used to grey the sky. */
  severity(): number {
    const wind = this.windKph / WIND_MAX_KPH
    const flood = this.floodLevelM / FLOOD_MAX_M
    const wave = WAVE_LEVELS[this.waveStrength]
    return clamp(wind * 0.5 + flood * 0.3 + wave * 0.2, 0, 1)
  }

  /**
   * Number of particles needed to hold the field at the target level. Derived
   * from the actual terrain profile, so a deep basin needs more than a shallow
   * one and the slider means the same thing on any map.
   */
  private targetParticleCount(): number {
    if (this.floodLevelM <= 0) return 0
    const t = this.field.terrain
    const spacing = this.sim.fluid.spacing
    const step = spacing
    let area = 0
    for (let x = this.field.left; x < this.field.right; x += step) {
      const depth = this.floodLevelM - t.heightAt(x)
      if (depth > 0) area += depth * step
    }
    return Math.floor(area / (spacing * spacing))
  }

  update(dt: number): void {
    this.sim.wind.baseSpeed = kphToMs(this.windKph)
    this.maintainFlood()
    this.driveWaves(dt)
  }

  private maintainFlood(): void {
    const target = this.targetParticleCount()
    const current = this.sim.fluidCount

    if (current < target) {
      this.admitFromEdges(Math.min(this.inflowRate, target - current))
    } else if (current > target + this.inflowRate * 2) {
      this.drain(Math.min(this.inflowRate, current - target))
    }
  }

  /**
   * Water enters at the field edges and flows inward: a JITTERED band of
   * admission slots at each edge, rained in rather than stacked.
   *
   * The jitter is load-bearing, not cosmetic. A ladder of particles admitted
   * at exact rest-density spacing is self-supporting under the unilateral
   * density constraint - leftover admissions formed a levitating lattice
   * shelf above the surface, which then blocked every admission slot and
   * stalled the flood at a fraction of the slider's target. Jittered spawns
   * are sub-rest-density almost everywhere, get no pressure support, and
   * fall as rain into the bulk.
   *
   * The band also sits a few spacings inside the wall: the boundary clamp
   * kills x-velocity on contact, and water admitted right into the corner
   * stagnated into a lip.
   */
  private admitFromEdges(count: number): void {
    if (count <= 0) return
    const t = this.field.terrain
    const p = this.sim.particles
    const spacing = this.sim.fluid.spacing
    const mass = this.sim.fluid.particleMass
    const COLUMNS = 4

    const sides: { x: number; dir: number }[] = []
    for (let k = 0; k < COLUMNS; k++) {
      sides.push({ x: this.field.left + spacing * (4 + k * 2), dir: 1 })
      sides.push({ x: this.field.right - spacing * (4 + k * 2), dir: -1 })
    }

    // The occupancy guard reads LAST frame's spatial hash, so spawns made
    // earlier in this same call are invisible to it - and two jittered
    // admissions landing within a fraction of a spacing of each other are an
    // over-density pocket that discharges at the speed cap. Track this call's
    // spawns and keep clear of them too.
    const newX: number[] = []
    const newY: number[] = []
    const clearOfNew = (x: number, y: number, r: number): boolean => {
      const r2 = r * r
      for (let i = 0; i < newX.length; i++) {
        const dx = newX[i]! - x
        const dy = newY[i]! - y
        if (dx * dx + dy * dy < r2) return false
      }
      return true
    }

    let spawned = 0
    let ring = 0
    while (spawned < count && ring < 60) {
      for (const side of sides) {
        if (spawned >= count) break
        const ground = t.heightAt(side.x)
        const top = this.floodLevelM
        if (top <= ground) continue
        const y = ground + spacing * 0.5 + ring * spacing + (this.rng.next() - 0.5) * spacing * 0.7
        if (y > top || y < ground + spacing * 0.4) continue

        const x = side.x + (this.rng.next() - 0.5) * spacing * 0.7
        // Never admit water where there is already water. Overlapping spawns
        // produce a density error the solver can only answer with a violent
        // correction, and that is what made raising the flood slider detonate.
        if (this.sim.hasFluidNear(x, y, spacing * 0.85)) continue
        if (!clearOfNew(x, y, spacing * 0.85)) continue

        const i = p.create({
          x,
          y,
          invMass: 1 / mass,
          radius: spacing * 0.5,
          kind: KIND_FLUID,
        })
        // A gentle inward push, so it reads as rolling in rather than welling up.
        p.velX[i] = side.dir * 2.5
        newX.push(x)
        newY.push(y)
        spawned++
      }
      ring++
    }

    // The edge band saturates once the basin is nearly at level: with only a
    // small head left, lateral spreading across an 80 m field is a
    // minutes-long process, and the slider would take minutes to mean what it
    // says. Top up as storm rain over whichever columns are still short.
    if (spawned < count) this.rainIn(count - spawned, clearOfNew, newX, newY)
  }

  /** Sparse rain over the flooding region, wherever the level is still short. */
  private rainIn(
    count: number,
    clearOfNew: (x: number, y: number, r: number) => boolean,
    newX: number[],
    newY: number[],
  ): void {
    const t = this.field.terrain
    const p = this.sim.particles
    const spacing = this.sim.fluid.spacing
    const mass = this.sim.fluid.particleMass
    const target = this.floodLevelM

    let spawned = 0
    const attempts = count * 6
    for (let a = 0; a < attempts && spawned < count; a++) {
      const x = this.field.left + 2 + this.rng.next() * (this.field.widthM - 4)
      const ground = t.heightAt(x)
      if (target <= ground) continue
      const local = this.sim.water.surfaceAt(x)
      const surface = local === -Infinity ? ground : Math.max(ground, local)
      if (surface >= target - spacing * 0.25) continue

      const y = surface + spacing * (1 + this.rng.next() * 0.5)
      // Wider clearance than the edge band: while a column holds fewer than
      // three particles its surface reads as dry, so early rain re-targets
      // the bed frame after frame, and rain needs a full spacing of clearance
      // to stay sub-rest density and just fall.
      if (this.sim.hasFluidNear(x, y, spacing * 1.05)) continue
      if (!clearOfNew(x, y, spacing * 1.05)) continue

      p.create({
        x,
        y,
        invMass: 1 / mass,
        radius: spacing * 0.5,
        kind: KIND_FLUID,
      })
      newX.push(x)
      newY.push(y)
      spawned++
    }
  }

  private drain(count: number): void {
    const p = this.sim.particles
    let removed = 0
    for (let i = p.highWater - 1; i >= 0 && removed < count; i--) {
      if (p.slots.alive[i] !== 1 || p.kind[i] !== KIND_FLUID) continue
      const x = p.posX[i]!
      if (x < this.field.left + 4 || x > this.field.right - 4) {
        p.destroy(i)
        removed++
      }
    }
  }

  /**
   * Waves: a paddle at the seaward edge. Driving the water rather than moving
   * structures directly means everything downstream - shoving a raft, breaking
   * over a wall - falls out of the fluid rather than needing its own code.
   *
   * This computes the paddle; the SOLVER applies it (SimWorld.waveDrive),
   * because the blend mutates velocities and the GPU backend owns those
   * device-side. Frame-level either way, same oscillator, same 0.08 blend.
   */
  private driveWaves(dt: number): void {
    const strength = WAVE_LEVELS[this.waveStrength]
    if (strength <= 0) {
      this.sim.waveDrive = null
      return
    }

    const period = 6.5 - strength * 2.5
    this.wavePhase += (dt / period) * Math.PI * 2
    const amplitude = strength * 9
    // OSCILLATORY, like a real paddle: full stroke shoreward (-x), a softer
    // return stroke, so the zone produces genuine back-and-forth flow with a
    // net shoreward transport. The old driver only ever pushed shoreward,
    // which made "waves" a one-way current that never came back.
    const sin = Math.sin(this.wavePhase)
    const push = sin < 0 ? sin * amplitude : sin * amplitude * 0.55

    // Blend rather than set, so the paddle nudges the water instead of
    // teleporting its velocity and injecting a shock.
    this.sim.waveDrive = { x0: this.field.right - 8, push, blend: 0.08 }
  }
}
