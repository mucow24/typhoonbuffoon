import { clamp, kphToMs } from '../core/math'
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
  /** Target water level in metres. Water rolls in from the field edges. */
  floodLevelM = 0
  waveStrength: WaveStrength = 'none'

  /** Particles admitted per frame, so a flood arrives rather than appearing. */
  inflowRate = 90
  private wavePhase = 0

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
    const spacing = this.sim.fluid.spacing

    if (current < target) {
      this.admitFromEdges(Math.min(this.inflowRate, target - current))
    } else if (current > target + this.inflowRate * 2) {
      this.drain(Math.min(this.inflowRate, current - target))
    }
    void spacing
  }

  /** Water enters at the field edges and flows inward. */
  private admitFromEdges(count: number): void {
    if (count <= 0) return
    const t = this.field.terrain
    const p = this.sim.particles
    const spacing = this.sim.fluid.spacing
    const mass = this.sim.fluid.particleMass

    const sides: { x: number; dir: number }[] = [
      { x: this.field.left + spacing, dir: 1 },
      { x: this.field.right - spacing, dir: -1 },
    ]

    let spawned = 0
    let ring = 0
    while (spawned < count && ring < 60) {
      for (const side of sides) {
        if (spawned >= count) break
        const ground = t.heightAt(side.x)
        const top = this.floodLevelM
        if (top <= ground) continue
        const y = ground + spacing * 0.5 + ring * spacing
        if (y > top) continue

        const i = p.create({
          x: side.x + side.dir * ring * spacing * 0.15,
          y,
          invMass: 1 / mass,
          radius: spacing * 0.5,
          kind: KIND_FLUID,
        })
        // A gentle inward push, so it reads as rolling in rather than welling up.
        p.velX[i] = side.dir * 2.5
        spawned++
      }
      ring++
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
   */
  private driveWaves(dt: number): void {
    const strength = WAVE_LEVELS[this.waveStrength]
    if (strength <= 0) return

    const period = 6.5 - strength * 2.5
    this.wavePhase += (dt / period) * Math.PI * 2
    const amplitude = strength * 9
    const push = Math.sin(this.wavePhase) * amplitude

    const p = this.sim.particles
    const zoneX = this.field.right - 8
    for (let i = 0; i < p.highWater; i++) {
      if (p.slots.alive[i] !== 1 || p.kind[i] !== KIND_FLUID) continue
      if (p.posX[i]! < zoneX) continue
      // Blend rather than set, so the paddle nudges the water instead of
      // teleporting its velocity and injecting a shock.
      p.velX[i]! += (-Math.abs(push) - p.velX[i]!) * 0.08 * (push > 0 ? 1 : 0.25)
    }
  }
}
