import type { Container } from 'pixi.js'
import { clamp, type Vec2 } from '../core/math'

/**
 * World space is metres, Y UP. Screen space is CSS pixels, Y down. The flip
 * lives here so the physics code can read naturally (gravity is -Y, buoyancy
 * is +Y) instead of being written upside down.
 *
 * Parallax multiplies the CAMERA position, not the world position:
 *
 *     screen = (world - camera.pos * factor) * scale + viewportCentre
 *
 * so a layer at factor 0.2 drifts a fifth as fast as the play field. Zoom is
 * applied AFTER the parallax offset, which is the usual place this goes wrong -
 * scaling the offset too makes distant layers slide as you zoom.
 */
export class Camera {
  /** Camera centre, world metres. */
  x = 0
  y = 0
  /** Multiplier on top of pxPerMetre. */
  zoom = 1
  /** Base render scale at zoom 1. */
  pxPerMetre = 24

  minZoom = 0.08
  maxZoom = 4

  get scale(): number {
    return this.pxPerMetre * this.zoom
  }

  /** Position a parallax layer for the current camera state. */
  applyTo(container: Container, factor: number, viewW: number, viewH: number): void {
    const s = this.scale
    container.scale.set(s, -s)
    container.position.set(viewW * 0.5 - this.x * factor * s, viewH * 0.5 + this.y * factor * s)
  }

  worldToScreen(wx: number, wy: number, viewW: number, viewH: number): Vec2 {
    const s = this.scale
    return {
      x: (wx - this.x) * s + viewW * 0.5,
      y: viewH * 0.5 - (wy - this.y) * s,
    }
  }

  screenToWorld(sx: number, sy: number, viewW: number, viewH: number): Vec2 {
    const s = this.scale
    return {
      x: (sx - viewW * 0.5) / s + this.x,
      y: (viewH * 0.5 - sy) / s + this.y,
    }
  }

  setZoom(z: number): void {
    this.zoom = clamp(z, this.minZoom, this.maxZoom)
  }

  /** Zoom about a screen point, keeping the world point under it fixed. */
  zoomAt(sx: number, sy: number, factor: number, viewW: number, viewH: number): void {
    const before = this.screenToWorld(sx, sy, viewW, viewH)
    this.setZoom(this.zoom * factor)
    const after = this.screenToWorld(sx, sy, viewW, viewH)
    this.x += before.x - after.x
    this.y += before.y - after.y
  }

  /**
   * Overview zoom is derived from the field width rather than fixed, so retyping
   * the width refits rather than leaving you lost off-screen.
   */
  fitWidth(fieldWidthM: number, viewW: number, margin = 1.08): void {
    if (fieldWidthM <= 0 || viewW <= 0) return
    this.setZoom(viewW / (fieldWidthM * margin) / this.pxPerMetre)
  }
}
