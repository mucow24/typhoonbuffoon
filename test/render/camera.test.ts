import { describe, it, expect } from 'vitest'
import type { Container } from 'pixi.js'
import { Camera } from '../../src/render/camera'

/**
 * Camera MATHS - the transform, not the drawing. CLAUDE.md names this file as
 * explicitly not exempt: parallax-under-zoom is "the usual place parallax
 * implementations go wrong", and the audit found it shipped untested anyway.
 */

const VIEW_W = 1280
const VIEW_H = 800

describe('world/screen round trip', () => {
  it('inverts exactly at any zoom and position', () => {
    const cam = new Camera()
    cam.x = 12.5
    cam.y = -3.75
    cam.setZoom(1.7)

    for (const [wx, wy] of [
      [0, 0],
      [45.2, 9.9],
      [-60, -11],
    ] as const) {
      const s = cam.worldToScreen(wx, wy, VIEW_W, VIEW_H)
      const w = cam.screenToWorld(s.x, s.y, VIEW_W, VIEW_H)
      expect(w.x).toBeCloseTo(wx, 6)
      expect(w.y).toBeCloseTo(wy, 6)
    }
  })

  it('puts the camera centre at the viewport centre, with Y up on screen-down', () => {
    const cam = new Camera()
    cam.x = 7
    cam.y = 2
    const centre = cam.worldToScreen(7, 2, VIEW_W, VIEW_H)
    expect(centre.x).toBeCloseTo(VIEW_W / 2)
    expect(centre.y).toBeCloseTo(VIEW_H / 2)

    // World UP must be screen UP (smaller screen y): the y-flip lives here.
    const above = cam.worldToScreen(7, 3, VIEW_W, VIEW_H)
    expect(above.y).toBeLessThan(centre.y)
  })
})

describe('zoomAt', () => {
  it('keeps the world point under the cursor fixed', () => {
    const cam = new Camera()
    cam.x = -4
    cam.y = 6
    cam.setZoom(0.8)

    const cursor = { x: 300, y: 650 }
    const before = cam.screenToWorld(cursor.x, cursor.y, VIEW_W, VIEW_H)
    cam.zoomAt(cursor.x, cursor.y, 1.6, VIEW_W, VIEW_H)
    const after = cam.screenToWorld(cursor.x, cursor.y, VIEW_W, VIEW_H)

    expect(after.x).toBeCloseTo(before.x, 6)
    expect(after.y).toBeCloseTo(before.y, 6)
    expect(cam.zoom).toBeCloseTo(0.8 * 1.6, 6)
  })

  it('clamps to the zoom limits', () => {
    const cam = new Camera()
    cam.zoomAt(0, 0, 1e9, VIEW_W, VIEW_H)
    expect(cam.zoom).toBe(cam.maxZoom)
    cam.zoomAt(0, 0, 1e-9, VIEW_W, VIEW_H)
    expect(cam.zoom).toBe(cam.minZoom)
  })
})

describe('fitWidth', () => {
  it('fits the field width into the viewport with the documented margin', () => {
    const cam = new Camera()
    cam.fitWidth(120, VIEW_W)
    // At the fitted zoom, the field spans viewW / margin pixels.
    const fieldPx = 120 * cam.scale
    expect(fieldPx).toBeCloseTo(VIEW_W / 1.08, 3)
  })

  it('ignores degenerate inputs', () => {
    const cam = new Camera()
    const zoom = cam.zoom
    cam.fitWidth(0, VIEW_W)
    cam.fitWidth(120, 0)
    expect(cam.zoom).toBe(zoom)
  })
})

describe('parallax layer transform', () => {
  /** Minimal stand-in for a pixi Container: applyTo only writes scale/position. */
  function fakeContainer() {
    const c = {
      scale: { x: 0, y: 0, set(x: number, y: number) { c.scale.x = x; c.scale.y = y } },
      position: { x: 0, y: 0, set(x: number, y: number) { c.position.x = x; c.position.y = y } },
    }
    return c
  }

  it('matches worldToScreen exactly for the gameplay layer (factor 1)', () => {
    const cam = new Camera()
    cam.x = 9
    cam.y = -2
    cam.setZoom(1.3)
    const c = fakeContainer()
    cam.applyTo(c as unknown as Container, 1, VIEW_W, VIEW_H)

    // The container transform must send world (wx, wy) to the same screen
    // point worldToScreen reports - otherwise picking and rendering disagree.
    const wx = 14.2
    const wy = 3.1
    const viaContainer = {
      x: c.position.x + wx * c.scale.x,
      y: c.position.y + wy * c.scale.y,
    }
    const direct = cam.worldToScreen(wx, wy, VIEW_W, VIEW_H)
    expect(viaContainer.x).toBeCloseTo(direct.x, 4)
    expect(viaContainer.y).toBeCloseTo(direct.y, 4)
  })

  it('scrolls distant layers slower, and zoom does not slide them', () => {
    const cam = new Camera()
    cam.setZoom(1)
    const layer = fakeContainer()

    // Pan the camera: a factor-0.2 layer moves a fifth as far on screen.
    cam.x = 0
    cam.applyTo(layer as unknown as Container, 0.2, VIEW_W, VIEW_H)
    const before = layer.position.x
    cam.x = 10
    cam.applyTo(layer as unknown as Container, 0.2, VIEW_W, VIEW_H)
    const panShift = before - layer.position.x
    expect(panShift).toBeCloseTo(10 * 0.2 * cam.scale, 4)

    // Zoom about the same camera position: the layer's WORLD anchor point
    // (camera.pos * factor maps to viewport centre) must stay at the centre -
    // "zoom applied after the parallax offset". Scaling the offset too is the
    // classic slide bug.
    const anchorWorldX = cam.x * 0.2
    const anchorBefore = layer.position.x + anchorWorldX * layer.scale.x
    cam.setZoom(2)
    cam.applyTo(layer as unknown as Container, 0.2, VIEW_W, VIEW_H)
    const anchorAfter = layer.position.x + anchorWorldX * layer.scale.x
    expect(anchorBefore).toBeCloseTo(VIEW_W / 2, 4)
    expect(anchorAfter).toBeCloseTo(VIEW_W / 2, 4)
  })
})
