import { Container, Graphics } from 'pixi.js'
import type { EditorController } from '../editor/tools'
import { nodePosition } from '../editor/viewModel'
import { MATERIALS } from '../sim/materials'
import type { SimClient } from '../runtime/client'
import type { Camera } from './camera'

/**
 * Editor overlay: anchors, the drag preview, and hover highlights. Drawn above
 * the structure so build affordances are never hidden by what you are building.
 * Anchor and hover positions come from the snapshot's structure view-model.
 */
export class EditorView {
  readonly container = new Container()
  private readonly g = new Graphics()

  constructor(
    parent: Container,
    private readonly client: SimClient,
    private readonly editor: EditorController,
  ) {
    this.container.addChild(this.g)
    parent.addChild(this.container)
  }

  update(camera: Camera, viewW: number, viewH: number): void {
    camera.applyTo(this.container, 1, viewW, viewH)
    const g = this.g
    g.clear()

    const scale = 1 / camera.zoom
    const vm = this.client.latest?.structure

    if (vm) {
      // Anchors: the fixed connection points, riding their live positions.
      for (const anchor of vm.anchors) {
        const r = 0.34 * Math.min(Math.max(scale, 0.5), 3)
        g.circle(anchor.x, anchor.y, r).fill(anchor.attachedTo ? 0xffc46b : 0xff9d5c)
        g.circle(anchor.x, anchor.y, r * 1.7).stroke({
          width: 0.07 * Math.max(scale, 1),
          color: 0x000000,
          alpha: 0.35,
        })
      }

      // Hover highlight.
      if (this.editor.hoverNode) {
        const pos = nodePosition(vm, this.editor.hoverNode)
        if (pos) {
          g.circle(pos.x, pos.y, 0.5 * Math.max(scale, 0.6)).stroke({
            width: 0.1 * Math.max(scale, 1),
            color: 0x6fd3ff,
          })
        }
      }
    }

    // Drag preview.
    const p = this.editor.preview
    if (p.kind === 'member') {
      const colour = p.valid ? MATERIALS[this.editor.material].colour : 0xe2483c
      g.moveTo(p.from.x, p.from.y)
      g.lineTo(p.to.x, p.to.y)
      g.stroke({
        width: MATERIALS[this.editor.material].section,
        color: colour,
        alpha: 0.65,
        cap: 'round',
      })
    } else if (p.kind === 'object') {
      const x = Math.min(p.from.x, p.to.x)
      const y = Math.min(p.from.y, p.to.y)
      const w = Math.abs(p.to.x - p.from.x)
      const h = Math.abs(p.to.y - p.from.y)
      g.rect(x, y, w, h).fill({ color: 0xc99a5b, alpha: 0.4 })
      g.rect(x, y, w, h).stroke({ width: 0.08 * Math.max(scale, 1), color: 0xffffff, alpha: 0.6 })
    }
  }
}
