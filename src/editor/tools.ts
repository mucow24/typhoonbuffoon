import type { Camera } from '../render/camera'
import type { Session } from '../game/session'
import type { MaterialId } from '../sim/materials'
import type { Field } from '../world/field'
import type { Vec2 } from '../core/math'

export type ToolName = 'pan' | 'build' | 'anchor' | 'object' | 'delete'

export interface EditorPreview {
  kind: 'none' | 'member' | 'object'
  from: Vec2
  to: Vec2
  valid: boolean
  snappedFrom: string | null
  snappedTo: string | null
}

export interface ViewSize {
  width: number
  height: number
}

/**
 * Build-mode mouse handling. Everything it does goes through the Session, so
 * edits land in the document and the live sim together - which is what makes
 * building during the sim work rather than being a special case.
 */
export class EditorController {
  tool: ToolName = 'build'
  material: MaterialId = 'wood'
  /** Snap radius in screen pixels, converted to world units per-use. */
  snapPixels = 18
  gridSnap = 0

  readonly preview: EditorPreview = {
    kind: 'none',
    from: { x: 0, y: 0 },
    to: { x: 0, y: 0 },
    valid: false,
    snappedFrom: null,
    snappedTo: null,
  }

  hoverNode: string | null = null
  hoverMember: string | null = null

  private dragging = false
  private dragFrom: Vec2 = { x: 0, y: 0 }
  private dragFromNode: string | null = null
  private readonly disposers: (() => void)[] = []

  constructor(
    private readonly element: HTMLElement,
    private readonly camera: Camera,
    private readonly view: ViewSize,
    private readonly session: Session,
    private readonly field: Field,
  ) {
    this.on('pointerdown', this.onDown as EventListener)
    this.on('pointermove', this.onMove as EventListener, window)
    this.on('pointerup', this.onUp as EventListener, window)
    this.on('keydown', this.onKey as EventListener, window)
  }

  private on(type: string, handler: EventListener, target: EventTarget = this.element): void {
    target.addEventListener(type, handler)
    this.disposers.push(() => target.removeEventListener(type, handler))
  }

  dispose(): void {
    for (const d of this.disposers) d()
    this.disposers.length = 0
  }

  get active(): boolean {
    return this.tool !== 'pan'
  }

  private toWorld(e: PointerEvent): Vec2 {
    const rect = this.element.getBoundingClientRect()
    return this.camera.screenToWorld(
      e.clientX - rect.left,
      e.clientY - rect.top,
      this.view.width,
      this.view.height,
    )
  }

  private snapRadiusWorld(): number {
    return this.snapPixels / this.camera.scale
  }

  private applyGrid(p: Vec2): Vec2 {
    if (this.gridSnap <= 0) return p
    return {
      x: Math.round(p.x / this.gridSnap) * this.gridSnap,
      y: Math.round(p.y / this.gridSnap) * this.gridSnap,
    }
  }

  private onDown = (e: PointerEvent): void => {
    if (e.button !== 0 || !this.active) return
    const world = this.toWorld(e)
    const snap = this.snapRadiusWorld()

    switch (this.tool) {
      case 'build': {
        this.dragging = true
        this.dragFromNode = this.session.pickNode(world.x, world.y, snap)
        const at = this.dragFromNode
          ? this.session.nodePosition(this.dragFromNode)!
          : this.applyGrid(world)
        this.dragFrom = at
        this.preview.kind = 'member'
        this.preview.from = at
        this.preview.to = at
        this.preview.snappedFrom = this.dragFromNode
        break
      }

      case 'anchor': {
        // Dropping an anchor on an object binds it to that object; a structure
        // attached to it then genuinely holds the object up.
        const objectId = this.session.pickObject(world.x, world.y)
        const at = objectId ? world : this.snapToGround(world)
        this.session.addAnchor(at.x, at.y, objectId)
        break
      }

      case 'object': {
        this.dragging = true
        this.dragFrom = this.applyGrid(world)
        this.preview.kind = 'object'
        this.preview.from = this.dragFrom
        this.preview.to = this.dragFrom
        break
      }

      case 'delete': {
        // Most specific first: a member is a thin target, an object a large
        // one, so picking the object first would make members unclickable.
        const member = this.session.pickMember(world.x, world.y, snap)
        if (member) {
          this.session.removeMember(member)
          break
        }
        const anchor = this.session.pickAnchor(world.x, world.y, snap * 1.5)
        if (anchor) {
          this.session.removeAnchor(anchor)
          break
        }
        const object = this.session.pickObject(world.x, world.y)
        if (object) this.session.removeObject(object)
        break
      }
    }
  }

  /** Sit an anchor on the terrain surface if it was dropped near it. */
  private snapToGround(world: Vec2): Vec2 {
    const ground = this.field.terrain.heightAt(world.x)
    if (Math.abs(world.y - ground) < 2.5) return { x: world.x, y: ground }
    return this.applyGrid(world)
  }

  private onMove = (e: PointerEvent): void => {
    if (!this.active) return
    const world = this.toWorld(e)
    const snap = this.snapRadiusWorld()

    this.hoverNode = this.session.pickNode(world.x, world.y, snap)
    this.hoverMember = this.tool === 'delete' ? this.session.pickMember(world.x, world.y, snap) : null

    if (!this.dragging) return

    if (this.preview.kind === 'member') {
      const targetNode = this.session.pickNode(world.x, world.y, snap)
      const at = targetNode ? this.session.nodePosition(targetNode)! : this.applyGrid(world)
      this.preview.to = at
      this.preview.snappedTo = targetNode
      this.preview.valid =
        targetNode !== this.preview.snappedFrom &&
        Math.hypot(at.x - this.dragFrom.x, at.y - this.dragFrom.y) > 0.5
    } else if (this.preview.kind === 'object') {
      this.preview.to = this.applyGrid(world)
      this.preview.valid =
        Math.abs(this.preview.to.x - this.dragFrom.x) > 0.4 &&
        Math.abs(this.preview.to.y - this.dragFrom.y) > 0.4
    }
  }

  private onUp = (e: PointerEvent): void => {
    if (e.button !== 0 || !this.dragging) return
    this.dragging = false

    if (this.preview.kind === 'member' && this.preview.valid) {
      const a = this.preview.snappedFrom ?? this.session.addNode(this.dragFrom.x, this.dragFrom.y)
      const b = this.preview.snappedTo ?? this.session.addNode(this.preview.to.x, this.preview.to.y)
      this.session.addMember(a, b, this.material)
    } else if (this.preview.kind === 'object' && this.preview.valid) {
      const x = (this.dragFrom.x + this.preview.to.x) * 0.5
      const y = (this.dragFrom.y + this.preview.to.y) * 0.5
      this.session.addWorldObject({
        x,
        y,
        width: Math.abs(this.preview.to.x - this.dragFrom.x),
        height: Math.abs(this.preview.to.y - this.dragFrom.y),
        density: 400,
      })
    }

    this.preview.kind = 'none'
    this.preview.valid = false
    this.preview.snappedFrom = null
    this.preview.snappedTo = null
  }

  private onKey = (e: KeyboardEvent): void => {
    if ((e.target as HTMLElement)?.tagName === 'INPUT') return
    const mod = e.ctrlKey || e.metaKey
    if (mod && e.key.toLowerCase() === 'z') {
      e.preventDefault()
      if (e.shiftKey) this.session.redo()
      else this.session.undo()
      return
    }
    switch (e.key) {
      case '1':
        this.tool = 'build'
        break
      case '2':
        this.tool = 'anchor'
        break
      case '3':
        this.tool = 'object'
        break
      case '4':
        this.tool = 'delete'
        break
      case '5':
        this.tool = 'pan'
        break
      case 'q':
        this.material = 'wood'
        break
      case 'w':
        this.material = 'steel'
        break
    }
  }
}
