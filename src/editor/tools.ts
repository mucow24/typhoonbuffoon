import type { Camera } from '../render/camera'
import type { MaterialId } from '../sim/materials'
import type { Vec2 } from '../core/math'

export type ToolName = 'pan' | 'build' | 'anchor' | 'object' | 'delete' | 'water'

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
 * Everything the editor needs from the far side of the worker boundary.
 * Picks and positions answer synchronously from the latest snapshot's
 * structure view-model (at most one sim frame old); mutations are
 * fire-and-forget commands whose results come back in the next snapshot.
 * Implemented in app.ts over SimClient + editor/viewModel.ts.
 */
export interface EditorGateway {
  pickNode(x: number, y: number, radius: number): string | null
  pickMember(x: number, y: number, radius: number): string | null
  pickAnchor(x: number, y: number, radius: number): string | null
  pickObject(x: number, y: number): string | null
  nodePosition(id: string): Vec2 | null
  groundHeight(x: number): number
  buildMember(
    fromNode: string | null,
    from: Vec2,
    toNode: string | null,
    to: Vec2,
    material: MaterialId,
  ): void
  addAnchor(x: number, y: number, attachedTo: string | null): void
  addObject(x: number, y: number, width: number, height: number, density: number): void
  removeMember(id: string): void
  removeAnchor(id: string): void
  removeObject(id: string): void
  undo(): void
  redo(): void
  splash(x: number, y: number): void
  setStream(x: number, y: number): void
  clearStream(): void
}

/**
 * Build-mode mouse handling. Everything it does goes through the gateway, so
 * edits land in the worker's document and live sim together - which is what
 * makes building during the sim work rather than being a special case.
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

  private waterHeld = false
  private dragging = false
  private dragFrom: Vec2 = { x: 0, y: 0 }
  private dragFromNode: string | null = null
  private readonly disposers: (() => void)[] = []

  constructor(
    private readonly element: HTMLElement,
    private readonly camera: Camera,
    private readonly view: ViewSize,
    private readonly gateway: EditorGateway,
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
        this.dragFromNode = this.gateway.pickNode(world.x, world.y, snap)
        const at = this.dragFromNode
          ? this.gateway.nodePosition(this.dragFromNode) ?? this.applyGrid(world)
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
        const objectId = this.gateway.pickObject(world.x, world.y)
        const at = objectId ? world : this.snapToGround(world)
        this.gateway.addAnchor(at.x, at.y, objectId)
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

      case 'water': {
        this.waterHeld = true
        this.gateway.splash(world.x, world.y)
        this.gateway.setStream(world.x, world.y)
        break
      }

      case 'delete': {
        // Most specific first: a member is a thin target, an object a large
        // one, so picking the object first would make members unclickable.
        const member = this.gateway.pickMember(world.x, world.y, snap)
        if (member) {
          this.gateway.removeMember(member)
          break
        }
        const anchor = this.gateway.pickAnchor(world.x, world.y, snap * 1.5)
        if (anchor) {
          this.gateway.removeAnchor(anchor)
          break
        }
        const object = this.gateway.pickObject(world.x, world.y)
        if (object) this.gateway.removeObject(object)
        break
      }
    }
  }

  /** Sit an anchor on the terrain surface if it was dropped near it. */
  private snapToGround(world: Vec2): Vec2 {
    const ground = this.gateway.groundHeight(world.x)
    if (Math.abs(world.y - ground) < 2.5) return { x: world.x, y: ground }
    return this.applyGrid(world)
  }

  private onMove = (e: PointerEvent): void => {
    if (!this.active) return
    const world = this.toWorld(e)
    const snap = this.snapRadiusWorld()

    this.hoverNode = this.gateway.pickNode(world.x, world.y, snap)
    this.hoverMember =
      this.tool === 'delete' ? this.gateway.pickMember(world.x, world.y, snap) : null

    if (this.waterHeld) this.gateway.setStream(world.x, world.y)

    if (!this.dragging) return

    if (this.preview.kind === 'member') {
      const targetNode = this.gateway.pickNode(world.x, world.y, snap)
      const at = targetNode
        ? this.gateway.nodePosition(targetNode) ?? this.applyGrid(world)
        : this.applyGrid(world)
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

  private releaseWater(): void {
    if (!this.waterHeld) return
    this.waterHeld = false
    this.gateway.clearStream()
  }

  private onUp = (e: PointerEvent): void => {
    if (e.button === 0) this.releaseWater()
    if (e.button !== 0 || !this.dragging) return
    this.dragging = false

    if (this.preview.kind === 'member' && this.preview.valid) {
      this.gateway.buildMember(
        this.preview.snappedFrom,
        this.dragFrom,
        this.preview.snappedTo,
        this.preview.to,
        this.material,
      )
    } else if (this.preview.kind === 'object' && this.preview.valid) {
      const x = (this.dragFrom.x + this.preview.to.x) * 0.5
      const y = (this.dragFrom.y + this.preview.to.y) * 0.5
      this.gateway.addObject(
        x,
        y,
        Math.abs(this.preview.to.x - this.dragFrom.x),
        Math.abs(this.preview.to.y - this.dragFrom.y),
        400,
      )
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
      if (e.shiftKey) this.gateway.redo()
      else this.gateway.undo()
      return
    }
    const tools: Record<string, ToolName> = {
      '1': 'build',
      '2': 'anchor',
      '3': 'object',
      '4': 'delete',
      '5': 'pan',
      '6': 'water',
    }
    const next = tools[e.key]
    if (next) {
      // Switching away from the water tool mid-hold must stop the stream.
      if (next !== 'water') this.releaseWater()
      this.tool = next
      return
    }
    if (e.key === 'q') this.material = 'wood'
    else if (e.key === 'w') this.material = 'steel'
  }
}
