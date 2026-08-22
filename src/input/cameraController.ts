import type { Camera } from '../render/camera'

export interface ViewSize {
  width: number
  height: number
}

/**
 * Pan and zoom. Left-drag pans until the build tools claim the left button, at
 * which point `panWithLeft` goes false and panning is middle-drag or the
 * dedicated pan tool. Space is deliberately NOT a pan modifier - it is the
 * play/pause key.
 */
export class CameraController {
  panWithLeft = true

  private dragging = false
  private dragButton = -1
  private lastX = 0
  private lastY = 0
  private readonly disposers: (() => void)[] = []

  constructor(
    private readonly camera: Camera,
    private readonly element: HTMLElement,
    private readonly view: ViewSize,
  ) {
    this.on(element, 'pointerdown', this.onPointerDown as EventListener)
    this.on(window, 'pointermove', this.onPointerMove as EventListener)
    this.on(window, 'pointerup', this.onPointerUp as EventListener)
    this.on(element, 'wheel', this.onWheel as EventListener, { passive: false })
    this.on(element, 'contextmenu', this.onContextMenu as EventListener)
  }

  private on(
    target: EventTarget,
    type: string,
    handler: EventListener,
    opts?: AddEventListenerOptions,
  ): void {
    target.addEventListener(type, handler, opts)
    this.disposers.push(() => target.removeEventListener(type, handler, opts))
  }

  dispose(): void {
    for (const d of this.disposers) d()
    this.disposers.length = 0
  }

  /** True while the camera is consuming drag input, so tools can stand down. */
  get isPanning(): boolean {
    return this.dragging
  }

  private canPanWith(button: number): boolean {
    if (button === 1) return true // middle always pans
    if (button === 0) return this.panWithLeft
    return false
  }

  private onPointerDown = (e: PointerEvent): void => {
    if (!this.canPanWith(e.button)) return
    this.dragging = true
    this.dragButton = e.button
    this.lastX = e.clientX
    this.lastY = e.clientY
    this.element.style.cursor = 'grabbing'
  }

  private onPointerMove = (e: PointerEvent): void => {
    if (!this.dragging) return
    const s = this.camera.scale
    const dx = e.clientX - this.lastX
    const dy = e.clientY - this.lastY
    this.lastX = e.clientX
    this.lastY = e.clientY
    this.camera.x -= dx / s
    this.camera.y += dy / s // world Y is up, screen Y is down
  }

  private onPointerUp = (e: PointerEvent): void => {
    if (!this.dragging || e.button !== this.dragButton) return
    this.dragging = false
    this.dragButton = -1
    this.element.style.cursor = ''
  }

  private onWheel = (e: WheelEvent): void => {
    e.preventDefault()
    const rect = this.element.getBoundingClientRect()
    const factor = Math.exp(-e.deltaY * 0.0015)
    this.camera.zoomAt(
      e.clientX - rect.left,
      e.clientY - rect.top,
      factor,
      this.view.width,
      this.view.height,
    )
  }

  private onContextMenu = (e: Event): void => {
    e.preventDefault()
  }
}
