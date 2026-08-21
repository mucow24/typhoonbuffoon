export interface LoopStats {
  /** Wall-clock ms for the last frame (raw). */
  frameMs: number
  /** Smoothed frame time, for a readable HUD. */
  smoothedFrameMs: number
  /** Fixed steps executed during the last frame. */
  stepsLastFrame: number
  /** Total fixed steps since start. */
  totalSteps: number
  /** Seconds of simulated time elapsed. */
  simTime: number
  /** True when the loop had to drop simulated time to keep up. */
  starved: boolean
}

export interface LoopOptions {
  /** Fixed simulation rate. 60 Hz unless there is a reason. */
  fixedHz?: number
  /**
   * Largest wall-clock slice we will ever try to catch up on, in ms. Beyond this
   * we drop time rather than spiral: a long GC pause or a tab switch must not
   * queue up hundreds of steps.
   */
  maxFrameMs?: number
  fixedUpdate(dt: number): void
  /** `alpha` is the 0..1 interpolation factor between the last two fixed states. */
  render(alpha: number, stats: Readonly<LoopStats>): void
}

/**
 * Fixed-timestep loop with an accumulator and decoupled render interpolation.
 * The sim never sees a variable dt - stability rule 1 in docs/PLAN.md is "add
 * substeps, never grow dt", and that only holds if dt is genuinely fixed.
 */
export class GameLoop {
  readonly fixedDt: number
  private readonly maxFrameMs: number
  private readonly fixedUpdate: (dt: number) => void
  private readonly renderFn: (alpha: number, stats: Readonly<LoopStats>) => void

  private accumulator = 0
  private lastTime = 0
  private rafHandle = 0
  private running = false

  readonly stats: LoopStats = {
    frameMs: 0,
    smoothedFrameMs: 0,
    stepsLastFrame: 0,
    totalSteps: 0,
    simTime: 0,
    starved: false,
  }

  constructor(opts: LoopOptions) {
    this.fixedDt = 1 / (opts.fixedHz ?? 60)
    this.maxFrameMs = opts.maxFrameMs ?? 250
    this.fixedUpdate = opts.fixedUpdate
    this.renderFn = opts.render
  }

  /**
   * Run exactly one fixed step outside requestAnimationFrame. Used for headless
   * verification, where the browser pane may not be compositing at all.
   */
  step(): void {
    this.fixedUpdate(this.fixedDt)
    this.stats.simTime += this.fixedDt
    this.stats.totalSteps++
  }

  start(): void {
    if (this.running) return
    this.running = true
    this.lastTime = performance.now()
    this.accumulator = 0
    this.rafHandle = requestAnimationFrame(this.tick)
  }

  stop(): void {
    this.running = false
    if (this.rafHandle) cancelAnimationFrame(this.rafHandle)
    this.rafHandle = 0
  }

  private tick = (now: number): void => {
    if (!this.running) return
    this.rafHandle = requestAnimationFrame(this.tick)

    let frameMs = now - this.lastTime
    this.lastTime = now

    const starved = frameMs > this.maxFrameMs
    if (starved) frameMs = this.maxFrameMs

    this.stats.frameMs = frameMs
    this.stats.smoothedFrameMs = this.stats.smoothedFrameMs === 0
      ? frameMs
      : this.stats.smoothedFrameMs * 0.9 + frameMs * 0.1
    this.stats.starved = starved

    this.accumulator += frameMs / 1000

    let steps = 0
    while (this.accumulator >= this.fixedDt) {
      this.step()
      this.accumulator -= this.fixedDt
      steps++
    }
    this.stats.stepsLastFrame = steps

    this.renderFn(this.accumulator / this.fixedDt, this.stats)
  }
}
