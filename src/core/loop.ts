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
  /**
   * Most fixed steps one frame may run. THE overload valve: when a step costs
   * more than the frame budget, wall-clock debt accrues faster than steps can
   * pay it down, and an uncapped (or generously capped) catch-up loop makes
   * every frame slower than the last until it pins at maxFrameMs' worth of
   * steps - measured in the app as a 60%-over-budget sim becoming a sub-1-fps
   * freeze. Beyond this many steps the remaining debt is DROPPED: the game
   * runs in slow motion and stays responsive. Small hiccups (a dropped frame
   * or two) still catch up fully.
   */
  maxCatchUpSteps?: number
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
  private readonly maxCatchUpSteps: number
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
    this.maxCatchUpSteps = Math.max(1, opts.maxCatchUpSteps ?? 3)
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
    this.beginFrames(performance.now())
    this.rafHandle = requestAnimationFrame(this.tick)
  }

  stop(): void {
    this.running = false
    if (this.rafHandle) cancelAnimationFrame(this.rafHandle)
    this.rafHandle = 0
  }

  /** Reset frame timing so the next advance() measures from `now`. */
  beginFrames(now: number): void {
    this.lastTime = now
    this.accumulator = 0
  }

  /**
   * One frame's worth of work at wall-clock time `now` (ms). Extracted from
   * the rAF callback so the loop's overload behaviour is testable - the
   * accumulator maths is exactly the kind of logic that quietly turns "a bit
   * over budget" into "frozen for many seconds".
   */
  advance(now: number): void {
    let frameMs = now - this.lastTime
    this.lastTime = now

    let starved = frameMs > this.maxFrameMs
    if (starved) frameMs = this.maxFrameMs

    this.stats.frameMs = frameMs
    this.stats.smoothedFrameMs = this.stats.smoothedFrameMs === 0
      ? frameMs
      : this.stats.smoothedFrameMs * 0.9 + frameMs * 0.1

    this.accumulator += frameMs / 1000

    let steps = 0
    while (this.accumulator >= this.fixedDt && steps < this.maxCatchUpSteps) {
      this.step()
      this.accumulator -= this.fixedDt
      steps++
    }
    // Debt we chose not to run is dropped, not carried: carrying it means the
    // next frame starts behind too, and an over-budget sim never sees a
    // healthy frame again. Dropping it is the slow-motion contract.
    if (this.accumulator >= this.fixedDt) {
      this.accumulator = this.accumulator % this.fixedDt
      starved = true
    }
    this.stats.stepsLastFrame = steps
    this.stats.starved = starved

    this.renderFn(this.accumulator / this.fixedDt, this.stats)
  }

  private tick = (now: number): void => {
    if (!this.running) return
    this.rafHandle = requestAnimationFrame(this.tick)
    this.advance(now)
  }
}
