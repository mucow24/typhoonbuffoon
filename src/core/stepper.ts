export interface StepperStats {
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
  /** True when the stepper had to drop simulated time to keep up. */
  starved: boolean
}

export interface StepperOptions {
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
  /**
   * Called once per advance(), after the frame's steps. `alpha` is the 0..1
   * interpolation factor between the last two fixed states. In the worker this
   * is where the snapshot is built and posted.
   */
  frameEnd(alpha: number, stats: Readonly<StepperStats>): void
}

/**
 * Fixed-timestep accumulator. The sim never sees a variable dt - stability
 * rule 1 in docs/PLAN.md is "add substeps, never grow dt", and that only holds
 * if dt is genuinely fixed.
 *
 * Deliberately free of any scheduler: the sim worker drives advance() from a
 * timer, tests drive it with synthetic clocks, and step() runs one fixed step
 * synchronously for headless verification. (Its rAF-driven ancestor lived on
 * the main thread as core/loop.ts until the sim moved behind the worker
 * boundary - docs/GPU_PLAN.md.)
 */
export class FixedStepper {
  readonly fixedDt: number
  private readonly maxFrameMs: number
  private readonly maxCatchUpSteps: number
  private readonly fixedUpdate: (dt: number) => void
  private readonly frameEnd: (alpha: number, stats: Readonly<StepperStats>) => void

  private accumulator = 0
  private lastTime = 0

  readonly stats: StepperStats = {
    frameMs: 0,
    smoothedFrameMs: 0,
    stepsLastFrame: 0,
    totalSteps: 0,
    simTime: 0,
    starved: false,
  }

  constructor(opts: StepperOptions) {
    this.fixedDt = 1 / (opts.fixedHz ?? 60)
    this.maxFrameMs = opts.maxFrameMs ?? 250
    this.maxCatchUpSteps = Math.max(1, opts.maxCatchUpSteps ?? 3)
    this.fixedUpdate = opts.fixedUpdate
    this.frameEnd = opts.frameEnd
  }

  /**
   * Run exactly one fixed step outside the accumulator. Used for headless
   * verification (the pump command), where no wall clock is involved.
   */
  step(): void {
    this.fixedUpdate(this.fixedDt)
    this.stats.simTime += this.fixedDt
    this.stats.totalSteps++
  }

  /** Reset frame timing so the next advance() measures from `now`. */
  beginFrames(now: number): void {
    this.lastTime = now
    this.accumulator = 0
  }

  /**
   * One frame's worth of work at wall-clock time `now` (ms). The accumulator
   * maths is exactly the kind of logic that quietly turns "a bit over budget"
   * into "frozen for many seconds", which is why it is testable in isolation.
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

    this.frameEnd(this.accumulator / this.fixedDt, this.stats)
  }
}
