/**
 * A deliberately tiny DOM control kit. The editor panels are a palette, a few
 * sliders and a readout; a framework would be friction here (docs/PLAN.md 5.1).
 */

import { snapToDetent } from '../core/math'
import { THUMB_PX, TRACK_PX, atFraction, fillOrigin, fillRange, fraction } from './sliderGeometry'

const uiRoot = (): HTMLElement => {
  const el = document.getElementById('ui')
  if (!el) throw new Error('#ui root missing from index.html')
  return el
}

function style(el: HTMLElement, css: Partial<CSSStyleDeclaration>): void {
  Object.assign(el.style, css)
}

export interface PanelOptions {
  title: string
  side?: 'left' | 'right'
  width?: number
  collapsed?: boolean
}

export class Panel {
  readonly root: HTMLDivElement
  readonly body: HTMLDivElement
  private collapsed: boolean

  constructor(opts: PanelOptions) {
    this.collapsed = opts.collapsed ?? false

    this.root = document.createElement('div')
    style(this.root, {
      position: 'absolute',
      top: '10px',
      [opts.side === 'right' ? 'right' : 'left']: '10px',
      width: `${opts.width ?? 220}px`,
      background: 'var(--panel-bg)',
      border: '1px solid var(--panel-border)',
      borderRadius: '6px',
      fontSize: '11px',
      lineHeight: '1.5',
      backdropFilter: 'blur(6px)',
      userSelect: 'none',
      maxHeight: 'calc(100vh - 20px)',
      display: 'flex',
      flexDirection: 'column',
    })

    const header = document.createElement('div')
    header.textContent = opts.title
    style(header, {
      padding: '6px 9px',
      fontWeight: '600',
      letterSpacing: '0.06em',
      textTransform: 'uppercase',
      fontSize: '10px',
      color: 'var(--text-dim)',
      cursor: 'pointer',
      borderBottom: '1px solid var(--panel-border)',
      flex: '0 0 auto',
    })
    header.addEventListener('click', () => this.toggle())
    this.root.appendChild(header)

    this.body = document.createElement('div')
    style(this.body, {
      padding: '8px 9px',
      display: this.collapsed ? 'none' : 'block',
      overflowY: 'auto',
      flex: '1 1 auto',
    })
    this.root.appendChild(this.body)

    uiRoot().appendChild(this.root)
  }

  toggle(): void {
    this.collapsed = !this.collapsed
    this.body.style.display = this.collapsed ? 'none' : 'block'
  }

  /** Stack this panel below another one on the same side. */
  below(other: Panel, gap = 8): this {
    const reposition = () => {
      const r = other.root.getBoundingClientRect()
      this.root.style.top = `${r.bottom + gap}px`
    }
    reposition()
    new ResizeObserver(reposition).observe(other.root)
    return this
  }

  section(label: string): HTMLDivElement {
    const el = document.createElement('div')
    el.textContent = label
    style(el, {
      marginTop: '10px',
      marginBottom: '4px',
      fontSize: '9px',
      letterSpacing: '0.08em',
      textTransform: 'uppercase',
      color: 'var(--text-dim)',
      borderTop: '1px solid var(--panel-border)',
      paddingTop: '7px',
    })
    if (this.body.childElementCount === 0) {
      el.style.marginTop = '0'
      el.style.borderTop = 'none'
      el.style.paddingTop = '0'
    }
    this.body.appendChild(el)
    return el
  }

  note(text: string): HTMLDivElement {
    const el = document.createElement('div')
    el.textContent = text
    style(el, { color: 'var(--text-dim)', fontSize: '10px', margin: '4px 0' })
    this.body.appendChild(el)
    return el
  }
}

function labelledRow(parent: HTMLElement, label: string): { row: HTMLDivElement; value: HTMLSpanElement } {
  const row = document.createElement('div')
  style(row, { margin: '5px 0' })

  const head = document.createElement('div')
  style(head, { display: 'flex', justifyContent: 'space-between', gap: '6px' })

  const name = document.createElement('span')
  name.textContent = label
  style(name, { color: 'var(--text-dim)' })

  const value = document.createElement('span')
  style(value, { color: 'var(--accent)', fontVariantNumeric: 'tabular-nums' })

  head.appendChild(name)
  head.appendChild(value)
  row.appendChild(head)
  parent.appendChild(row)
  return { row, value }
}

/**
 * The kit draws its own slider track.
 *
 * A native range fills from `min` to the thumb, which is wrong for anything
 * signed: at dead calm the wind slider read half full, as though the storm
 * were half on. Neither `accent-color` nor any other property moves that
 * origin, and the fill lives in a shadow pseudo-element, so the only way to
 * put it where it belongs is to paint the track ourselves - which also makes
 * the thumb size a number we KNOW rather than one we assume, so the tick
 * marks and the fill can be placed against it exactly.
 *
 * Injected once, lazily: the kit is otherwise inline-styled, and
 * pseudo-elements cannot be.
 */
let sliderCssInjected = false
function ensureSliderCss(): void {
  if (sliderCssInjected) return
  sliderCssInjected = true
  const el = document.createElement('style')
  el.textContent = `
.tb-slider {
  -webkit-appearance: none;
  appearance: none;
  width: 100%;
  height: ${THUMB_PX}px;
  margin: 2px 0 0;
  background: transparent;
  cursor: pointer;
  --tb-groove: rgba(255, 255, 255, 0.14);
  --tb-fill: var(--tb-groove);
}
.tb-slider::-webkit-slider-runnable-track {
  height: ${TRACK_PX}px;
  border-radius: ${TRACK_PX / 2}px;
  background: var(--tb-fill);
}
.tb-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  box-sizing: border-box;
  width: ${THUMB_PX}px;
  height: ${THUMB_PX}px;
  /* Centre the thumb on the track rather than on the track's top edge. */
  margin-top: ${(TRACK_PX - THUMB_PX) / 2}px;
  border-radius: 50%;
  border: 1px solid rgba(0, 0, 0, 0.45);
  background: var(--accent);
}
.tb-slider::-moz-range-track {
  height: ${TRACK_PX}px;
  border-radius: ${TRACK_PX / 2}px;
  background: var(--tb-fill);
}
/* Firefox paints its own left-origin progress bar over the track. Not here. */
.tb-slider::-moz-range-progress { height: ${TRACK_PX}px; background: transparent; }
.tb-slider::-moz-range-thumb {
  box-sizing: border-box;
  width: ${THUMB_PX}px;
  height: ${THUMB_PX}px;
  border-radius: 50%;
  border: 1px solid rgba(0, 0, 0, 0.45);
  background: var(--accent);
}
.tb-slider:hover::-webkit-slider-thumb { filter: brightness(1.15); }
.tb-slider:hover::-moz-range-thumb { filter: brightness(1.15); }
.tb-slider:focus { outline: none; }
.tb-slider:focus-visible::-webkit-slider-thumb { box-shadow: 0 0 0 3px rgba(111, 211, 255, 0.3); }
.tb-slider:focus-visible::-moz-range-thumb { box-shadow: 0 0 0 3px rgba(111, 211, 255, 0.3); }
`
  document.head.appendChild(el)
}

/**
 * The cosmetic tick row under a slider. Marks only: a tick you can feel is a
 * detent, and a slider that snapped to every 50 kph could not be set to 175.
 */
function tickStrip(opts: SliderOptions, detents: readonly number[]): HTMLDivElement {
  const strip = document.createElement('div')
  style(strip, { position: 'relative', height: '5px', margin: '-1px 0 2px' })

  for (const t of opts.ticks ?? []) {
    const detent = detents.includes(t)
    const mark = document.createElement('div')
    style(mark, {
      position: 'absolute',
      top: '0',
      left: atFraction(fraction(t, opts.min, opts.max)),
      transform: 'translateX(-50%)',
      width: '1px',
      height: detent ? '5px' : '3px',
      background: detent ? 'var(--accent)' : 'var(--text-dim)',
      opacity: detent ? '0.85' : '0.45',
    })
    strip.appendChild(mark)
  }
  return strip
}

export interface SliderOptions {
  label: string
  min: number
  max: number
  step?: number
  value: number
  /** Values to draw a mark at. Purely a scale - they do not snap. */
  ticks?: number[]
  /** Values the thumb sticks to while it is being dragged. */
  detents?: number[]
  /** Detent catchment in value units. Defaults to two steps. */
  detentRadius?: number
  /** Where the filled part of the bar grows from. Defaults to zero. */
  origin?: number
  format?: (v: number) => string
  onInput: (v: number) => void
}

export class Slider {
  readonly input: HTMLInputElement
  private readonly valueEl: HTMLSpanElement
  private readonly format: (v: number) => string
  private readonly min: number
  private readonly max: number
  private readonly origin: number
  /** Detents apply to the pointer only - see the listener below. */
  private dragging = false

  constructor(parent: HTMLElement, opts: SliderOptions) {
    ensureSliderCss()
    const { row, value } = labelledRow(parent, opts.label)
    this.valueEl = value
    this.format = opts.format ?? ((v) => v.toFixed(2))
    this.min = opts.min
    this.max = opts.max
    this.origin = opts.origin ?? fillOrigin(opts.min, opts.max)

    const step = opts.step ?? (opts.max - opts.min) / 100
    const detents = opts.detents ?? []
    const detentRadius = opts.detentRadius ?? step * 2

    this.input = document.createElement('input')
    this.input.type = 'range'
    this.input.className = 'tb-slider'
    this.input.min = String(opts.min)
    this.input.max = String(opts.max)
    this.input.step = String(step)
    this.input.value = String(opts.value)

    if (detents.length > 0) {
      this.input.addEventListener('pointerdown', () => {
        this.dragging = true
      })
      // The pointer is usually released off the input - a range drag wanders
      // far from a 200px control - so the release has to be watched globally.
      const release = () => {
        this.dragging = false
      }
      window.addEventListener('pointerup', release)
      window.addEventListener('pointercancel', release)
    }

    this.input.addEventListener('input', () => {
      let v = Number(this.input.value)
      // Pointer only. Arrow keys step by exactly one step, and a detent wider
      // than a step would swallow the keypress and trap the slider on zero.
      if (this.dragging) {
        const snapped = snapToDetent(v, detents, detentRadius)
        if (snapped !== v) {
          v = snapped
          this.input.value = String(v)
        }
      }
      this.valueEl.textContent = this.format(v)
      this.paint()
      opts.onInput(v)
    })

    row.appendChild(this.input)
    if (opts.ticks && opts.ticks.length > 0) row.appendChild(tickStrip(opts, detents))
    this.valueEl.textContent = this.format(opts.value)
    this.paint()
  }

  /**
   * Repaint the track. The fill runs between the origin and the thumb CENTRE,
   * so it ends under the thumb rather than short of it or past it.
   */
  private paint(): void {
    const [lo, hi] = fillRange(Number(this.input.value), this.origin, this.min, this.max)
    const a = atFraction(lo)
    const b = atFraction(hi)
    this.input.style.setProperty(
      '--tb-fill',
      `linear-gradient(to right, var(--tb-groove) 0 ${a}, var(--accent) ${a} ${b}, var(--tb-groove) ${b} 100%)`,
    )
  }

  get value(): number {
    return Number(this.input.value)
  }

  set(v: number, fire = false): void {
    this.input.value = String(v)
    this.valueEl.textContent = this.format(v)
    this.paint()
    if (fire) this.input.dispatchEvent(new Event('input'))
  }
}

export interface NumberFieldOptions {
  label: string
  value: number
  min?: number
  max?: number
  step?: number
  suffix?: string
  onCommit: (v: number) => void
}

/** Typed entry. Commits on Enter or blur, not on every keystroke. */
export class NumberField {
  readonly input: HTMLInputElement

  constructor(parent: HTMLElement, opts: NumberFieldOptions) {
    const row = document.createElement('div')
    style(row, { margin: '5px 0', display: 'flex', alignItems: 'center', gap: '6px' })

    const name = document.createElement('span')
    name.textContent = opts.label
    style(name, { color: 'var(--text-dim)', flex: '1 1 auto' })

    this.input = document.createElement('input')
    this.input.type = 'number'
    this.input.value = String(opts.value)
    if (opts.min !== undefined) this.input.min = String(opts.min)
    if (opts.max !== undefined) this.input.max = String(opts.max)
    if (opts.step !== undefined) this.input.step = String(opts.step)
    style(this.input, {
      width: '68px',
      flex: '0 0 auto',
      background: 'rgba(0,0,0,0.35)',
      border: '1px solid var(--panel-border)',
      borderRadius: '3px',
      color: 'var(--text)',
      font: 'inherit',
      padding: '2px 4px',
      textAlign: 'right',
    })

    const commit = () => {
      let v = Number(this.input.value)
      if (!Number.isFinite(v)) return
      if (opts.min !== undefined) v = Math.max(opts.min, v)
      if (opts.max !== undefined) v = Math.min(opts.max, v)
      this.input.value = String(v)
      opts.onCommit(v)
    }
    this.input.addEventListener('change', commit)
    this.input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        commit()
        this.input.blur()
      }
      e.stopPropagation()
    })

    row.appendChild(name)
    row.appendChild(this.input)
    if (opts.suffix) {
      const suffix = document.createElement('span')
      suffix.textContent = opts.suffix
      style(suffix, { color: 'var(--text-dim)', flex: '0 0 auto' })
      row.appendChild(suffix)
    }
    parent.appendChild(row)
  }

  set(v: number): void {
    this.input.value = String(v)
  }
}

export function button(parent: HTMLElement, label: string, onClick: () => void): HTMLButtonElement {
  const el = document.createElement('button')
  el.textContent = label
  style(el, {
    display: 'block',
    width: '100%',
    margin: '4px 0 0',
    padding: '5px 6px',
    background: 'rgba(255,255,255,0.07)',
    border: '1px solid var(--panel-border)',
    borderRadius: '3px',
    color: 'var(--text)',
    font: 'inherit',
    cursor: 'pointer',
  })
  el.addEventListener('mouseenter', () => (el.style.background = 'rgba(255,255,255,0.14)'))
  el.addEventListener('mouseleave', () => (el.style.background = 'rgba(255,255,255,0.07)'))
  el.addEventListener('click', onClick)
  parent.appendChild(el)
  return el
}

export interface ChoiceOptions<T extends string> {
  label: string
  options: readonly { value: T; label: string }[]
  value: T
  onChange: (v: T) => void
}

export class Choice<T extends string> {
  readonly select: HTMLSelectElement

  constructor(parent: HTMLElement, opts: ChoiceOptions<T>) {
    const row = document.createElement('div')
    style(row, { margin: '5px 0', display: 'flex', alignItems: 'center', gap: '6px' })

    const name = document.createElement('span')
    name.textContent = opts.label
    style(name, { color: 'var(--text-dim)', flex: '1 1 auto' })

    this.select = document.createElement('select')
    style(this.select, {
      flex: '0 0 auto',
      background: 'rgba(0,0,0,0.35)',
      border: '1px solid var(--panel-border)',
      borderRadius: '3px',
      color: 'var(--text)',
      font: 'inherit',
      padding: '2px 4px',
    })
    for (const o of opts.options) {
      const opt = document.createElement('option')
      opt.value = o.value
      opt.textContent = o.label
      this.select.appendChild(opt)
    }
    this.select.value = opts.value
    this.select.addEventListener('change', () => opts.onChange(this.select.value as T))

    row.appendChild(name)
    row.appendChild(this.select)
    parent.appendChild(row)
  }

  set(v: T): void {
    this.select.value = v
  }
}

export function toggle(
  parent: HTMLElement,
  label: string,
  value: boolean,
  onChange: (v: boolean) => void,
): HTMLInputElement {
  const row = document.createElement('label')
  style(row, { margin: '5px 0', display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' })

  const input = document.createElement('input')
  input.type = 'checkbox'
  input.checked = value
  style(input, { accentColor: 'var(--accent)', margin: '0' })
  input.addEventListener('change', () => onChange(input.checked))

  const name = document.createElement('span')
  name.textContent = label
  style(name, { color: 'var(--text-dim)' })

  row.appendChild(input)
  row.appendChild(name)
  parent.appendChild(row)
  return input
}
