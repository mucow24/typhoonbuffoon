import { Panel } from './controls'

type StatFn = () => string

/**
 * Live stats HUD. Particle count and frame time live here permanently: field
 * width and sim resolution are unclamped by design, so when a configuration
 * tanks the framerate the cause has to be visible (docs/PLAN.md 8).
 */
export class DebugOverlay {
  readonly panel: Panel
  private readonly rows = new Map<string, { el: HTMLSpanElement; fn: StatFn }>()

  constructor(title = 'stats') {
    this.panel = new Panel({ title, side: 'right', width: 190 })
  }

  add(label: string, fn: StatFn): this {
    if (this.rows.has(label)) {
      this.rows.get(label)!.fn = fn
      return this
    }

    const row = document.createElement('div')
    Object.assign(row.style, {
      display: 'flex',
      justifyContent: 'space-between',
      gap: '8px',
      margin: '2px 0',
    } satisfies Partial<CSSStyleDeclaration>)

    const name = document.createElement('span')
    name.textContent = label
    name.style.color = 'var(--text-dim)'

    const value = document.createElement('span')
    value.style.color = 'var(--accent)'
    value.style.fontVariantNumeric = 'tabular-nums'

    row.appendChild(name)
    row.appendChild(value)
    this.panel.body.appendChild(row)

    this.rows.set(label, { el: value, fn })
    return this
  }

  update(): void {
    for (const { el, fn } of this.rows.values()) {
      const next = fn()
      if (el.textContent !== next) el.textContent = next
    }
  }
}
