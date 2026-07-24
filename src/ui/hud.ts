import type { RollingStats } from '../gpu/timing.ts'
import { SPECIES_BUDGET_BYTES, type VramReport } from '../gpu/resources.ts'

const MB = 1024 * 1024

export interface HudOptions {
  container: HTMLElement
  stats: RollingStats
  vram: () => VramReport
  /** Labels shown with a "contended" warning (A/B pages). */
  contended?: boolean
  /** timestamp-query missing — GPU rows replaced by a notice. */
  gpuTimingAvailable: boolean
  extraLines?: () => string[]
}

/** Live stats HUD — GPU per-pass percentiles, CPU frame time, VRAM budget bars. */
export class Hud {
  private el: HTMLElement
  private interval: ReturnType<typeof setInterval>

  constructor(private opts: HudOptions) {
    this.el = document.createElement('div')
    this.el.className = 'hud'
    opts.container.appendChild(this.el)
    this.interval = setInterval(() => this.render(), 250)
    this.render()
  }

  private render(): void {
    const { stats } = this.opts
    const fmt = (v: number): string => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2))
    const rows: string[] = []

    const cpu = stats.stats('cpu/frame')
    const js = stats.stats('cpu/js')
    rows.push(`<table><tr><th></th><th>p50</th><th>p95</th><th>p99</th></tr>`)
    if (cpu) {
      rows.push(
        `<tr><td>cpu frame</td><td>${fmt(cpu.p50)}</td><td>${fmt(cpu.p95)}</td><td>${fmt(cpu.p99)}</td></tr>`,
      )
      rows.push(`<tr><td class="muted">├ fps ~${(1000 / Math.max(cpu.p50, 0.01)).toFixed(0)}</td><td colspan="3" class="muted">js ${js ? fmt(js.p50) : '–'}ms</td></tr>`)
    }
    const gpuLabels = stats.labels().filter((l) => l.startsWith('gpu/'))
    if (this.opts.gpuTimingAvailable) {
      for (const label of gpuLabels) {
        const s = stats.stats(label)!
        rows.push(
          `<tr><td>${label.slice(4)}</td><td>${fmt(s.p50)}</td><td>${fmt(s.p95)}</td><td>${fmt(s.p99)}</td></tr>`,
        )
      }
      const total = gpuLabels.reduce((a, l) => a + (stats.stats(l)?.p50 ?? 0), 0)
      rows.push(`<tr><td class="muted">gpu Σp50</td><td class="muted">${fmt(total)}</td><td colspan="2"></td></tr>`)
    } else {
      rows.push(`<tr><td colspan="4" class="muted">no timestamp-query — CPU timing only</td></tr>`)
    }
    rows.push('</table>')
    if (this.opts.contended) {
      rows.push(`<div class="section">A/B timings are contended — bench solo for real numbers</div>`)
    }

    const vram = this.opts.vram()
    const species = Object.entries(vram.bySpecies)
    rows.push(`<div class="section">vram ${(vram.totalBytes / MB).toFixed(1)}MB total</div>`)
    for (const [id, bytes] of species) {
      const frac = bytes / SPECIES_BUDGET_BYTES
      const cls = frac > 1 ? 'over' : frac > 0.8 ? 'warn' : ''
      rows.push(
        `<div>${id} <span class="muted">${(bytes / MB).toFixed(1)}/25MB</span></div>` +
          `<div class="vram-bar"><div class="${cls}" style="width:${Math.min(frac * 100, 100)}%"></div></div>`,
      )
    }
    for (const line of this.opts.extraLines?.() ?? []) rows.push(`<div class="section">${line}</div>`)
    this.el.innerHTML = rows.join('')
  }

  dispose(): void {
    clearInterval(this.interval)
    this.el.remove()
  }
}
