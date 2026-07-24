/** Toasts + fatal error surface — WebGPU errors land here, not just the console. */
export class Overlay {
  private toasts: HTMLElement
  private recent = new Map<string, number>()

  constructor(private root: HTMLElement) {
    this.toasts = document.createElement('div')
    this.toasts.className = 'toasts'
    document.body.appendChild(this.toasts)
  }

  toast(text: string, kind: 'error' | 'warn' | 'info' = 'info', ttlMs = 6000): void {
    // Collapse identical repeats within 2s (per-frame validation errors spam).
    const now = performance.now()
    const last = this.recent.get(text)
    if (last !== undefined && now - last < 2000) return
    this.recent.set(text, now)

    const el = document.createElement('div')
    el.className = `toast ${kind}`
    el.textContent = text
    this.toasts.appendChild(el)
    setTimeout(() => el.remove(), ttlMs)
    if (kind === 'error') console.error(text)
  }

  fatal(title: string, detail?: string): void {
    const wrap = document.createElement('div')
    wrap.className = 'fatal'
    const box = document.createElement('div')
    box.className = 'box'
    const h = document.createElement('h3')
    h.textContent = title
    box.appendChild(h)
    if (detail) {
      const p = document.createElement('pre')
      p.className = 'mono'
      p.style.whiteSpace = 'pre-wrap'
      p.textContent = detail
      box.appendChild(p)
    }
    const back = document.createElement('a')
    back.href = '#/'
    back.textContent = '← back to experiments'
    box.appendChild(back)
    wrap.appendChild(box)
    this.root.appendChild(wrap)
  }

  dispose(): void {
    this.toasts.remove()
  }
}
