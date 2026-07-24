import { parsePose, type CameraPose } from '../../scene/camera.ts'
import { SPECIES } from '../../scene/species.ts'
import { SCATTER_MAX_DENSITY } from '../../scene/scatter.ts'
import { updateQuery } from '../../url/state.ts'
import type { Coverage } from '../registry.ts'

export interface View {
  dispose(): void
}

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag)
  if (className) node.className = className
  if (text !== undefined) node.textContent = text
  return node
}

export function link(href: string, text: string): HTMLAnchorElement {
  const a = el('a', undefined, text)
  a.href = href
  return a
}

export function topbar(title: string, right: (HTMLElement | string)[] = []): HTMLElement {
  const bar = el('div', 'topbar')
  bar.appendChild(link('#/', 'groundcover lab'))
  bar.appendChild(el('span', 'title', title))
  bar.appendChild(el('span', 'spacer'))
  for (const item of right) bar.appendChild(typeof item === 'string' ? el('span', 'hint', item) : item)
  return bar
}

export function button(label: string, onClick: () => void, className = ''): HTMLButtonElement {
  const b = el('button', className, label)
  b.addEventListener('click', onClick)
  return b
}

export function readSeed(q: URLSearchParams): number {
  const v = Number(q.get('seed'))
  return Number.isFinite(v) && v > 0 ? Math.floor(v) : 42
}

/** Workload from URL: `radius` (m, half-size) and `dscale` (density multiplier). */
export function readCoverage(q: URLSearchParams): Coverage {
  // Number(null) is 0 — treat absent params as absent, not zero.
  const radius = q.has('radius') ? Number(q.get('radius')) : NaN
  const dscale = q.has('dscale') ? Number(q.get('dscale')) : NaN
  return {
    radius: Number.isFinite(radius) && radius > 0 ? Math.min(radius, 8192) : 128,
    densityScale: Number.isFinite(dscale) && dscale >= 0 ? Math.min(dscale, 64) : 1,
  }
}

/** Estimated plant count for a species set under a coverage (scatter clamps at 8/m²). */
export function estimatePlants(speciesIds: string[], coverage: Coverage): number {
  const area = (coverage.radius * 2) ** 2
  let total = 0
  for (const id of speciesIds) {
    const s = SPECIES.find((s) => s.id === id)
    if (s) total += Math.min(s.density * coverage.densityScale, SCATTER_MAX_DENSITY) * area
  }
  return total
}

export function formatCount(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}k`
  return `${Math.round(n)}`
}

/**
 * Toolbar inputs for the shared workload — writing them reloads the route so
 * every experiment on the page rebuilds against the new coverage.
 */
export function coverageControls(coverage: Coverage): HTMLElement[] {
  const make = (labelText: string, value: number, key: 'radius' | 'dscale', step: number): HTMLElement => {
    const label = el('label', undefined, labelText)
    const input = el('input')
    input.type = 'number'
    input.min = '0'
    input.step = String(step)
    input.value = String(value)
    input.addEventListener('change', () => {
      const v = Number(input.value)
      if (!Number.isFinite(v) || v < 0) return
      updateQuery({ [key]: String(v) })
      location.reload()
    })
    label.appendChild(input)
    return label
  }
  return [make('radius m', coverage.radius, 'radius', 16), make('density ×', coverage.densityScale, 'dscale', 0.25)]
}

/** `cam` param: a bookmark name or a serialized pose. */
export function resolveCam(
  q: URLSearchParams,
  bookmarks: Record<string, CameraPose>,
): { pose: CameraPose | undefined; bookmark: string | undefined } {
  const raw = q.get('cam')
  if (!raw) return { pose: undefined, bookmark: undefined }
  const bookmark = bookmarks[raw]
  if (bookmark) return { pose: { ...bookmark }, bookmark: raw }
  return { pose: parsePose(raw) ?? undefined, bookmark: undefined }
}

export async function copyToClipboard(text: string): Promise<void> {
  await navigator.clipboard.writeText(text)
}

// ---------------------------------------------------------------------------

import { serializePose } from '../../scene/camera.ts'
import { saveUserBookmark } from '../../scene/bookmarks.ts'
import { makeDebouncedQueryWriter } from '../../url/state.ts'
import type { LabApp } from '../loop.ts'
import type { Overlay } from '../../ui/overlay.ts'

export const STANDARD_CAM_KEYS = ['grazing', 'topdown', 'inside-plant', 'far-horizon']

/**
 * Camera <-> URL sync + bookmark hotkeys, shared by run/ab views:
 * keys 1-4 jump to standard cams (URL keeps the name), Shift+1-9 saves user
 * bookmarks, free movement serializes the pose into `cam=` (debounced).
 */
export function setupCameraSync(app: LabApp, initialCam: string | null, overlay: Overlay): () => void {
  const writeQuery = makeDebouncedQueryWriter()
  let lastCam = initialCam ?? ''

  const interval = setInterval(() => {
    const s = serializePose(app.camera.pose)
    const onBookmark = app.scene.bookmarks[lastCam]
    if (onBookmark ? s !== serializePose(onBookmark) : s !== lastCam) {
      lastCam = s
      writeQuery({ cam: s })
    }
  }, 250)

  const onKey = (ev: KeyboardEvent): void => {
    if (ev.target instanceof HTMLInputElement || ev.target instanceof HTMLTextAreaElement) return
    const digit = Number(ev.code.replace('Digit', ''))
    if (!Number.isInteger(digit) || digit < 1 || digit > 9) return
    if (ev.shiftKey) {
      const name = `u${digit}`
      saveUserBookmark(name, app.camera.pose)
      app.scene.bookmarks[name] = { ...app.camera.pose }
      overlay.toast(`saved bookmark ${name} (Shift+${digit} to overwrite, ${digit} to recall)`)
    } else {
      const names = [...STANDARD_CAM_KEYS, ...Object.keys(app.scene.bookmarks).filter((k) => k.startsWith('u')).sort()]
      const name = names[digit - 1]
      const pose = name !== undefined ? app.scene.bookmarks[name] : undefined
      if (pose && name) {
        app.camera.setPose(pose)
        lastCam = name
        writeQuery({ cam: name })
      }
    }
  }
  window.addEventListener('keydown', onKey)

  return () => {
    clearInterval(interval)
    window.removeEventListener('keydown', onKey)
  }
}

/** Current cam name if the pose sits on a named bookmark. */
export function currentBookmarkName(app: LabApp): string | null {
  const s = serializePose(app.camera.pose)
  for (const [name, pose] of Object.entries(app.scene.bookmarks)) {
    if (serializePose(pose) === s) return name
  }
  return null
}
