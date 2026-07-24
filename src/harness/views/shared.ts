import { parsePose, type CameraPose } from '../../scene/camera.ts'

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
