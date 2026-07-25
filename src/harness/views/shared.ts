import { parsePose, type CameraPose } from '../../scene/camera.ts'
import { MESH_NOT_DEPLOYED } from '../../mesh/catalog.ts'
import { DEFAULT_PREVIEW_OBJECT, isPreviewObjectId, type PreviewObjectId } from '../../scene/preview.ts'
import { standById, standPlantCounts, STANDS, type Stand } from '../../scene/stands.ts'
import { updateQuery } from '../../url/state.ts'
import { isMaterialStage, materialStage } from '../materialStage.ts'
import type { RegistryEntry } from '../registry.ts'
import { standStage, type StageFactory } from '../stage.ts'

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

/** The active stand from the URL (`stand=<id>`); unknown/absent → 'default'. */
export function readStand(q: URLSearchParams): Stand {
  const id = q.get('stand')
  if (id === null) return standById('default')
  try {
    return standById(id)
  } catch {
    console.warn(`unknown stand "${id}" — falling back to default`)
    return standById('default')
  }
}

export function formatCount(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}k`
  return `${Math.round(n)}`
}

/** One-line stand summary for HUDs/topbars. */
export function standSummary(stand: Stand): string {
  const counts = standPlantCounts(stand)
  const per = counts.bySpecies.map((s) => `${formatCount(s.count)} ${s.species}`).join(' · ')
  return `stand ${stand.id} · ±${stand.radius}m · ~${formatCount(counts.total)} plants (${per})`
}

/**
 * Toolbar stand picker — switching reloads the route so every experiment on
 * the page rebuilds against the newly selected placement setup.
 */
export function standPicker(current: Stand): HTMLElement {
  const label = el('label', undefined, 'stand')
  const select = el('select')
  for (const stand of STANDS) {
    const option = el('option', undefined, `${stand.id} (~${formatCount(standPlantCounts(stand).total)})`)
    option.value = stand.id
    option.selected = stand.id === current.id
    select.appendChild(option)
  }
  select.addEventListener('change', () => {
    updateQuery({ stand: select.value === 'default' ? null : select.value })
    location.reload()
  })
  label.appendChild(select)
  return label
}

// ---------------------------------------------------------------------------
// Choosing the stage from the experiment's kind
// ---------------------------------------------------------------------------

/**
 * What a route needs to know once it has looked at `manifest.kind`: which
 * Stage to build, and which partition key the bench result should record.
 * Exactly one of `stand` / `previewObject` is set.
 */
export interface StageChoice {
  kind: 'stand' | 'material'
  stage: StageFactory
  /** Renderers only — the placement setup being drawn. */
  stand: Stand | null
  /** Materials only — the preview object the stage opens on. */
  previewObject: PreviewObjectId | null
  /** One line for the topbar/HUD. */
  summary: string
}

/** The preview object from the URL (`obj=`), else the manifest's, else sphere. */
export function readPreviewObject(q: URLSearchParams, entry?: RegistryEntry): PreviewObjectId {
  const fromUrl = q.get('obj')
  if (isPreviewObjectId(fromUrl)) return fromUrl
  if (fromUrl !== null) console.warn(`unknown preview object "${fromUrl}" — falling back to ${DEFAULT_PREVIEW_OBJECT}`)
  const fromManifest = entry?.manifest?.previewObject
  return isPreviewObjectId(fromManifest) ? fromManifest : DEFAULT_PREVIEW_OBJECT
}

/**
 * Pick the stage for an experiment. This is the ONLY place a route branches on
 * kind — `#/run`, `#/ab` and `#/bench` are otherwise identical for a material
 * and for a renderer, which is the point of the Stage seam.
 */
export function stageFor(entry: RegistryEntry, q: URLSearchParams, seed: number): StageChoice {
  if (entry.kind === 'material') {
    const previewObject = readPreviewObject(q, entry)
    return {
      kind: 'material',
      stage: materialStage(previewObject),
      stand: null,
      previewObject,
      summary: `material preview · ${previewObject}`,
    }
  }
  const stand = readStand(q)
  return {
    kind: 'stand',
    stage: standStage(stand, seed),
    stand,
    previewObject: null,
    summary: standSummary(stand),
  }
}

/**
 * The preview object a running app is showing RIGHT NOW (the picker changes it
 * live), or the default for a page that has no material stage.
 */
export function currentPreviewObject(app: LabApp): PreviewObjectId {
  return isMaterialStage(app.stage) ? app.stage.previewObject : DEFAULT_PREVIEW_OBJECT
}

/**
 * One-line summary of the active preview object, for HUDs/topbars — the
 * material counterpart of `standSummary`. Reports the uv-to-world density
 * because that is what a material turns into a life-size tile repeat.
 */
export function previewSummary(app: LabApp): string {
  const stage = app.stage
  if (!isMaterialStage(stage)) return 'no preview object'
  const m = stage.geometry.mesh
  const wrap = m.uvPeriod.map((p, i) => (p > 0 ? `${'uv'[i]} wraps @${p.toFixed(2)}m` : `${'uv'[i]} open`)).join(' · ')
  return (
    `${m.title} · ${formatCount(m.indexCount / 3)} tris · uv in metres ` +
    `[${m.uvBounds.min[0].toFixed(2)},${m.uvBounds.min[1].toFixed(2)}]–` +
    `[${m.uvBounds.max[0].toFixed(2)},${m.uvBounds.max[1].toFixed(2)}] · ${wrap}`
  )
}

/**
 * Toolbar preview-object picker — the material counterpart of `standPicker`.
 * Switching is live (the stage holds all four meshes and the experiment reads
 * `ctx.preview.mesh` per frame), so unlike the stand picker it does not reload.
 *
 * Deliberately the same unstyled label+select as everything else in this
 * toolbar: the UI rework is a later phase.
 */
export function previewObjectPicker(app: LabApp): HTMLElement {
  const stage = app.stage
  const label = el('label', undefined, 'object')
  const select = el('select')
  if (!isMaterialStage(stage)) return label
  for (const mesh of stage.geometry.objects) {
    const option = el('option', undefined, mesh.id)
    option.value = mesh.id
    option.title = mesh.purpose
    option.selected = mesh.id === stage.previewObject
    select.appendChild(option)
  }
  select.addEventListener('change', () => {
    if (!isPreviewObjectId(select.value)) return
    stage.previewObject = select.value
    updateQuery({ obj: select.value === DEFAULT_PREVIEW_OBJECT ? null : select.value })
  })
  label.appendChild(select)
  return label
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

/**
 * Detail text for overlay.fatal(). Normally the stack — but an experiment
 * that wanted a raw source mesh on a deployment built without them fails for
 * a boring, explainable reason, so say that instead of dumping a minified
 * stack the visitor cannot act on.
 */
export function fatalDetail(err: unknown): string {
  const message = err instanceof Error ? err.message : String(err)
  if (message.includes(MESH_NOT_DEPLOYED)) {
    return (
      `${message}\n\n` +
      'This view reads the raw GCMESH1 source mesh, and the ~357MB of source binaries are not part of ' +
      'this deployment. Rebuild with GC_DEPLOY_MESHES=1, or run the lab locally (npm run dev). ' +
      'Experiments with committed baked artifacts are unaffected.'
    )
  }
  return err instanceof Error ? (err.stack ?? err.message) : String(err)
}

// ---------------------------------------------------------------------------

import { serializePose } from '../../scene/camera.ts'
import { saveUserBookmark } from '../../scene/bookmarks.ts'
import { makeDebouncedQueryWriter } from '../../url/state.ts'
import { DEBUG_VIEW_MODES, debugModeIndex, parseDebugMode } from '../debug.ts'
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
    const onBookmark = app.bookmarks[lastCam]
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
      app.bookmarks[name] = { ...app.camera.pose }
      overlay.toast(`saved bookmark ${name} (Shift+${digit} to overwrite, ${digit} to recall)`)
    } else {
      const names = [...STANDARD_CAM_KEYS, ...Object.keys(app.bookmarks).filter((k) => k.startsWith('u')).sort()]
      const name = names[digit - 1]
      const pose = name !== undefined ? app.bookmarks[name] : undefined
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

/**
 * The global debug-view dropdown. Applies live (it is just a uniform), so no
 * reload — and it drives every renderer on the page identically.
 */
export function debugPicker(app: LabApp, initial: string | null): HTMLElement {
  const mode = parseDebugMode(initial)
  app.debugMode = debugModeIndex(mode)
  const label = el('label', undefined, 'view')
  const select = el('select')
  for (const name of DEBUG_VIEW_MODES) {
    const option = el('option', undefined, name)
    option.value = name
    option.selected = name === mode
    select.appendChild(option)
  }
  select.addEventListener('change', () => {
    const picked = parseDebugMode(select.value)
    app.debugMode = debugModeIndex(picked)
    updateQuery({ debug: picked === 'off' ? null : picked })
  })
  label.appendChild(select)
  return label
}

/** Current cam name if the pose sits on a named bookmark. */
export function currentBookmarkName(app: LabApp): string | null {
  const s = serializePose(app.camera.pose)
  for (const [name, pose] of Object.entries(app.bookmarks)) {
    if (serializePose(pose) === s) return name
  }
  return null
}
