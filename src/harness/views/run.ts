import { parsePose } from '../../scene/camera.ts'
import { HAS_DEV_SINK } from '../../util/env.ts'
import { buildHash, makeDebouncedQueryWriter, type HashState } from '../../url/state.ts'
import { Overlay } from '../../ui/overlay.ts'
import { Hud } from '../../ui/hud.ts'
import { buildParamsPanel } from '../../ui/panel.ts'
import { captureViewPng, downloadBlob, uploadCapture } from '../capture.ts'
import { fetchRating, saveRating } from '../ratings.ts'
import { pipsRow } from '../../ui/pips.ts'
import { LabApp } from '../loop.ts'
import { standStage } from '../stage.ts'
import { paramsToQuery } from '../params.ts'
import { findExperiment } from '../registry.ts'
import {
  button,
  copyToClipboard,
  currentBookmarkName,
  debugPicker,
  el,
  fatalDetail,
  readSeed,
  readStand,
  resolveCam,
  setupCameraSync,
  standPicker,
  standSummary,
  topbar,
  type View,
} from './shared.ts'

/** #/run/<id> — single-experiment runner. */
export async function runView(root: HTMLElement, state: HashState): Promise<View> {
  const id = state.path[1] ?? ''
  const page = el('div', 'page')
  root.appendChild(page)
  const overlay = new Overlay(page)
  const cleanups: (() => void)[] = [() => overlay.dispose(), () => page.remove()]
  const dispose = (): void => {
    for (const fn of [...cleanups].reverse()) fn()
  }

  try {
    const entry = await findExperiment(id)
    const manifest = entry.manifest!
    const seed = readSeed(state.q)
    const stand = readStand(state.q)
    const isReference = manifest.status === 'reference'

    page.appendChild(
      topbar(manifest.title, [
        el('span', 'hint', isReference ? 'reference — ignores the stand' : `${stand.id} · seed ${seed}`),
        el('span', 'hint', 'drag=look · WASD/QE=fly · Tab=orbit · 1-4 cams · Shift+N save cam'),
      ]),
    )
    const content = el('div', 'content')
    const viewer = el('div', 'viewer')
    const canvas = el('canvas')
    viewer.appendChild(canvas)
    content.appendChild(viewer)
    page.appendChild(content)

    // Plain status line while the page starts — a multi-minute bake otherwise
    // shows nothing at all. Removed as soon as the app exists.
    const status = el('div', 'hud', 'starting…')
    viewer.appendChild(status)

    const det = state.q.get('det') === '1'
    const t = Number(state.q.get('t'))
    const app = await LabApp.create({
      canvas,
      seed,
      stage: standStage(stand, seed),
      experiments: [{ entry, ns: 'p', query: state.q }],
      onProgress: (fraction, note) => {
        status.textContent = `${Math.round(fraction * 100)}% · ${note ?? ''}`
      },
      onError: (m) => overlay.toast(m, 'error'),
      onShaderMessage: (m) =>
        overlay.toast(m.text, m.kind === 'error' ? 'error' : 'info', m.kind === 'error' ? 10000 : 2500),
      onDeviceLost: (reason) => {
        overlay.toast(`device lost (${reason}) — reloading`, 'error')
        setTimeout(() => location.reload(), 1200)
      },
      ...(det && { fixedDt: 1 / 60 }),
    })
    cleanups.push(() => app.dispose())
    status.remove()
    const view = app.views[0]!

    for (const [name, poseStr] of Object.entries(manifest.cams ?? {})) {
      const pose = parsePose(poseStr)
      if (pose) app.bookmarks[name] = pose
    }
    const { pose } = resolveCam(state.q, app.bookmarks)
    if (pose) app.camera.setPose(pose)
    if (det && Number.isFinite(t)) {
      app.time = t
      app.paused = true
    }

    cleanups.push(setupCameraSync(app, state.q.get('cam'), overlay))

    const hud = new Hud({
      container: viewer,
      stats: app.stats,
      vram: () => app.vramReport(),
      gpuTimingAvailable: app.gpu.hasTimestamps,
      extraLines: () => [isReference ? 'reference — ignores the stand' : standSummary(stand)],
    })
    cleanups.push(() => hud.dispose())

    const paneHolder = el('div', 'pane-holder')
    viewer.appendChild(paneHolder)
    const writeQuery = makeDebouncedQueryWriter()
    const panel = buildParamsPanel({
      container: paneHolder,
      title: manifest.title,
      schema: manifest.params,
      values: view.params,
      onChange: (key) => {
        app.setParam(view, key, view.params[key])
        writeQuery(paramsToQuery(manifest.params, view.params, 'p'))
      },
    })
    cleanups.push(() => panel.dispose())

    const toolbar = el('div', 'toolbar')
    toolbar.append(
      button('copy link', () => {
        void copyToClipboard(location.href).then(() => overlay.toast('link copied'))
      }),
    )
    // Capture buttons write into the repo through the dev-sink — a static
    // deployment has nowhere to put them, so they simply do not exist there.
    if (HAS_DEV_SINK) {
      toolbar.append(
        button('thumbnail', () => {
          void captureViewPng(app, 0, 640)
            .then((blob) => uploadCapture(id, blob))
            .then((saved) => overlay.toast(`saved ${saved} — shows on the browser card`))
            .catch(() => {
              overlay.toast('dev server sink unavailable — downloading instead', 'warn')
              void captureViewPng(app, 0, 640).then((b) => downloadBlob(b, `${id}-thumbnail.png`))
            })
        }),
        button('golden @cam', () => {
          const name = currentBookmarkName(app)
          if (!name) {
            overlay.toast('goldens need a named cam — press 1-4 first', 'warn')
            return
          }
          void captureViewPng(app)
            .then((blob) => uploadCapture(id, blob, name))
            .then((saved) => overlay.toast(`saved ${saved}`))
            .catch((err: unknown) => overlay.toast(String(err), 'error'))
        }),
      )
    }
    toolbar.append(
      button('bench', () => {
        location.hash = buildHash(['bench', id], { seed: String(seed), stand: stand.id })
      }),
      standPicker(stand),
      debugPicker(app, state.q.get('debug')),
    )
    if (!isReference) {
      const pips = pipsRow({
        value: await fetchRating(id),
        readOnly: !HAS_DEV_SINK,
        onSet: (value) => {
          void saveRating(id, value)
            .then(() => overlay.toast(value === null ? 'rating cleared' : `rated ${value}/5`))
            .catch(() => overlay.toast('rating needs the dev server', 'warn'))
        },
      })
      toolbar.appendChild(pips.el)
    }
    viewer.appendChild(toolbar)

    app.start()
    return { dispose }
  } catch (err) {
    overlay.fatal(`Failed to start "${id}"`, fatalDetail(err))
    return { dispose }
  }
}
