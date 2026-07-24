import { updateQuery, makeDebouncedQueryWriter, navigate, type HashState } from '../../url/state.ts'
import { Overlay } from '../../ui/overlay.ts'
import { Hud } from '../../ui/hud.ts'
import { buildParamsPanel } from '../../ui/panel.ts'
import { CompareCompositor, type CompareMode } from '../compare.ts'
import { LabApp, type ViewSlot } from '../loop.ts'
import { paramsToQuery } from '../params.ts'
import { findExperiment } from '../registry.ts'
import {
  button,
  copyToClipboard,
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

/**
 * #/ab/<idA>/<idB> — A/B compare. ONE camera, ONE terrain/seed, base pass
 * encoded once and copied, so the two experiments differ in nothing but
 * their own passes. Modes: wipe (drag the divider), flicker (Space), diff
 * (heatmap + numeric readout).
 */
export async function abView(root: HTMLElement, state: HashState): Promise<View> {
  const [idA, idB] = [state.path[1] ?? '', state.path[2] ?? '']
  const page = el('div', 'page')
  root.appendChild(page)
  const overlay = new Overlay(page)
  const cleanups: (() => void)[] = [() => overlay.dispose(), () => page.remove()]
  const dispose = (): void => {
    for (const fn of [...cleanups].reverse()) fn()
  }

  try {
    const [entryA, entryB] = await Promise.all([findExperiment(idA), findExperiment(idB)])
    const seed = readSeed(state.q)
    const stand = readStand(state.q)

    // Surface renderers that do not cover part of the stand.
    const warnings: string[] = []
    for (const [side, entry] of [['A', entryA], ['B', entryB]] as const) {
      if (entry.manifest!.status === 'reference') {
        warnings.push(`${side} is a reference — it ignores the stand`)
        continue
      }
      const missing = stand.species.filter((e) => !entry.manifest!.species.includes(e.species))
      if (missing.length > 0) warnings.push(`${side} does not render: ${missing.map((e) => e.species).join(', ')}`)
    }

    page.appendChild(
      topbar(`A/B — timings contended, bench solo for truth`, [
        el('span', 'hint', `${stand.id} · seed ${seed}`),
        ...warnings.map((w) => el('span', 'hint', w)),
      ]),
    )
    const content = el('div', 'content')
    const viewer = el('div', 'viewer')
    const canvas = el('canvas')
    viewer.appendChild(canvas)
    content.appendChild(viewer)
    page.appendChild(content)

    const app = await LabApp.create({
      canvas,
      seed,
      stand,
      experiments: [
        { entry: entryA, ns: 'a', query: state.q },
        { entry: entryB, ns: 'b', query: state.q },
      ],
      onError: (m) => overlay.toast(m, 'error'),
      onShaderMessage: (m) =>
        overlay.toast(m.text, m.kind === 'error' ? 'error' : 'info', m.kind === 'error' ? 10000 : 2500),
      onDeviceLost: (reason) => {
        overlay.toast(`device lost (${reason}) — reloading`, 'error')
        setTimeout(() => location.reload(), 1200)
      },
    })
    cleanups.push(() => app.dispose())

    const compositor = app.tracker.exempt(() => new CompareCompositor(app.gpu.device, app.shaders, app.gpu.format))
    app.setCompositor(compositor)
    const initialMode = state.q.get('mode')
    if (initialMode === 'flicker' || initialMode === 'diff') compositor.mode = initialMode

    const { pose } = resolveCam(state.q, app.scene.bookmarks)
    if (pose) app.camera.setPose(pose)
    cleanups.push(setupCameraSync(app, state.q.get('cam'), overlay))

    // Labels over the wipe halves.
    const labelA = el('div', 'wipe-label', `A · ${entryA.manifest!.title}`)
    labelA.style.left = '12px'
    const labelB = el('div', 'wipe-label', `B · ${entryB.manifest!.title}`)
    labelB.style.right = '12px'
    viewer.append(labelA, labelB)

    // Wipe divider drag (camera drag stays on the canvas itself).
    const divider = el('div')
    Object.assign(divider.style, {
      position: 'absolute',
      top: '0',
      bottom: '0',
      width: '14px',
      cursor: 'ew-resize',
      transform: 'translateX(-7px)',
      zIndex: '5',
    } satisfies Partial<CSSStyleDeclaration>)
    viewer.appendChild(divider)
    const placeDivider = (): void => {
      divider.style.left = `${compositor.split * 100}%`
      divider.style.display = compositor.mode === 'wipe' ? '' : 'none'
    }
    placeDivider()
    divider.addEventListener('pointerdown', (down) => {
      divider.setPointerCapture(down.pointerId)
      const onMove = (ev: PointerEvent): void => {
        const rect = viewer.getBoundingClientRect()
        compositor.split = Math.min(0.98, Math.max(0.02, (ev.clientX - rect.left) / rect.width))
        placeDivider()
      }
      const onUp = (): void => {
        divider.removeEventListener('pointermove', onMove)
        divider.removeEventListener('pointerup', onUp)
      }
      divider.addEventListener('pointermove', onMove)
      divider.addEventListener('pointerup', onUp)
    })

    const setMode = (mode: CompareMode): void => {
      compositor.mode = mode
      updateQuery({ mode: mode === 'wipe' ? null : mode })
      placeDivider()
      for (const b of modeButtons) b.classList.toggle('primary', b.dataset['mode'] === mode)
      labelA.style.display = mode === 'diff' ? 'none' : ''
      labelB.style.display = mode === 'diff' ? 'none' : ''
    }
    const modeButtons = (['wipe', 'flicker', 'diff'] as const).map((mode) => {
      const b = button(mode, () => setMode(mode))
      b.dataset['mode'] = mode
      return b
    })

    const onSpace = (ev: KeyboardEvent): void => {
      if (ev.code !== 'Space' || compositor.mode !== 'flicker') return
      if (ev.target instanceof HTMLInputElement || ev.target instanceof HTMLTextAreaElement) return
      ev.preventDefault()
      compositor.autoFlicker = false
      compositor.side = 1 - compositor.side
    }
    window.addEventListener('keydown', onSpace)
    cleanups.push(() => window.removeEventListener('keydown', onSpace))

    const hud = new Hud({
      container: viewer,
      stats: app.stats,
      vram: () => app.vramReport(),
      gpuTimingAvailable: app.gpu.hasTimestamps,
      contended: true,
      extraLines: () => {
        const lines = [standSummary(stand)]
        if (compositor.mode === 'flicker') lines.push(`showing ${compositor.side === 0 ? 'A' : 'B'} — Space to swap`)
        if (compositor.mode === 'diff' && compositor.lastDiff) {
          const d = compositor.lastDiff
          lines.push(`diff mean ${(d.mean * 255).toFixed(2)}/255 · ${(d.overFrac * 100).toFixed(2)}% px differ`)
        }
        return lines
      },
    })
    cleanups.push(() => hud.dispose())

    // Two param panels, stacked.
    const paneHolder = el('div', 'pane-holder')
    viewer.appendChild(paneHolder)
    const writeQuery = makeDebouncedQueryWriter()
    const addPanel = (view: ViewSlot, label: 'A' | 'B', ns: 'a' | 'b'): void => {
      const manifest = view.entry.manifest!
      const panel = buildParamsPanel({
        container: paneHolder,
        title: `${label} · ${manifest.title}`,
        schema: manifest.params,
        values: view.params,
        onChange: (key) => {
          app.setParam(view, key, view.params[key])
          writeQuery(paramsToQuery(manifest.params, view.params, ns))
        },
      })
      cleanups.push(() => panel.dispose())
    }
    addPanel(app.views[0]!, 'A', 'a')
    addPanel(app.views[1]!, 'B', 'b')

    const toolbar = el('div', 'toolbar')
    toolbar.append(
      ...modeButtons,
      button('swap A/B', () => {
        const q = Object.fromEntries(state.q)
        navigate(['ab', idB, idA], q)
      }),
      button('copy link', () => {
        void copyToClipboard(location.href).then(() => overlay.toast('link copied'))
      }),
      standPicker(stand),
      debugPicker(app, state.q.get('debug')),
    )
    viewer.appendChild(toolbar)
    setMode(compositor.mode)

    app.start()
    return { dispose }
  } catch (err) {
    overlay.fatal(`Failed to start A/B ${idA} vs ${idB}`, fatalDetail(err))
    return { dispose }
  }
}
