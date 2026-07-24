import { BENCH_SCHEMA_VERSION, type BenchResult } from '../../bench/schema.ts'
import { downloadResult, postBenchResult } from '../../bench/store.ts'
import { summarize } from '../../gpu/timing.ts'
import { buildHash, type HashState } from '../../url/state.ts'
import { Overlay } from '../../ui/overlay.ts'
import { LabApp } from '../loop.ts'
import { paramsHash } from '../params.ts'
import { findExperiment, HARNESS_API } from '../registry.ts'
import { button, el, link, readSeed, readStand, topbar, type View } from './shared.ts'

const CANVAS: [number, number] = [1600, 900]

/**
 * #/bench/<id>?spline=orbit-low&warmup=120&frames=600 — the ONLY
 * authoritative timing source. Fixed canvas size, fixed timestep, spline-
 * driven camera: every run of every experiment sees the same pixels.
 * The result auto-saves to results/ via the dev-server sink.
 */
export async function benchView(root: HTMLElement, state: HashState): Promise<View> {
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
    const seed = readSeed(state.q)
    const stand = readStand(state.q)
    const splineName = state.q.get('spline') ?? 'orbit-low'
    const warmup = Math.max(1, Number(state.q.get('warmup')) || 120)
    const frames = Math.max(1, Number(state.q.get('frames')) || 600)

    page.appendChild(
      topbar(`bench: ${entry.manifest!.title}`, [
        el(
          'span',
          'hint',
          `${stand.id} · ${splineName} · seed ${seed} · ${warmup}+${frames} frames · ${CANVAS[0]}×${CANVAS[1]}`,
        ),
        link(buildHash(['results']), 'results table'),
      ]),
    )
    const content = el('div', 'content')
    const viewer = el('div', 'viewer')
    const canvas = el('canvas')
    canvas.style.objectFit = 'contain'
    canvas.style.background = '#000'
    viewer.appendChild(canvas)
    content.appendChild(viewer)
    page.appendChild(content)

    const status = el('div', 'hud', 'starting…')
    viewer.appendChild(status)

    const app = await LabApp.create({
      canvas,
      seed,
      stand,
      experiments: [{ entry, ns: 'p', query: state.q }],
      onError: (m) => overlay.toast(m, 'error'),
      onShaderMessage: (m) => overlay.toast(m.text, m.kind === 'error' ? 'error' : 'info'),
      onDeviceLost: (r) => overlay.toast(`device lost: ${r}`, 'error'),
      fixedDt: 1 / 60,
      fixedSize: { width: CANVAS[0], height: CANVAS[1] },
      timerRingSize: 8,
    })
    cleanups.push(() => app.dispose())
    app.camera.inputEnabled = false

    const spline = app.scene.splines[splineName]
    if (!spline) {
      throw new Error(`unknown spline "${splineName}" — known: ${Object.keys(app.scene.splines).join(', ')}`)
    }
    const total = warmup + frames
    const duration = total / 60
    app.cameraDriver = (time) => spline.sample(time / duration)

    if (!app.gpu.hasTimestamps) {
      overlay.toast('timestamp-query unavailable — CPU timing only, result marked accordingly', 'warn', 10000)
    }

    // Collect measured-window samples.
    const cpuSamples: number[] = []
    const jsSamples: number[] = []
    const gpuSamples = new Map<string, number[]>()
    let cancelled = false
    let finishing = false

    const finish = async (): Promise<void> => {
      finishing = true
      app.stop()
      status.textContent = 'draining GPU readbacks…'
      await app.gpu.device.queue.onSubmittedWorkDone()
      // A couple of extra polls for the timing ring to flush.
      for (let i = 0; i < 10; i++) {
        await new Promise((r) => setTimeout(r, 50))
        app.timer.afterSubmit()
        for (const t of app.timer.poll()) {
          if (t.frameIndex >= warmup && t.frameIndex < total) {
            for (const p of t.passes) {
              let arr = gpuSamples.get(p.label)
              if (!arr) gpuSamples.set(p.label, (arr = []))
              arr.push(p.ms)
            }
          }
        }
      }

      const view = app.views[0]!
      const vram = app.vramReport()
      const gpu = app.gpu.hasTimestamps
        ? Object.fromEntries([...gpuSamples.entries()].map(([label, arr]) => [label, summarize(arr)]))
        : null
      const result: BenchResult = {
        schemaVersion: BENCH_SCHEMA_VERSION,
        experiment: { id, params: { ...view.params }, spline: splineName, seed, stand: stand.id },
        meta: {
          date: new Date().toISOString(),
          paramsHash: paramsHash(view.params),
          adapterSlug: app.gpu.info.slug,
          adapter: {
            vendor: app.gpu.info.vendor,
            architecture: app.gpu.info.architecture,
            description: app.gpu.info.description,
          },
          ua: navigator.userAgent,
          canvas: CANVAS,
          timestampQuery: app.gpu.hasTimestamps,
          harnessApi: HARNESS_API,
          warmup,
          frames,
        },
        cpu: { frameMs: summarize(cpuSamples), jsMs: summarize(jsSamples) },
        gpu,
        vram: { totalBytes: vram.totalBytes, bySpecies: vram.bySpecies },
        samples: { cpuFrameMs: cpuSamples, gpu: Object.fromEntries(gpuSamples) },
      }

      const summaryRows: string[] = [
        `cpu frame p50 ${result.cpu.frameMs.p50.toFixed(2)}ms · p95 ${result.cpu.frameMs.p95.toFixed(2)}ms`,
      ]
      if (gpu) {
        for (const [label, s] of Object.entries(gpu)) {
          summaryRows.push(`gpu ${label}: p50 ${s.p50.toFixed(3)}ms · p95 ${s.p95.toFixed(3)}ms · p99 ${s.p99.toFixed(3)}ms`)
        }
      }
      status.textContent = summaryRows.join('\n')

      try {
        const saved = await postBenchResult(result)
        overlay.toast(`saved ${saved}`)
      } catch {
        overlay.toast('dev server sink unavailable — downloading result instead', 'warn')
        downloadResult(result)
      }

      const toolbar = el('div', 'toolbar')
      toolbar.append(
        button('run again', () => location.reload()),
        button('download JSON', () => {
          downloadResult(result)
        }),
        button('results table →', () => {
          location.hash = buildHash(['results'])
        }),
      )
      viewer.appendChild(toolbar)
    }

    app.onAfterFrame = (timings, cpuFrameMs, cpuJsMs) => {
      if (finishing || cancelled) return
      const i = app.frameIndex // frame just rendered
      if (i >= warmup && i < total) {
        cpuSamples.push(cpuFrameMs)
        jsSamples.push(cpuJsMs)
      }
      for (const t of timings) {
        if (t.frameIndex >= warmup && t.frameIndex < total) {
          for (const p of t.passes) {
            let arr = gpuSamples.get(p.label)
            if (!arr) gpuSamples.set(p.label, (arr = []))
            arr.push(p.ms)
          }
        }
      }
      status.textContent =
        i < warmup ? `warmup ${i}/${warmup}` : `measuring ${Math.min(i - warmup, frames)}/${frames}`
      if (i >= total) void finish()
    }

    const cancelBar = el('div', 'toolbar')
    cancelBar.appendChild(
      button('cancel', () => {
        cancelled = true
        location.hash = buildHash(['run', id], { seed: String(seed) })
      }),
    )
    viewer.appendChild(cancelBar)

    app.start()
    return { dispose }
  } catch (err) {
    overlay.fatal(`Bench failed for "${id}"`, err instanceof Error ? (err.stack ?? err.message) : String(err))
    return { dispose }
  }
}
