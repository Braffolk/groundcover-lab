import type { VramScope } from '../../gpu/resources.ts'
import {
  chainStats,
  decodeTexel,
  levelToRgba8,
  nodeTextureFor,
  readMipChain,
  statsVerdict,
  texelAt,
  textureBytesOf,
  type ChannelPick,
  type LevelStats,
  type MipChain,
  type NodeTextureResult,
} from '../../material/inspect.ts'
import { materialInstance, onMaterialInstance, type MaterialInstance } from '../../material/instance.ts'
import {
  materialReport,
  onMaterialReport,
  type MaterialNodeReport,
  type MaterialReport,
} from '../../material/report.ts'
import type { ChannelSpec } from '../../material/types.ts'
import type { TexelSemantics } from '../../gpu/texture.ts'
import { buildHash, makeDebouncedQueryWriter, type HashState } from '../../url/state.ts'
import { Overlay } from '../../ui/overlay.ts'
import { buildParamsPanel } from '../../ui/panel.ts'
import { downloadBlob, imageDataToPngBlob } from '../capture.ts'
import { LabApp } from '../loop.ts'
import { paramsToQuery } from '../params.ts'
import { findExperiment } from '../registry.ts'
import {
  button,
  copyToClipboard,
  currentPreviewObject,
  debugPicker,
  el,
  fatalDetail,
  link,
  previewObjectPicker,
  previewSummary,
  readSeed,
  stageFor,
  topbar,
  type View,
} from './shared.ts'

/**
 * #/material/<id> — the material inspector.
 *
 * Four panes over ONE material: the node DAG, an inspector for the selected
 * node's actual output, the flat channel list, and the generated WGSL — plus
 * the live preview and the validator report.
 *
 * Everything here is drawn from `MaterialReport` (the resolved graph as data)
 * and `MaterialInstance` (the same graph's live GPU handles). There is no
 * per-material code anywhere on this page and there must never be: a material
 * an agent writes tomorrow is inspectable the moment it exists, or the graph
 * was not worth declaring.
 *
 * The one thing worth knowing before changing it: a MATERIALIZED node shows
 * the texture the fragment shader actually samples, while a LIVE node has no
 * texture at all and is evaluated on demand into an offscreen target by the
 * generated `cs_materialize` entry point — the same `n_<id>_eval` the fragment
 * inlines (see src/material/inspect.ts). Nothing is re-implemented for
 * display, so what you see cannot drift from what renders.
 */

const NODE_W = 202
const NODE_H = 76
const H_GAP = 58
const V_GAP = 12
const GRAPH_PAD = 12

const CHANNEL_PICKS: ChannelPick[] = ['composite', 'r', 'g', 'b', 'a']

export async function materialView(root: HTMLElement, state: HashState): Promise<View> {
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
    const choice = stageFor(entry, state.q, seed)

    // ---- shell ------------------------------------------------------------
    const topRight: HTMLElement[] = [
      el('span', 'hint', entry.kind === 'material' ? 'material graph inspector' : `kind: ${entry.kind}`),
      link(buildHash(['run', id], state.q), 'open in runner'),
    ]
    page.appendChild(topbar(manifest.title, topRight))

    const content = el('div', 'content')
    const layout = el('div', 'material')
    const left = el('div', 'mat-left')
    const right = el('div', 'mat-right')
    layout.append(left, right)
    content.appendChild(layout)
    page.appendChild(content)

    if (entry.kind !== 'material') {
      left.appendChild(heading('not a material'))
      left.appendChild(
        note(
          `"${id}" is a ${entry.kind}, not a material — it draws a stand's plants and has no channel graph. ` +
            'The material inspector only has anything to show for `kind: material`.',
        ),
      )
      return { dispose }
    }

    // ---- live preview (the existing material stage, not a second renderer) -
    const previewBox = el('div', 'mat-preview')
    const canvas = el('canvas')
    previewBox.appendChild(canvas)
    right.appendChild(previewBox)
    const previewNote = el('div', 'hint mono', 'starting…')
    right.appendChild(previewNote)

    const paneHolder = el('div', 'mat-params')
    right.appendChild(paneHolder)

    const nodePane = el('div', 'mat-node-pane')
    right.appendChild(nodePane)

    const graphPane = el('div')
    const channelsPane = el('div')
    const wgslPane = el('div')
    const reportPane = el('div')
    left.append(graphPane, channelsPane, wgslPane, reportPane)

    // ---- start the material ----------------------------------------------
    let app: LabApp | null = null
    let startError: unknown = null
    try {
      app = await LabApp.create({
        canvas,
        seed,
        stage: choice.stage,
        experiments: [{ entry, ns: 'p', query: state.q }],
        onProgress: (fraction, n) => {
          previewNote.textContent = `${Math.round(fraction * 100)}% · ${n ?? ''}`
        },
        onError: (m) => overlay.toast(m, 'error'),
        onShaderMessage: (m) => overlay.toast(m.text, m.kind === 'error' ? 'error' : 'info'),
        onDeviceLost: (reason) => overlay.toast(`device lost (${reason})`, 'error'),
      })
    } catch (err) {
      startError = err
    }

    // A material that failed validation has still published its report — that
    // is the whole reason the report is data. Keep the page, drop the preview.
    if (!app) {
      previewBox.remove()
      previewNote.className = 'mono'
      previewNote.style.whiteSpace = 'pre-wrap'
      previewNote.textContent = `the material did not start:\n${
        startError instanceof Error ? startError.message : String(startError)
      }`
    }

    const runningApp = app
    if (runningApp) {
      cleanups.push(() => runningApp.dispose())
      const view = runningApp.views[0]!
      previewNote.textContent = previewSummary(runningApp)

      const writeQuery = makeDebouncedQueryWriter()
      const panel = buildParamsPanel({
        container: paneHolder,
        title: 'params',
        schema: manifest.params,
        values: view.params,
        onChange: (key) => {
          runningApp.setParam(view, key, view.params[key])
          writeQuery(paramsToQuery(manifest.params, view.params, 'p'))
          refreshLiveNode()
        },
      })
      cleanups.push(() => panel.dispose())

      const toolbar = el('div', 'mat-toolbar')
      toolbar.append(
        previewObjectPicker(runningApp),
        debugPicker(runningApp, state.q.get('debug')),
        button('bench', () => {
          location.hash = buildHash(['bench', id], { seed: String(seed), obj: currentPreviewObject(runningApp) })
        }),
      )
      // Below the preview, not overlaid on it: at this column width an
      // overlaid row clips its own first label instead of wrapping.
      right.insertBefore(toolbar, previewNote)
      runningApp.start()
    }

    // ---- inspector state --------------------------------------------------
    const tracker = runningApp?.tracker ?? null
    let selected: string | null = null
    let selection = 0
    let graph: GraphView = { boxes: new Map(), svg: null }

    /**
     * A plain (non-structural) param does not rebuild the material, so nothing
     * publishes and nothing re-renders — but it DOES change what a live node
     * evaluates to. Re-evaluate the selected node in that case, debounced,
     * because it is a full-resolution compute pass plus a chain readback.
     * Texture-mode nodes are left alone: only a rebuild or a variant swap can
     * change one, and both of those publish.
     */
    let liveRefresh: ReturnType<typeof setTimeout> | null = null
    const refreshLiveNode = (): void => {
      const node = materialReport(id)?.nodes.find((n) => n.id === selected)
      if (!node || node.mode !== 'live') return
      if (liveRefresh !== null) clearTimeout(liveRefresh)
      liveRefresh = setTimeout(() => {
        liveRefresh = null
        void select(selected)
      }, 350)
    }
    cleanups.push(() => {
      if (liveRefresh !== null) clearTimeout(liveRefresh)
    })

    // ---- rendering --------------------------------------------------------
    const render = (): void => {
      const report = materialReport(id)
      const inst = materialInstance(id)
      graph = renderGraph(graphPane, report, (nodeId) => void select(nodeId))
      renderChannels(channelsPane, report, inst, tracker, (nodeId) => void select(nodeId))
      renderWgsl(wgslPane, report, overlay)
      renderReport(reportPane, report, id)
      if (!report) void select(null)
      else if (selected === null || !report.nodes.some((n) => n.id === selected)) void select(defaultNode(report))
      else markSelected(selected)
    }

    const markSelected = (nodeId: string | null): void => {
      for (const [boxId, box] of graph.boxes) box.classList.toggle('selected', boxId === nodeId)
      if (!graph.svg) return
      for (const edge of graph.svg.querySelectorAll('path')) {
        const from = edge.getAttribute('data-from')
        const to = edge.getAttribute('data-to')
        edge.classList.toggle('active', nodeId !== null && (from === nodeId || to === nodeId))
      }
    }

    const select = async (nodeId: string | null): Promise<void> => {
      selected = nodeId
      markSelected(nodeId)
      const token = ++selection
      const report = materialReport(id)
      const inst = materialInstance(id)
      const node = report?.nodes.find((n) => n.id === nodeId)
      if (!node) {
        nodePane.replaceChildren(
          heading('node'),
          note(report ? 'select a node in the graph' : 'no graph — there are no nodes to inspect'),
        )
        return
      }
      nodePane.replaceChildren(heading(`node · ${node.id}`), ...nodeSummary(node, report!))
      if (!inst || !tracker) {
        nodePane.appendChild(note('no live instance — the material is not running, so it has no textures to show'))
        return
      }
      const status = el('div', 'hint mono', 'reading back…')
      nodePane.appendChild(status)
      // The scope lives exactly as long as the readback: once the mip chain is
      // on the CPU, an evaluated or on-demand-loaded texture has done its job,
      // and holding it would leave two 48 MiB map sets resident for a page
      // whose whole subject is VRAM honesty. (Destroy() lets in-flight copies
      // finish, and GPUTexture keeps reporting its own dimensions afterwards.)
      const scope = tracker.scope(`material-inspector:${node.id}`)
      try {
        const provided = await nodeTextureFor(inst, node.id, { res: scope })
        const chain = await readMipChain(inst.device, provided.texture, { exempt: (fn) => tracker.exempt(fn) })
        if (token !== selection) return
        status.remove()
        nodePane.appendChild(nodeInspector(node, provided, chain, overlay))
      } catch (err) {
        if (token !== selection) return
        status.className = 'mono'
        status.style.whiteSpace = 'pre-wrap'
        status.textContent = `cannot show this node: ${err instanceof Error ? err.message : String(err)}`
      } finally {
        scope.dispose()
      }
    }

    render()
    cleanups.push(onMaterialReport((r) => {
      if (r.id === id) render()
    }))
    cleanups.push(onMaterialInstance((_inst, changed) => {
      if (changed === id) render()
    }))

    return { dispose }
  } catch (err) {
    overlay.fatal(`Material inspector failed for "${id}"`, fatalDetail(err))
    return { dispose }
  }
}

// ---------------------------------------------------------------------------
// Small shared bits
// ---------------------------------------------------------------------------

function heading(text: string): HTMLElement {
  return el('h2', undefined, text)
}

function note(text: string): HTMLElement {
  const n = el('div', 'hint')
  n.style.cssText = 'margin-top:8px;max-width:88ch'
  n.textContent = text
  return n
}

function row(label: string, value: string): HTMLElement {
  const r = el('div', 'mat-row')
  r.append(el('span', 'k', label), el('span', 'v mono', value))
  return r
}

function bytesLabel(n: number): string {
  if (n >= 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(2)} MiB`
  if (n >= 1024) return `${(n / 1024).toFixed(1)} KiB`
  return `${n} B`
}

function semanticsLabel(s: TexelSemantics): string {
  switch (s.kind) {
    case 'color-with-coverage':
      return `color-with-coverage (weight ${s.weight})`
    case 'premultiplied-color':
      return 'premultiplied-color'
    case 'unit-vector':
      return `unit-vector (hemisphere ${s.hemisphere.join(', ')})`
    case 'scalar-field':
      return `scalar-field (holes: ${s.empty})`
    case 'mask':
      return `mask (${s.reduce})`
  }
}

/** LIVE / MATERIALIZED, plus where the texture came from. */
function modeLabel(node: MaterialNodeReport): string {
  if (node.mode === 'live') return 'LIVE'
  switch (node.source) {
    case 'image':
      return 'MATERIALIZED · png'
    case 'materialize':
      return 'MATERIALIZED · baked'
    case 'variant':
      return 'MATERIALIZED · variant slot'
    default:
      return 'MATERIALIZED'
  }
}

function specLabel(spec: ChannelSpec): string {
  return `${spec.type} · ${semanticsLabel(spec.semantics)}${spec.srgb ? ' · srgb' : ''}`
}

/** The channel(s) a node feeds, plus the view-uv stage if it feeds that. */
function outputsOf(node: MaterialNodeReport, report: MaterialReport): string[] {
  const out = report.channels.filter((c) => c.nodeId === node.id).map((c) => c.channel)
  if (report.viewUvNodes.includes(node.id)) out.push(`view-uv:${report.viewUv}`)
  return out
}

function defaultNode(report: MaterialReport): string | null {
  return report.channels[0]?.nodeId ?? report.nodes[report.nodes.length - 1]?.id ?? null
}

// ---------------------------------------------------------------------------
// Pane 1 — the graph
// ---------------------------------------------------------------------------

/** Longest-path layering: a node sits one column right of its deepest input. */
function layerOf(report: MaterialReport): Map<string, number> {
  const byId = new Map(report.nodes.map((n) => [n.id, n]))
  const depth = new Map<string, number>()
  const visit = (nodeId: string, guard: Set<string>): number => {
    const cached = depth.get(nodeId)
    if (cached !== undefined) return cached
    if (guard.has(nodeId)) return 0 // a cycle is a validator error; do not hang on it
    const node = byId.get(nodeId)
    if (!node) return 0
    guard.add(nodeId)
    let d = 0
    for (const dep of node.deps) d = Math.max(d, visit(dep, guard) + 1)
    guard.delete(nodeId)
    depth.set(nodeId, d)
    return d
  }
  for (const n of report.nodes) visit(n.id, new Set())
  return depth
}

/** The DOM the graph pane produced, so the selection highlight can find it. */
interface GraphView {
  boxes: Map<string, HTMLElement>
  svg: SVGSVGElement | null
}

function renderGraph(
  host: HTMLElement,
  report: MaterialReport | undefined,
  onSelect: (id: string) => void,
): GraphView {
  host.replaceChildren(heading('graph'))
  const boxes = new Map<string, HTMLElement>()
  if (!report) {
    host.appendChild(
      note(
        'no material report — this experiment is a material, but it is not GRAPH-authored: it builds its own ' +
          'pipeline in main.ts instead of declaring a MaterialDef, so there are no nodes, no generated WGSL and no ' +
          'validator report to show. The live preview on the right is still the real thing.',
      ),
    )
    return { boxes, svg: null }
  }
  host.appendChild(
    note(
      `${report.nodes.length} nodes · ${report.channels.length} channels · view-uv ${report.viewUv} · ` +
        'dependencies flow left to right; click a node to inspect its output.',
    ),
  )

  const depth = layerOf(report)
  const deepest = Math.max(0, ...report.nodes.map((n) => depth.get(n.id) ?? 0))
  // Dense, so a hole can never sneak into the width/height reduction below.
  const columns: MaterialNodeReport[][] = Array.from({ length: deepest + 1 }, () => [])
  for (const node of report.nodes) columns[depth.get(node.id) ?? 0]!.push(node)

  const width = Math.max(1, columns.length) * (NODE_W + H_GAP) - H_GAP + GRAPH_PAD * 2
  const height = Math.max(...columns.map((c) => c.length), 1) * (NODE_H + V_GAP) - V_GAP + GRAPH_PAD * 2

  const scroll = el('div', 'mat-graph')
  const stage = el('div', 'mat-graph-stage')
  stage.style.width = `${width}px`
  stage.style.height = `${height}px`

  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
  svg.setAttribute('width', String(width))
  svg.setAttribute('height', String(height))
  svg.classList.add('mat-edges')
  stage.appendChild(svg)

  const centreOf = new Map<string, { x: number; y: number; left: number; right: number }>()
  columns.forEach((column, col) => {
    column.forEach((node, i) => {
      const x = GRAPH_PAD + col * (NODE_W + H_GAP)
      const y = GRAPH_PAD + i * (NODE_H + V_GAP)
      centreOf.set(node.id, { x: x + NODE_W / 2, y: y + NODE_H / 2, left: x, right: x + NODE_W })
    })
  })

  // Edges first, so a node box always sits above its wires.
  for (const node of report.nodes) {
    const to = centreOf.get(node.id)
    if (!to) continue
    for (const dep of node.deps) {
      const from = centreOf.get(dep)
      if (!from) continue
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path')
      const dx = Math.max(18, (to.left - from.right) * 0.6)
      path.setAttribute('d', `M ${from.right} ${from.y} C ${from.right + dx} ${from.y}, ${to.left - dx} ${to.y}, ${to.left} ${to.y}`)
      path.setAttribute('data-from', dep)
      path.setAttribute('data-to', node.id)
      svg.appendChild(path)
    }
  }

  columns.forEach((column, col) => {
    column.forEach((node, i) => {
      const box = el('div', 'mat-node')
      if (node.ownedBy !== undefined) box.classList.add('owned')
      box.style.left = `${GRAPH_PAD + col * (NODE_W + H_GAP)}px`
      box.style.top = `${GRAPH_PAD + i * (NODE_H + V_GAP)}px`
      box.append(
        el('div', 'id mono', node.id),
        el('div', 'meta mono', `${node.kind} · ${node.spec.type} · ${node.spec.semantics.kind}`),
      )
      const tags = el('div', 'tags')
      tags.appendChild(el('span', 'tag mode', modeLabel(node)))
      for (const out of outputsOf(node, report)) tags.appendChild(el('span', 'tag', `→ ${out}`))
      if (node.ownedBy !== undefined) tags.appendChild(el('span', 'tag', `variant of ${node.ownedBy}`))
      box.appendChild(tags)
      box.title = [node.doc, node.spec.doc].filter(Boolean).join(' — ')
      box.addEventListener('click', () => onSelect(node.id))
      stage.appendChild(box)
      boxes.set(node.id, box)
    })
  })

  scroll.appendChild(stage)
  host.appendChild(scroll)
  return { boxes, svg }
}

// ---------------------------------------------------------------------------
// Pane 2 — the selected node
// ---------------------------------------------------------------------------

function nodeSummary(node: MaterialNodeReport, report: MaterialReport): HTMLElement[] {
  const out: HTMLElement[] = [
    row('kind', node.kind),
    row('resolved', modeLabel(node)),
    row('spec', specLabel(node.spec)),
  ]
  if (node.deps.length > 0) out.push(row('inputs', node.deps.join(', ')))
  const feeds = outputsOf(node, report)
  if (feeds.length > 0) out.push(row('feeds', feeds.join(', ')))
  const mat = report.materializations.find((m) => m.nodeId === node.id)
  if (mat) {
    out.push(row('bake', `${mat.source} · ${mat.ms}ms · key ${mat.key}`))
    out.push(row('cache key', `subtree [${mat.subtree.join(', ')}] · params ${JSON.stringify(mat.params)}`))
  }
  const doc = [node.doc, node.spec.doc].filter((d): d is string => Boolean(d)).join(' — ')
  if (doc) out.push(note(doc))
  return out
}

function nodeInspector(
  node: MaterialNodeReport,
  provided: NodeTextureResult,
  chain: MipChain,
  overlay: Overlay,
): HTMLElement {
  const wrap = el('div', 'mat-inspect')
  wrap.appendChild(note(provided.note))

  const tex = provided.texture
  const stats = chainStats(chain, node.spec.type)
  let level = 0
  let pick: ChannelPick = 'composite'
  let decoded = true

  const canvas = el('canvas', 'mat-canvas')
  const canvasBox = el('div', 'mat-canvas-box')
  canvasBox.appendChild(canvas)
  const ctx2d = canvas.getContext('2d')!

  const controls = el('div', 'mat-controls')
  const pickButtons = CHANNEL_PICKS.map((p) => {
    const label = p === 'composite' ? 'composite' : p.toUpperCase()
    const b = button(label, () => {
      pick = p
      draw()
    })
    b.dataset['pick'] = p
    if (p !== 'composite' && CHANNEL_PICKS.indexOf(p) > chain.components) b.disabled = true
    return b
  })
  controls.append(el('span', 'hint', 'channel'), ...pickButtons)

  const decodeBtn = button('decoded', () => {
    decoded = !decoded
    draw()
  })
  controls.appendChild(decodeBtn)

  const mipRow = el('div', 'mat-controls')
  const slider = el('input')
  slider.type = 'range'
  slider.min = '0'
  slider.max = String(chain.levels.length - 1)
  slider.value = '0'
  slider.step = '1'
  slider.style.flex = '1'
  slider.disabled = chain.levels.length < 2
  const mipLabel = el('span', 'mono hint')
  slider.addEventListener('input', () => {
    level = Number(slider.value)
    draw()
  })
  mipRow.append(el('span', 'hint', 'mip'), slider, mipLabel)

  const numbers = el('div', 'mat-numbers')
  const probe = el('div', 'mat-probe mono', 'hover the image to probe a texel')

  const draw = (): void => {
    for (const b of pickButtons) b.classList.toggle('primary', b.dataset['pick'] === pick)
    decodeBtn.classList.toggle('primary', decoded)
    decodeBtn.textContent = decoded ? 'decoded' : 'stored'
    const l = chain.levels[level]!
    const image = levelToRgba8(chain, level, pick, node.spec.type, decoded ? node.spec.semantics : null)
    canvas.width = image.width
    canvas.height = image.height
    ctx2d.putImageData(new ImageData(image.data, image.width, image.height), 0, 0)
    canvas.classList.toggle('pixelated', image.width <= 256)
    mipLabel.textContent = `L${l.level} · ${l.width}×${l.height}`
    const s = stats[level]!
    numbers.replaceChildren(
      row('dimensions', `${tex.width}×${tex.height} · ${chain.levels.length} mip levels`),
      row('format', `${tex.format} · ${chain.components} channel${chain.components === 1 ? '' : 's'}`),
      row('bytes', `${bytesLabel(textureBytesOf(tex))} with mips · ${bytesLabel(tex.width * tex.height * (chain.levels[0]!.bytesPerTexel))} at level 0`),
      row('this level', `${l.width}×${l.height} · luma ${s.luma.toFixed(4)} · min ${s.min.toFixed(4)} · max ${s.max.toFixed(4)}`),
    )
  }

  canvas.addEventListener('mousemove', (ev) => {
    const rect = canvas.getBoundingClientRect()
    const x = Math.floor(((ev.clientX - rect.left) / rect.width) * canvas.width)
    const y = Math.floor(((ev.clientY - rect.top) / rect.height) * canvas.height)
    const stored = texelAt(chain, level, x, y)
    const dec = decodeTexel(node.spec.semantics, stored)
    const fmt = (v: readonly number[]): string => v.map((n) => n.toFixed(4)).join(', ')
    const l = chain.levels[level]!
    probe.textContent =
      `texel ${x},${y} of ${l.width}×${l.height} · uv ${((x + 0.5) / l.width).toFixed(4)},${((y + 0.5) / l.height).toFixed(4)}\n` +
      `stored  ${fmt(stored)}\ndecoded ${fmt(dec)}`
  })
  canvas.addEventListener('mouseleave', () => {
    probe.textContent = 'hover the image to probe a texel'
  })

  const save = button('save png', () => {
    const image = levelToRgba8(chain, level, pick, node.spec.type, decoded ? node.spec.semantics : null)
    void imageDataToPngBlob(new ImageData(image.data, image.width, image.height))
      .then((blob) => downloadBlob(blob, `${node.id}-L${level}-${pick}.png`))
      .catch((err: unknown) => overlay.toast(String(err), 'error'))
  })
  controls.appendChild(save)

  wrap.append(canvasBox, controls, mipRow, numbers, probe, levelTable(stats, node.spec.semantics))
  draw()
  return wrap
}

/**
 * Per-level mean luma — the coverage/premultiplied trap, as a table. The one
 * question it answers is "is the stored luma flat across levels", which is
 * exactly the test CLAUDE.md prescribes by hand.
 */
function levelTable(stats: readonly LevelStats[], semantics: TexelSemantics): HTMLElement {
  const wrap = el('div', 'mat-levels')
  wrap.appendChild(
    el(
      'div',
      'hint',
      'per-level stored luma — flat or falling is the mip-convention test, not a cosmetic detail. ' +
        'luma is rgb-weighted for a vec3/vec4 node and the r channel for a scalar, so a node reads the same ' +
        'whether it is stored as r8unorm or materialized into rgba8unorm.',
    ),
  )
  const table = el('table', 'mono')
  const head = el('tr')
  for (const h of ['level', 'size', 'luma', 'Σ(luma·a)/Σa', 'mean a']) head.appendChild(el('th', undefined, h))
  table.appendChild(head)
  const base = stats[0]
  for (const s of stats) {
    const tr = el('tr')
    const drift = base && Math.abs(base.coveredLuma) > 1e-5 ? (s.coveredLuma - base.coveredLuma) / base.coveredLuma : 0
    tr.append(
      el('td', undefined, `L${s.level}`),
      el('td', undefined, `${s.width}×${s.height}`),
      el('td', undefined, s.luma.toFixed(4)),
      el('td', undefined, `${s.coveredLuma.toFixed(4)} (${drift >= 0 ? '+' : ''}${(drift * 100).toFixed(1)}%)`),
      el('td', undefined, s.meanAlpha.toFixed(4)),
    )
    table.appendChild(tr)
  }
  wrap.appendChild(table)
  wrap.appendChild(note(statsVerdict(semantics, stats)))
  return wrap
}

// ---------------------------------------------------------------------------
// Pane 3 — the flat channel list
// ---------------------------------------------------------------------------

function renderChannels(
  host: HTMLElement,
  report: MaterialReport | undefined,
  inst: MaterialInstance | undefined,
  tracker: { scope(owner: string): VramScope; exempt<T>(fn: () => T): T } | null,
  onSelect: (id: string) => void,
): void {
  host.replaceChildren()
  if (!report) return
  host.appendChild(heading('channels'))
  host.appendChild(
    note(
      'every channel this material DECLARES it produces — the generated `Channels` struct has exactly these ' +
        'fields, in this order, and the surface stage receives it.',
    ),
  )
  const grid = el('div', 'mat-channels')
  host.appendChild(grid)

  for (const channel of report.channels) {
    const node = report.nodes.find((n) => n.id === channel.nodeId)
    const card = el('div', 'mat-chan')
    const thumb = el('canvas', 'thumb')
    thumb.width = 96
    thumb.height = 96
    card.appendChild(thumb)
    const body = el('div', 'body')
    body.append(
      el('div', 'name mono', channel.channel),
      el('div', 'hint mono', node ? specLabel(node.spec) : channel.type),
      el('div', 'hint mono', `from "${channel.nodeId}" · ${node ? modeLabel(node) : channel.mode}`),
    )
    const size = el('div', 'hint mono', '…')
    body.appendChild(size)
    card.appendChild(body)
    card.addEventListener('click', () => onSelect(channel.nodeId))
    grid.appendChild(card)

    if (!inst || !tracker || !node) {
      size.textContent = 'no live instance'
      continue
    }
    // Thumbnails evaluate a LIVE node at 256² rather than at map resolution:
    // the generated entry point builds its EvalCtx from the DESTINATION texel,
    // so any size is a legal, honest evaluation of the same function.
    const scope = tracker.scope(`material-inspector/thumb:${channel.channel}`)
    void (async () => {
      try {
        const provided = await nodeTextureFor(inst, channel.nodeId, { res: scope, size: [256, 256] })
        // ONE level, the one closest to 128²: reading a 2048² chain for a 72px
        // thumbnail is 21 MB of readback per channel, three times over.
        const level = Math.max(0, provided.texture.mipLevelCount - 1 - Math.round(Math.log2(128)))
        const chain = await readMipChain(inst.device, provided.texture, {
          exempt: (fn) => tracker.exempt(fn),
          levels: [level],
        })
        const image = levelToRgba8(chain, 0, 'composite', node.spec.type, node.spec.semantics)
        thumb.width = image.width
        thumb.height = image.height
        thumb.getContext('2d')!.putImageData(new ImageData(image.data, image.width, image.height), 0, 0)
        size.textContent =
          provided.origin === 'evaluated'
            ? `live — evaluated at ${provided.texture.width}×${provided.texture.height} for this thumbnail`
            : `${provided.texture.width}×${provided.texture.height} · ${provided.texture.format} · ${bytesLabel(textureBytesOf(provided.texture))}`
      } catch (err) {
        size.textContent = err instanceof Error ? err.message : String(err)
      } finally {
        scope.dispose()
      }
    })()
  }
}

// ---------------------------------------------------------------------------
// Pane 4 — the generated WGSL
// ---------------------------------------------------------------------------

function renderWgsl(host: HTMLElement, report: MaterialReport | undefined, overlay: Overlay): void {
  host.replaceChildren()
  if (!report || !report.code) return
  const details = el('details', 'group')
  details.open = false
  const summary = el('summary')
  summary.append(
    el('span', 'label', 'generated WGSL'),
    el('span', 'count', `${report.code.split('\n').length} lines · ${report.moduleId}`),
  )
  details.appendChild(summary)

  const bar = el('div', 'mat-controls')
  bar.append(
    button('copy', () => {
      void copyToClipboard(report.code)
        .then(() => overlay.toast('generated WGSL copied'))
        .catch(() => overlay.toast('clipboard unavailable', 'warn'))
    }),
    el(
      'span',
      'hint',
      'what codegen actually produced: the node functions, the Channels struct, the view-uv stage, the ' +
        "material's own surface/shade sources and the generated entry points.",
    ),
  )
  details.appendChild(bar)

  const pre = el('pre', 'mono mat-wgsl')
  pre.textContent = report.code
  details.appendChild(pre)
  host.appendChild(details)
}

// ---------------------------------------------------------------------------
// The validator report
// ---------------------------------------------------------------------------

function renderReport(host: HTMLElement, report: MaterialReport | undefined, id: string): void {
  host.replaceChildren()
  if (!report) return
  const errors = report.issues.filter((i) => i.level === 'error').length
  const warnings = report.issues.length - errors
  const details = el('details', 'group')
  details.open = errors > 0
  const summary = el('summary')
  summary.append(
    el('span', 'label', 'validator report'),
    el('span', 'count', `${errors} errors · ${warnings} warnings · ${report.materializations.length} materialized`),
  )
  details.appendChild(summary)

  const body = el('div', 'mat-report')
  body.append(
    row('module', report.moduleId || '(not compiled)'),
    row('view-uv', `${report.viewUv}${report.viewUvNodes.length ? ` (reads ${report.viewUvNodes.join(', ')})` : ''}`),
    row('resident textures', bytesLabel(report.textureBytes)),
    row('status', report.ok ? 'ok' : 'FAILED — the material did not start'),
    note(`the same object is on globalThis.__materialReports['${id}'], for headless probing.`),
  )

  if (report.issues.length > 0) {
    const table = el('table', 'mono')
    const head = el('tr')
    for (const h of ['level', 'code', 'node', 'message']) head.appendChild(el('th', undefined, h))
    table.appendChild(head)
    for (const issue of report.issues) {
      const tr = el('tr')
      if (issue.level === 'error') tr.className = 'err'
      tr.append(
        el('td', undefined, issue.level),
        el('td', undefined, issue.code),
        el('td', undefined, issue.nodeId ?? issue.channel ?? ''),
        el('td', 'msg', issue.message),
      )
      table.appendChild(tr)
    }
    body.appendChild(table)
  } else {
    body.appendChild(note('no issues.'))
  }

  if (report.materializations.length > 0) {
    const table = el('table', 'mono')
    const head = el('tr')
    for (const h of ['node', 'source', 'ms', 'size', 'subtree', 'params']) head.appendChild(el('th', undefined, h))
    table.appendChild(head)
    for (const m of report.materializations) {
      const tr = el('tr')
      tr.append(
        el('td', undefined, m.nodeId),
        el('td', undefined, m.source),
        el('td', undefined, String(m.ms)),
        el('td', undefined, `${m.size[0]}×${m.size[1]}`),
        el('td', undefined, m.subtree.join(', ')),
        el('td', 'msg', JSON.stringify(m.params)),
      )
      table.appendChild(tr)
    }
    body.appendChild(el('div', 'hint', 'materialized nodes — the cache key is the node\'s TRANSITIVE subtree, not the whole graph'))
    body.appendChild(table)
  }

  details.appendChild(body)
  host.appendChild(details)
}
