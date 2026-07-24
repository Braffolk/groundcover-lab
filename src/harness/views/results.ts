import { gpuTotalP50, type BenchResult } from '../../bench/schema.ts'
import { fetchResult, listResultNames } from '../../bench/store.ts'
import { el, topbar, type View } from './shared.ts'

interface Row {
  name: string
  r: BenchResult
}

type SortKey = 'name' | 'date' | 'cpu50' | 'cpu95' | 'gpu50' | 'vram'

const val = (row: Row, key: SortKey): number | string => {
  switch (key) {
    case 'name':
      return row.r.experiment.id
    case 'date':
      return row.r.meta.date
    case 'cpu50':
      return row.r.cpu.frameMs.p50
    case 'cpu95':
      return row.r.cpu.frameMs.p95
    case 'gpu50':
      return gpuTotalP50(row.r) ?? Number.POSITIVE_INFINITY
    case 'vram':
      return Math.max(0, ...Object.values(row.r.vram.bySpecies))
  }
}

/** #/results — every bench run in results/, sortable, with baseline deltas. */
export async function resultsView(root: HTMLElement): Promise<View> {
  const page = el('div', 'page')
  root.appendChild(page)
  page.appendChild(topbar('bench results', [el('span', 'hint', 'click a row = baseline · drop JSON files to add')]))
  const content = el('div', 'content')
  const container = el('div', 'results')
  content.appendChild(container)
  page.appendChild(content)

  const rows: Row[] = []
  let baseline: string | null = null
  let sortKey: SortKey = 'date'
  let sortDir = -1
  let comparableOnly = false

  try {
    const names = await listResultNames()
    const loaded = await Promise.allSettled(names.map(async (name) => ({ name, r: await fetchResult(name) })))
    for (const l of loaded) if (l.status === 'fulfilled') rows.push(l.value)
  } catch {
    container.appendChild(el('div', 'hint', 'dev server sink unavailable — drop result JSON files here'))
  }

  // Drag-drop fallback (works on static builds too).
  container.addEventListener('dragover', (ev) => ev.preventDefault())
  container.addEventListener('drop', (ev) => {
    ev.preventDefault()
    for (const file of ev.dataTransfer?.files ?? []) {
      void file.text().then((text) => {
        rows.push({ name: file.name, r: JSON.parse(text) as BenchResult })
        render()
      })
    }
  })

  const table = el('table')
  container.appendChild(table)

  const render = (): void => {
    const base = rows.find((row) => row.name === baseline)
    const visible = rows.filter((row) => {
      if (!comparableOnly || !base || row === base) return true
      return (
        row.r.experiment.spline === base.r.experiment.spline &&
        row.r.experiment.seed === base.r.experiment.seed &&
        row.r.experiment.coverage?.radius === base.r.experiment.coverage?.radius &&
        row.r.experiment.coverage?.densityScale === base.r.experiment.coverage?.densityScale &&
        row.r.meta.canvas.join() === base.r.meta.canvas.join() &&
        row.r.meta.timestampQuery === base.r.meta.timestampQuery
      )
    })
    visible.sort((a, b) => {
      const [va, vb] = [val(a, sortKey), val(b, sortKey)]
      return (va < vb ? -1 : va > vb ? 1 : 0) * sortDir
    })

    const fmt = (v: number): string => v.toFixed(2)
    const delta = (mine: number | null, theirs: number | null): string => {
      if (mine === null || theirs === null || theirs === 0 || mine === theirs) return ''
      const pct = ((mine - theirs) / theirs) * 100
      const cls = pct < 0 ? 'better' : 'worse'
      return ` <span class="delta ${cls}">${pct > 0 ? '+' : ''}${pct.toFixed(0)}%</span>`
    }
    const baseGpu = base ? gpuTotalP50(base.r) : null
    const baseCpu = base?.r.cpu.frameMs.p50 ?? null

    const heads: [SortKey | null, string][] = [
      ['name', 'experiment'],
      [null, 'adapter'],
      [null, 'spline'],
      ['date', 'date'],
      ['cpu50', 'cpu p50'],
      ['cpu95', 'cpu p95'],
      ['gpu50', 'gpu Σp50'],
      ['vram', 'vram max/species'],
    ]
    const mb = (b: number): string => `${(b / (1024 * 1024)).toFixed(1)}MB`

    const parts: string[] = ['<thead><tr>']
    for (const [key, label] of heads) {
      const arrow = key === sortKey ? (sortDir > 0 ? ' ↑' : ' ↓') : ''
      parts.push(`<th data-key="${key ?? ''}">${label}${arrow}</th>`)
    }
    parts.push('</tr></thead><tbody>')
    for (const row of visible) {
      const g = gpuTotalP50(row.r)
      const cls = row.name === baseline ? 'baseline' : ''
      parts.push(
        `<tr class="${cls}" data-name="${row.name}">` +
          `<td>${row.r.experiment.id}<br><span class="hint mono">p-${row.r.meta.paramsHash} seed ${row.r.experiment.seed}${row.r.experiment.coverage ? ` r${row.r.experiment.coverage.radius} d${row.r.experiment.coverage.densityScale}` : ''}</span></td>` +
          `<td>${row.r.meta.adapterSlug}${row.r.meta.timestampQuery ? '' : ' <span class="delta worse">no-ts</span>'}</td>` +
          `<td>${row.r.experiment.spline}</td>` +
          `<td>${row.r.meta.date.slice(0, 16).replace('T', ' ')}</td>` +
          `<td>${fmt(row.r.cpu.frameMs.p50)}${delta(row.r.cpu.frameMs.p50, baseCpu)}</td>` +
          `<td>${fmt(row.r.cpu.frameMs.p95)}</td>` +
          `<td>${g === null ? '–' : fmt(g)}${delta(g, baseGpu)}</td>` +
          `<td>${mb(Math.max(0, ...Object.values(row.r.vram.bySpecies)))}</td>` +
          `</tr>`,
      )
      if (row.name === baseline && row.r.gpu) {
        const passes = Object.entries(row.r.gpu)
          .map(([label, s]) => `${label}: p50 ${s.p50.toFixed(3)} · p95 ${s.p95.toFixed(3)} · p99 ${s.p99.toFixed(3)}`)
          .join('<br>')
        parts.push(
          `<tr class="baseline"><td colspan="8" class="mono" style="text-align:left">${passes}<br>params ${JSON.stringify(row.r.experiment.params)}</td></tr>`,
        )
      }
    }
    parts.push('</tbody>')
    table.innerHTML = parts.join('')

    table.querySelectorAll('th').forEach((th) => {
      th.addEventListener('click', () => {
        const key = th.dataset['key'] as SortKey | ''
        if (!key) return
        if (key === sortKey) sortDir *= -1
        else {
          sortKey = key
          sortDir = key === 'name' || key === 'date' ? 1 : 1
        }
        render()
      })
    })
    table.querySelectorAll('tbody tr[data-name]').forEach((tr) => {
      tr.addEventListener('click', () => {
        const name = (tr as HTMLElement).dataset['name']!
        baseline = baseline === name ? null : name
        render()
      })
    })
  }

  const controls = el('div')
  controls.style.margin = '0 0 10px'
  const comparableToggle = el('input')
  comparableToggle.type = 'checkbox'
  comparableToggle.id = 'cmp'
  comparableToggle.addEventListener('change', () => {
    comparableOnly = comparableToggle.checked
    render()
  })
  const lbl = el('label', 'hint', ' comparable to baseline only (same spline/seed/coverage/canvas/timing class)')
  lbl.htmlFor = 'cmp'
  controls.append(comparableToggle, lbl)
  container.insertBefore(controls, table)

  if (rows.length === 0) {
    container.appendChild(el('div', 'hint', 'No results yet — run a bench from any experiment.'))
  }
  render()

  return { dispose: () => page.remove() }
}
