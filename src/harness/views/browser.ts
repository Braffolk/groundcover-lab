import { discoverExperiments, type ExperimentKind, type RegistryEntry } from '../registry.ts'
import { meshCatalog, type MeshInfo } from '../../mesh/catalog.ts'
import { fetchAllRatings, saveRating } from '../ratings.ts'
import { balanceScore, loadPerfIndex } from '../scoring.ts'
import { currentState, navigate, updateQuery } from '../../url/state.ts'
import { HAS_DEV_SINK, RAW_MESHES_AVAILABLE } from '../../util/env.ts'
import { pipsRow } from '../../ui/pips.ts'
import { button, el, formatCount, link, topbar, type View } from './shared.ts'

type SortMode = 'id' | 'visual' | 'perf' | 'balance'
const SORT_MODES: SortMode[] = ['id', 'visual', 'perf', 'balance']

/**
 * Where an experiment with no taxonomy (one dropped straight into
 * `experiments/`) is filed. Every experiment lives under a kind root now, so
 * this is a fallback rather than the normal path — but it means a stray folder
 * shows up in the right place instead of inventing a nameless top-level group.
 */
const KIND_GROUP: Record<ExperimentKind, string> = { renderer: 'renderers', material: 'materials' }

/** Top-level groups pinned ahead of alphabetical order; the rest sort by name. */
const GROUP_ORDER = ['renderers', 'materials']

interface CardRef {
  entry: RegistryEntry
  card: HTMLElement
  badge: HTMLElement
  selectBtn: HTMLButtonElement
  scoreLabel: HTMLElement
}

/**
 * One node of the taxonomy tree. Depth is whatever the directory layout says —
 * `renderers` is one level, `materials/mosses/sphagnum` is three, and nothing
 * here knows or cares which.
 */
interface GroupNode {
  name: string
  /** Slash-joined taxonomy path — the key collapse state round-trips under. */
  key: string
  children: Map<string, GroupNode>
  /** Experiments filed directly at this level. */
  entries: RegistryEntry[]
}

function emptyGroup(name: string, key: string): GroupNode {
  return { name, key, children: new Map(), entries: [] }
}

/** Group entries by `taxonomy` (falling back to the kind's root), any depth. */
function buildTree(entries: readonly RegistryEntry[]): GroupNode {
  const root = emptyGroup('', '')
  for (const entry of entries) {
    const path = entry.taxonomy.length > 0 ? entry.taxonomy : [KIND_GROUP[entry.kind]]
    let node = root
    for (const name of path) {
      let child = node.children.get(name)
      if (!child) node.children.set(name, (child = emptyGroup(name, node.key ? `${node.key}/${name}` : name)))
      node = child
    }
    node.entries.push(entry)
  }
  return root
}

function groupCount(node: GroupNode): number {
  let n = node.entries.length
  for (const child of node.children.values()) n += groupCount(child)
  return n
}

function orderedChildren(node: GroupNode): GroupNode[] {
  const rank = (name: string): number => {
    const i = GROUP_ORDER.indexOf(name)
    return i === -1 ? GROUP_ORDER.length : i
  }
  return [...node.children.values()].sort((a, b) => rank(a.name) - rank(b.name) || a.name.localeCompare(b.name))
}

/** #/ — the experiment browser: a collapsible tree of rated, ranked cards. */
export async function browserView(root: HTMLElement): Promise<View> {
  const page = el('div', 'page')
  page.appendChild(topbar('experiments', [link('#/results', 'bench results')]))
  const content = el('div', 'content')
  const browser = el('div', 'browser')
  content.appendChild(browser)
  page.appendChild(content)
  root.appendChild(page)

  const entries = await discoverExperiments()
  const rendered = entries.filter((e) => e.manifest?.status !== 'reference')
  const references = entries.filter((e) => e.manifest?.status === 'reference')
  const [ratings, perf] = await Promise.all([
    fetchAllRatings(rendered),
    loadPerfIndex(),
  ])

  // --- A/B selection (order matters: first = A, second = B) ---------------
  // Keyed by id across every grid, so picking A in one group and B in another
  // works exactly as it did when there was a single flat list.
  const selection: string[] = []
  const cards = new Map<string, CardRef>()
  const compareBar = el('div', 'compare-bar')
  const compareLabel = el('span', 'hint')
  compareBar.append(
    compareLabel,
    button('compare A/B', () => {
      const [a, b] = selection
      if (a && b) navigate(['ab', a, b])
    }, 'primary'),
    button('clear', () => {
      selection.length = 0
      syncSelection()
    }),
  )
  compareBar.style.display = 'none'
  page.appendChild(compareBar)

  const syncSelection = (): void => {
    for (const [id, ref] of cards) {
      const idx = selection.indexOf(id)
      ref.card.classList.toggle('selected', idx !== -1)
      ref.badge.style.display = idx === -1 ? 'none' : ''
      ref.badge.textContent = idx === 0 ? 'A' : 'B'
      ref.selectBtn.textContent = idx === -1 ? 'compare' : `selected: ${idx === 0 ? 'A' : 'B'}`
    }
    compareBar.style.display = selection.length === 2 ? '' : 'none'
    if (selection.length === 2) compareLabel.textContent = `${selection[0]} vs ${selection[1]}`
  }
  const toggleSelect = (id: string): void => {
    const idx = selection.indexOf(id)
    if (idx !== -1) selection.splice(idx, 1)
    else if (selection.length < 2) selection.push(id)
    syncSelection()
  }

  // --- sort control --------------------------------------------------------
  let sortMode = (currentState().q.get('sort') as SortMode | null) ?? 'id'
  if (!SORT_MODES.includes(sortMode)) sortMode = 'id'

  const sortBar = el('div')
  sortBar.style.cssText = 'display:flex;gap:8px;align-items:center;flex-wrap:wrap'
  sortBar.appendChild(el('span', 'hint', 'sort'))
  const sortButtons = SORT_MODES.map((mode) => {
    const b = button(mode, () => {
      sortMode = mode
      updateQuery({ sort: mode === 'id' ? null : mode })
      applySort()
    })
    b.dataset['mode'] = mode
    return b
  })
  sortBar.append(...sortButtons)
  sortBar.appendChild(
    el('span', 'hint', `· sorts within each group · perf & balance from latest bench @ ${perf.context} · balance = √(visual × perf), absolute anchors`),
  )
  browser.appendChild(sortBar)

  // --- collapse state, round-tripped through the URL like sort/stand/debug --
  const closed = new Set((currentState().q.get('closed') ?? '').split(',').filter((s) => s.length > 0))
  const writeClosed = (): void => {
    updateQuery({ closed: closed.size > 0 ? [...closed].sort().join(',') : null })
  }

  /** A collapsible section: header with a text marker, a name and a count. */
  const section = (name: string, key: string, count: number): HTMLDetailsElement => {
    const details = el('details', 'group')
    details.open = !closed.has(key)
    const summary = el('summary')
    summary.appendChild(el('span', 'label', name))
    summary.appendChild(el('span', 'count', String(count)))
    details.appendChild(summary)
    details.addEventListener('toggle', () => {
      if (details.open) closed.delete(key)
      else closed.add(key)
      writeClosed()
    })
    return details
  }

  const sortValue = (id: string): number | null => {
    const rating = ratings.get(id) ?? null
    const ms = perf.byExperiment.get(id) ?? null
    switch (sortMode) {
      case 'id':
        return 0
      case 'visual':
        return rating === null ? null : -rating
      case 'perf':
        return ms
      case 'balance':
        return rating === null || ms === null ? null : -balanceScore(rating, ms)
    }
  }

  const scoreText = (id: string): string => {
    const ms = perf.byExperiment.get(id) ?? null
    const rating = ratings.get(id) ?? null
    const perfPart = ms === null ? 'no bench' : `${ms.toFixed(2)}ms Σp50`
    const balPart = ms !== null && rating !== null ? `bal ${balanceScore(rating, ms)}` : 'bal –'
    return `${perfPart} · ${balPart}`
  }

  // Every card grid on the page, each with the entries it owns. Sorting is
  // per-grid: one comparator, applied inside each group independently, so a
  // group's members never migrate out of their group.
  const grids: { grid: HTMLElement; entries: readonly RegistryEntry[] }[] = []

  const applySort = (): void => {
    for (const b of sortButtons) b.classList.toggle('primary', b.dataset['mode'] === sortMode)
    for (const { grid, entries: members } of grids) {
      const order = [...members].sort((a, b) => {
        const [va, vb] = [sortValue(a.id), sortValue(b.id)]
        if (va === null && vb === null) return a.id.localeCompare(b.id, undefined, { numeric: true })
        if (va === null) return 1
        if (vb === null) return -1
        return va - vb || a.id.localeCompare(b.id, undefined, { numeric: true })
      })
      for (const entry of order) {
        const ref = cards.get(entry.id)
        if (ref) grid.appendChild(ref.card) // appendChild moves — reorders in place
      }
    }
  }

  const makeCard = (entry: RegistryEntry, rated: boolean): CardRef => {
    const ref = experimentCard(entry, toggleSelect, rated && entry.manifest ? {
      rating: ratings.get(entry.id) ?? null,
      onRate: (value) => {
        void saveRating(entry.dir, value)
          .then(() => {
            if (value === null) ratings.delete(entry.id)
            else ratings.set(entry.id, value)
            ref.scoreLabel.textContent = scoreText(entry.id)
            applySort()
          })
          .catch(() => {
            ref.scoreLabel.textContent = 'rating needs the dev server'
          })
      },
    } : null)
    ref.scoreLabel.textContent = entry.manifest && rated ? scoreText(entry.id) : ''
    cards.set(entry.id, ref)
    return ref
  }

  const addGrid = (parent: HTMLElement, members: readonly RegistryEntry[], rated: boolean): void => {
    const grid = el('div', 'cards')
    parent.appendChild(grid)
    for (const entry of members) grid.appendChild(makeCard(entry, rated).card)
    grids.push({ grid, entries: members })
  }

  const emitGroup = (node: GroupNode, parent: HTMLElement): void => {
    const details = section(node.name, node.key, groupCount(node))
    parent.appendChild(details)
    if (node.entries.length > 0) addGrid(details, node.entries, true)
    for (const child of orderedChildren(node)) emitGroup(child, details)
  }

  const tree = buildTree(rendered)
  // Only reachable if an experiment sits directly in experiments/ with no kind
  // root above it — kept so such a folder still shows rather than vanishing.
  if (tree.entries.length > 0) addGrid(browser, tree.entries, true)
  for (const child of orderedChildren(tree)) emitGroup(child, browser)

  if (entries.length === 0) {
    browser.appendChild(el('div', 'hint', 'No experiments yet — run `npm run new -- <slug>` to scaffold one.'))
  }

  if (references.length > 0) {
    const details = section('references — ignore the stand', 'references', references.length)
    browser.appendChild(details)
    if (!RAW_MESHES_AVAILABLE) {
      details.appendChild(
        el('div', 'hint', 'raw source meshes are not part of this deployment — references that render them cannot start here'),
      )
    }
    addGrid(details, references, false)
  }

  const meshes = meshCatalog.list()
  if (meshes.length > 0) {
    const details = section('source meshes', 'meshes', meshes.length)
    browser.appendChild(details)
    const meshGrid = el('div', 'cards')
    details.appendChild(meshGrid)
    for (const mesh of meshes) meshGrid.appendChild(meshCard(mesh))
  }

  applySort()

  return { dispose: () => page.remove() }
}

function experimentCard(
  entry: RegistryEntry,
  onToggleSelect: (id: string) => void,
  rating: { rating: number | null; onRate: (value: number | null) => void } | null,
): CardRef {
  const m = entry.manifest
  const card = el('div', m ? 'card' : 'card broken')

  const badge = el('div', 'ab-badge')
  badge.style.display = 'none'
  card.appendChild(badge)

  const thumb = el('div', 'thumb')
  if (entry.thumbnailUrl && m) {
    const img = el('img')
    img.src = entry.thumbnailUrl
    img.alt = entry.id
    img.addEventListener('error', () => {
      img.remove()
      thumb.textContent = 'no thumbnail yet — capture one in the runner'
    })
    thumb.appendChild(img)
  } else {
    thumb.textContent = m ? 'no thumbnail yet — capture one in the runner' : 'broken'
  }
  card.appendChild(thumb)

  const body = el('div', 'body')
  const head = el('div', 'head')
  const name = el('span', 'name', m?.title ?? entry.id)
  head.appendChild(name)
  head.appendChild(el('span', `status ${m?.status ?? 'abandoned'}`, m?.status ?? 'error'))
  body.appendChild(head)
  body.appendChild(el('div', 'desc', m?.description ?? entry.error ?? ''))

  const scoreLabel = el('span', 'mono hint')
  if (rating) {
    const scores = el('div', 'scores')
    // Read-only deployment: pips still show the owner's verdict (and still
    // drive the `visual`/`balance` sorts) but cannot be clicked.
    scores.appendChild(pipsRow({ value: rating.rating, onSet: rating.onRate, readOnly: !HAS_DEV_SINK }).el)
    scores.appendChild(scoreLabel)
    body.appendChild(scores)
  }

  const foot = el('div', 'foot')
  foot.appendChild(el('code', 'mono', entry.id))
  foot.appendChild(el('span', 'spacer'))
  const selectBtn = button('compare', () => onToggleSelect(entry.id), 'select-btn')
  if (m) {
    // A material's node graph and every intermediate texture are inspectable —
    // that page is the reason the graph is declarative, so it gets a door here.
    if (entry.kind === 'material') {
      foot.appendChild(button('inspect', () => navigate(['material', entry.id]), 'select-btn'))
    }
    foot.appendChild(selectBtn)
    const open = (): void => navigate(['run', entry.id])
    thumb.addEventListener('click', open)
    name.addEventListener('click', open)
  }
  body.appendChild(foot)
  card.appendChild(body)
  return { entry, card, badge, selectBtn, scoreLabel }
}

function meshCard(mesh: MeshInfo): HTMLElement {
  const card = el('div', 'card')
  const thumb = el('div', 'thumb', 'open inspector')
  card.appendChild(thumb)
  const body = el('div', 'body')
  const head = el('div', 'head')
  head.appendChild(el('span', 'name', mesh.id))
  head.appendChild(el('span', 'status reference', 'gcmesh1'))
  body.appendChild(head)
  const size = mesh.bytes >= 1e6 ? `${(mesh.bytes / 1e6).toFixed(1)}MB` : `${(mesh.bytes / 1e3).toFixed(0)}KB`
  const extent = mesh.tileSize
    ? `tile ${mesh.tileSize[0].toFixed(2)}×${mesh.tileSize[1].toFixed(2)}m`
    : 'single specimen'
  body.appendChild(
    el(
      'div',
      'desc',
      `${formatCount(mesh.vertexCount)} verts · ${formatCount(mesh.triangleCount)} tris · ${size} · ${extent}`,
    ),
  )
  if (!mesh.available) {
    thumb.textContent = 'source .bin not deployed'
    body.appendChild(el('div', 'hint', 'manifest only — the binary is not part of this deployment'))
  }
  const foot = el('div', 'foot')
  foot.appendChild(el('code', 'mono', `mesh/raw/${mesh.id}/`))
  body.appendChild(foot)
  card.appendChild(body)
  const open = (): void => navigate(['mesh', mesh.id])
  thumb.addEventListener('click', open)
  card.querySelector('.name')!.addEventListener('click', open)
  return card
}
