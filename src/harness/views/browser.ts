import { discoverExperiments, type RegistryEntry } from '../registry.ts'
import { meshCatalog, type MeshInfo } from '../../mesh/catalog.ts'
import { fetchAllRatings, saveRating } from '../ratings.ts'
import { balanceScore, loadPerfIndex } from '../scoring.ts'
import { currentState, navigate, updateQuery } from '../../url/state.ts'
import { HAS_DEV_SINK, RAW_MESHES_AVAILABLE } from '../../util/env.ts'
import { pipsRow } from '../../ui/pips.ts'
import { button, el, formatCount, link, topbar, type View } from './shared.ts'

type SortMode = 'id' | 'visual' | 'perf' | 'balance'
const SORT_MODES: SortMode[] = ['id', 'visual', 'perf', 'balance']

interface CardRef {
  entry: RegistryEntry
  card: HTMLElement
  badge: HTMLElement
  selectBtn: HTMLButtonElement
  scoreLabel: HTMLElement
}

/** #/ — the experiment browser: rated, ranked, A/B-selectable cards + meshes. */
export async function browserView(root: HTMLElement): Promise<View> {
  const page = el('div', 'page')
  page.appendChild(topbar('experiments', [link('#/results', 'bench results')]))
  const content = el('div', 'content')
  const browser = el('div', 'browser')
  content.appendChild(browser)
  page.appendChild(content)
  root.appendChild(page)

  const entries = await discoverExperiments()
  const renderers = entries.filter((e) => e.manifest?.status !== 'reference')
  const references = entries.filter((e) => e.manifest?.status === 'reference')
  const [ratings, perf] = await Promise.all([
    fetchAllRatings(renderers.map((e) => e.id)),
    loadPerfIndex(),
  ])

  // --- A/B selection (order matters: first = A, second = B) ---------------
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
    el('span', 'hint', `· perf & balance from latest bench @ ${perf.context} · balance = √(visual × perf), absolute anchors`),
  )
  browser.appendChild(sortBar)

  const grid = el('div', 'cards')
  browser.appendChild(grid)
  if (renderers.length === 0) {
    browser.appendChild(el('div', 'hint', 'No experiments yet — run `npm run new -- <slug>` to scaffold one.'))
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

  const applySort = (): void => {
    for (const b of sortButtons) b.classList.toggle('primary', b.dataset['mode'] === sortMode)
    const order = [...renderers].sort((a, b) => {
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

  const scoreText = (id: string): string => {
    const ms = perf.byExperiment.get(id) ?? null
    const rating = ratings.get(id) ?? null
    const perfPart = ms === null ? 'no bench' : `${ms.toFixed(2)}ms Σp50`
    const balPart = ms !== null && rating !== null ? `bal ${balanceScore(rating, ms)}` : 'bal –'
    return `${perfPart} · ${balPart}`
  }

  for (const entry of renderers) {
    const ref = experimentCard(entry, toggleSelect, entry.manifest ? {
      rating: ratings.get(entry.id) ?? null,
      onRate: (value) => {
        void saveRating(entry.id, value)
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
    ref.scoreLabel.textContent = entry.manifest ? scoreText(entry.id) : ''
    cards.set(entry.id, ref)
    grid.appendChild(ref.card)
  }
  applySort()

  if (references.length > 0) {
    browser.appendChild(el('h2', undefined, 'references — ignore the stand'))
    if (!RAW_MESHES_AVAILABLE) {
      browser.appendChild(
        el('div', 'hint', 'raw source meshes are not part of this deployment — references that render them cannot start here'),
      )
    }
    const refGrid = el('div', 'cards')
    browser.appendChild(refGrid)
    for (const entry of references) {
      const ref = experimentCard(entry, toggleSelect, null)
      cards.set(entry.id, ref)
      refGrid.appendChild(ref.card)
    }
  }

  const meshes = meshCatalog.list()
  if (meshes.length > 0) {
    browser.appendChild(el('h2', undefined, 'source meshes'))
    const meshGrid = el('div', 'cards')
    browser.appendChild(meshGrid)
    for (const mesh of meshes) meshGrid.appendChild(meshCard(mesh))
  }

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
