import { discoverExperiments, type RegistryEntry } from '../registry.ts'
import { meshCatalog, type MeshInfo } from '../../mesh/catalog.ts'
import { navigate } from '../../url/state.ts'
import { button, el, formatCount, link, topbar, type View } from './shared.ts'

/** #/ — the experiment browser: experiment cards (select two for A/B), mesh cards. */
export async function browserView(root: HTMLElement): Promise<View> {
  const page = el('div', 'page')
  page.appendChild(topbar('experiments', [link('#/results', 'bench results')]))
  const content = el('div', 'content')
  const browser = el('div', 'browser')
  content.appendChild(browser)
  page.appendChild(content)
  root.appendChild(page)

  const entries = await discoverExperiments()

  // A/B selection: order matters — first pick is A, second is B.
  const selection: string[] = []
  const cardsById = new Map<string, { card: HTMLElement; badge: HTMLElement; btn: HTMLButtonElement }>()

  const compareBar = el('div', 'compare-bar')
  const compareLabel = el('span', 'hint')
  const goCompare = button('compare A/B', () => {
    const [a, b] = selection
    if (a && b) navigate(['ab', a, b])
  }, 'primary')
  const clearBtn = button('clear', () => {
    for (const id of [...selection]) toggleSelect(id)
  })
  compareBar.append(compareLabel, goCompare, clearBtn)
  compareBar.style.display = 'none'
  page.appendChild(compareBar)

  const syncSelection = (): void => {
    for (const [id, ui] of cardsById) {
      const idx = selection.indexOf(id)
      ui.card.classList.toggle('selected', idx !== -1)
      ui.badge.style.display = idx === -1 ? 'none' : ''
      ui.badge.textContent = idx === 0 ? 'A' : 'B'
      ui.btn.textContent = idx === -1 ? 'compare' : `selected: ${idx === 0 ? 'A' : 'B'}`
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

  if (entries.length === 0) {
    browser.appendChild(el('div', 'hint', 'No experiments yet — run `npm run new -- <slug>` to scaffold one.'))
  }
  // Stand renderers first; references (which ignore the stand) get their own row.
  const renderers = entries.filter((e) => e.manifest?.status !== 'reference')
  const references = entries.filter((e) => e.manifest?.status === 'reference')
  const grid = el('div', 'cards')
  browser.appendChild(grid)
  for (const entry of renderers) {
    const ui = experimentCard(entry, toggleSelect)
    grid.appendChild(ui.card)
    if (entry.manifest) cardsById.set(entry.id, ui)
  }
  if (references.length > 0) {
    browser.appendChild(el('h2', undefined, 'references — ignore the stand'))
    const refGrid = el('div', 'cards')
    browser.appendChild(refGrid)
    for (const entry of references) {
      const ui = experimentCard(entry, toggleSelect)
      refGrid.appendChild(ui.card)
      if (entry.manifest) cardsById.set(entry.id, ui)
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
): { card: HTMLElement; badge: HTMLElement; btn: HTMLButtonElement } {
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

  const foot = el('div', 'foot')
  foot.appendChild(el('code', 'mono', entry.id))
  foot.appendChild(el('span', 'spacer'))
  const btn = button('compare', () => onToggleSelect(entry.id), 'select-btn')
  if (m) {
    foot.appendChild(btn)
    const open = (): void => navigate(['run', entry.id])
    thumb.addEventListener('click', open)
    name.addEventListener('click', open)
  }
  body.appendChild(foot)
  card.appendChild(body)
  return { card, badge, btn }
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
  body.appendChild(
    el(
      'div',
      'desc',
      `${formatCount(mesh.vertexCount)} verts · ${formatCount(mesh.triangleCount)} tris · ${size} · ` +
        `tile ${mesh.tileSize[0].toFixed(2)}×${mesh.tileSize[1].toFixed(2)}m`,
    ),
  )
  const foot = el('div', 'foot')
  foot.appendChild(el('code', 'mono', `mesh/raw/${mesh.id}/`))
  body.appendChild(foot)
  card.appendChild(body)
  const open = (): void => navigate(['mesh', mesh.id])
  thumb.addEventListener('click', open)
  card.querySelector('.name')!.addEventListener('click', open)
  return card
}
