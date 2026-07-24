import { discoverExperiments, type RegistryEntry } from '../registry.ts'
import { meshCatalog } from '../../mesh/catalog.ts'
import { buildHash, navigate } from '../../url/state.ts'
import { button, el, link, topbar, type View } from './shared.ts'

/** #/ — the experiment browser: cards, pick two for A/B. */
export async function browserView(root: HTMLElement): Promise<View> {
  const page = el('div', 'page')
  const meshLinks = meshCatalog.list().map((m) => link(buildHash(['mesh', m.id]), `mesh: ${m.id}`))
  page.appendChild(topbar('experiments', [...meshLinks, link('#/results', 'bench results')]))
  const content = el('div', 'content')
  const browser = el('div', 'browser')
  content.appendChild(browser)
  page.appendChild(content)
  root.appendChild(page)

  const entries = await discoverExperiments()
  const selected = new Set<string>()

  const header = el('div')
  const compareBtn = button('compare A/B →', () => {
    const [a, b] = [...selected]
    if (a && b) navigate(['ab', a, b])
  }, 'primary')
  compareBtn.style.display = 'none'
  const headline = el('span', undefined, entries.length === 0 ? 'No experiments yet — run `npm run new -- <slug>` to scaffold one.' : 'Select two cards to compare A/B.')
  headline.className = 'hint'
  header.append(headline, ' ', compareBtn)
  browser.appendChild(header)

  const grid = el('div', 'cards')
  browser.appendChild(grid)
  for (const entry of entries) grid.appendChild(card(entry, selected, () => {
    compareBtn.style.display = selected.size === 2 ? '' : 'none'
  }))

  return { dispose: () => page.remove() }
}

function card(entry: RegistryEntry, selected: Set<string>, onSelect: () => void): HTMLElement {
  const m = entry.manifest
  const c = el('div', m ? 'card' : 'card broken')

  const thumb = el('div', 'thumb')
  if (entry.thumbnailUrl) {
    const img = el('img')
    img.src = entry.thumbnailUrl
    img.alt = entry.id
    thumb.appendChild(img)
  } else {
    thumb.textContent = m ? 'no thumbnail yet — 📷 in the runner' : '⚠ broken'
  }
  c.appendChild(thumb)

  const body = el('div', 'body')
  const head = el('div', 'head')
  const name = el('span', 'name', m?.title ?? entry.id)
  head.appendChild(name)
  head.appendChild(el('span', `status ${m?.status ?? 'abandoned'}`, m?.status ?? 'error'))
  body.appendChild(head)
  body.appendChild(el('div', 'desc', m?.description ?? entry.error ?? ''))

  const foot = el('div', 'foot')
  foot.appendChild(el('code', 'mono', entry.id))
  if (m) {
    const pick = el('input')
    pick.type = 'checkbox'
    pick.title = 'select for A/B compare'
    pick.addEventListener('change', () => {
      if (pick.checked) {
        if (selected.size >= 2) {
          pick.checked = false
          return
        }
        selected.add(entry.id)
      } else {
        selected.delete(entry.id)
      }
      onSelect()
    })
    foot.appendChild(el('span', 'spacer'))
    foot.appendChild(pick)
    const open = (): void => navigate(['run', entry.id])
    thumb.addEventListener('click', open)
    name.addEventListener('click', open)
  }
  body.appendChild(foot)
  c.appendChild(body)
  return c
}
