import '../ui/style.css'
import { currentState, onHashChange } from '../url/state.ts'
import { browserView } from './views/browser.ts'
import { runView } from './views/run.ts'
import type { View } from './views/shared.ts'

/**
 * Hash router. Routes:
 *   #/                      experiment browser
 *   #/run/<id>              single-experiment runner
 *   #/ab/<idA>/<idB>        A/B compare (shared camera)
 *   #/bench/<id>            benchmark runner
 *   #/results               bench results table
 *   #/mesh/<meshId>         raw source-mesh inspector
 */
const root = document.getElementById('app')!

let current: View | null = null
let generation = 0

async function route(): Promise<void> {
  const gen = ++generation
  current?.dispose()
  current = null
  root.replaceChildren()

  const state = currentState()
  const name = state.path[0] ?? ''
  try {
    let view: View
    switch (name) {
      case '':
        view = await browserView(root)
        break
      case 'run':
        view = await runView(root, state)
        break
      case 'ab':
        view = await (await import('./views/ab.ts')).abView(root, state)
        break
      case 'bench':
        view = await (await import('./views/bench.ts')).benchView(root, state)
        break
      case 'results':
        view = await (await import('./views/results.ts')).resultsView(root)
        break
      case 'mesh':
        view = await (await import('./views/mesh.ts')).meshView(root, state)
        break
      default:
        location.hash = '#/'
        return
    }
    // A newer navigation happened while this view was loading.
    if (gen !== generation) view.dispose()
    else current = view
  } catch (err) {
    if (gen !== generation) return
    const pre = document.createElement('pre')
    pre.style.padding = '24px'
    pre.textContent = `route error: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}`
    root.appendChild(pre)
  }
}

onHashChange(() => void route())
void route()
