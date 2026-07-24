/**
 * Hash-based URL state. Every interesting view in the lab is a link:
 *   #/run/001-foo?cam=grazing&seed=42&p.density=2
 *   #/ab/000-ground-truth/001-foo?mode=wipe&cam=8.00,2.10,...
 *   #/bench/001-foo?spline=orbit-low&frames=600
 */

export interface HashState {
  /** Path segments after '#/', e.g. ['run', '001-foo']. */
  path: string[]
  q: URLSearchParams
}

export function parseHash(hash: string): HashState {
  const raw = hash.replace(/^#\/?/, '')
  const qIndex = raw.indexOf('?')
  const pathPart = qIndex === -1 ? raw : raw.slice(0, qIndex)
  const queryPart = qIndex === -1 ? '' : raw.slice(qIndex + 1)
  return {
    path: pathPart.split('/').filter((s) => s.length > 0).map(decodeURIComponent),
    q: new URLSearchParams(queryPart),
  }
}

export function currentState(): HashState {
  return parseHash(location.hash)
}

export function buildHash(path: string[], q?: Record<string, string> | URLSearchParams): string {
  const search = q ? new URLSearchParams(q).toString() : ''
  return `#/${path.map(encodeURIComponent).join('/')}${search ? `?${search}` : ''}`
}

/**
 * Patch query params of the current hash in place (null deletes a key).
 * Uses replaceState so camera movement does not spam browser history.
 */
export function updateQuery(patch: Record<string, string | null>): void {
  const { path, q } = currentState()
  for (const [key, value] of Object.entries(patch)) {
    if (value === null) q.delete(key)
    else q.set(key, value)
  }
  history.replaceState(null, '', buildHash(path, q))
}

/** Debounced updateQuery for high-frequency sources (camera drag). */
export function makeDebouncedQueryWriter(delayMs = 250): (patch: Record<string, string | null>) => void {
  let pending: Record<string, string | null> = {}
  let timer: ReturnType<typeof setTimeout> | null = null
  return (patch) => {
    pending = { ...pending, ...patch }
    timer ??= setTimeout(() => {
      timer = null
      const batch = pending
      pending = {}
      updateQuery(batch)
    }, delayMs)
  }
}

export function navigate(path: string[], q?: Record<string, string>): void {
  location.hash = buildHash(path, q)
}

export function onHashChange(cb: () => void): () => void {
  window.addEventListener('hashchange', cb)
  return () => window.removeEventListener('hashchange', cb)
}
