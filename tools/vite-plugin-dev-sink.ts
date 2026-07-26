import { mkdirSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import type { IncomingMessage, ServerResponse } from 'node:http'
import type { Plugin } from 'vite'

/**
 * Dev-server endpoints that let browser tooling persist artifacts into the
 * repo with one click (static builds degrade to download/drag-drop):
 *
 *   POST /__bench                        JSON BenchResult -> results/<derived-name>.json
 *   GET  /__bench/list                   JSON string[] of results/ filenames
 *   POST /__thumb?exp=<dir>              PNG body -> <dir>/thumbnail.png
 *   POST /__thumb?exp=<dir>&cam=<name>   PNG body -> goldens/<id>/<name>.png (id = last segment)
 *   POST /__bake?exp=<id>&key=<key>[&ext=bin|png|json]
 *                                        binary body -> mesh/baked/<id>/<key>.<ext> (ext defaults to bin)
 *   POST /__rating?exp=<dir>             {"visual":1..5|null} -> <dir>/rating.json (null deletes)
 */
const BAKE_EXTS = ['bin', 'png', 'json']

export function devSink(): Plugin {
  let root = process.cwd()

  const segment = (value: string | null, label: string): string => {
    if (!value || !/^[a-zA-Z0-9._-]+$/.test(value) || value.includes('..')) {
      throw new Error(`invalid ${label}: ${JSON.stringify(value)}`)
    }
    return value
  }

  /**
   * A repo-relative experiment directory, e.g. `experiments/renderers/001-billboard-smoke`
   * or `experiments/materials/mosses/sphagnum/003-nd-fuzz`. Experiments gained a
   * hierarchy, so this has to admit `/` — which means it must be validated
   * segment by segment, and pinned under `experiments/`, or it becomes a
   * write-anywhere primitive on the owner's disk.
   */
  const experimentDir = (value: string | null, label: string): string => {
    if (!value) throw new Error(`invalid ${label}: ${JSON.stringify(value)}`)
    const parts = value.split('/')
    if (parts[0] !== 'experiments' || parts.length < 2) {
      throw new Error(`invalid ${label}: must be a path under experiments/, got ${JSON.stringify(value)}`)
    }
    for (const part of parts) segment(part, label)
    return parts.join('/')
  }

  const readBody = (req: IncomingMessage): Promise<Buffer> =>
    new Promise((resolve, reject) => {
      const chunks: Buffer[] = []
      req.on('data', (c: Buffer) => chunks.push(c))
      req.on('end', () => resolve(Buffer.concat(chunks)))
      req.on('error', reject)
    })

  const write = (relDir: string, name: string, data: Buffer | string): string => {
    const dir = path.join(root, relDir)
    mkdirSync(dir, { recursive: true })
    const file = path.join(dir, name)
    writeFileSync(file, data)
    return path.relative(root, file)
  }

  const json = (res: ServerResponse, status: number, body: unknown): void => {
    res.statusCode = status
    res.setHeader('content-type', 'application/json')
    res.end(JSON.stringify(body))
  }

  return {
    name: 'dev-sink',
    configResolved(config) {
      root = config.root
    },
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = new URL(req.url ?? '/', 'http://localhost')
        const route = url.pathname

        const handle = async (): Promise<void> => {
          if (route === '/__bench/list' && req.method === 'GET') {
            let names: string[] = []
            try {
              names = readdirSync(path.join(root, 'results')).filter((n) => n.endsWith('.json'))
            } catch {
              /* results/ does not exist yet */
            }
            json(res, 200, names.sort())
            return
          }
          if (route === '/__bench' && req.method === 'POST') {
            const body = JSON.parse((await readBody(req)).toString('utf8')) as {
              experiment?: { id?: string; stand?: string; context?: string }
              meta?: { paramsHash?: string; adapterSlug?: string; date?: string }
            }
            const exp = segment(body.experiment?.id ?? null, 'experiment id')
            // Renderers file under their stand; kinds with no stand (materials)
            // file under `context` — the preview object. Same slot in the name,
            // so every historical filename still parses the same way.
            const partition = segment(
              body.experiment?.stand ?? body.experiment?.context ?? 'none',
              'stand/context id',
            )
            const hash = segment(body.meta?.paramsHash ?? null, 'params hash')
            const adapter = segment(body.meta?.adapterSlug ?? null, 'adapter slug')
            const date = (body.meta?.date ?? new Date().toISOString()).replace(/[:.]/g, '-')
            const name = `${exp}__${partition}__p-${hash}__${adapter}__${segment(date, 'date')}.json`
            const saved = write('results', name, JSON.stringify(body, null, 2))
            json(res, 200, { saved })
            return
          }
          if (route === '/__thumb' && req.method === 'POST') {
            const dir = experimentDir(url.searchParams.get('exp'), 'exp')
            const cam = url.searchParams.get('cam')
            const body = await readBody(req)
            // Goldens stay keyed by ID (the directory's last segment), not by
            // path, so a taxonomy reshuffle never orphans a golden.
            const id = dir.slice(dir.lastIndexOf('/') + 1)
            const saved = cam
              ? write(path.join('goldens', id), `${segment(cam, 'cam')}.png`, body)
              : write(dir, 'thumbnail.png', body)
            json(res, 200, { saved })
            return
          }
          if (route === '/__rating' && req.method === 'POST') {
            const dir = experimentDir(url.searchParams.get('exp'), 'exp')
            const { visual } = JSON.parse((await readBody(req)).toString('utf8')) as { visual: number | null }
            const file = path.join(root, dir, 'rating.json')
            if (visual === null) {
              rmSync(file, { force: true })
              json(res, 200, { cleared: true })
              return
            }
            if (!Number.isInteger(visual) || visual < 1 || visual > 5) throw new Error(`invalid rating ${visual}`)
            writeFileSync(file, `${JSON.stringify({ visual })}\n`)
            json(res, 200, { saved: path.relative(root, file) })
            return
          }
          if (route === '/__bake' && req.method === 'POST') {
            const exp = segment(url.searchParams.get('exp'), 'exp')
            const key = segment(url.searchParams.get('key'), 'key')
            // Extension is a separate parameter on purpose: segment() allows
            // '.' inside a key, so an extension smuggled into the key would
            // land as `foo.png.bin`.
            const ext = url.searchParams.get('ext') ?? 'bin'
            if (!BAKE_EXTS.includes(ext)) throw new Error(`invalid ext: ${JSON.stringify(ext)} (allowed: ${BAKE_EXTS.join(', ')})`)
            const saved = write(path.join('mesh', 'baked', exp), `${key}.${ext}`, await readBody(req))
            json(res, 200, { saved })
            return
          }
          next()
        }

        handle().catch((err: unknown) => {
          json(res, 400, { error: String(err) })
        })
      })
    },
  }
}
