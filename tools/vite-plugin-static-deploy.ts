import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import type { Plugin, ResolvedConfig } from 'vite'

/**
 * Everything the deployed lab fetches at runtime that is NOT in the module
 * graph, copied into dist/ at build time:
 *
 *   experiments/<id>/thumbnail.png      browser cards
 *   experiments/<id>/rating.json        owner's visual verdict (display + sort)
 *   results/*.json + results/index.json bench results table (index replaces
 *                                       the dev-only GET /__bench/list)
 *   mesh/baked/**                       committed baked artifacts (~256MB) —
 *                                       without them a visitor would have to
 *                                       bake from raw meshes in-browser
 *   mesh/raw/<id>/manifest.json         mesh catalog listing (tiny)
 *   mesh/raw/<id>/*.bin                 ONLY with GC_DEPLOY_MESHES=1 (~357MB;
 *                                       poa-pratensis alone is 229MB)
 *
 * goldens/ is deliberately NOT copied — nothing fetches it at runtime.
 *
 * Also defines __GC_RAW_MESHES__ (see src/util/env.ts) so the mesh inspector
 * can say "not included in this deployment" instead of failing on a 404 or,
 * worse, parsing an SPA-fallback index.html as GCMESH1.
 */
export interface StaticDeployOptions {
  /** Include the raw source-mesh binaries (default: GC_DEPLOY_MESHES=1). */
  rawMeshes?: boolean
}

interface CopyStat {
  label: string
  files: number
  bytes: number
}

export function staticDeploy(options: StaticDeployOptions = {}): Plugin {
  const rawMeshes = options.rawMeshes ?? process.env['GC_DEPLOY_MESHES'] === '1'
  let config: ResolvedConfig
  let root = process.cwd()
  let outDir = path.resolve(root, 'dist')

  const mb = (bytes: number): string => `${(bytes / (1024 * 1024)).toFixed(1)}MB`

  /** Copy root-relative `rel` (file or directory) into the same place in dist/. */
  const copy = (rel: string, stat: CopyStat): void => {
    const from = path.join(root, rel)
    if (!existsSync(from)) return
    const to = path.join(outDir, rel)
    mkdirSync(path.dirname(to), { recursive: true })
    if (statSync(from).isDirectory()) {
      cpSync(from, to, { recursive: true })
      for (const file of walk(from)) {
        stat.files++
        stat.bytes += statSync(file).size
      }
    } else {
      cpSync(from, to)
      stat.files++
      stat.bytes += statSync(from).size
    }
  }

  const walk = (dir: string): string[] => {
    const out: string[] = []
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name)
      if (entry.isDirectory()) out.push(...walk(full))
      else out.push(full)
    }
    return out
  }

  const dirs = (rel: string): string[] => {
    const full = path.join(root, rel)
    if (!existsSync(full)) return []
    return readdirSync(full, { withFileTypes: true })
      .filter((e) => e.isDirectory() && !e.name.startsWith('_') && !e.name.startsWith('.'))
      .map((e) => e.name)
      .sort()
  }

  return {
    name: 'static-deploy',
    config(_userConfig, env) {
      return { define: { __GC_RAW_MESHES__: JSON.stringify(env.command === 'serve' || rawMeshes) } }
    },
    configResolved(resolved) {
      config = resolved
      root = resolved.root
      outDir = path.resolve(root, resolved.build.outDir)
    },
    closeBundle() {
      if (config.command !== 'build') return
      const stats: CopyStat[] = []

      // --- experiment thumbnails + owner ratings ---------------------------
      // Experiments are a tree (materials/<class>/<subject>/<attempt>), so this
      // walks for manifest.ts at any depth rather than reading one level.
      const experimentDirs = (rel: string): string[] => {
        const out: string[] = []
        for (const name of dirs(rel)) {
          const child = path.join(rel, name)
          if (existsSync(path.join(root, child, 'manifest.ts'))) out.push(child)
          else out.push(...experimentDirs(child))
        }
        return out
      }
      const experiments: CopyStat = { label: 'experiments (thumbnails + ratings)', files: 0, bytes: 0 }
      for (const dir of experimentDirs('experiments')) {
        copy(path.join(dir, 'thumbnail.png'), experiments)
        copy(path.join(dir, 'rating.json'), experiments)
      }
      stats.push(experiments)

      // --- bench results + the static index that replaces /__bench/list ----
      const results: CopyStat = { label: 'results (+ index.json)', files: 0, bytes: 0 }
      const names = existsSync(path.join(root, 'results'))
        ? readdirSync(path.join(root, 'results'))
            .filter((n) => n.endsWith('.json') && n !== 'index.json')
            .sort()
        : []
      for (const name of names) copy(path.join('results', name), results)
      mkdirSync(path.join(outDir, 'results'), { recursive: true })
      const index = `${JSON.stringify(names, null, 2)}\n`
      writeFileSync(path.join(outDir, 'results', 'index.json'), index)
      results.files++
      results.bytes += Buffer.byteLength(index)
      stats.push(results)

      // --- committed baked artifacts (the bulk of the upload) -------------
      const baked: CopyStat = { label: 'mesh/baked', files: 0, bytes: 0 }
      copy(path.join('mesh', 'baked'), baked)
      stats.push(baked)

      // --- shared material assets (maps + measured scalars) ---------------
      const assets: CopyStat = { label: 'assets', files: 0, bytes: 0 }
      copy('assets', assets)
      stats.push(assets)

      // --- raw source meshes: manifests always, binaries on request -------
      const raw: CopyStat = { label: rawMeshes ? 'mesh/raw (manifests + binaries)' : 'mesh/raw (manifests only)', files: 0, bytes: 0 }
      const missing: string[] = []
      for (const id of dirs(path.join('mesh', 'raw'))) {
        const manifestRel = path.join('mesh', 'raw', id, 'manifest.json')
        copy(manifestRel, raw)
        if (!rawMeshes) continue
        const manifest = JSON.parse(readFileSync(path.join(root, manifestRel), 'utf8')) as {
          binary?: string | { file?: string }
        }
        const file = typeof manifest.binary === 'string' ? manifest.binary : manifest.binary?.file
        if (!file) continue
        const rel = path.join('mesh', 'raw', id, file)
        if (existsSync(path.join(root, rel))) copy(rel, raw)
        else missing.push(rel)
      }
      stats.push(raw)

      const total = stats.reduce((sum, s) => sum + s.bytes, 0)
      const distTotal = walk(outDir).reduce((sum, f) => sum + statSync(f).size, 0)
      config.logger.info(
        [
          `\nstatic-deploy → ${path.relative(root, outDir)}/ (base ${config.base})`,
          ...stats.map((s) => `  ${s.label.padEnd(32)} ${String(s.files).padStart(5)} files  ${mb(s.bytes).padStart(8)}`),
          `  ${'copied total'.padEnd(32)} ${String(stats.reduce((n, s) => n + s.files, 0)).padStart(5)} files  ${mb(total).padStart(8)}`,
          `  ${'dist total (incl. bundle)'.padEnd(32)} ${''.padStart(5)}        ${mb(distTotal).padStart(8)}`,
          rawMeshes
            ? '  raw source meshes INCLUDED (GC_DEPLOY_MESHES=1)'
            : '  raw source meshes omitted — set GC_DEPLOY_MESHES=1 to include them',
          ...missing.map((m) => `  ! ${m} not found locally — deployed site will report it as missing`),
        ].join('\n'),
      )
    },
  }
}
