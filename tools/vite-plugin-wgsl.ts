import { readFileSync } from 'node:fs'
import path from 'node:path'
import type { Plugin } from 'vite'

/**
 * WGSL import support with `#include` resolution and hot reload.
 *
 * `import shader from './x.wgsl'` yields `{ id, code }` where `code` has all
 * `#include "..."` directives recursively inlined (include-once). Include paths
 * are project-root-relative (`#include "src/wgsl/wind.wgsl"`) or relative to
 * the including file (`#include "./local.wgsl"`).
 *
 * Editing any included file hot-updates every module that (transitively)
 * inlined it. Modules self-accept and forward the fresh source to
 * `globalThis.__wgslHot`, which the ShaderRegistry hooks at runtime.
 */
export function wgsl(): Plugin {
  let root = process.cwd()
  // include file (abs) -> module files (abs) whose expansion inlined it
  const dependents = new Map<string, Set<string>>()

  const expand = (file: string, visited: Set<string>, rootModule: string): string => {
    const source = readFileSync(file, 'utf8')
    return source.replace(/^[ \t]*#include\s+"([^"]+)"[ \t]*$/gm, (_, ref: string) => {
      const target = ref.startsWith('.')
        ? path.resolve(path.dirname(file), ref)
        : path.resolve(root, ref)
      if (visited.has(target)) return `// #include "${ref}" (already included)`
      visited.add(target)
      let deps = dependents.get(target)
      if (!deps) dependents.set(target, (deps = new Set()))
      deps.add(rootModule)
      const body = expand(target, visited, rootModule)
      return `// -- begin #include "${ref}"\n${body}\n// -- end #include "${ref}"`
    })
  }

  return {
    name: 'wgsl',
    configResolved(config) {
      root = config.root
    },
    transform(_code, id) {
      if (!id.endsWith('.wgsl')) return null
      const file = id.split('?')[0]!
      const visited = new Set<string>([file])
      const code = expand(file, visited, file)
      for (const dep of visited) if (dep !== file) this.addWatchFile(dep)
      const rel = path.relative(root, file)
      return {
        code: [
          `const src = { id: ${JSON.stringify(rel)}, code: ${JSON.stringify(code)} };`,
          `export default src;`,
          `if (import.meta.hot) {`,
          `  import.meta.hot.accept((m) => { globalThis.__wgslHot?.(src.id, m?.default?.code); });`,
          `}`,
        ].join('\n'),
        map: null,
      }
    },
    handleHotUpdate({ file, server, modules }) {
      if (!file.endsWith('.wgsl')) return
      const affected = new Set(modules)
      for (const modFile of dependents.get(file) ?? []) {
        for (const mod of server.moduleGraph.getModulesByFile(modFile) ?? []) affected.add(mod)
      }
      return [...affected]
    },
  }
}
