export interface WgslSource {
  id: string
  code: string
}

export interface ShaderMessage {
  shaderId: string
  kind: 'error' | 'warning'
  text: string
}

/**
 * Shader module cache with hot reload. The WGSL Vite plugin calls
 * `globalThis.__wgslHot(id, code)` when a .wgsl file (or anything it
 * `#include`s) changes; the registry validates the new source, keeps the old
 * module on compile errors (surfacing them via onMessage), and otherwise
 * notifies reload subscribers so they can rebuild pipelines.
 *
 * `module(src)` is keyed by `src.id` and always returns the LATEST code for
 * that id — callers may hold stale `src` objects across hot reloads.
 */
export class ShaderRegistry {
  private latest = new Map<string, string>()
  private cache = new Map<string, { code: string; module: GPUShaderModule }>()
  private reloadSubscribers = new Set<() => void>()

  constructor(
    private device: GPUDevice,
    private onMessage: (msg: ShaderMessage) => void,
  ) {
    globalThis.__wgslHot = (id, code) => {
      if (code !== undefined) void this.hotReload(id, code)
    }
  }

  module(src: WgslSource): GPUShaderModule {
    if (!this.latest.has(src.id)) this.latest.set(src.id, src.code)
    const code = this.latest.get(src.id)!
    const cached = this.cache.get(src.id)
    if (cached && cached.code === code) return cached.module
    const module = this.device.createShaderModule({ label: src.id, code })
    this.cache.set(src.id, { code, module })
    void this.reportCompilation(src.id, module)
    return module
  }

  /** Subscribe to successful hot reloads; returns unsubscribe. */
  onReload(rebuild: () => void): () => void {
    this.reloadSubscribers.add(rebuild)
    return () => this.reloadSubscribers.delete(rebuild)
  }

  private async hotReload(id: string, code: string): Promise<void> {
    const module = this.device.createShaderModule({ label: id, code })
    const info = await module.getCompilationInfo()
    const errors = info.messages.filter((m) => m.type === 'error')
    if (errors.length > 0) {
      for (const e of errors) {
        this.onMessage({ shaderId: id, kind: 'error', text: `${id}:${e.lineNum}:${e.linePos} ${e.message}` })
      }
      return // keep the old module running
    }
    this.latest.set(id, code)
    this.cache.set(id, { code, module })
    this.onMessage({ shaderId: id, kind: 'warning', text: `${id} reloaded` })
    for (const rebuild of this.reloadSubscribers) {
      try {
        rebuild()
      } catch (err) {
        this.onMessage({ shaderId: id, kind: 'error', text: `pipeline rebuild failed: ${String(err)}` })
      }
    }
  }

  private async reportCompilation(id: string, module: GPUShaderModule): Promise<void> {
    const info = await module.getCompilationInfo()
    for (const m of info.messages) {
      if (m.type === 'info') continue
      this.onMessage({ shaderId: id, kind: m.type === 'error' ? 'error' : 'warning', text: `${id}:${m.lineNum}:${m.linePos} ${m.message}` })
    }
  }
}
