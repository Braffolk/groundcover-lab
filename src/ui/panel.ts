import { Pane } from 'tweakpane'
import type { ParamSchema } from '../harness/params.ts'

export interface ParamsPanelOptions {
  container: HTMLElement
  title: string
  schema: ParamSchema
  values: Record<string, unknown>
  onChange: (key: string, value: unknown) => void
}

/** Tweakpane auto-built from a manifest's param schema. */
export function buildParamsPanel(opts: ParamsPanelOptions): { dispose: () => void; refresh: () => void } {
  const pane = new Pane({ container: opts.container, title: opts.title })
  for (const [key, def] of Object.entries(opts.schema)) {
    const binding =
      def.kind === 'number'
        ? pane.addBinding(opts.values, key, { min: def.min, max: def.max, ...(def.step !== undefined && { step: def.step }) })
        : def.kind === 'enum'
          ? pane.addBinding(opts.values, key, { options: Object.fromEntries(def.options.map((o) => [o, o])) })
          : pane.addBinding(opts.values, key)
    binding.on('change', (ev) => opts.onChange(key, ev.value))
  }
  return {
    dispose: () => pane.dispose(),
    refresh: () => pane.refresh(),
  }
}
