/**
 * Typed experiment parameter schemas. Declared once in the manifest; the
 * harness derives the Tweakpane UI, URL round-tripping, and bench-result
 * provenance from the same declaration.
 */

export interface NumberParam {
  kind: 'number'
  def: number
  min: number
  max: number
  step?: number
}

export interface BooleanParam {
  kind: 'boolean'
  def: boolean
}

export interface EnumParam<T extends string = string> {
  kind: 'enum'
  def: T
  options: readonly T[]
}

export type ParamDef = NumberParam | BooleanParam | EnumParam

export type ParamSchema = Record<string, ParamDef>

export type ParamValue<D extends ParamDef> = D extends NumberParam
  ? number
  : D extends BooleanParam
    ? boolean
    : D extends EnumParam<infer T>
      ? T
      : never

export type ParamValues<S extends ParamSchema> = { [K in keyof S]: ParamValue<S[K]> }

export const p = {
  num: (def: number, o: { min: number; max: number; step?: number }): NumberParam => ({ kind: 'number', def, ...o }),
  bool: (def: boolean): BooleanParam => ({ kind: 'boolean', def }),
  enum: <T extends string>(def: T, options: readonly T[]): EnumParam<T> => ({ kind: 'enum', def, options }),
}

export function paramDefaults<S extends ParamSchema>(schema: S): ParamValues<S> {
  const out: Record<string, unknown> = {}
  for (const [key, def] of Object.entries(schema)) out[key] = def.def
  return out as ParamValues<S>
}

/** Read `<ns>.<key>` query params over defaults; invalid values are ignored. */
export function paramsFromQuery<S extends ParamSchema>(schema: S, q: URLSearchParams, ns: string): ParamValues<S> {
  const values: Record<string, unknown> = paramDefaults(schema)
  for (const [key, def] of Object.entries(schema)) {
    const raw = q.get(`${ns}.${key}`)
    if (raw === null) continue
    switch (def.kind) {
      case 'number': {
        const v = Number(raw)
        if (Number.isFinite(v)) values[key] = Math.min(def.max, Math.max(def.min, v))
        break
      }
      case 'boolean':
        values[key] = raw === '1' || raw === 'true'
        break
      case 'enum':
        if ((def.options as readonly string[]).includes(raw)) values[key] = raw
        break
    }
  }
  return values as ParamValues<S>
}

/** Non-default values as `<ns>.<key>` entries (null = delete from URL). */
export function paramsToQuery<S extends ParamSchema>(
  schema: S,
  values: ParamValues<S>,
  ns: string,
): Record<string, string | null> {
  const out: Record<string, string | null> = {}
  for (const [key, def] of Object.entries(schema)) {
    const value = values[key]
    out[`${ns}.${key}`] = value === def.def ? null : String(value)
  }
  return out
}

/** Stable 8-hex-digit hash of a params object, for bench result filenames. */
export function paramsHash(values: Record<string, unknown>): string {
  const stable = JSON.stringify(values, Object.keys(values).sort())
  let h = 0x811c9dc5 // FNV-1a
  for (let i = 0; i < stable.length; i++) {
    h ^= stable.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return (h >>> 0).toString(16).padStart(8, '0')
}
