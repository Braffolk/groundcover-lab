/**
 * The GPU + numeric half of the material inspector.
 *
 * It answers three questions about ANY node of a built material, with no
 * per-material knowledge whatsoever:
 *
 *   1. what texture does this node's value live in?  `nodeTextureFor()`
 *   2. what is actually in it, per mip level?        `readMipChain()`
 *   3. is the mip chain telling the truth?           `chainStats()`
 *
 * (1) is the interesting one. A texture-mode node HAS a texture — the one the
 * fragment shader samples — so the inspector shows that, not a re-render. A
 * LIVE node has none, so it is evaluated into an offscreen target by the SAME
 * generated `cs_materialize` entry point a materialized node would have used,
 * through `runMaterializePass` and deliberately NOT through `materializeNode`:
 * looking at a node must never write a PNG into the bake cache under a key
 * that asserts the node was materialized.
 *
 * (3) is the automated form of the check CLAUDE.md prescribes by hand — stored
 * rgb luma per level, which is exactly how the coverage-vs-premultiplied trap
 * announces itself. `checkMipConvention` in src/gpu/texture.ts runs the same
 * test in DEV for the two colour semantics; this one runs for every format and
 * every semantics, and REPORTS rather than warns, because on this page the
 * numbers are the point.
 *
 * No DOM here: the view turns these arrays into canvases.
 */

import type { VramScope } from '../gpu/resources.ts'
import { createTexture2D, generateMips, mipLevelsFor, readTexture, type TexelSemantics } from '../gpu/texture.ts'
import { generateMaterializeModule, uniformLayout } from './codegen.ts'
import { loadNodeImage } from './images.ts'
import { MATERIALIZE_FORMAT, runMaterializePass } from './materialize.ts'
import type { MaterialInstance } from './instance.ts'
import { CHANNEL_COMPONENTS, type ChannelType } from './types.ts'

/** Size a live node is evaluated at when nothing in its subtree implies one. */
export const DEFAULT_EVAL_SIZE = 512

// ---------------------------------------------------------------------------
// 1. A texture for any node
// ---------------------------------------------------------------------------

export type NodeTextureOrigin =
  /** The very texture the fragment shader samples. */
  | 'resident'
  /** Evaluated on demand into an offscreen target by the generated entry point. */
  | 'evaluated'
  /** An inactive variant's PNG, loaded on demand just to look at it. */
  | 'loaded'

export interface NodeTextureResult {
  nodeId: string
  texture: GPUTexture
  origin: NodeTextureOrigin
  /** Provenance, shown verbatim in the UI — never inferred there. */
  note: string
  /** True when the texture was created for this call and the caller owns it. */
  owned: boolean
}

/**
 * The texture holding `nodeId`'s value, creating one if the node is live.
 * Anything created lands in `opts.res`, which the caller disposes.
 */
export async function nodeTextureFor(
  inst: MaterialInstance,
  nodeId: string,
  opts: { res: VramScope; size?: readonly [number, number] },
): Promise<NodeTextureResult> {
  const resolved = inst.resolved.byId.get(nodeId)
  if (!resolved) throw new Error(`node "${nodeId}" is not in the resolved graph (unreachable nodes are dropped)`)

  const slot = inst.slots.find((s) => s.nodeId === nodeId)
  if (slot) {
    return {
      nodeId,
      texture: slot.texture,
      origin: 'resident',
      owned: false,
      note:
        slot.variant === undefined
          ? `bound at @group(1) @binding(${slot.binding}) — this IS the texture the fragment samples`
          : `bound at @group(1) @binding(${slot.binding}), holding the active variant "${slot.variant}" (${slot.sourceNodeId})`,
    }
  }

  const owner = inst.slots.find((s) => s.sourceNodeId === nodeId)
  if (owner) {
    return {
      nodeId,
      texture: owner.texture,
      origin: 'resident',
      owned: false,
      note: `the ACTIVE variant of "${owner.nodeId}" — bound at @group(1) @binding(${owner.binding})`,
    }
  }

  if (resolved.node.kind === 'image') {
    // An inactive variant of a discrete blend: only one is ever resident, so
    // looking at another one means loading it. Through loadNodeImage, so the
    // format/mip decision is the material's own, not the inspector's.
    const loaded = await loadNodeImage({
      res: opts.res,
      device: inst.device,
      shaders: inst.shaders,
      url: resolved.node.url,
      label: `material-inspector/${inst.id}/${nodeId}`,
      spec: resolved.node.spec,
      attr: { tag: 'inspector' },
    })
    return {
      nodeId,
      texture: loaded.texture,
      origin: 'loaded',
      owned: true,
      note:
        `not resident — a discrete variantBlend binds ONE slot and loads only the active variant. ` +
        `Loaded here through the material's own image path just to look at it.`,
    }
  }

  return evaluateNode(inst, nodeId, opts)
}

/**
 * Run a node's generated `cs_materialize` entry point into a fresh texture.
 * This is the same function the fragment shader would have inlined — that
 * duality is the whole design, and it is what makes a live node inspectable
 * without a second implementation of anything.
 */
async function evaluateNode(
  inst: MaterialInstance,
  nodeId: string,
  opts: { res: VramScope; size?: readonly [number, number] },
): Promise<NodeTextureResult> {
  const resolved = inst.resolved.byId.get(nodeId)!
  const generated = generateMaterializeModule({
    materialId: `${inst.id}/inspect`,
    schema: inst.schema,
    resolved: inst.resolved,
    nodeId,
    format: MATERIALIZE_FORMAT,
    shaders: inst.shaders,
  })

  const dependencies = new Map<string, GPUTexture>()
  let inherited: readonly [number, number] | undefined
  for (const dep of generated.dependencies) {
    const tex = inst.slots.find((s) => s.nodeId === dep.node.id)?.texture
    if (!tex) {
      throw new Error(
        `"${nodeId}" cannot be evaluated: its dependency "${dep.node.id}" has no resident texture ` +
          '(an inactive variant of a discrete blend has none by design)',
      )
    }
    dependencies.set(dep.node.id, tex)
    inherited ??= [tex.width, tex.height]
  }

  const [width, height] = opts.size ?? resolved.node.size ?? inherited ?? [DEFAULT_EVAL_SIZE, DEFAULT_EVAL_SIZE]
  const texture = createTexture2D(opts.res, {
    label: `material-inspector/${inst.id}/${nodeId}`,
    size: [width, height],
    format: MATERIALIZE_FORMAT,
    mipLevelCount: mipLevelsFor(width, height),
    encoding: resolved.node.spec.semantics,
    usage: GPUTextureUsage.STORAGE_BINDING,
    attr: { tag: 'inspector' },
  })

  runMaterializePass({
    device: inst.device,
    shaders: inst.shaders,
    frameLayout: inst.frame.layout,
    frameBindGroup: inst.frame.bindGroup,
    sampler: inst.sampler,
    uniformBuffer: inst.uniformBuffer,
    uniformSize: uniformLayout(inst.schema).size,
    generated,
    dependencies,
    target: texture,
    width,
    height,
  })
  generateMips(opts.res, inst.shaders, texture, { semantics: resolved.node.spec.semantics })

  const source = opts.size ? 'inspector-chosen' : resolved.node.size ? 'declared' : inherited ? 'inherited' : 'default'
  return {
    nodeId,
    texture,
    origin: 'evaluated',
    owned: true,
    note:
      `LIVE — no texture exists. Evaluated into a ${width}x${height} ${MATERIALIZE_FORMAT} target (${source} size) by ` +
      `the generated n_${nodeId.replace(/[^A-Za-z0-9_]/g, '_')}_eval, i.e. the exact function the fragment inlines. ` +
      'Nothing was written to the bake cache.',
  }
}

// ---------------------------------------------------------------------------
// 2. Reading it back
// ---------------------------------------------------------------------------

export interface MipLevelData {
  /** The TRUE mip index in the texture — not necessarily this entry's position. */
  level: number
  width: number
  height: number
  /** Tightly packed texels, exactly as stored. */
  data: Uint8Array
  bytesPerTexel: number
}

/**
 * Mip levels read back to the CPU. Everything downstream indexes `levels`
 * POSITIONALLY (`levels[0]` is the first entry read, whatever mip that was);
 * `MipLevelData.level` carries the true mip index for display. The two
 * coincide for a full chain and deliberately do not for a single-level read —
 * a thumbnail wants one 128² level, not 21 MB of readback.
 */
export interface MipChain {
  format: GPUTextureFormat
  /** Channels the format actually stores (1, 2 or 4). */
  components: number
  levels: MipLevelData[]
  /** Bytes these levels occupy in VRAM. */
  bytes: number
}

interface FormatInfo {
  components: number
  sample: 'unorm8' | 'snorm8' | 'float16' | 'float32'
  bgra: boolean
  srgb: boolean
}

const FORMAT_INFO: Record<string, FormatInfo> = {
  r8unorm: { components: 1, sample: 'unorm8', bgra: false, srgb: false },
  rg8unorm: { components: 2, sample: 'unorm8', bgra: false, srgb: false },
  rgba8unorm: { components: 4, sample: 'unorm8', bgra: false, srgb: false },
  'rgba8unorm-srgb': { components: 4, sample: 'unorm8', bgra: false, srgb: true },
  bgra8unorm: { components: 4, sample: 'unorm8', bgra: true, srgb: false },
  'bgra8unorm-srgb': { components: 4, sample: 'unorm8', bgra: true, srgb: true },
  r8snorm: { components: 1, sample: 'snorm8', bgra: false, srgb: false },
  rgba8snorm: { components: 4, sample: 'snorm8', bgra: false, srgb: false },
  r16float: { components: 1, sample: 'float16', bgra: false, srgb: false },
  rg16float: { components: 2, sample: 'float16', bgra: false, srgb: false },
  rgba16float: { components: 4, sample: 'float16', bgra: false, srgb: false },
  r32float: { components: 1, sample: 'float32', bgra: false, srgb: false },
  rg32float: { components: 2, sample: 'float32', bgra: false, srgb: false },
  rgba32float: { components: 4, sample: 'float32', bgra: false, srgb: false },
}

export function formatInfoOf(format: GPUTextureFormat): FormatInfo {
  const info = FORMAT_INFO[format]
  if (!info) {
    throw new Error(
      `the material inspector cannot read ${format} back yet — add it to FORMAT_INFO in src/material/inspect.ts`,
    )
  }
  return info
}

/** Read mip levels of a texture back to the CPU — the whole chain by default. */
export async function readMipChain(
  device: GPUDevice,
  tex: GPUTexture,
  opts: { exempt?: <T>(fn: () => T) => T; levels?: readonly number[] } = {},
): Promise<MipChain> {
  const info = formatInfoOf(tex.format)
  const wanted = opts.levels ?? [...Array(tex.mipLevelCount).keys()]
  const levels: MipLevelData[] = []
  let bytes = 0
  for (const level of wanted) {
    const read = await readTexture(device, tex, { mipLevel: level, ...(opts.exempt && { exempt: opts.exempt }) })
    levels.push({
      level,
      width: read.width,
      height: read.height,
      data: read.data,
      bytesPerTexel: read.bytesPerTexel,
    })
    bytes += read.width * read.height * read.bytesPerTexel
  }
  return { format: tex.format, components: info.components, levels, bytes }
}

export type Rgba = [number, number, number, number]

function halfToFloat(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1
  const exponent = (bits >> 10) & 0x1f
  const fraction = bits & 0x3ff
  if (exponent === 0) return sign * fraction * 2 ** -24
  if (exponent === 0x1f) return fraction ? Number.NaN : sign * Infinity
  return sign * (fraction + 1024) * 2 ** (exponent - 25)
}

/**
 * Fills `out` with the stored value at a texel, as the sampler sees it BEFORE
 * the generated decoder runs. Channels the format does not store read
 * (0, 0, 1) — the same convention `TexelSemantics` documents for alpha.
 *
 * A closure over pre-built typed-array views rather than a plain function: a
 * 2048² chain is 5.6M texels, and a DataView per texel is the difference
 * between "instant" and "the tab stalls".
 */
export type TexelReader = (level: number, x: number, y: number, out: Rgba) => Rgba

export function texelReaderFor(chain: MipChain): TexelReader {
  const info = formatInfoOf(chain.format)
  const views = chain.levels.map((l) =>
    info.sample === 'float32'
      ? new Float32Array(l.data.buffer, l.data.byteOffset, l.data.byteLength / 4)
      : info.sample === 'float16'
        ? new Uint16Array(l.data.buffer, l.data.byteOffset, l.data.byteLength / 2)
        : l.data,
  )
  const n = info.components
  return (level, x, y, out) => {
    const l = chain.levels[level]
    const view = views[level]
    out[0] = 0
    out[1] = 0
    out[2] = 0
    out[3] = 1
    if (!l || !view) return out
    const cx = x < 0 ? 0 : x >= l.width ? l.width - 1 : x | 0
    const cy = y < 0 ? 0 : y >= l.height ? l.height - 1 : y | 0
    const base = (cy * l.width + cx) * n
    for (let c = 0; c < n; c++) {
      const raw = view[base + c]!
      out[c] =
        info.sample === 'unorm8'
          ? raw / 255
          : info.sample === 'snorm8'
            ? Math.max(-1, (raw > 127 ? raw - 256 : raw) / 127)
            : info.sample === 'float16'
              ? halfToFloat(raw)
              : raw
    }
    if (info.bgra) {
      const r = out[0]
      out[0] = out[2]
      out[2] = r
    }
    return out
  }
}

/** One texel, allocating. For a probe, never for a loop over an image. */
export function texelAt(chain: MipChain, level: number, x: number, y: number): Rgba {
  return texelReaderFor(chain)(level, x, y, [0, 0, 0, 1])
}

/**
 * The generated decoder, in TypeScript. Deliberately the same four cases as
 * `decoderBody()` in src/gpu/texture.ts: this is what the shader sees after
 * the read, and showing it beside the stored bytes is the whole point of the
 * probe — the difference between the two IS the semantics.
 */
export function decodeTexel(semantics: TexelSemantics, t: Rgba): Rgba {
  switch (semantics.kind) {
    case 'color-with-coverage':
      return [...t] as Rgba
    case 'premultiplied-color': {
      const a = Math.max(t[3], 1e-5)
      return [t[0] / a, t[1] / a, t[2] / a, t[3]]
    }
    case 'unit-vector': {
      const v: [number, number, number] = [t[0] * 2 - 1, t[1] * 2 - 1, t[2] * 2 - 1]
      const len = Math.hypot(v[0], v[1], v[2])
      if (len <= 1e-5) {
        const h = semantics.hemisphere
        const hl = Math.hypot(h[0], h[1], h[2]) || 1
        return [h[0] / hl, h[1] / hl, h[2] / hl, t[3]]
      }
      return [v[0] / len, v[1] / len, v[2] / len, t[3]]
    }
    case 'scalar-field':
    case 'mask':
      return [...t] as Rgba
  }
}

// ---------------------------------------------------------------------------
// 3. Per-level statistics — the mip-convention check, as numbers
// ---------------------------------------------------------------------------

export interface LevelStats {
  level: number
  width: number
  height: number
  /** Mean stored luma — rgb-weighted for a vec3/vec4 node, r for a scalar. */
  luma: number
  /** Σ(luma·a)/Σa — flat across levels iff the colour is coverage-normalised. */
  coveredLuma: number
  /** mean(a). Exactly preserved by a box filter. */
  meanAlpha: number
  min: number
  max: number
}

/**
 * `type` is the NODE's channel type, not the texture's channel count, and the
 * difference matters: a scalar node materializes into rgba8unorm (the only
 * 8-bit storage format WebGPU core allows), so weighting its single value as
 * rgb luma would report 0.2126x the truth and make the same quantity
 * incomparable between its image form (r8unorm) and its baked form.
 */
export function chainStats(chain: MipChain, type: ChannelType): LevelStats[] {
  const colorish = CHANNEL_COMPONENTS[type] >= 3
  const read = texelReaderFor(chain)
  const t: Rgba = [0, 0, 0, 1]
  return chain.levels.map((l, index) => {
    let lumaSum = 0
    let lumaA = 0
    let aSum = 0
    let min = Infinity
    let max = -Infinity
    for (let y = 0; y < l.height; y++) {
      for (let x = 0; x < l.width; x++) {
        read(index, x, y, t)
        const luma = colorish ? 0.2126 * t[0] + 0.7152 * t[1] + 0.0722 * t[2] : t[0]
        lumaSum += luma
        lumaA += luma * t[3]
        aSum += t[3]
        if (luma < min) min = luma
        if (luma > max) max = luma
      }
    }
    const n = Math.max(1, l.width * l.height)
    return {
      level: l.level,
      width: l.width,
      height: l.height,
      luma: lumaSum / n,
      coveredLuma: aSum > 0 ? lumaA / aSum : 0,
      meanAlpha: aSum / n,
      min: min === Infinity ? 0 : min,
      max: max === -Infinity ? 0 : max,
    }
  })
}

/** One line saying what the numbers above should look like, and whether they do. */
export function statsVerdict(semantics: TexelSemantics, stats: readonly LevelStats[]): string {
  const base = stats[0]
  if (!base || stats.length < 2) return 'single level — nothing to compare'
  const drift = (pick: (s: LevelStats) => number): number => {
    const b = pick(base)
    if (Math.abs(b) < 1e-5) return 0
    return Math.max(...stats.map((s) => Math.abs(pick(s) - b) / Math.abs(b)))
  }
  const lumaDrift = drift((s) => s.coveredLuma)
  const alphaDrift = drift((s) => s.meanAlpha)
  const pct = (v: number): string => `${(v * 100).toFixed(1)}%`
  switch (semantics.kind) {
    case 'color-with-coverage':
      return (
        `coverage-weighted: Σ(luma·a)/Σa must be FLAT across levels (the filter already normalised by coverage, so ` +
        `the decoder must not divide again). Worst drift ${pct(lumaDrift)}.`
      )
    case 'premultiplied-color':
      return (
        `premultiplied: stored luma must FALL in step with coverage — the decoder divides it back out. ` +
        `luma drift ${pct(lumaDrift)}, alpha drift ${pct(alphaDrift)}; flat luma over falling alpha is the 017 bug.`
      )
    case 'unit-vector':
      return `unit vectors: filtered by flipping into the hemisphere, averaging and renormalising. Level luma drift ${pct(lumaDrift)}.`
    case 'scalar-field':
      return `scalar field (holes: ${semantics.empty}): the mean should track the field, not sink toward zero. Drift ${pct(lumaDrift)}.`
    case 'mask':
      return `mask (${semantics.reduce}): ${semantics.reduce === 'max' ? 'conservative — thin features survive, so the mean RISES' : 'honest area average'}. Drift ${pct(lumaDrift)}.`
  }
}

// ---------------------------------------------------------------------------
// Display bytes (RGBA8, for an ImageData the view owns)
// ---------------------------------------------------------------------------

export type ChannelPick = 'composite' | 'r' | 'g' | 'b' | 'a'

/**
 * One mip level as RGBA8 display bytes. `composite` shows the node's value the
 * way its TYPE means it (a scalar as grey, a vec3 as colour); an isolated
 * channel is always grey, so R of a normal map and R of an albedo read the
 * same way.
 */
export function levelToRgba8(
  chain: MipChain,
  level: number,
  channel: ChannelPick,
  type: ChannelType,
  decode: TexelSemantics | null,
): { data: Uint8ClampedArray<ArrayBuffer>; width: number; height: number } {
  const l = chain.levels[level]
  if (!l) return { data: new Uint8ClampedArray(4), width: 1, height: 1 }
  const out = new Uint8ClampedArray(l.width * l.height * 4)
  const scalar = type === 'f32'
  const read = texelReaderFor(chain)
  const raw: Rgba = [0, 0, 0, 1]
  for (let y = 0; y < l.height; y++) {
    for (let x = 0; x < l.width; x++) {
      read(level, x, y, raw)
      const t = decode ? decodeTexel(decode, raw) : raw
      const d = (y * l.width + x) * 4
      let r: number
      let g: number
      let b: number
      if (channel === 'composite') {
        // A decoded unit vector is in [-1,1]; show it the way a normal map is
        // always shown, remapped, rather than clipping half of it to black.
        const bias = decode?.kind === 'unit-vector' ? 0.5 : 0
        const gain = decode?.kind === 'unit-vector' ? 0.5 : 1
        r = scalar ? t[0] : t[0] * gain + bias
        g = scalar ? t[0] : t[1] * gain + bias
        b = scalar ? t[0] : type === 'vec2f' ? 0 : t[2] * gain + bias
      } else {
        const v = channel === 'r' ? raw[0] : channel === 'g' ? raw[1] : channel === 'b' ? raw[2] : raw[3]
        r = v
        g = v
        b = v
      }
      out[d] = r * 255
      out[d + 1] = g * 255
      out[d + 2] = b * 255
      out[d + 3] = 255
    }
  }
  return { data: out, width: l.width, height: l.height }
}

/** Bytes a texture occupies in VRAM, mips included. */
export function textureBytesOf(tex: GPUTexture): number {
  const info = FORMAT_INFO[tex.format]
  const perTexel = info
    ? info.components * (info.sample === 'float32' ? 4 : info.sample === 'float16' ? 2 : 1)
    : 4
  let bytes = 0
  let w = tex.width
  let h = tex.height
  for (let level = 0; level < tex.mipLevelCount; level++) {
    bytes += w * h * perTexel
    w = Math.max(1, w >> 1)
    h = Math.max(1, h >> 1)
  }
  return bytes
}
