/**
 * Texel semantics, mip generation and the WGSL decoder that matches it.
 *
 * ONE declaration produces BOTH the mip filter and the shader-side decoder, so
 * the two cannot disagree. Twenty experiments hand-rolled this and it produced
 * two documented classes of wrong result:
 *
 *   - un-premultiplying colour that was ALREADY normalised by coverage
 *     (017-cached-clusters was inflated 2.42x at its deepest mip);
 *   - box-filtering OCTAHEDRAL normals, which lands near (0,0) and decodes to
 *     exactly straight up (017 lit its whole far field with an up-normal, and
 *     `debug=albedo` looked perfect the whole time).
 *
 * Both are made unrepresentable here rather than documented:
 *   - the un-premultiply divide exists ONLY inside the generated decoder;
 *   - `TexelSemantics` has no octahedral member at all, and asking
 *     `createTexture2D` for a mipped octahedral texture throws at init.
 *
 * DEV additionally runs the check CLAUDE.md prescribes by hand — stored rgb
 * luma per level — and warns with the per-level numbers when it drifts.
 */

import type { ShaderRegistry } from './shaders.ts'
import type { VramAttr, VramScope } from './resources.ts'
import { textureFormatBytes } from './resources.ts'

// ---------------------------------------------------------------------------
// Semantics
// ---------------------------------------------------------------------------

export type TexelSemantics =
  /**
   * Straight (un-premultiplied) colour in rgb, coverage in a, filtered as
   * `rgb = Σ(rgb·w)/Σw`, `a = Σa/4`. The stored rgb is ALREADY normalised by
   * coverage at every level — the decoder must NOT divide by alpha again.
   */
  | { kind: 'color-with-coverage'; weight: 'a' }
  /**
   * Premultiplied colour (`rgb = C·a`) filtered with a plain box. The decoder
   * DOES divide by alpha to recover C.
   */
  | { kind: 'premultiplied-color' }
  /**
   * Plain unit vectors stored as `xyz*0.5+0.5`. Filtered by flipping each
   * sample into `hemisphere`, averaging, and renormalising. This is the ONLY
   * mip-safe way to store normals — see the octahedral note above.
   */
  | { kind: 'unit-vector'; hemisphere: [number, number, number] }
  /**
   * A scalar (height, thickness, SDF, …) in r (and g/b as further independent
   * scalars), validity in a. `empty: 'zero'` averages holes in as 0;
   * `empty: 'nearest'` averages only the valid samples so the field does not
   * sink into its holes (falling back to the plain mean when all four are
   * empty). Formats without an alpha channel read a = 1, so the two agree.
   */
  | { kind: 'scalar-field'; empty: 'zero' | 'nearest' }
  /**
   * An occupancy/coverage mask. `mean` is the honest area average; `max` is
   * the conservative reduction that keeps thin features alive through the
   * chain (use it when the mask gates a ray march, not when it weights a blend).
   */
  | { kind: 'mask'; reduce: 'mean' | 'max' }

/**
 * Non-mip-averageable encodings. They are deliberately NOT part of
 * `TexelSemantics`, so `MipPlan` can never name one; `createTexture2D` throws
 * if you ask for mips anyway.
 */
export type NonMippableEncoding = { kind: 'octahedral-normal' }

export type TextureEncoding = TexelSemantics | NonMippableEncoding

export interface MipPlan {
  semantics: TexelSemantics
  /** Levels to fill, including level 0. Defaults to the texture's own count. */
  levels?: number
}

// ---------------------------------------------------------------------------
// Texture creation
// ---------------------------------------------------------------------------

/** Full mip chain length for a 2D texture. */
export function mipLevelsFor(width: number, height: number): number {
  return 1 + Math.floor(Math.log2(Math.max(1, Math.max(width, height))))
}

export interface Texture2DDesc {
  label: string
  size: readonly [number, number]
  format: GPUTextureFormat
  /** Default 1. `mipLevelsFor(w, h)` for a full chain. */
  mipLevelCount?: number
  /**
   * What the texels MEAN. Pass the same value you will pass to
   * `generateMips`/`wgslDecoder`; it is what makes the octahedral guard fire.
   */
  encoding?: TextureEncoding
  /**
   * Extra usage flags. TEXTURE_BINDING | COPY_DST | COPY_SRC |
   * RENDER_ATTACHMENT are always included: COPY_DST for `writeTexture`,
   * RENDER_ATTACHMENT for `copyExternalImageToTexture` and the mip filter
   * (omitting either is a validation error, and WebGPU validation errors do
   * NOT throw — they arrive as a toast), COPY_SRC for readback and the DEV
   * convention self-check.
   */
  usage?: number
  attr?: VramAttr
}

const BASE_TEXTURE_USAGE = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC

/**
 * Colour formats that cannot be a render attachment in WebGPU core. Adding
 * RENDER_ATTACHMENT to one is a validation error — which does NOT throw, it
 * arrives as a toast — so we leave the flag off and say so up front if mips
 * were requested (the shared filter renders into each level).
 */
const NON_RENDERABLE = new Set<GPUTextureFormat>([
  'r8snorm',
  'rg8snorm',
  'rgba8snorm',
  'rgb9e5ufloat',
  'rg11b10ufloat',
])

const OCTAHEDRAL_MIP_ERROR = [
  'octahedral normals are NOT mip-averageable, so a mipped octahedral texture cannot be correct.',
  'Octahedral encoding is non-linear and a blade\'s front/back faces encode to roughly OPPOSITE pairs;',
  'box-filtering them lands near (0,0), which decodes to *exactly* straight up. 017-cached-clusters lit',
  'its entire far field with that up-normal — flat, blown out, worse with distance — while debug=albedo',
  'looked perfect throughout, so nobody caught it for a week.',
  '',
  'Fix: store plain unit vectors instead —',
  "  createTexture2D(res, { ..., encoding: { kind: 'unit-vector', hemisphere: [0, 1, 0] } })",
  'which filters in a linear space (flip into the hemisphere, average, renormalise) and whose decoder',
  'is generated to match. Octahedral is still fine for NON-mipped storage: wgslOctahedral()/octEncode().',
].join('\n')

export function createTexture2D(res: VramScope, desc: Texture2DDesc): GPUTexture {
  const levels = desc.mipLevelCount ?? 1
  if (desc.encoding?.kind === 'octahedral-normal' && levels > 1) {
    throw new Error(`createTexture2D("${desc.label}"): mipLevelCount=${levels} with octahedral-normal — ${OCTAHEDRAL_MIP_ERROR}`)
  }
  const [width, height] = desc.size
  const maxLevels = mipLevelsFor(width, height)
  if (levels > maxLevels) {
    throw new Error(`createTexture2D("${desc.label}"): mipLevelCount=${levels} exceeds ${maxLevels} for ${width}x${height}`)
  }
  const renderable = !NON_RENDERABLE.has(desc.format)
  if (!renderable && levels > 1) {
    throw new Error(
      `createTexture2D("${desc.label}"): ${desc.format} cannot be a render attachment, so generateMips cannot fill its levels. ` +
        'Pick a renderable format (rgba8unorm, rgba16float, r32float, …) or build the chain on the CPU and upload every level.',
    )
  }
  return res.createTexture(
    {
      label: desc.label,
      size: [width, height],
      format: desc.format,
      mipLevelCount: levels,
      usage: BASE_TEXTURE_USAGE | (renderable ? GPUTextureUsage.RENDER_ATTACHMENT : 0) | (desc.usage ?? 0),
    },
    desc.attr ?? {},
  )
}

/**
 * Upload an ImageBitmap into `tex`. The bitmap MUST have been decoded with
 * `{ premultiplyAlpha: 'none', colorSpaceConversion: 'none' }` (see
 * `loadImageTexture`) — the createImageBitmap defaults premultiply and apply
 * the display profile, which is the 017 colour-convention bug arriving through
 * a new door.
 */
export function uploadImageBitmap(
  device: GPUDevice,
  tex: GPUTexture,
  bitmap: ImageBitmap,
  opts: { flipY?: boolean; mipLevel?: number; origin?: [number, number] } = {},
): void {
  device.queue.copyExternalImageToTexture(
    { source: bitmap, flipY: opts.flipY ?? false },
    {
      texture: tex,
      mipLevel: opts.mipLevel ?? 0,
      origin: opts.origin ?? [0, 0],
      premultipliedAlpha: false,
      colorSpace: 'srgb',
    },
    [bitmap.width, bitmap.height],
  )
}

// ---------------------------------------------------------------------------
// Mip generation
// ---------------------------------------------------------------------------

const UINT_FORMAT = /(?:uint|sint)$/

export function semanticsTag(s: TexelSemantics): string {
  switch (s.kind) {
    case 'color-with-coverage':
      return `color-with-coverage-${s.weight}`
    case 'premultiplied-color':
      return 'premultiplied-color'
    case 'unit-vector':
      return `unit-vector-${s.hemisphere.map((v) => v.toFixed(3)).join('_')}`
    case 'scalar-field':
      return `scalar-field-${s.empty}`
    case 'mask':
      return `mask-${s.reduce}`
  }
}

function f32(v: number): string {
  return Number.isInteger(v) ? `${v}.0` : String(v)
}

/** The 2x2 reduction for one semantics kind, as a WGSL fragment body. */
function mipFilterBody(s: TexelSemantics): string {
  switch (s.kind) {
    case 'color-with-coverage':
      return `
  // rgb = Σ(rgb·a)/Σa — the stored colour stays coverage-normalised at every
  // level, so it is ALREADY straight colour and must not be divided again.
  // A fully empty 2x2 keeps the plain mean instead of going black, so the
  // dilation bleed survives up the chain rather than being eaten at level 1.
  let w = vec4f(t0.a, t1.a, t2.a, t3.a);
  let wsum = w.x + w.y + w.z + w.w;
  let weighted = (t0.rgb * w.x + t1.rgb * w.y + t2.rgb * w.z + t3.rgb * w.w) / max(wsum, 1e-6);
  let plain = (t0.rgb + t1.rgb + t2.rgb + t3.rgb) * 0.25;
  return vec4f(select(plain, weighted, wsum > 1e-6), wsum * 0.25);`
    case 'premultiplied-color':
      return `
  // Plain box over premultiplied colour: mean(rgb) and mean(a) are both exactly
  // preserved, and the DECODER divides rgb by a to recover straight colour.
  return (t0 + t1 + t2 + t3) * 0.25;`
    case 'unit-vector': {
      const h = `vec3f(${s.hemisphere.map(f32).join(', ')})`
      return `
  // Flip into the capture hemisphere, average, renormalise. Averaging raw unit
  // vectors is linear and therefore mip-safe; averaging an octahedral pair is
  // NOT (it lands at (0,0) = straight up). See the module header.
  let h = ${h};
  var acc = mip_flip(t0.xyz * 2.0 - 1.0, h)
          + mip_flip(t1.xyz * 2.0 - 1.0, h)
          + mip_flip(t2.xyz * 2.0 - 1.0, h)
          + mip_flip(t3.xyz * 2.0 - 1.0, h);
  let len = length(acc);
  let n = select(normalize(h), acc / len, len > 1e-6);
  return vec4f(n * 0.5 + 0.5, (t0.a + t1.a + t2.a + t3.a) * 0.25);`
    }
    case 'scalar-field':
      return s.empty === 'zero'
        ? `
  // Holes are genuinely zero — average them in.
  return (t0 + t1 + t2 + t3) * 0.25;`
        : `
  // Holes carry no value: average only the valid samples (a > 0) so the field
  // does not sink toward zero as the chain deepens.
  let w = vec4f(t0.a, t1.a, t2.a, t3.a);
  let wsum = w.x + w.y + w.z + w.w;
  let plain = (t0 + t1 + t2 + t3) * 0.25;
  let valid = (t0 * w.x + t1 * w.y + t2 * w.z + t3 * w.w) / max(wsum, 1e-6);
  let v = select(plain.rgb, valid.rgb, wsum > 1e-6);
  return vec4f(v, wsum * 0.25);`
    case 'mask':
      return s.reduce === 'mean'
        ? `
  return (t0 + t1 + t2 + t3) * 0.25;`
        : `
  // Conservative: a thin feature that survives at level 0 survives everywhere.
  return max(max(t0, t1), max(t2, t3));`
  }
}

/**
 * The scaffolding every mip filter shares: the 2x2 gather and the hemisphere
 * flip. A CONSTANT rather than inline text because the portable material export
 * (src/material/export.ts) emits the same filter into its bundle — the mip
 * convention is the one thing a consumer cannot re-derive from the texels, so
 * it travels as CODE, from this single source, not as prose.
 */
export const MIP_FILTER_HELPERS = `fn mip_texel(base: vec2i, dx: i32, dy: i32) -> vec4f {
  let dims = vec2i(textureDimensions(mip_src));
  return textureLoad(mip_src, min(base + vec2i(dx, dy), dims - vec2i(1)), 0);
}

fn mip_flip(v: vec3f, h: vec3f) -> vec3f {
  return select(-v, v, dot(v, h) >= 0.0);
}
`

/**
 * The 2x2 reduction for one semantics kind, as a standalone WGSL function
 * `fn <name>_reduce(t0, t1, t2, t3: vec4f) -> vec4f`. Needs `MIP_FILTER_HELPERS`
 * in scope (the unit-vector reduction calls `mip_flip`).
 */
export function wgslMipReduce(name: string, s: TexelSemantics): string {
  return `// GENERATED by wgslMipReduce("${name}") — 2x2 reduction for ${semanticsTag(s)}.
fn ${name}_reduce(t0: vec4f, t1: vec4f, t2: vec4f, t3: vec4f) -> vec4f {
${mipFilterBody(s)}
}
`
}

function mipFilterWgsl(s: TexelSemantics): string {
  return `// GENERATED by src/gpu/texture.ts — mip filter for ${semanticsTag(s)}.
// Do not hand-edit; change the TexelSemantics declaration instead.

@group(0) @binding(0) var mip_src: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var tri = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(3.0, 1.0), vec2f(-1.0, 1.0));
  return vec4f(tri[vi], 0.0, 1.0);
}

${MIP_FILTER_HELPERS}
${wgslMipReduce('mip', s)}
@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let base = vec2i(pos.xy) * 2;
  let t0 = mip_texel(base, 0, 0);
  let t1 = mip_texel(base, 1, 0);
  let t2 = mip_texel(base, 0, 1);
  let t3 = mip_texel(base, 1, 1);
  return mip_reduce(t0, t1, t2, t3);
}
`
}

interface MipPipeline {
  bgl: GPUBindGroupLayout
  pipeline: GPURenderPipeline
}

const mipPipelineCache = new WeakMap<GPUDevice, Map<string, MipPipeline>>()

function mipPipeline(device: GPUDevice, shaders: ShaderRegistry, s: TexelSemantics, format: GPUTextureFormat): MipPipeline {
  let byKey = mipPipelineCache.get(device)
  if (!byKey) mipPipelineCache.set(device, (byKey = new Map()))
  const tag = semanticsTag(s)
  const key = `${tag}|${format}`
  const hit = byKey.get(key)
  if (hit) return hit

  // The shader id must be unique per generated variant: ShaderRegistry.module()
  // is keyed by id and returns the FIRST code it saw for that id forever.
  const module = shaders.module({ id: `gpu/texture.ts#mipgen/${tag}`, code: mipFilterWgsl(s) })
  const bgl = device.createBindGroupLayout({
    label: `mipgen/${tag}`,
    // 'unfilterable-float' accepts filterable-float formats too, and covers
    // r32float without the float32-filterable feature. textureLoad only.
    entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } }],
  })
  const pipeline = device.createRenderPipeline({
    label: `mipgen/${key}`,
    layout: device.createPipelineLayout({ label: `mipgen/${key}`, bindGroupLayouts: [bgl] }),
    vertex: { module, entryPoint: 'vs' },
    fragment: { module, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  })
  const made = { bgl, pipeline }
  byKey.set(key, made)
  return made
}

/**
 * Fill mip levels 1..N-1 of `tex` from level 0, using the filter that `plan`
 * names. Submits its own command buffer.
 *
 * In DEV this also runs the stored-luma convention check (see
 * `checkMipConvention`) and warns with the per-level numbers on drift.
 */
export function generateMips(res: VramScope, shaders: ShaderRegistry, tex: GPUTexture, plan: MipPlan): void {
  const device = res.device
  if (UINT_FORMAT.test(tex.format)) {
    throw new Error(
      `generateMips("${tex.label}"): format ${tex.format} is an integer format; the shared filter is float-only. ` +
        `Store a data map as r32float/rgba16float if it needs mips, or read the integer texture with textureLoad and no mips.`,
    )
  }
  const levels = Math.min(plan.levels ?? tex.mipLevelCount, tex.mipLevelCount)
  if (levels <= 1) return
  if ((tex.usage & GPUTextureUsage.RENDER_ATTACHMENT) === 0) {
    throw new Error(`generateMips("${tex.label}"): texture lacks RENDER_ATTACHMENT usage (use createTexture2D)`)
  }

  const { bgl, pipeline } = mipPipeline(device, shaders, plan.semantics, tex.format)
  const enc = device.createCommandEncoder({ label: `mipgen/${tex.label}` })
  for (let level = 1; level < levels; level++) {
    const pass = enc.beginRenderPass({
      label: `mipgen/${tex.label}/${level}`,
      colorAttachments: [
        {
          view: tex.createView({ baseMipLevel: level, mipLevelCount: 1 }),
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    })
    pass.setPipeline(pipeline)
    pass.setBindGroup(
      0,
      device.createBindGroup({
        label: `mipgen/${tex.label}/${level}`,
        layout: bgl,
        entries: [{ binding: 0, resource: tex.createView({ baseMipLevel: level - 1, mipLevelCount: 1 }) }],
      }),
    )
    pass.draw(3)
    pass.end()
  }
  device.queue.submit([enc.finish()])

  if (import.meta.env.DEV) {
    void checkMipConvention(res, tex, plan.semantics, levels).then((report) => {
      if (report && !report.ok) console.warn(formatMipCheck(tex.label, report))
    })
  }
}

// ---------------------------------------------------------------------------
// The DEV convention self-check
// ---------------------------------------------------------------------------

export interface MipCheckLevel {
  level: number
  width: number
  height: number
  /** Σ(luma·a)/Σa — the mean stored luma over the COVERED area. */
  coveredLuma: number
  /** Σ(a·a)/Σa — the mean coverage a covered texel actually has. */
  coveredAlpha: number
  /** mean(a) over the whole level. Exactly preserved by a box filter. */
  meanAlpha: number
}

export interface MipCheckReport {
  semantics: string
  ok: boolean
  /** Worst relative drift of the invariant, as a fraction (0.02 = 2%). */
  drift: number
  message: string
  levels: MipCheckLevel[]
}

const CHECK_TOLERANCE = 0.02

function formatMipCheck(label: string, r: MipCheckReport): string {
  const rows = r.levels
    .map(
      (l) =>
        `    L${l.level} ${String(l.width).padStart(5)}x${String(l.height).padEnd(5)} ` +
        `coveredLuma=${l.coveredLuma.toFixed(4)}  coveredAlpha=${l.coveredAlpha.toFixed(4)}  meanAlpha=${l.meanAlpha.toFixed(4)}`,
    )
    .join('\n')
  return `[mip] "${label}" declared ${r.semantics}: ${r.message} (drift ${(r.drift * 100).toFixed(1)}%)\n${rows}`
}

const SRGB_FORMAT = /-srgb$/
const CHECKABLE_FORMAT = /^(?:rgba8unorm|bgra8unorm)(?:-srgb)?$/

function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
}

/**
 * The test CLAUDE.md prescribes by hand, automated: read the stored rgb luma
 * per mip level and check it against the invariant of the declared semantics.
 *
 *   - `color-with-coverage`: Σ(luma·a)/Σa is EXACTLY preserved by the filter,
 *     so it must be flat across levels.
 *   - `premultiplied-color`: the stored rgb is C·a, so Σ(luma·a)/Σa must FALL
 *     in step with the covered alpha; `coveredLuma / coveredAlpha` is what
 *     stays flat. If the luma is flat while the alpha falls, the data is
 *     straight/coverage-weighted, not premultiplied — and the generated decoder
 *     is about to divide it by alpha a second time. That is the 017 bug.
 *
 * Returns null when nothing can be measured (no alpha semantics, unsupported
 * format, or the texture lacks COPY_SRC).
 *
 * What each direction is actually worth (measured, not assumed):
 *  - For a chain THIS module generated, `color-with-coverage` satisfies its
 *    own invariant by construction — Σ(rgb·a)/Σa is algebraically exact for a
 *    coverage-weighted 2x2 — so that branch is a regression guard on the
 *    filter, not a detector of wrong input data.
 *  - `premultiplied-color` is a genuine data detector: the statistic is not
 *    preserved by a box filter, so straight (or dilated) colour declared
 *    premultiplied shows up as flat luma over falling coverage. That is the
 *    dangerous direction, because it is the one where the decoder divides.
 *  - On a chain built somewhere ELSE (a CPU mip loop in a bake, all levels
 *    uploaded by hand — which is what most experiments do) BOTH directions
 *    detect a mismatch. Call this explicitly after such an upload.
 *
 * Note also that with hard alpha (0 or 255) and black holes, straight and
 * premultiplied storage are the SAME bytes, so nothing can distinguish them —
 * and nothing needs to. The conventions only diverge once alpha is fractional
 * or the holes carry dilated colour.
 */
export async function checkMipConvention(
  res: VramScope,
  tex: GPUTexture,
  semantics: TexelSemantics,
  levels = tex.mipLevelCount,
): Promise<MipCheckReport | null> {
  if (semantics.kind !== 'color-with-coverage' && semantics.kind !== 'premultiplied-color') return null
  if (levels < 2) return null
  if (!CHECKABLE_FORMAT.test(tex.format)) return null
  if ((tex.usage & GPUTextureUsage.COPY_SRC) === 0) return null

  const device = res.device
  const bgra = tex.format.startsWith('bgra')
  const linearize = SRGB_FORMAT.test(tex.format)

  interface Region {
    offset: number
    bytesPerRow: number
    width: number
    height: number
  }
  const regions: Region[] = []
  let total = 0
  for (let level = 0; level < levels; level++) {
    const width = Math.max(1, tex.width >> level)
    const height = Math.max(1, tex.height >> level)
    const bytesPerRow = Math.ceil((width * 4) / 256) * 256
    regions.push({ offset: total, bytesPerRow, width, height })
    total += bytesPerRow * height
  }

  const readback = res.exempt(() =>
    device.createBuffer({
      label: `mipcheck/${tex.label}`,
      size: total,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    }),
  )
  const enc = device.createCommandEncoder({ label: `mipcheck/${tex.label}` })
  for (let level = 0; level < levels; level++) {
    const r = regions[level]!
    enc.copyTextureToBuffer(
      { texture: tex, mipLevel: level },
      { buffer: readback, offset: r.offset, bytesPerRow: r.bytesPerRow, rowsPerImage: r.height },
      [r.width, r.height],
    )
  }
  device.queue.submit([enc.finish()])
  await readback.mapAsync(GPUMapMode.READ)
  const bytes = new Uint8Array(readback.getMappedRange()).slice()
  readback.unmap()
  readback.destroy()

  const out: MipCheckLevel[] = []
  for (let level = 0; level < levels; level++) {
    const r = regions[level]!
    let lumaA = 0
    let aa = 0
    let aSum = 0
    let n = 0
    for (let y = 0; y < r.height; y++) {
      for (let x = 0; x < r.width; x++) {
        const s = r.offset + y * r.bytesPerRow + x * 4
        let cr = bytes[bgra ? s + 2 : s]! / 255
        let cg = bytes[s + 1]! / 255
        let cb = bytes[bgra ? s : s + 2]! / 255
        const a = bytes[s + 3]! / 255
        if (linearize) {
          cr = srgbToLinear(cr)
          cg = srgbToLinear(cg)
          cb = srgbToLinear(cb)
        }
        const luma = 0.2126 * cr + 0.7152 * cg + 0.0722 * cb
        lumaA += luma * a
        aa += a * a
        aSum += a
        n++
      }
    }
    out.push({
      level,
      width: r.width,
      height: r.height,
      coveredLuma: aSum > 0 ? lumaA / aSum : 0,
      coveredAlpha: aSum > 0 ? aa / aSum : 0,
      meanAlpha: n > 0 ? aSum / n : 0,
    })
  }

  const base = out[0]!
  const tag = semanticsTag(semantics)
  if (base.coveredLuma <= 1e-5) {
    return { semantics: tag, ok: true, drift: 0, message: 'level 0 is black — nothing to check', levels: out }
  }

  if (semantics.kind === 'color-with-coverage') {
    let drift = 0
    for (const l of out) drift = Math.max(drift, Math.abs(l.coveredLuma - base.coveredLuma) / base.coveredLuma)
    return {
      semantics: tag,
      ok: drift <= CHECK_TOLERANCE,
      drift,
      message:
        drift <= CHECK_TOLERANCE
          ? 'stored luma is flat across levels, as coverage-weighted filtering requires'
          : 'stored luma is NOT flat across levels. Either the source is premultiplied (then declare ' +
            "{ kind: 'premultiplied-color' }), or the field is genuinely losing colour with depth. A decoder " +
            'that divides this by alpha would inflate the far field — that is the 017-cached-clusters bug.',
      levels: out,
    }
  }

  // premultiplied-color
  const baseRatio = base.coveredLuma / Math.max(base.coveredAlpha, 1e-6)
  let ratioDrift = 0
  for (const l of out) {
    ratioDrift = Math.max(ratioDrift, Math.abs(l.coveredLuma / Math.max(l.coveredAlpha, 1e-6) - baseRatio) / baseRatio)
  }
  let alphaFall = 0
  let lumaFall = 0
  for (const l of out) {
    alphaFall = Math.max(alphaFall, Math.abs(l.coveredAlpha - base.coveredAlpha) / Math.max(base.coveredAlpha, 1e-6))
    lumaFall = Math.max(lumaFall, Math.abs(l.coveredLuma - base.coveredLuma) / base.coveredLuma)
  }
  const looksStraight = alphaFall > 5 * CHECK_TOLERANCE && lumaFall <= CHECK_TOLERANCE
  const ok = ratioDrift <= CHECK_TOLERANCE && !looksStraight
  return {
    semantics: tag,
    ok,
    drift: Math.max(ratioDrift, looksStraight ? alphaFall : 0),
    message: looksStraight
      ? 'covered alpha falls with depth but the stored luma does NOT — the data is straight (or already ' +
        'coverage-normalised) colour, not premultiplied. The generated decoder divides by alpha, so the far ' +
        "field will be inflated by 1/coverage. Declare { kind: 'color-with-coverage', weight: 'a' } instead."
      : ok
        ? 'stored luma falls in step with coverage, as premultiplied filtering requires'
        : 'stored luma does not track coverage — un-premultiplying will not restore a flat colour.',
    levels: out,
  }
}

// ---------------------------------------------------------------------------
// The generated decoder
// ---------------------------------------------------------------------------

function decoderBody(s: TexelSemantics): string {
  switch (s.kind) {
    case 'color-with-coverage':
      return `  // The mip filter already normalised rgb by coverage at EVERY level.
  // There is deliberately no divide here — adding one inflates the far field
  // by 1/coverage (017-cached-clusters: 2.42x at its deepest mip).
  return t;`
    case 'premultiplied-color':
      return `  // Plain box over premultiplied colour: THIS is where the un-premultiply
  // lives, and the only place in the codebase that should have one.
  return vec4f(t.rgb / max(t.a, 1e-5), t.a);`
    case 'unit-vector': {
      const h = `vec3f(${s.hemisphere.map(f32).join(', ')})`
      return `  let v = t.xyz * 2.0 - 1.0;
  let len = length(v);
  return vec4f(select(normalize(${h}), v / len, len > 1e-5), t.a);`
    }
    case 'scalar-field':
    case 'mask':
      return `  return t;`
  }
}

function decoderDoc(s: TexelSemantics): string {
  switch (s.kind) {
    case 'color-with-coverage':
      return 'rgb = straight colour (already coverage-normalised), a = coverage.'
    case 'premultiplied-color':
      return 'rgb = straight colour recovered by dividing out a, a = coverage.'
    case 'unit-vector':
      return 'xyz = renormalised unit vector in world/tangent space, w = the stored a channel.'
    case 'scalar-field':
      return 'r/g/b = scalars, a = validity.'
    case 'mask':
      return 'the mask value, unchanged.'
  }
}

/**
 * Emit the WGSL that reads a texture filtered with `plan`. Paste the result
 * into your shader source (or `#include` a file you wrote it into):
 *
 *   fn <name>_decode(t: vec4f) -> vec4f
 *   fn <name>_read(tex: texture_2d<f32>, samp: sampler, uv: vec2f, ddx: vec2f, ddy: vec2f) -> vec4f
 *   fn <name>_read_level(tex: texture_2d<f32>, samp: sampler, uv: vec2f, level: f32) -> vec4f
 *
 * `_read` takes EXPLICIT gradients on purpose. A parallax/POM material offsets
 * the uv per fragment, which makes the implicit uv derivative the heightmap's
 * own gradient rather than the surface's — the sampler then jumps to near-top
 * mips at grazing angles and the far field turns into flat paint. Compute
 * ddx/ddy from the UNPERTURBED surface uv and pass them here; with this
 * signature there is no way to forget.
 */
export function wgslDecoder(name: string, plan: MipPlan): string {
  const s = plan.semantics
  return `// GENERATED by wgslDecoder("${name}") — semantics: ${semanticsTag(s)}.
// Returns ${decoderDoc(s)}
// Do not hand-edit: this decoder is the exact inverse of the mip filter that
// generateMips() built from the same TexelSemantics declaration.
fn ${name}_decode(t: vec4f) -> vec4f {
${decoderBody(s)}
}

// Gradients are EXPLICIT: pass ddx/ddy of the UNPERTURBED surface uv, even when
// the uv argument itself carries a parallax offset. Implicit derivatives of an
// offset uv are the heightmap's gradient, not the surface's, and collapse the
// mip chain at grazing angles.
fn ${name}_read(tex: texture_2d<f32>, samp: sampler, uv: vec2f, ddx: vec2f, ddy: vec2f) -> vec4f {
  return ${name}_decode(textureSampleGrad(tex, samp, uv, ddx, ddy));
}

fn ${name}_read_level(tex: texture_2d<f32>, samp: sampler, uv: vec2f, level: f32) -> vec4f {
  return ${name}_decode(textureSampleLevel(tex, samp, uv, level));
}
`
}

/**
 * The exact INVERSE of `wgslDecoder`: takes a decoded value and returns the
 * texel to store. A pass that WRITES a texture this convention will later be
 * read through (a materialized graph node, a bake) needs it, and it has to
 * live next to the decoder or the pair drifts — a premultiplied encoder with a
 * coverage-weighted decoder is the 017 bug written from the other end.
 *
 * Emits `fn <name>_encode(v: vec4f) -> vec4f`.
 */
export function wgslEncoder(name: string, plan: MipPlan): string {
  const s = plan.semantics
  let body: string
  switch (s.kind) {
    case 'color-with-coverage':
      // Straight colour is stored verbatim; the filter, not the store, is what
      // keeps it coverage-normalised.
      body = '  return v;'
      break
    case 'premultiplied-color':
      body = '  return vec4f(v.rgb * v.a, v.a);'
      break
    case 'unit-vector':
      body = '  return vec4f(normalize(v.xyz) * 0.5 + 0.5, v.w);'
      break
    case 'scalar-field':
    case 'mask':
      body = '  return v;'
      break
  }
  return `// GENERATED by wgslEncoder("${name}") — semantics: ${semanticsTag(s)}.
// The exact inverse of ${name}_decode: encode(decode(t)) == t.
fn ${name}_encode(v: vec4f) -> vec4f {
${body}
}
`
}

// ---------------------------------------------------------------------------
// Octahedral — NON-MIPPED storage only
// ---------------------------------------------------------------------------

/**
 * Octahedral encode/decode for a NON-MIPPED normal texture (or a direction
 * packed into a buffer). Never put this on a texture with mipLevelCount > 1 —
 * `createTexture2D` throws if you try, and the reason is in
 * `OCTAHEDRAL_MIP_ERROR` / the module header.
 *
 * **Y-primary** — the stored pair is (n.x, n.z) and n.y is the derived
 * component, folded when negative. Encode returns [0,1]; decode takes [0,1].
 * The sign uses `select(-1, 1, v >= 0)`, not `sign()`, which returns 0 at
 * exactly 0 (a dozen experiment shaders have that latent difference).
 *
 * This pair is SELF-CONSISTENT and unrelated to the source-mesh convention:
 * `encode(decode(t)) == t`, so which axis it derives is a free choice and
 * nothing here needs to change. It is emphatically NOT the GCMESH1 decode —
 * that one derives **z**, lives in `decodeGcMeshNormal()` /
 * `src/wgsl/gcmesh.wgsl`, and was measured rather than assumed. This comment
 * used to claim the two matched, which was the wrong half of the truth: they
 * were both y-derived, and only one of them was allowed to be.
 *
 * Emits `<name>_oct_encode(n: vec3f) -> vec2f` and `<name>_oct_decode(e: vec2f) -> vec3f`,
 * matching `octEncode`/`octDecode` below.
 */
export function wgslOctahedral(name: string): string {
  return `// GENERATED by wgslOctahedral("${name}").
// NON-MIPPED storage only: octahedral values are not mip-averageable.
// Y-primary and self-consistent with its own encode; NOT the GCMESH1 source
// convention (that one is gcmesh_normal_decode()). Range [0,1] both ways.
fn ${name}_oct_sign(v: f32) -> f32 {
  return select(-1.0, 1.0, v >= 0.0);
}

fn ${name}_oct_encode(n: vec3f) -> vec2f {
  let s = max(abs(n.x) + abs(n.y) + abs(n.z), 1e-8);
  var u = n.x / s;
  var v = n.z / s;
  if (n.y < 0.0) {
    let fu = (1.0 - abs(v)) * ${name}_oct_sign(u);
    let fv = (1.0 - abs(u)) * ${name}_oct_sign(v);
    u = fu;
    v = fv;
  }
  return vec2f(u, v) * 0.5 + 0.5;
}

fn ${name}_oct_decode(e01: vec2f) -> vec3f {
  let e = e01 * 2.0 - 1.0;
  var x = e.x;
  var z = e.y;
  let y = 1.0 - abs(e.x) - abs(e.y);
  if (y < 0.0) {
    x = (1.0 - abs(e.y)) * ${name}_oct_sign(e.x);
    z = (1.0 - abs(e.x)) * ${name}_oct_sign(e.y);
  }
  return normalize(vec3f(x, y, z));
}
`
}

/** TS twin of `<name>_oct_encode` — Y-primary, [0,1] range, for bake-time packing. */
export function octEncode(x: number, y: number, z: number): [number, number] {
  const s = Math.max(Math.abs(x) + Math.abs(y) + Math.abs(z), 1e-8)
  let u = x / s
  let v = z / s
  if (y < 0) {
    const fu = (1 - Math.abs(v)) * (u >= 0 ? 1 : -1)
    const fv = (1 - Math.abs(u)) * (v >= 0 ? 1 : -1)
    u = fu
    v = fv
  }
  return [u * 0.5 + 0.5, v * 0.5 + 0.5]
}

/** TS twin of `<name>_oct_decode`. */
export function octDecode(e0: number, e1: number): [number, number, number] {
  const u = e0 * 2 - 1
  const v = e1 * 2 - 1
  let x = u
  let z = v
  const y = 1 - Math.abs(u) - Math.abs(v)
  if (y < 0) {
    x = (1 - Math.abs(v)) * (u >= 0 ? 1 : -1)
    z = (1 - Math.abs(u)) * (v >= 0 ? 1 : -1)
  }
  const len = Math.hypot(x, y, z) || 1
  return [x / len, y / len, z / len]
}

// ---------------------------------------------------------------------------
// Wrapping dilation
// ---------------------------------------------------------------------------

export interface DilateOptions {
  /** Row stride of `pixels`, in texels. */
  width: number
  height: number
  /** Values per texel. Default 4 (RGBA8). */
  channels?: number
  /** Channel index holding coverage. Default `channels - 1`. */
  alphaChannel?: number
  /** Texels with coverage <= this are holes. Default 0. */
  threshold?: number
  /** Rings to grow. Default: until every hole is filled. */
  iterations?: number
  /**
   * Torus wrap (default true) — what a tiling carpet atlas needs, since its
   * left edge really does neighbour its right edge. Set false for a card
   * atlas, where bleeding across a tile boundary would smear neighbouring
   * cards into each other.
   */
  wrap?: boolean
  /**
   * Restrict the whole operation to one sub-rect (an atlas tile). Neighbour
   * lookups wrap or clamp within the REGION, not the full image — which is
   * what an atlas of independently-tiling tiles actually wants.
   */
  region?: { x: number; y: number; width: number; height: number }
}

/**
 * Bleed colour outward into transparent texels, IN PLACE, leaving the coverage
 * channel untouched. Without this, bilinear filtering and every mip level pull
 * the (usually black) colour of uncovered texels into the silhouette edge, and
 * the plant gets a dark halo that gets worse with distance.
 *
 * Wrapping is the default because a tiling carpet texture must match itself
 * across its seam; pass `wrap: false` (and usually a `region`) for a card
 * atlas.
 */
export function dilateWrap(
  pixels: Uint8Array | Uint8ClampedArray | Float32Array,
  opts: DilateOptions,
): void {
  const stride = opts.width
  const ch = opts.channels ?? 4
  const alphaCh = opts.alphaChannel ?? ch - 1
  const threshold = opts.threshold ?? 0
  const wrap = opts.wrap ?? true
  const rx = opts.region?.x ?? 0
  const ry = opts.region?.y ?? 0
  const rw = opts.region?.width ?? opts.width
  const rh = opts.region?.height ?? opts.height
  const maxIterations = opts.iterations ?? Math.max(rw, rh)
  const quantize = !(pixels instanceof Float32Array)

  const at = (lx: number, ly: number): number => ((ry + ly) * stride + (rx + lx)) * ch

  const count = rw * rh
  const filled = new Uint8Array(count)
  let holes = 0
  for (let ly = 0; ly < rh; ly++) {
    for (let lx = 0; lx < rw; lx++) {
      const solid = pixels[at(lx, ly) + alphaCh]! > threshold
      filled[ly * rw + lx] = solid ? 1 : 0
      if (!solid) holes++
    }
  }
  if (holes === 0 || holes === count) return

  const acc = new Float32Array(ch)
  for (let iteration = 0; iteration < maxIterations && holes > 0; iteration++) {
    const newly: number[] = []
    for (let ly = 0; ly < rh; ly++) {
      for (let lx = 0; lx < rw; lx++) {
        const li = ly * rw + lx
        if (filled[li] === 1) continue
        acc.fill(0)
        let n = 0
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue
            let sx = lx + dx
            let sy = ly + dy
            if (wrap) {
              sx = (sx + rw) % rw
              sy = (sy + rh) % rh
            } else if (sx < 0 || sy < 0 || sx >= rw || sy >= rh) {
              continue
            }
            if (filled[sy * rw + sx] !== 1) continue
            const s = at(sx, sy)
            for (let c = 0; c < ch; c++) acc[c] = acc[c]! + pixels[s + c]!
            n++
          }
        }
        if (n === 0) continue
        const d = at(lx, ly)
        for (let c = 0; c < ch; c++) {
          if (c === alphaCh) continue
          const v = acc[c]! / n
          pixels[d + c] = quantize ? Math.round(v) : v
        }
        newly.push(li)
      }
    }
    if (newly.length === 0) break
    for (const li of newly) filled[li] = 1
    holes -= newly.length
  }
}

// ---------------------------------------------------------------------------
// Readback
// ---------------------------------------------------------------------------

export interface ReadTextureResult {
  /** Tightly packed rows (the 256-byte padding is already removed). */
  data: Uint8Array
  width: number
  height: number
  bytesPerTexel: number
}

export interface ReadTextureOptions {
  mipLevel?: number
  /**
   * Suppress the VRAM tracker's untracked-allocation warning for the transient
   * staging buffer. Pass `res.exempt.bind(res)` (or `tracker.exempt`).
   */
  exempt?: <T>(fn: () => T) => T
}

/**
 * Read a colour texture level back to the CPU. Handles the two chores every
 * caller got to rediscover: `bytesPerRow` must be a multiple of 256, and the
 * padding has to be stripped row by row afterwards.
 */
export async function readTexture(
  device: GPUDevice,
  tex: GPUTexture,
  opts: ReadTextureOptions = {},
): Promise<ReadTextureResult> {
  if ((tex.usage & GPUTextureUsage.COPY_SRC) === 0) {
    throw new Error(`readTexture("${tex.label}"): texture lacks COPY_SRC usage (use createTexture2D)`)
  }
  const bpt = textureFormatBytes(tex.format)
  if (bpt === undefined) throw new Error(`readTexture("${tex.label}"): unknown texel size for format ${tex.format}`)

  const level = opts.mipLevel ?? 0
  const width = Math.max(1, tex.width >> level)
  const height = Math.max(1, tex.height >> level)
  const bytesPerRow = Math.ceil((width * bpt) / 256) * 256
  const run = opts.exempt ?? (<T>(fn: () => T): T => fn())

  const readback = run(() =>
    device.createBuffer({
      label: `readTexture/${tex.label}`,
      size: bytesPerRow * height,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    }),
  )
  const enc = device.createCommandEncoder({ label: `readTexture/${tex.label}` })
  enc.copyTextureToBuffer({ texture: tex, mipLevel: level }, { buffer: readback, bytesPerRow, rowsPerImage: height }, [
    width,
    height,
  ])
  device.queue.submit([enc.finish()])
  await readback.mapAsync(GPUMapMode.READ)
  const padded = new Uint8Array(readback.getMappedRange())

  const data = new Uint8Array(width * height * bpt)
  for (let y = 0; y < height; y++) {
    data.set(padded.subarray(y * bytesPerRow, y * bytesPerRow + width * bpt), y * width * bpt)
  }
  readback.unmap()
  readback.destroy()
  return { data, width, height, bytesPerTexel: bpt }
}
