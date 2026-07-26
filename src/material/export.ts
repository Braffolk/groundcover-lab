/**
 * PORTABLE EXPORT — a graph material as a self-contained bundle.
 *
 * The goal, in the owner's words: dropping a material into an unrelated WebGPU
 * codebase should be a COPY, not a port. So the bundle carries everything that
 * cannot be re-derived from looking at the pictures:
 *
 *   material.wgsl   the generated shader, standalone (see portable.ts) — the
 *                   lab's @group(0) swapped for a documented host struct, with
 *                   the shared lighting model and the debug views INLINED
 *                   rather than dropped.
 *   textures/*.png  level 0 of every texture the fragment shader samples, at
 *                   the resolution it samples them.
 *   mipgen.wgsl     the lab's own mip reductions, one per semantics in use.
 *                   The mip CONVENTION is the one thing a consumer cannot read
 *                   off the texels — a coverage-weighted chain and a
 *                   premultiplied one are the same bytes at level 0 and differ
 *                   by 1/coverage in the far field (the documented 2.42x
 *                   inflation) — so it travels as code.
 *   material.json   the manifest: channels + TexelSemantics, texture formats,
 *                   the uniform layout with the exact float block the lab is
 *                   running, the uv contract (METRES of surface, not [0,1]),
 *                   and the measured scalars the material carries.
 *   README.md       bind groups and a draw call, for someone who has never
 *                   seen this repo.
 *
 * WHAT IS DELIBERATELY NOT IN IT: the inactive variants of a discrete
 * `variantBlend`. Only one is ever resident — that is the whole point of the
 * node kind — so the bundle carries the ACTIVE habitat and says so. Exporting
 * again with a different selector produces the other one.
 */

import { encodePng, type PngChannels } from '../gpu/image.ts'
import {
  MIP_FILTER_HELPERS,
  mipLevelsFor,
  readTexture,
  semanticsTag,
  wgslMipReduce,
  type TexelSemantics,
} from '../gpu/texture.ts'
import { makeZip, type ZipEntry } from '../gpu/zip.ts'
import type { ParamDef, ParamSchema } from '../harness/params.ts'
import { PREVIEW_VERTEX_LAYOUT } from '../scene/preview.ts'
import { paramFloat, uniformLayout, wgslIdent, wgslParamName } from './codegen.ts'
import type { MaterialInstance, MaterialTextureSlot } from './instance.ts'
import { MAT_HOST_FIELDS, MAT_HOST_SIZE, portableMaterialWgsl } from './portable.ts'
import type { MaterialNodeReport, MaterialReport } from './report.ts'
import {
  MATERIAL_FIRST_TEXTURE_BINDING,
  MATERIAL_SAMPLER_BINDING,
  MATERIAL_UNIFORM_BINDING,
  dependenciesOf,
  viewUvDependencies,
  type ResolvedGraph,
} from './resolve.ts'
import { MATERIAL_SAMPLER_DESC } from './runtime.ts'
import type { MaterialNode } from './types.ts'

export const BUNDLE_FORMAT = 'groundcover-material-bundle/1'

// ---------------------------------------------------------------------------
// Refusal
// ---------------------------------------------------------------------------

/**
 * Why this material cannot be exported, or null if it can.
 *
 * A material with no graph (001-flat-maps builds its own pipeline in main.ts)
 * has no generated WGSL, no declared channels and no TexelSemantics on
 * anything — there is nothing to put in a manifest, and a bundle that guessed
 * would be worse than no bundle. Refuse by name rather than emit one.
 */
export function materialExportRefusal(
  id: string,
  report: MaterialReport | undefined,
  instance: MaterialInstance | undefined,
): string | null {
  if (!report) {
    return (
      `"${id}" is not a GRAPH material: it builds its own pipeline in main.ts instead of declaring a MaterialDef, ` +
      'so there is no generated shader, no declared channel list and no TexelSemantics to export. A bundle would ' +
      'have to guess the mip convention and the uv contract, which is exactly the part a consumer cannot re-derive. ' +
      'Port it to a MaterialDef (see 002-graph-maps) and it exports for free.'
    )
  }
  if (!report.ok) {
    return `"${id}" failed validation, so what is running is not what it declares. Fix the validator report first.`
  }
  if (!instance) {
    return `"${id}" has no live instance — the material is not running, so there are no textures to read back.`
  }
  return null
}

// ---------------------------------------------------------------------------
// Textures
// ---------------------------------------------------------------------------

/** How one texture leaves the lab: the PNG, and the format to rebuild it in. */
interface TextureExport {
  entry: ZipEntry
  manifest: Record<string, unknown>
  semantics: TexelSemantics
}

const PNG_CHANNELS_FOR: Record<number, PngChannels> = { 1: 1, 2: 2, 3: 3, 4: 4 }

function safeName(id: string): string {
  return id.replace(/[^A-Za-z0-9._-]/g, '_')
}

/**
 * Read level 0 back and encode it.
 *
 * The format branch is the same one `loadNodeImage` made on the way in, run
 * backwards: an 8-bit texture's bytes ARE the PNG's bytes, and a float32
 * texture only exists because the source was a 16-bit PNG, so it goes back out
 * as one. Anything else throws rather than quietly losing precision — a
 * heightmap that silently became 8-bit is the exact failure mode
 * `createImageBitmap` is guarded against elsewhere in this repo.
 */
async function exportTexture(
  device: GPUDevice,
  slot: MaterialTextureSlot,
  node: MaterialNodeReport,
  exempt?: <T>(fn: () => T) => T,
): Promise<TextureExport> {
  const tex = slot.texture
  const read = await readTexture(device, tex, { mipLevel: 0, ...(exempt && { exempt }) })
  const isFloat32 = tex.format.endsWith('32float')
  const components = isFloat32 ? read.bytesPerTexel / 4 : read.bytesPerTexel
  const channels = PNG_CHANNELS_FOR[components]
  if (!channels || (!isFloat32 && !tex.format.startsWith('r8') && !tex.format.startsWith('rgba8'))) {
    throw new Error(
      `cannot export node "${node.id}": texture format ${tex.format} has no lossless PNG form here. ` +
        'Supported: r8unorm / rgba8unorm[-srgb] (8-bit PNG) and r/rg/rgba32float (16-bit PNG, which is what a ' +
        '16-bit source PNG became on load).',
    )
  }

  let bytes: Uint8Array
  let bitDepth: 8 | 16
  if (isFloat32) {
    // The float texture holds a 16-bit integer normalised to [0,1] (see
    // textureFromPng16), so *65535 is exact, not a requantisation.
    const src = new Float32Array(read.data.buffer, read.data.byteOffset, read.data.byteLength / 4)
    const out = new Uint16Array(src.length)
    for (let i = 0; i < src.length; i++) out[i] = Math.round(Math.min(1, Math.max(0, src[i]!)) * 65535)
    bytes = await encodePng({ width: read.width, height: read.height, channels, bitDepth: 16, data: out })
    bitDepth = 16
  } else {
    bytes = await encodePng({ width: read.width, height: read.height, channels, bitDepth: 8, data: read.data })
    bitDepth = 8
  }

  const file = `textures/${safeName(slot.nodeId)}.png`
  const tag = semanticsTag(node.spec.semantics)
  return {
    entry: { name: file, bytes, compress: false },
    semantics: node.spec.semantics,
    manifest: {
      binding: slot.binding,
      node: slot.nodeId,
      ...(slot.sourceNodeId !== slot.nodeId && { dataFrom: slot.sourceNodeId }),
      ...(slot.variant !== undefined && { activeVariant: slot.variant }),
      file,
      width: read.width,
      height: read.height,
      png: { channels, bitDepth },
      gpuFormat: tex.format,
      channelType: node.spec.type,
      srgb: node.spec.srgb ?? false,
      semantics: node.spec.semantics,
      mip: {
        levels: mipLevelsFor(read.width, read.height),
        /** The reduction to build levels 1..N-1 with — see mipgen.wgsl. */
        entryPoint: `fs_${safeName(tag).replace(/[.-]/g, '_')}`,
        note: mipNoteFor(node.spec.semantics),
      },
      ...(node.doc && { doc: node.doc }),
      ...(node.spec.doc && { specDoc: node.spec.doc }),
    },
  }
}

function mipNoteFor(s: TexelSemantics): string {
  switch (s.kind) {
    case 'color-with-coverage':
      return (
        'rgb = SUM(rgb*a)/SUM(a): the stored colour stays coverage-normalised at every level, so the shader must ' +
        'NOT divide by alpha again. (Doing so inflates the far field by 1/coverage — measured at 2.42x.)'
      )
    case 'premultiplied-color':
      return 'plain box over premultiplied colour; the shader DOES divide rgb by a to recover straight colour.'
    case 'unit-vector':
      return `flip each sample into the hemisphere (${s.hemisphere.join(', ')}), average, renormalise. Never box-average an octahedral pair.`
    case 'scalar-field':
      return s.empty === 'zero' ? 'plain box average; holes are genuinely zero.' : 'average only samples with a > 0, so the field does not sink into its holes.'
    case 'mask':
      return s.reduce === 'max' ? 'conservative max, so thin features survive the chain.' : 'plain box average.'
  }
}

// ---------------------------------------------------------------------------
// Measured scalars
// ---------------------------------------------------------------------------

const CONST_RETURN = /^return\s+([\s\S]+?);$/
const VEC_LITERAL = /^vec([234])f\(([^()]*)\)$/

/**
 * A `procedural` node with no inputs whose body is a bare literal IS a measured
 * number — 003-nd-fuzz puts `tipColor`, `baseColor`, the mean cushion plane and
 * the cushion relief in the graph exactly this way, so that one selector moves
 * the maps and the numbers together. Pulling them back out is a text match on
 * the declaration, which is honest: if it does not parse it is not reported.
 */
function constantValueOf(node: MaterialNode): number[] | null {
  if (node.kind !== 'procedural') return null
  if (node.inputs && Object.keys(node.inputs).length > 0) return null
  const m = CONST_RETURN.exec(node.body.trim())
  if (!m) return null
  const expr = m[1]!.trim()
  if (/\bu\.p\.|\bc\.|\bin_|\bframe\.|texture/.test(expr)) return null
  const vec = VEC_LITERAL.exec(expr)
  if (vec) {
    const want = Number(vec[1])
    const parts = vec[2]!.split(',').map((s) => Number(s.trim()))
    if (!parts.every((v) => Number.isFinite(v))) return null
    if (parts.length === 1) return Array.from({ length: want }, () => parts[0]!)
    return parts.length === want ? parts : null
  }
  const n = Number(expr)
  return Number.isFinite(n) ? [n] : null
}

interface MeasuredScalar {
  node: string
  value: number[]
  doc?: string
  /** Set when the value came from the ACTIVE variant of a discrete blend. */
  selectedBy?: { selector: string; variant: string; variantNode: string }
}

function measuredScalarsOf(inst: MaterialInstance): MeasuredScalar[] {
  const out: MeasuredScalar[] = []
  const values = inst.values as Record<string, unknown>
  const seen = new Set<string>()

  for (const r of inst.resolved.byId.values()) {
    if (r.node.kind !== 'variantBlend') continue
    const active = String(values[r.node.selector])
    const variantId = r.node.variants[active] ?? Object.values(r.node.variants)[0]
    const variant = variantId ? inst.resolved.byId.get(variantId) : undefined
    if (!variant) continue
    const value = constantValueOf(variant.node)
    if (!value) continue
    out.push({
      node: r.node.id,
      value,
      ...(r.node.doc && { doc: r.node.doc }),
      selectedBy: { selector: r.node.selector, variant: active, variantNode: variant.node.id },
    })
    seen.add(r.node.id)
    for (const id of Object.values(r.node.variants)) seen.add(id)
  }

  for (const r of inst.resolved.byId.values()) {
    if (seen.has(r.node.id) || r.ownedBy !== undefined) continue
    const value = constantValueOf(r.node)
    if (!value) continue
    out.push({ node: r.node.id, value, ...(r.node.doc && { doc: r.node.doc }) })
  }
  return out.sort((a, b) => a.node.localeCompare(b.node))
}

// ---------------------------------------------------------------------------
// The manifest
// ---------------------------------------------------------------------------

function paramManifest(schema: ParamSchema, values: Record<string, unknown>): Record<string, unknown>[] {
  const layout = uniformLayout(schema)
  return layout.keys.map((key, i) => {
    const def: ParamDef = schema[key]!
    const floatIndex = layout.paramOffset + i
    const common = {
      name: key,
      wgslName: `u.p.${wgslParamName(key)}`,
      floatIndex,
      byteOffset: floatIndex * 4,
      kind: def.kind,
      /** What the exported uniform block actually holds for this param. */
      exportedValue: paramFloat(def, values[key]),
    }
    switch (def.kind) {
      case 'number':
        return { ...common, default: def.def, min: def.min, max: def.max, ...(def.step !== undefined && { step: def.step }) }
      case 'boolean':
        return { ...common, default: def.def ? 1 : 0, encoding: '0 or 1' }
      case 'enum':
        return {
          ...common,
          default: def.options.indexOf(def.def),
          options: def.options,
          exportedOption: String(values[key]),
          encoding: 'index into `options`',
        }
    }
  })
}

/**
 * The vertex layout, read off `PREVIEW_VERTEX_LAYOUT` rather than retyped —
 * the offsets are the one part of a bundle a consumer cannot debug by looking
 * at the picture, and a hand-copied stride goes stale silently.
 */
const VERTEX_ATTR_NOTES: Record<number, { name: string; note?: string }> = {
  0: { name: 'position', note: 'WORLD space — vs_main has no model matrix' },
  1: { name: 'normal' },
  2: { name: 'tangent', note: 'xyz along +u, w = handedness for bitangent = cross(N,T)*w' },
  3: { name: 'uv', note: 'METRES of surface — see `uv`' },
}

function vertexLayoutManifest(): Record<string, unknown> {
  return {
    arrayStride: PREVIEW_VERTEX_LAYOUT.arrayStride,
    attributes: [...PREVIEW_VERTEX_LAYOUT.attributes].map((a) => ({
      shaderLocation: a.shaderLocation,
      offset: a.offset,
      format: a.format,
      ...VERTEX_ATTR_NOTES[a.shaderLocation],
    })),
  }
}

const UV_CONTRACT =
  'The vertex uv this shader was authored against is in METRES OF SURFACE, not [0,1]. One unit of u or v is one ' +
  'metre measured along the surface, on every object and in both axes. A consumer WILL assume [0,1]: if you feed a ' +
  'normalised chart, the material tiles once per polygon island instead of at its physical period, and the tile ' +
  'stops being square wherever the chart is not. Multiply your metre uv by `tilesPerMetre` (ONE isotropic number, ' +
  'never a per-axis fit) — the shader does that itself from uv_scale in the material uniform.'

// ---------------------------------------------------------------------------
// The bundle
// ---------------------------------------------------------------------------

/**
 * Nodes whose `n_<id>_eval` the FRAGMENT shader actually calls.
 *
 * Codegen emits a function for every reachable node, called or not, so a text
 * scan of the generated WGSL says "used" for a texture nothing samples — which
 * is how the first version of this shipped a 3 MB map that broke `layout:
 * 'auto'`. The truth is in the resolved graph, and the walk mirrors emitNode()
 * exactly: a LIVE node calls its dependencies; a TEXTURE-mode node is a reader
 * that samples its own texture and calls nothing. So the walk stops there, and
 * a node reachable only THROUGH a texture-mode node was consumed at bake time,
 * not at draw time.
 */
function sampledNodes(resolved: ResolvedGraph): Set<string> {
  const called = new Set<string>()
  const walk = (id: string): void => {
    if (called.has(id)) return
    const r = resolved.byId.get(id)
    if (!r) return
    called.add(id)
    if (r.mode === 'texture') return
    for (const dep of dependenciesOf(r.node)) walk(dep)
  }
  for (const root of Object.values(resolved.graph.channels)) walk(root)
  for (const root of viewUvDependencies(resolved.graph)) walk(root)
  return called
}

export interface MaterialBundleInput {
  instance: MaterialInstance
  report: MaterialReport
  meta: {
    title: string
    description?: string
    /** The preview object the export was taken on — uv_min/uv_max come from it. */
    previewObject?: string
    /** Render-target formats the lab's pipeline was built with. */
    colorFormat: GPUTextureFormat
    depthFormat: GPUTextureFormat
  }
  /** Suppress the VRAM tracker's warning for the readback staging buffers. */
  exempt?: <T>(fn: () => T) => T
}

export interface MaterialBundle {
  blob: Blob
  filename: string
  files: { name: string; bytes: number }[]
  totalBytes: number
}

export async function exportMaterialBundle(input: MaterialBundleInput): Promise<MaterialBundle> {
  const { instance: inst, report, meta } = input
  const refusal = materialExportRefusal(inst.id, report, inst)
  if (refusal) throw new Error(refusal)

  const nodeById = new Map(report.nodes.map((n) => [n.id, n]))
  const allSlots = [...inst.slots].sort((a, b) => a.binding - b.binding)

  // --- which bound textures does the FRAGMENT shader actually sample? -------
  const sampled = sampledNodes(inst.resolved)
  const deadNodes = inst.resolved.order.filter((id) => !sampled.has(id)).map((id) => wgslIdent(id))
  const unusedVars: string[] = []
  const slots: MaterialTextureSlot[] = []
  for (const slot of allSlots) {
    if (sampled.has(slot.nodeId)) slots.push(slot)
    else unusedVars.push(`t_${wgslIdent(slot.nodeId)}`)
  }

  // --- textures ------------------------------------------------------------
  const textures: TextureExport[] = []
  for (const slot of slots) {
    const node = nodeById.get(slot.nodeId)
    if (!node) throw new Error(`export: bound node "${slot.nodeId}" is not in the report`)
    textures.push(await exportTexture(inst.device, slot, node, input.exempt))
  }

  // --- mip filters, one per distinct semantics -----------------------------
  const bySemantics = new Map<string, TexelSemantics>()
  for (const t of textures) bySemantics.set(semanticsTag(t.semantics), t.semantics)
  const mipgen = mipgenWgsl(bySemantics)

  // --- shader --------------------------------------------------------------
  const wgsl = portableMaterialWgsl(report.code, {
    id: inst.id,
    moduleId: report.moduleId,
    deadNodes,
  })

  // --- manifest ------------------------------------------------------------
  const layout = uniformLayout(inst.schema)
  const uniforms = inst.uniformFloats()
  const manifest = {
    format: BUNDLE_FORMAT,
    id: inst.id,
    title: meta.title,
    ...(meta.description && { description: meta.description }),
    source: {
      lab: 'groundcover-experiments',
      /** Content hash of the generated WGSL — the exact shader this came from. */
      moduleId: report.moduleId,
      viewUv: report.viewUv,
      ...(meta.previewObject && { previewObject: meta.previewObject }),
    },
    shader: {
      file: 'material.wgsl',
      vertex: 'vs_main',
      fragment: 'fs_main',
      // Matched against the exact statements the generator emits, never a bare
      // word: `discard` and `frag_depth` both appear in COMMENTS in material
      // sources that do neither (002's surface.wgsl explains why it does not
      // discard), and a manifest that mis-declares either sends a consumer
      // hunting for a depth attachment write that is not there.
      writesFragDepth: report.code.includes('@builtin(frag_depth) depth: f32,'),
      discards: report.code.includes('if (vu.clip) { discard; }'),
      vertexBufferLayout: vertexLayoutManifest(),
      pipeline: {
        topology: 'triangle-list',
        cullMode: 'back',
        colorTarget: { format: meta.colorFormat, note: 'the lab renders to the preferred canvas format; any non-sRGB 8-bit unorm target matches' },
        depthStencil: { format: meta.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
      },
    },
    hostBindGroup: {
      group: 0,
      binding: 0,
      struct: 'MatHost',
      sizeBytes: MAT_HOST_SIZE,
      fields: MAT_HOST_FIELDS,
      labValues: {
        sun_dir: [0.35, 0.75, 0.3].map((v) => v / Math.hypot(0.35, 0.75, 0.3)),
        sun_color: [1.15, 1.02, 0.82],
        ambient: [0.21, 0.25, 0.32],
        note: 'What the lab was lit by. Use your own — but a material tuned against these will not look the same under others.',
      },
    },
    materialBindGroup: {
      group: 1,
      uniform: {
        binding: MATERIAL_UNIFORM_BINDING,
        sizeBytes: layout.size,
        header: [
          { name: 'uv_scale', floatIndex: 0, floats: 2, doc: 'tiles per metre, isotropic (both components equal)' },
          { name: 'uv_rot', floatIndex: 2, floats: 1, doc: 'radians, applied to the mesh uv before scaling' },
          { name: '_pad', floatIndex: 3, floats: 1, doc: '' },
          { name: 'uv_min', floatIndex: 4, floats: 2, doc: 'the mesh uv rectangle in METRES; a silhouette POM clips against it' },
          { name: 'uv_max', floatIndex: 6, floats: 2, doc: '' },
        ],
        paramsFloatOffset: layout.paramOffset,
        /**
         * The EXACT float block the lab is running right now: write this and
         * the exported shader reproduces the exported picture.
         */
        exportedFloats: [...uniforms],
      },
      sampler: { binding: MATERIAL_SAMPLER_BINDING, ...MATERIAL_SAMPLER_DESC },
      firstTextureBinding: MATERIAL_FIRST_TEXTURE_BINDING,
      /**
       * Binding indices are the lab's and may be SPARSE — a texture the lab
       * bound but the fragment shader never samples is not here and not in the
       * shader. Bind exactly this list.
       */
      textures: textures.map((t) => t.manifest),
      ...(unusedVars.length > 0 && {
        omittedTextures: {
          vars: unusedVars,
          why:
            'the lab binds these but the fragment shader never samples them (their only consumer was a node that ' +
            "got baked into another texture). Their declarations are removed from material.wgsl too, so `layout: " +
            "'auto'` and an explicit layout agree.",
        },
      }),
    },
    params: paramManifest(inst.schema, inst.values as Record<string, unknown>),
    channels: report.channels.map((c) => {
      const node = nodeById.get(c.nodeId)
      return {
        name: c.channel,
        node: c.nodeId,
        type: c.type,
        mode: c.mode,
        ...(node && { semantics: node.spec.semantics, srgb: node.spec.srgb ?? false }),
        ...(node?.spec.doc && { doc: node.spec.doc }),
      }
    }),
    uv: {
      unit: 'metres-of-surface',
      contract: UV_CONTRACT,
      tilesPerMetre: uniforms[0],
      rotationRadians: uniforms[2],
      meshUvBoundsMetres: { min: [uniforms[4], uniforms[5]], max: [uniforms[6], uniforms[7]] },
    },
    measuredScalars: measuredScalarsOf(inst),
    mipgen: {
      file: 'mipgen.wgsl',
      note:
        'Level 0 is the only level in the PNGs. Build the rest with these reductions, not with a generic box filter: ' +
        'the reduction is chosen by the texture\'s TexelSemantics, and getting it wrong is invisible at level 0 and ' +
        'wrong everywhere else.',
      entryPoints: [...bySemantics.keys()].map((tag) => ({ semantics: tag, entryPoint: `fs_${safeName(tag).replace(/[.-]/g, '_')}` })),
    },
    limitations: [
      'vs_main has no model matrix: vertex positions are taken as world space. Fold your model transform into ' +
        'view_proj and pre-transform position/normal/tangent, or edit vs_main.',
      'Only the ACTIVE variant of each discrete variantBlend is included — one habitat, not all three. Switch the ' +
        'selector in the lab and export again for another.',
      'Mip levels 1..N are not shipped; regenerate them with mipgen.wgsl.',
      'The debug views need frame.debug_mode; set it to 0 for normal rendering.',
      'A view-uv stage other than `identity` makes the result sensitive to VERTEX TESSELLATION. Measured: on the ' +
        "same flat 1x1 m patch, a mesh matching the lab's 64x64 grid reproduces the lab bit-for-bit (0 of 684,000 " +
        'pixels differ), while a 2-triangle quad of the same patch differs on 17% of pixels with max |d| 189 — ' +
        'barycentric interpolation of uv rounds differently over a large triangle, and a parallax step turns a ' +
        '1-ULP uv difference into a texel-sized lookup shift. Without a view-uv stage the same coarse quad is ' +
        'within 2/255 everywhere. Not a bug in the bundle; a property of dependent texture reads.',
    ],
  }

  // --- assemble ------------------------------------------------------------
  const encoder = new TextEncoder()
  const readme = readmeFor(manifest, textures)
  const entries: ZipEntry[] = [
    { name: 'README.md', bytes: encoder.encode(readme) },
    { name: 'material.json', bytes: encoder.encode(`${JSON.stringify(manifest, null, 2)}\n`) },
    { name: 'material.wgsl', bytes: encoder.encode(wgsl) },
    { name: 'mipgen.wgsl', bytes: encoder.encode(mipgen) },
    ...textures.map((t) => t.entry),
  ]
  const blob = await makeZip(entries)
  return {
    blob,
    filename: `${inst.id}-material-bundle.zip`,
    files: entries.map((e) => ({ name: e.name, bytes: e.bytes.length })),
    totalBytes: blob.size,
  }
}

// ---------------------------------------------------------------------------
// mipgen.wgsl
// ---------------------------------------------------------------------------

function mipgenWgsl(bySemantics: ReadonlyMap<string, TexelSemantics>): string {
  const parts: string[] = [
    `// GENERATED — the lab's own mip reductions, one per TexelSemantics this
// material uses. Emitted from the SAME source as the filter that built the
// chain the exported textures were sampled through (src/gpu/texture.ts), so the
// two cannot drift.
//
// Usage: for each level L in 1..N-1, run a fullscreen pass with level L-1 bound
// at @group(0) @binding(0) and level L as the colour attachment, picking the
// fragment entry point named in material.json for that texture.

@group(0) @binding(0) var mip_src: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var tri = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(3.0, 1.0), vec2f(-1.0, 1.0));
  return vec4f(tri[vi], 0.0, 1.0);
}

${MIP_FILTER_HELPERS}`,
  ]
  for (const [tag, semantics] of bySemantics) {
    const name = safeName(tag).replace(/[.-]/g, '_')
    parts.push(wgslMipReduce(name, semantics))
    parts.push(`@fragment
fn fs_${name}(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let base = vec2i(pos.xy) * 2;
  return ${name}_reduce(
    mip_texel(base, 0, 0),
    mip_texel(base, 1, 0),
    mip_texel(base, 0, 1),
    mip_texel(base, 1, 1),
  );
}
`)
  }
  return parts.join('\n')
}

// ---------------------------------------------------------------------------
// README.md
// ---------------------------------------------------------------------------

function readmeFor(manifest: Record<string, unknown>, textures: readonly TextureExport[]): string {
  const m = manifest as {
    id: string
    title: string
    description?: string
    materialBindGroup: { uniform: { sizeBytes: number }; firstTextureBinding: number }
    hostBindGroup: { sizeBytes: number }
    uv: { tilesPerMetre: number }
    channels: { name: string; type: string }[]
    params: { name: string }[]
  }
  const texRows = textures
    .map((t) => {
      const tm = t.manifest as { binding: number; node: string; file: string; gpuFormat: string; width: number; height: number }
      return `| ${tm.binding} | \`${tm.node}\` | \`${tm.file}\` | ${tm.gpuFormat} | ${tm.width}x${tm.height} |`
    })
    .join('\n')

  return `# ${m.title}

${m.description ?? ''}

A self-contained WebGPU material. Nothing here imports anything: \`material.wgsl\`
is one module with \`vs_main\` and \`fs_main\`, and the only thing it wants from
you is a ${m.hostBindGroup.sizeBytes}-byte camera/light uniform.

## Files

- \`material.wgsl\` — the shader. Group 0 is your camera/light struct; group 1 is
  this material's uniform, sampler and textures.
- \`material.json\` — byte offsets, formats, semantics, and the exact uniform
  block this was exported with. Read it rather than hard-coding anything below.
- \`textures/*.png\` — **level 0 only**. Build the rest with \`mipgen.wgsl\`.
- \`mipgen.wgsl\` — the mip reduction per texture, which is NOT interchangeable
  with a box filter; see "Mips".

| binding | node | file | format | size |
| --- | --- | --- | --- | --- |
${texRows}

## Usage

\`\`\`js
const man = await (await fetch('material.json')).json()
const code = await (await fetch('material.wgsl')).text()
const mod = device.createShaderModule({ code })

// group 1: the material's own uniform, sampler and maps.
const matUbo = device.createBuffer({ size: ${m.materialBindGroup.uniform.sizeBytes}, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
device.queue.writeBuffer(matUbo, 0, new Float32Array(man.materialBindGroup.uniform.exportedFloats))
const sampler = device.createSampler(man.materialBindGroup.sampler) // repeat + linear + aniso 16
const entries = [{ binding: 0, resource: { buffer: matUbo } }, { binding: 1, resource: sampler }]
for (const t of man.materialBindGroup.textures) {
  const tex = device.createTexture({ size: [t.width, t.height], format: t.gpuFormat, mipLevelCount: t.mip.levels,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT })
  await uploadPngLevel0(tex, t)   // your PNG decode; see "Textures"
  await buildMips(tex, t.mip.entryPoint)  // mipgen.wgsl
  entries.push({ binding: t.binding, resource: tex.createView() })
}
const matGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(1), entries })

// group 0: your camera and light. Offsets are in man.hostBindGroup.fields.
const host = new Float32Array(${m.hostBindGroup.sizeBytes} / 4)
host.set(viewProj, 0); host.set(cameraPos, 16); host[19] = time
host.set(sunDir, 20);  host[23] = 0 /* debug_mode */
host.set(sunColor, 24); host.set(ambient, 28)
device.queue.writeBuffer(hostUbo, 0, host)

pass.setPipeline(pipeline)          // triangle-list, cullMode 'back', depth 'less' + write
pass.setBindGroup(0, hostGroup)
pass.setBindGroup(1, matGroup)
pass.setVertexBuffer(0, meshVerts)  // stride 48: pos3, normal3, tangent4, uv2
pass.setIndexBuffer(meshIndices, 'uint32')
pass.drawIndexed(indexCount)
\`\`\`

## UV — read this one

**The uv attribute is in METRES OF SURFACE, not [0, 1].** One unit of u or v is
one metre along the surface, in both axes, on every mesh. The material's own
period is applied inside the shader from \`uv_scale\` (${m.uv.tilesPerMetre.toFixed(4)} tiles per
metre in this export), so a [0, 1] chart makes it tile once per uv island and go
non-square wherever the chart is stretched. If your mesh has a normalised chart,
multiply by the island's size in metres before handing it over.

## Textures

Each PNG is level 0, exactly the bytes the lab's texture held: gray8 -> \`r8unorm\`,
RGBA8 -> \`rgba8unorm\`, 16-bit -> a float32 format (see \`gpuFormat\`). No decode,
no colour management, no premultiply — if you use \`createImageBitmap\`, pass
\`{ premultiplyAlpha: 'none', colorSpaceConversion: 'none' }\` or the values change
under you.

## Mips

The reduction is a property of what the texels MEAN, and it is in
\`material.json\` per texture as \`mip.entryPoint\`:

${[...new Set(textures.map((t) => `- **${semanticsTag(t.semantics)}** — ${mipNoteFor(t.semantics)}`))].join('\n')}

Using one filter for all of them is the classic way to be exactly right at
level 0 and wrong at every distance.

## Params

\`material.json -> params\` lists every parameter with its float index in the
material uniform, its range, and the value this bundle was exported at. Writing
a different float at that index is the whole mechanism — nothing recompiles.
Params: ${m.params.map((p) => `\`${p.name}\``).join(', ')}.

## Channels

${m.channels.map((c) => `- \`${c.name}\` (${c.type})`).join('\n')}

## Debug views

\`frame.debug_mode\`: 0 off, 1 albedo, 2 normals, 3 lighting, 4 coverage, 5 depth.
If \`normals\` looks like garbage or \`lighting\` is blown out, the material is
being fed something it did not expect — that is what these are for.
`
}
