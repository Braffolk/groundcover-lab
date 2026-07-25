import shaderSrc from './shaders/flat-maps.wgsl'
import {
  assetUrl,
  createTexture2D,
  decodePng,
  fetchPngBytes,
  generateMips,
  loadImageTexture,
  mipLevelsFor,
  PREVIEW_VERTEX_LAYOUT,
  wgslDecoder,
  type Experiment,
  type ExperimentContext,
  type FrameInfo,
  type MaterialContextExt,
  type TexelSemantics,
  type ViewTargets,
} from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Sphagnum, rendered from the measured maps and nothing else.
 *
 * The maps in assets/materials/sphagnum-* were measured off the 19.8M-tri
 * source mesh, which has since been deleted; they are source data now and
 * cannot be regenerated. This experiment's whole job is to put them on screen
 * honestly, so every later Sphagnum material has a floor to be measured
 * against — and to prove the material stage, the preview geometry, the image
 * path and the debug views all work end to end.
 */

const HABITATS = ['wet-vigorous', 'late-season', 'sun-exposed'] as const
type Habitat = (typeof HABITATS)[number]

/** What assets/materials/<dir>/measured.json carries that this material uses. */
interface Measured {
  speciesId: string
  texPx: number
  /** Period of the periodic tile, in metres. */
  tileM: number
  /** Height range the grey height map spans, in metres at mesh scale 1. */
  y0: number
  y1: number
  planeH: number
  apexH: number
  cover: number
  meanColor: [number, number, number]
}

// ---------------------------------------------------------------------------
// Texel semantics — declared ONCE, and used for BOTH the mip chain and the
// shader-side decoder (see wgslDecoder / generateMips in src/gpu/texture.ts).
// ---------------------------------------------------------------------------

/**
 * The albedo PNG has no alpha, so every texel reads a = 1 and this filter
 * degenerates to exactly a box filter over straight colour — which is what an
 * opaque measured map wants — while the generated decoder is the identity.
 * Declaring it `premultiplied-color` instead would make the decoder divide by
 * an alpha that is 1 everywhere: harmless here, and precisely the reasoning
 * that went wrong in 017-cached-clusters when alpha was not 1.
 */
const ALBEDO_SEMANTICS: TexelSemantics = { kind: 'color-with-coverage', weight: 'a' }

/**
 * measured.json is explicit that this map is plain xyz*0.5+0.5 and NOT
 * octahedral, "because this map is mipped and octahedral pairs are not
 * mip-averageable". `unit-vector` is the filter that matches: flip into the
 * hemisphere, average, renormalise. The hemisphere is +Y because the map is
 * mesh-frame with Y up (measured: the green channel never drops below 128).
 */
const NORMAL_SEMANTICS: TexelSemantics = { kind: 'unit-vector', hemisphere: [0, 1, 0] }

/** Cavity AO: a plain scalar with no holes, so a plain box average. */
const AO_SEMANTICS: TexelSemantics = { kind: 'scalar-field', empty: 'zero' }

interface MapSet {
  habitat: Habitat
  measured: Measured
  albedo: GPUTexture
  normal: GPUTexture
  ao: GPUTexture
}

function materialDir(habitat: Habitat): string {
  return `/assets/materials/sphagnum-${habitat}`
}

async function fetchMeasured(habitat: Habitat): Promise<Measured> {
  const url = assetUrl(`${materialDir(habitat)}/measured.json`)
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`)
  const text = await res.text()
  try {
    return JSON.parse(text) as Measured
  } catch {
    // The dev server answers a MISSING file with index.html at HTTP 200, so a
    // 200 is not proof the file exists.
    throw new Error(`${url}: not JSON (starts with ${JSON.stringify(text.slice(0, 40))}) — is the file there?`)
  }
}

/**
 * lib.dom types the writeTexture source as ArrayBuffer-backed only, while a
 * plain Uint8Array is `Uint8Array<ArrayBufferLike>` (it might sit on a
 * SharedArrayBuffer). Narrow, copying only in the shared case.
 */
function bufferSource(bytes: Uint8Array): Uint8Array<ArrayBuffer> {
  const buf = bytes.buffer
  return buf instanceof ArrayBuffer ? new Uint8Array(buf, bytes.byteOffset, bytes.byteLength) : new Uint8Array(bytes)
}

export async function create(
  ctx: ExperimentContext<typeof PARAMS, MaterialContextExt>,
): Promise<Experiment> {
  const { device } = ctx
  /** Guards against an earlier habitat load winning a race with a later one. */
  let habitatGeneration = 0

  // -------------------------------------------------------------------------
  // Map loading. Only ONE habitat is resident at a time — the three states are
  // ~50MB each and preloading them would triple that for a param nobody moves
  // per frame. Switching reloads asynchronously and keeps drawing the old maps
  // until the new ones are ready.
  // -------------------------------------------------------------------------
  const loadMaps = async (habitat: Habitat): Promise<MapSet> => {
    const dir = materialDir(habitat)
    ctx.progress(0.05, `sphagnum ${habitat}: measured.json`)
    const measured = await fetchMeasured(habitat)

    ctx.progress(0.2, `sphagnum ${habitat}: albedo`)
    const albedo = await loadImageTexture(ctx.res, device, `${dir}/albedo.png`, {
      // LINEAR, deliberately. These bytes ARE the measured linear albedo (the
      // byte mean matches measured.json meanColor to 3 decimal places), and
      // the lab writes fragment colour to a non-sRGB target throughout, so an
      // sRGB decode here would darken the moss by ~2.2 gamma for no reason.
      srgb: false,
      label: `${ctx.id}/albedo/${habitat}`,
      mips: { semantics: ALBEDO_SEMANTICS },
      shaders: ctx.shaders,
      attr: { species: ctx.id, tag: 'albedo' },
    })

    ctx.progress(0.5, `sphagnum ${habitat}: normal`)
    const normal = await loadImageTexture(ctx.res, device, `${dir}/normal.png`, {
      srgb: false,
      label: `${ctx.id}/normal/${habitat}`,
      mips: { semantics: NORMAL_SEMANTICS },
      shaders: ctx.shaders,
      attr: { species: ctx.id, tag: 'normal' },
    })

    // AO is one channel, so it goes in as r8unorm rather than through the
    // rgba8 ImageBitmap path: 5.6MB with mips instead of 22.4MB for three
    // channels of nothing. That means decoding the PNG by hand and driving
    // createTexture2D/generateMips directly.
    ctx.progress(0.8, `sphagnum ${habitat}: cavity AO`)
    const aoPng = await decodePng(await fetchPngBytes(`${dir}/ao.png`))
    if (aoPng.channels !== 1 || aoPng.bitDepth !== 8) {
      throw new Error(`${dir}/ao.png: expected gray8, got ${aoPng.channels}ch@${aoPng.bitDepth}`)
    }
    const ao = createTexture2D(ctx.res, {
      label: `${ctx.id}/ao/${habitat}`,
      size: [aoPng.width, aoPng.height],
      format: 'r8unorm',
      mipLevelCount: mipLevelsFor(aoPng.width, aoPng.height),
      encoding: AO_SEMANTICS,
      attr: { species: ctx.id, tag: 'ao' },
    })
    device.queue.writeTexture(
      { texture: ao },
      bufferSource(aoPng.data as Uint8Array),
      { bytesPerRow: aoPng.width, rowsPerImage: aoPng.height },
      [aoPng.width, aoPng.height],
    )
    generateMips(ctx.res, ctx.shaders, ao, { semantics: AO_SEMANTICS })

    ctx.progress(1, `sphagnum ${habitat}: ready`)
    return { habitat, measured, albedo, normal, ao }
  }

  let maps = await loadMaps(ctx.params.habitat as Habitat)

  // -------------------------------------------------------------------------
  // Pipeline
  // -------------------------------------------------------------------------
  const sampler = device.createSampler({
    label: `${ctx.id}/maps`,
    addressModeU: 'repeat',
    addressModeV: 'repeat',
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    maxAnisotropy: 16,
  })

  const paramsBuffer = ctx.res.createBuffer(
    { label: `${ctx.id}/params`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { species: ctx.id, tag: 'params' },
  )

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/maps`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
    ],
  })

  let bindGroup = makeBindGroup(maps)
  function makeBindGroup(set: MapSet): GPUBindGroup {
    return device.createBindGroup({
      label: `${ctx.id}/maps/${set.habitat}`,
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: sampler },
        { binding: 2, resource: set.albedo.createView() },
        { binding: 3, resource: set.normal.createView() },
        { binding: 4, resource: set.ao.createView() },
      ],
    })
  }

  /**
   * The shader is GENERATED: three decoders emitted from the same
   * TexelSemantics values the mip chains were built with, prepended to the
   * hand-written source. `latestCode` rather than `shaderSrc.code` so a hot
   * edit of the .wgsl (or of anything it includes) is picked up, and the
   * module id carries a hash of the result because ShaderRegistry.module()
   * returns the FIRST code it ever saw for an id.
   */
  const buildCode = (): string =>
    [
      wgslDecoder('albedo_map', { semantics: ALBEDO_SEMANTICS }),
      wgslDecoder('normal_map', { semantics: NORMAL_SEMANTICS }),
      wgslDecoder('ao_map', { semantics: AO_SEMANTICS }),
      ctx.shaders.latestCode(shaderSrc),
    ].join('\n')

  const fnv1a = (s: string): string => {
    let h = 0x811c9dc5
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i)
      h = Math.imul(h, 0x01000193)
    }
    return (h >>> 0).toString(16).padStart(8, '0')
  }

  let pipeline!: GPURenderPipeline
  const build = (): void => {
    const code = buildCode()
    const module = ctx.shaders.module({ id: `${ctx.id}/flat-maps#${fnv1a(code)}`, code })
    pipeline = device.createRenderPipeline({
      label: `${ctx.id}/flat-maps`,
      layout: device.createPipelineLayout({ label: ctx.id, bindGroupLayouts: [ctx.frame.layout, bgl] }),
      vertex: { module, entryPoint: 'vs_main', buffers: [PREVIEW_VERTEX_LAYOUT] },
      fragment: { module, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  // -------------------------------------------------------------------------
  // Per-frame uniforms
  // -------------------------------------------------------------------------
  const uniforms = new Float32Array(4)

  /**
   * TILES PER METRE — ONE isotropic number, never a per-axis fit.
   *
   * The preview uv is already in metres of surface, so life size is simply
   * 1/tileM and the tile is square on every object by construction. The single
   * correction is on an axis that CLOSES on itself: a fractional number of
   * tiles around a sphere or a cylinder butts two different parts of the tile
   * together at the seam, so the period is snapped to the nearest exact
   * divisor — and the snapped value is applied to BOTH axes, because scaling
   * one axis alone is exactly the "stretched to fit the polygons" failure this
   * whole convention exists to prevent. Cost: 0% on the plane and the cube,
   * 2.6% on the sphere, 1.8% on the cylinder.
   */
  const tilesPerMetre = (mesh: { uvPeriod: readonly [number, number] }, tileM: number, scale: number): number => {
    const wanted = 1 / Math.max(tileM * scale, 1e-4)
    const period = mesh.uvPeriod[0] > 0 ? mesh.uvPeriod[0] : mesh.uvPeriod[1]
    if (period <= 0) return wanted
    return Math.max(1, Math.round(period * wanted)) / period
  }

  return {
    update(_frame: FrameInfo): void {
      const mesh = ctx.preview.mesh
      const s = tilesPerMetre(mesh, maps.measured.tileM, ctx.params.tileScale)
      uniforms[0] = s
      uniforms[1] = s
      uniforms[2] = ctx.params.normalStrength
      uniforms[3] = ctx.params.aoStrength
      device.queue.writeBuffer(paramsBuffer, 0, uniforms)
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      // The harness base pass cleared colour to the studio backdrop and cleared
      // depth; the preview object is ours to draw, silhouette and all.
      const mesh = ctx.preview.mesh
      const pass = ctx.timing.renderPass(enc, 'flat-maps', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      pass.setBindGroup(1, bindGroup)
      pass.setVertexBuffer(0, mesh.vertices)
      pass.setIndexBuffer(mesh.indices, 'uint32')
      pass.drawIndexed(mesh.indexCount)
      pass.end()
    },

    onParamsChanged(keys: ReadonlySet<string>): void {
      if (!keys.has('habitat')) return
      const wanted = ctx.params.habitat as Habitat
      if (wanted === maps.habitat) return
      // Generation guard: rapid clicking through the enum must not let an
      // earlier load win the race and leave the wrong state resident.
      const generation = ++habitatGeneration
      void loadMaps(wanted)
        .then((next) => {
          if (generation !== habitatGeneration) {
            for (const t of [next.albedo, next.normal, next.ao]) t.destroy()
            return
          }
          const previous = maps
          maps = next
          bindGroup = makeBindGroup(next)
          for (const t of [previous.albedo, previous.normal, previous.ao]) t.destroy()
        })
        .catch((err: unknown) => {
          console.error(`[${ctx.id}] loading habitat "${wanted}" failed:`, err)
        })
    },

    dispose(): void {
      unsubscribe()
      // Textures and buffers are destroyed by the harness via ctx.res.
    },
  }
}
