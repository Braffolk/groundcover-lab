import shaderSrc from './shaders/ribbon.wgsl'
import {
  distillSpecies,
  serializeAtlas,
  parseAtlas,
  toF16,
  SPECIES_OPTIONS,
  K_COLS,
  RIBBONS,
  ROWS_PER_VARIANT,
  VARIANTS,
  ATLAS_VERSION,
} from './bake.ts'
import { commitBake, speciesById, type Experiment, type ExperimentContext, type FrameInfo, type ViewTargets } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Ribbon-skeleton renderer. Each species is distilled ONCE into a few dozen
 * curved ribbons whose shape lives in two tiny rgba16f textures; at runtime the
 * vertex shader unrolls each ribbon into a short triangle strip. Placement is
 * procedural over a bounded camera region (scatter twin), so per-frame cost is
 * independent of the stand's plant count. Two LODs (detail ribbons / a single
 * aggregate silhouette card) share one pipeline and cross-fade by collapsing
 * geometry — no per-frame source geometry, no raymarching, no stochastic alpha.
 */

const CELL = 4 // must equal SCATTER_CELL_SIZE

interface SpeciesGpu {
  posw: GPUTexture
  cola: GPUTexture
  rows: number
}

// One draw = one (entry, LOD) pair with its own meta buffer + bind group, so
// the two draws never contend for a single uniform buffer within the pass.
interface DrawSlot {
  entryIndex: number
  lodMode: 0 | 1
  metaBuffer: GPUBuffer
  bindGroup: GPUBindGroup
}

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx

  // --- distill / load one ribbon atlas per unique species --------------------
  const speciesCache = new Map<string, Promise<SpeciesGpu>>()
  const loadSpecies = (speciesId: string): Promise<SpeciesGpu> => {
    let cached = speciesCache.get(speciesId)
    if (cached) return cached
    const p = (async (): Promise<SpeciesGpu> => {
      const key = `${speciesId}-v${ATLAS_VERSION}-k${K_COLS}-r${RIBBONS}-va${VARIANTS}`
      // NOTE: the harness bakedArtifact() cache is intentionally bypassed — the
      // dev server answers a missing /mesh/baked/*.bin with a 200 index.html,
      // which poisons the OPFS cache. We validate every fetched buffer with
      // parseAtlas (magic+version) and only trust real committed artifacts.
      let atlas = null as ReturnType<typeof parseAtlas>
      const committed = await fetch(`/mesh/baked/${ctx.id}/${key}.bin`).catch(() => null)
      if (committed?.ok) atlas = parseAtlas(await committed.arrayBuffer())
      if (!atlas) {
        const mesh = await ctx.meshes.load(speciesById(speciesId).meshId)
        const rows = distillSpecies(
          {
            vertexCount: mesh.header.vertexCount,
            boundsMin: mesh.header.boundsMin,
            boundsMax: mesh.header.boundsMax,
            tileSize: mesh.header.tileSize,
            topH: mesh.header.topH,
            vertices: mesh.vertices,
          },
          { ...SPECIES_OPTIONS[speciesId], seed: 7 },
        )
        const buf = serializeAtlas(new Map([[0, rows]]))
        atlas = parseAtlas(buf)
        void commitBake(ctx.id, key, buf).catch(() => {})
      }
      if (!atlas) throw new Error(`ribbon-skeleton: bad atlas for ${speciesId}`)

      const rows = atlas.rowsTotal
      const mk = (label: string): GPUTexture =>
        ctx.res.createTexture(
          {
            label: `${ctx.id}/${speciesId}/${label}`,
            size: [K_COLS, rows],
            format: 'rgba16float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
          },
          { species: speciesId, tag: `ribbon-${label}` },
        )
      const posw = mk('posw')
      const cola = mk('cola')
      device.queue.writeTexture({ texture: posw }, toF16(atlas.t0), { bytesPerRow: K_COLS * 8, rowsPerImage: rows }, [K_COLS, rows])
      device.queue.writeTexture({ texture: cola }, toF16(atlas.t1), { bytesPerRow: K_COLS * 8, rowsPerImage: rows }, [K_COLS, rows])
      return { posw, cola, rows }
    })()
    speciesCache.set(speciesId, p)
    return p
  }

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, texture: { sampleType: 'unfilterable-float' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, texture: { sampleType: 'unfilterable-float' } },
    ],
  })

  const gpus = await Promise.all(ctx.stand.species.map((e) => loadSpecies(e.species)))

  const slots: DrawSlot[] = []
  ctx.stand.species.forEach((e, entryIndex) => {
    const gpu = gpus[entryIndex]!
    for (const lodMode of [0, 1] as const) {
      const metaBuffer = ctx.res.createBuffer(
        { label: `${ctx.id}/meta-${entryIndex}-${lodMode}`, size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
        { species: e.species, tag: 'meta' },
      )
      const bindGroup = device.createBindGroup({
        label: `${ctx.id}/bg-${entryIndex}-${lodMode}`,
        layout: bgl,
        entries: [
          { binding: 0, resource: { buffer: metaBuffer } },
          { binding: 1, resource: gpu.posw.createView() },
          { binding: 2, resource: gpu.cola.createView() },
        ],
      })
      slots.push({ entryIndex, lodMode, metaBuffer, bindGroup })
    }
  })

  let pipeline!: GPURenderPipeline
  const build = (): void => {
    const module = ctx.shaders.module(shaderSrc)
    pipeline = device.createRenderPipeline({
      label: `${ctx.id}/ribbons`,
      layout: device.createPipelineLayout({ label: ctx.id, bindGroupLayouts: [ctx.frame.layout, bgl] }),
      vertex: { module, entryPoint: 'vs_main' },
      fragment: { module, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
      primitive: { topology: 'triangle-strip', cullMode: 'none' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const meta = new Float32Array(16)
  let detailSide = 1
  let aggSide = 1

  return {
    update(frame: FrameInfo): void {
      const cam = frame.camera.pose
      const rDetail = ctx.params.detailRadius + ctx.params.blendBand
      const detailOriginX = Math.floor((cam.x - rDetail) / CELL)
      const detailOriginZ = Math.floor((cam.z - rDetail) / CELL)
      detailSide = Math.max(1, Math.ceil((2 * rDetail) / CELL) + 1)

      const rAgg = ctx.params.regionRadius
      const aggOriginX = Math.floor((cam.x - rAgg) / CELL)
      const aggOriginZ = Math.floor((cam.z - rAgg) / CELL)
      aggSide = Math.max(1, Math.ceil((2 * rAgg) / CELL) + 1)

      for (const s of slots) {
        const detail = s.lodMode === 0
        meta[0] = 0 // row_offset (one species per atlas)
        meta[1] = VARIANTS
        meta[2] = ROWS_PER_VARIANT
        meta[3] = RIBBONS
        meta[4] = detail ? detailOriginX : aggOriginX
        meta[5] = detail ? detailOriginZ : aggOriginZ
        meta[6] = detail ? detailSide : aggSide
        meta[7] = ctx.seed
        meta[8] = s.entryIndex
        meta[9] = s.lodMode
        meta[10] = ctx.params.detailRadius
        meta[11] = ctx.params.detailRadius + ctx.params.blendBand
        meta[12] = ctx.params.regionRadius
        meta[13] = K_COLS
        meta[14] = ctx.params.widthScale
        meta[15] = 0
        device.queue.writeBuffer(s.metaBuffer, 0, meta)
      }
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const pass = ctx.timing.renderPass(enc, 'ribbon-skeleton', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)

      const detailInstances = detailSide * detailSide * 128 * RIBBONS
      const aggInstances = aggSide * aggSide * 128
      for (const s of slots) {
        pass.setBindGroup(1, s.bindGroup)
        pass.draw(24, s.lodMode === 0 ? detailInstances : aggInstances)
      }
      pass.end()
    },

    dispose(): void {
      unsubscribe()
    },
  }
}
