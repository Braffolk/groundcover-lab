import cullSrc from './shaders/cull.wgsl'
import splatSrc from './shaders/splat.wgsl'
import depthInitSrc from './shaders/depthinit.wgsl'
import compositeSrc from './shaders/composite.wgsl'
import {
  BUCKET_BASES,
  LOD_COUNTS,
  LOD_OFFSETS,
  TOTAL_RECORDS,
  TOTAL_SPLATS,
  bakeSpeciesSplats,
  isValidSplatBake,
  parseSplatBake,
  type SplatBake,
} from './bake.ts'
import {
  SCATTER_CELL_SIZE,
  bakedArtifact,
  commitBake,
  speciesById,
  type Experiment,
  type ExperimentContext,
  type FrameInfo,
  type ViewTargets,
} from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Splat cloud + screen-space reconstruction.
 *
 * Each species bakes once into ~2.5k anisotropic covariance splats (16B each,
 * 5 LOD levels, ~41KB total). Per frame: (1) a compute pass procedurally
 * culls the stand's scatter inside a fixed camera-centered cell window into
 * per-(entry,LOD) plant buckets — work is window-bounded, never
 * plant-count-bounded; (2) indirect draws scatter dithered elliptical splats
 * into a HALF-RES buffer (albedo+height, normal+viewdepth) depth-tested
 * against downsampled scene depth; (3) a full-res reconstruction pass gathers
 * a 3x3 neighborhood with depth-affinity weights, closing the gaps between
 * sparse splats into continuous foliage, then lights + fogs once per pixel.
 */

const RINGS = [3, 7, 16, 36] as const
const FADE_END = 80
const LODS = LOD_COUNTS.length
const RECORD_BYTES = 16
const DRAWINFO_STRIDE = 256
const COLOR_FMT: GPUTextureFormat = 'rgba8unorm'
const AUX_FMT: GPUTextureFormat = 'rgba16float'
const DEPTH_FMT: GPUTextureFormat = 'depth32float'

interface ViewRes {
  width: number
  height: number
  sceneDepth: GPUTexture
  color: GPUTexture
  aux: GPUTexture
  depth: GPUTexture
  colorView: GPUTextureView
  auxView: GPUTextureView
  depthView: GPUTextureView
  depthInitBG: GPUBindGroup
  compositeBG: GPUBindGroup
}

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx
  const entries = ctx.stand.species
  const entryCount = entries.length

  // Overwrite the harness OPFS bake cache directly — needed to repair entries
  // poisoned by the dev server's SPA fallback (missing baked file => 200 html).
  const opfsPut = async (fullKey: string, data: ArrayBuffer): Promise<void> => {
    try {
      const root = await navigator.storage.getDirectory()
      const dir = await root.getDirectoryHandle('bake-cache', { create: true })
      const handle = await dir.getFileHandle(`${fullKey}.bin`, { create: true })
      const writable = await handle.createWritable()
      await writable.write(data)
      await writable.close()
    } catch {
      /* OPFS unavailable — cache misses just re-bake */
    }
  }

  // --- Baked splat sets (one per distinct species mesh) --------------------
  const bakes = new Map<string, SplatBake>()
  const splatBufs = new Map<string, GPUBuffer>()
  for (const entry of entries) {
    const species = speciesById(entry.species)
    if (bakes.has(species.meshId)) continue
    const key = `splats-v1-${species.meshId}`
    let fresh = false
    let data = await bakedArtifact({ expId: ctx.id, key }, async () => {
      fresh = true
      const mesh = await ctx.meshes.load(species.meshId)
      return bakeSpeciesSplats(mesh)
    })
    if (!isValidSplatBake(data)) {
      fresh = true
      const mesh = await ctx.meshes.load(species.meshId)
      data = bakeSpeciesSplats(mesh)
      await opfsPut(`${ctx.id}__${key}`, data)
    }
    if (fresh) {
      try {
        await commitBake(ctx.id, key, data)
      } catch (err) {
        console.warn(`[${ctx.id}] commitBake failed (static build?):`, err)
      }
    }
    const bake = parseSplatBake(data)
    bakes.set(species.meshId, bake)
    const buf = ctx.res.createBuffer(
      {
        label: `${ctx.id}/splats-${species.meshId}`,
        size: TOTAL_SPLATS * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      },
      { species: species.id, tag: 'splat-lods' },
    )
    device.queue.writeBuffer(buf, 0, bake.records)
    splatBufs.set(species.meshId, buf)
  }

  // --- Frame-persistent buffers -------------------------------------------
  const countsBuf = ctx.res.createBuffer(
    { label: `${ctx.id}/bucket-counts`, size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST },
    { tag: 'cull' },
  )
  const indirectBuf = ctx.res.createBuffer(
    {
      label: `${ctx.id}/indirect`,
      size: 3 * LODS * 16,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE,
    },
    { tag: 'cull' },
  )
  const paramsBuf = ctx.res.createBuffer(
    { label: `${ctx.id}/params`, size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'params' },
  )
  const recordBufs = entries.map((entry, e) =>
    ctx.res.createBuffer(
      {
        label: `${ctx.id}/records-${e}`,
        size: TOTAL_RECORDS * RECORD_BYTES,
        usage: GPUBufferUsage.STORAGE,
      },
      { species: entry.species, tag: 'plant-records' },
    ),
  )

  // Static per-(entry,lod) draw info, addressed via dynamic offsets.
  const drawInfoBuf = ctx.res.createBuffer(
    {
      label: `${ctx.id}/draw-info`,
      size: entryCount * LODS * DRAWINFO_STRIDE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    },
    { tag: 'draw-info' },
  )
  {
    const data = new ArrayBuffer(entryCount * LODS * DRAWINFO_STRIDE)
    const dv = new DataView(data)
    entries.forEach((entry, e) => {
      const bake = bakes.get(speciesById(entry.species).meshId)!
      for (let l = 0; l < LODS; l++) {
        const o = (e * LODS + l) * DRAWINFO_STRIDE
        dv.setFloat32(o + 0, bake.boundsMin[0], true)
        dv.setFloat32(o + 4, bake.boundsMin[1], true)
        dv.setFloat32(o + 8, bake.boundsMin[2], true)
        dv.setFloat32(o + 12, bake.topH, true)
        dv.setFloat32(o + 16, bake.boundsExt[0], true)
        dv.setFloat32(o + 20, bake.boundsExt[1], true)
        dv.setFloat32(o + 24, bake.boundsExt[2], true)
        dv.setFloat32(o + 28, bake.maxRa, true)
        dv.setFloat32(o + 32, bake.maxRb, true)
        dv.setUint32(o + 48, LOD_OFFSETS[l]!, true)
        dv.setUint32(o + 52, LOD_COUNTS[l]!, true)
        dv.setUint32(o + 56, BUCKET_BASES[l]!, true)
        dv.setUint32(o + 60, e, true)
      }
    })
    device.queue.writeBuffer(drawInfoBuf, 0, data)
  }

  // --- Bind group layouts --------------------------------------------------
  const cullBGL = device.createBindGroupLayout({
    label: `${ctx.id}/cull`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  })
  const finalizeBGL = device.createBindGroupLayout({
    label: `${ctx.id}/finalize`,
    entries: [
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  })
  const splatBGL = device.createBindGroupLayout({
    label: `${ctx.id}/splat`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 64 },
      },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    ],
  })
  const depthInitBGL = device.createBindGroupLayout({
    label: `${ctx.id}/depth-init`,
    entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } }],
  })
  const compositeBGL = device.createBindGroupLayout({
    label: `${ctx.id}/composite`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
    ],
  })

  const cullBG = device.createBindGroup({
    label: `${ctx.id}/cull`,
    layout: cullBGL,
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: countsBuf } },
      { binding: 2, resource: { buffer: recordBufs[0]! } },
      { binding: 3, resource: { buffer: recordBufs[Math.min(1, entryCount - 1)]! } },
      { binding: 4, resource: { buffer: recordBufs[Math.min(2, entryCount - 1)]! } },
    ],
  })
  const finalizeBG = device.createBindGroup({
    label: `${ctx.id}/finalize`,
    layout: finalizeBGL,
    entries: [
      { binding: 5, resource: { buffer: countsBuf } },
      { binding: 6, resource: { buffer: indirectBuf } },
    ],
  })
  const splatBGs = entries.map((entry, e) =>
    device.createBindGroup({
      label: `${ctx.id}/splat-${e}`,
      layout: splatBGL,
      entries: [
        { binding: 0, resource: { buffer: drawInfoBuf, offset: 0, size: 64 } },
        { binding: 1, resource: { buffer: recordBufs[e]! } },
        { binding: 2, resource: { buffer: splatBufs.get(speciesById(entry.species).meshId)! } },
        { binding: 3, resource: { buffer: paramsBuf } },
      ],
    }),
  )

  // --- Pipelines -----------------------------------------------------------
  let cullPipeline!: GPUComputePipeline
  let finalizePipeline!: GPUComputePipeline
  let splatPipeline!: GPURenderPipeline
  let depthInitPipeline!: GPURenderPipeline
  let compositePipeline!: GPURenderPipeline
  const build = (): void => {
    const cullModule = ctx.shaders.module(cullSrc)
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/cull`, bindGroupLayouts: [ctx.frame.layout, cullBGL] }),
      compute: { module: cullModule, entryPoint: 'cs_cull' },
    })
    finalizePipeline = device.createComputePipeline({
      label: `${ctx.id}/finalize`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/finalize`,
        bindGroupLayouts: [ctx.frame.layout, finalizeBGL],
      }),
      compute: { module: cullModule, entryPoint: 'cs_finalize' },
    })
    const splatModule = ctx.shaders.module(splatSrc)
    splatPipeline = device.createRenderPipeline({
      label: `${ctx.id}/splats`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/splats`, bindGroupLayouts: [ctx.frame.layout, splatBGL] }),
      vertex: { module: splatModule, entryPoint: 'vs_splat' },
      fragment: {
        module: splatModule,
        entryPoint: 'fs_splat',
        targets: [{ format: COLOR_FMT }, { format: AUX_FMT }],
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: DEPTH_FMT, depthCompare: 'less', depthWriteEnabled: true },
    })
    const depthInitModule = ctx.shaders.module(depthInitSrc)
    depthInitPipeline = device.createRenderPipeline({
      label: `${ctx.id}/depth-init`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/depth-init`,
        bindGroupLayouts: [ctx.frame.layout, depthInitBGL],
      }),
      vertex: { module: depthInitModule, entryPoint: 'vs_fullscreen' },
      fragment: {
        module: depthInitModule,
        entryPoint: 'fs_depth_init',
        targets: [
          { format: COLOR_FMT, writeMask: 0 },
          { format: AUX_FMT, writeMask: 0 },
        ],
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: DEPTH_FMT, depthCompare: 'always', depthWriteEnabled: true },
    })
    const compositeModule = ctx.shaders.module(compositeSrc)
    compositePipeline = device.createRenderPipeline({
      label: `${ctx.id}/reconstruct`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/reconstruct`,
        bindGroupLayouts: [ctx.frame.layout, compositeBGL],
      }),
      vertex: { module: compositeModule, entryPoint: 'vs_fullscreen' },
      fragment: {
        module: compositeModule,
        entryPoint: 'fs_main',
        targets: [
          {
            format: ctx.colorFormat,
            blend: {
              color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-list' },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  // --- Per-view offscreen targets ------------------------------------------
  const viewRes = new Map<string, ViewRes>()
  const ensureViewRes = (targets: ViewTargets): ViewRes => {
    const existing = viewRes.get(targets.view)
    if (
      existing &&
      existing.width === targets.width &&
      existing.height === targets.height &&
      existing.sceneDepth === targets.depthTexture
    ) {
      return existing
    }
    if (existing) {
      existing.color.destroy()
      existing.aux.destroy()
      existing.depth.destroy()
    }
    const hw = Math.max(1, Math.ceil(targets.width / 2))
    const hh = Math.max(1, Math.ceil(targets.height / 2))
    const mk = (label: string, format: GPUTextureFormat): GPUTexture =>
      ctx.res.createTexture(
        {
          label: `${ctx.id}/${label}-${targets.view}`,
          size: [hw, hh],
          format,
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        },
        { tag: 'halfres-targets' },
      )
    const color = mk('splat-color', COLOR_FMT)
    const aux = mk('splat-aux', AUX_FMT)
    const depth = mk('splat-depth', DEPTH_FMT)
    const colorView = color.createView()
    const auxView = aux.createView()
    const depthView = depth.createView()
    const sceneDepthView = targets.depthTexture.createView({ aspect: 'depth-only' })
    const res: ViewRes = {
      width: targets.width,
      height: targets.height,
      sceneDepth: targets.depthTexture,
      color,
      aux,
      depth,
      colorView,
      auxView,
      depthView,
      depthInitBG: device.createBindGroup({
        label: `${ctx.id}/depth-init-${targets.view}`,
        layout: depthInitBGL,
        entries: [{ binding: 0, resource: sceneDepthView }],
      }),
      compositeBG: device.createBindGroup({
        label: `${ctx.id}/composite-${targets.view}`,
        layout: compositeBGL,
        entries: [
          { binding: 0, resource: { buffer: paramsBuf } },
          { binding: 1, resource: colorView },
          { binding: 2, resource: auxView },
          { binding: 3, resource: sceneDepthView },
        ],
      }),
    }
    viewRes.set(targets.view, res)
    return res
  }

  // --- Per-frame state ------------------------------------------------------
  const paramsData = new ArrayBuffer(64)
  const pdv = new DataView(paramsData)
  let dimX = 0
  let dimZ = 0
  const debugModes = { off: 0, coverage: 1, normals: 2, depth: 3 } as const

  return {
    update(frame: FrameInfo): void {
      const p = ctx.params
      const fadeEnd = FADE_END * p.detail
      const fadeStart = fadeEnd * 0.85
      const windowR = Math.ceil(fadeEnd / SCATTER_CELL_SIZE) + 1
      const camCX = Math.floor(frame.camera.pose.x / SCATTER_CELL_SIZE)
      const camCZ = Math.floor(frame.camera.pose.z / SCATTER_CELL_SIZE)
      const r = ctx.stand.radius
      const c0 = Math.floor(-r / SCATTER_CELL_SIZE)
      const c1 = Math.floor(r / SCATTER_CELL_SIZE)
      const x0 = Math.max(camCX - windowR, c0)
      const x1 = Math.min(camCX + windowR, c1)
      const z0 = Math.max(camCZ - windowR, c0)
      const z1 = Math.min(camCZ + windowR, c1)
      dimX = Math.max(0, x1 - x0 + 1)
      dimZ = Math.max(0, z1 - z0 + 1)

      pdv.setInt32(0, x0, true)
      pdv.setInt32(4, z0, true)
      pdv.setUint32(8, ctx.seed >>> 0, true)
      pdv.setUint32(12, entryCount, true)
      RINGS.forEach((ring, i) => pdv.setFloat32(16 + i * 4, ring * p.detail, true))
      pdv.setFloat32(32, fadeStart, true)
      pdv.setFloat32(36, fadeEnd, true)
      pdv.setFloat32(40, p.splatScale, true)
      pdv.setFloat32(44, p.variance, true)
      pdv.setFloat32(48, p.surfaceTol, true)
      pdv.setFloat32(52, p.gapFill, true)
      pdv.setUint32(56, debugModes[p.debug], true)
      pdv.setUint32(60, p.reconstruct ? 1 : 0, true)
      device.queue.writeBuffer(paramsBuf, 0, paramsData)
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const res = ensureViewRes(targets)
      const active = dimX > 0 && dimZ > 0

      if (active) {
        enc.clearBuffer(countsBuf)
        const cull = ctx.timing.computePass(enc, 'cull')
        cull.setPipeline(cullPipeline)
        cull.setBindGroup(0, ctx.frame.bindGroup)
        cull.setBindGroup(1, cullBG)
        cull.dispatchWorkgroups(dimX, dimZ)
        cull.setPipeline(finalizePipeline)
        cull.setBindGroup(1, finalizeBG)
        cull.dispatchWorkgroups(1)
        cull.end()
      }

      const splat = ctx.timing.renderPass(enc, 'splats', {
        colorAttachments: [
          { view: res.colorView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
          { view: res.auxView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
        ],
        depthStencilAttachment: {
          view: res.depthView,
          depthClearValue: 1,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
      })
      splat.setPipeline(depthInitPipeline)
      splat.setBindGroup(0, ctx.frame.bindGroup)
      splat.setBindGroup(1, res.depthInitBG)
      splat.draw(3)
      if (active) {
        splat.setPipeline(splatPipeline)
        for (let e = 0; e < entryCount; e++) {
          for (let l = 0; l < LODS; l++) {
            const i = e * LODS + l
            splat.setBindGroup(1, splatBGs[e]!, [i * DRAWINFO_STRIDE])
            splat.drawIndirect(indirectBuf, i * 16)
          }
        }
      }
      splat.end()

      const composite = ctx.timing.renderPass(enc, 'reconstruct', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
      })
      composite.setPipeline(compositePipeline)
      composite.setBindGroup(0, ctx.frame.bindGroup)
      composite.setBindGroup(1, res.compositeBG)
      composite.draw(3)
      composite.end()
    },

    dispose(): void {
      unsubscribe()
      // GPU resources are reclaimed by the harness via ctx.res.
    },
  }
}
