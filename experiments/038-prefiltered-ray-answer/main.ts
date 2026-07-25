import canopySrc from './shaders/canopy.wgsl'
import mipgenSrc from './shaders/mipgen.wgsl'
import { BANDS, LAYERS_PER_BAND, loadRayTable, meanSway, RES, TILE_L } from './bake.ts'
import type { Experiment, ExperimentContext, FrameInfo, ViewTargets } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Runtime for the prefiltered ray-answer canopy.
 *
 * ONE draw of ONE coarse terrain-conformal grid (CARRIER_SIDE^2 quads — a fixed
 * constant: 557k plants and 134M plants rasterize exactly the same triangles
 * and read exactly the same table). Every plant exists only inside the baked
 * answer table; the fragment shader resolves each pixel's eye ray in a constant
 * 5 texture fetches with no marching. See NOTES.md.
 */

const CARRIER_SIDE = 240
const MIP_LEVELS = 8 // 192 96 48 24 12 6 3 1

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx
  const table = await loadRayTable(ctx, ctx.seed)
  const canopyH = table.canopyH
  const sway = meanSway(ctx.stand)
  const regionHalf = Math.min(ctx.stand.radius, ctx.scene.terrain.desc.size * 0.5 - 1)

  // --- table upload: three (surf, geom) pairs, one per species budget row ---
  const bandTex: { surf: GPUTexture; geom: GPUTexture }[] = []
  const bytesPerLayer = RES * RES * 4
  for (let b = 0; b < BANDS; b++) {
    const species = ctx.stand.species[b]?.species ?? ctx.stand.species[0]!.species
    const mk = (name: string): GPUTexture =>
      ctx.res.createTexture(
        {
          label: `${ctx.id}/${name}-${b}`,
          size: [RES, RES, LAYERS_PER_BAND],
          format: 'rgba8unorm',
          mipLevelCount: MIP_LEVELS,
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
        },
        { species, tag: 'ray-answer' },
      )
    const surf = mk('answer-surf')
    const geom = mk('answer-geom')
    const offset = b * LAYERS_PER_BAND * bytesPerLayer
    const span = LAYERS_PER_BAND * bytesPerLayer
    device.queue.writeTexture(
      { texture: surf },
      table.surf.subarray(offset, offset + span),
      { bytesPerRow: RES * 4, rowsPerImage: RES },
      [RES, RES, LAYERS_PER_BAND],
    )
    device.queue.writeTexture(
      { texture: geom },
      table.geom.subarray(offset, offset + span),
      { bytesPerRow: RES * 4, rowsPerImage: RES },
      [RES, RES, LAYERS_PER_BAND],
    )
    bandTex.push({ surf, geom })
  }

  // --- mip chain: THE prefilter (plain box over premultiplied channels) -----
  {
    const bgl = device.createBindGroupLayout({
      label: `${ctx.id}/mip-bgl`,
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: 'write-only', format: 'rgba8unorm', viewDimension: '2d-array' },
        },
      ],
    })
    const pipe = device.createComputePipeline({
      label: `${ctx.id}/mipgen`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/mipgen`, bindGroupLayouts: [bgl] }),
      compute: { module: ctx.shaders.module(mipgenSrc), entryPoint: 'main' },
    })
    const enc = device.createCommandEncoder({ label: `${ctx.id}/mipgen` })
    const pass = enc.beginComputePass({ label: `${ctx.id}/mipgen` })
    pass.setPipeline(pipe)
    for (const band of bandTex) {
      for (const t of [band.surf, band.geom]) {
        for (let level = 1; level < MIP_LEVELS; level++) {
          const size = Math.max(1, RES >> level)
          pass.setBindGroup(
            0,
            device.createBindGroup({
              label: `${ctx.id}/mip-${level}`,
              layout: bgl,
              entries: [
                {
                  binding: 0,
                  resource: t.createView({ dimension: '2d-array', baseMipLevel: level - 1, mipLevelCount: 1 }),
                },
                {
                  binding: 1,
                  resource: t.createView({ dimension: '2d-array', baseMipLevel: level, mipLevelCount: 1 }),
                },
              ],
            }),
          )
          pass.dispatchWorkgroups(Math.ceil(size / 8), Math.ceil(size / 8), LAYERS_PER_BAND)
        }
      }
    }
    pass.end()
    device.queue.submit([enc.finish()])
  }

  const sampler = device.createSampler({
    label: `${ctx.id}/answer-samp`,
    addressModeU: 'repeat',
    addressModeV: 'repeat',
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    maxAnisotropy: 16,
  })

  const cfgBuf = ctx.res.createBuffer(
    { label: `${ctx.id}/cfg`, size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'params' },
  )

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ...[2, 3, 4, 5, 6, 7].map((binding) => ({
        binding,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: 'float' as GPUTextureSampleType, viewDimension: '2d-array' as GPUTextureViewDimension },
      })),
    ],
  })
  const bindGroup = device.createBindGroup({
    label: `${ctx.id}/bindings`,
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: cfgBuf } },
      { binding: 1, resource: sampler },
      { binding: 2, resource: bandTex[0]!.surf.createView({ dimension: '2d-array' }) },
      { binding: 3, resource: bandTex[0]!.geom.createView({ dimension: '2d-array' }) },
      { binding: 4, resource: bandTex[1]!.surf.createView({ dimension: '2d-array' }) },
      { binding: 5, resource: bandTex[1]!.geom.createView({ dimension: '2d-array' }) },
      { binding: 6, resource: bandTex[2]!.surf.createView({ dimension: '2d-array' }) },
      { binding: 7, resource: bandTex[2]!.geom.createView({ dimension: '2d-array' }) },
    ],
  })

  let pipeline!: GPURenderPipeline
  const build = (): void => {
    const module = ctx.shaders.module(canopySrc)
    pipeline = device.createRenderPipeline({
      label: `${ctx.id}/canopy`,
      layout: device.createPipelineLayout({ label: ctx.id, bindGroupLayouts: [ctx.frame.layout, bgl] }),
      vertex: { module, entryPoint: 'vs_main' },
      fragment: {
        module,
        entryPoint: 'fs_main',
        targets: [
          {
            format: ctx.colorFormat,
            // Premultiplied: exactly one canopy layer resolves per pixel, so
            // honest partial coverage needs no sorting and no dithering.
            blend: {
              color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const cfg = new Float32Array(16)
  const vertexCount = CARRIER_SIDE * CARRIER_SIDE * 6

  return {
    update(frame: FrameInfo): void {
      // The rasterizer can only find a ray/carrier crossing IN FRONT of the eye,
      // so when the eye sits inside the canopy the carrier plane is dropped
      // below it. The canopy itself keeps its full height — only the surface
      // being intersected moves, and the entry point slides back up to the real
      // canopy top analytically.
      const camGround = ctx.scene.terrain.height(frame.camera.pose.x, frame.camera.pose.z)
      const clearance = frame.camera.pose.y - camGround - ctx.params.eyeClearance
      const carrierH = Math.max(0.12, Math.min(canopyH, clearance))
      cfg[0] = canopyH
      cfg[1] = TILE_L
      cfg[2] = carrierH
      cfg[3] = regionHalf
      cfg[4] = ctx.params.warpAmp
      cfg[5] = ctx.params.aoStrength
      cfg[6] = ctx.params.sharpen
      cfg[7] = ctx.params.lodBias
      cfg[8] = ctx.params.carrierCell
      cfg[9] = CARRIER_SIDE
      cfg[10] = ctx.params.snapTol
      cfg[11] = ctx.params.correct ? 1 : 0
      cfg[12] = sway
      cfg[13] = 0.35
      cfg[14] = ctx.params.azBlend ? 1 : 0
      cfg[15] = ctx.params.detail
      device.queue.writeBuffer(cfgBuf, 0, cfg)
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const pass = ctx.timing.renderPass(enc, 'ray-answer', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      pass.setBindGroup(1, bindGroup)
      pass.draw(vertexCount)
      pass.end()
    },

    dispose(): void {
      unsubscribe()
      // ctx.res owns every allocation.
    },
  }
}
