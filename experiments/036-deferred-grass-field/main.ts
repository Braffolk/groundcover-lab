import fieldSrc from './shaders/field.wgsl'
import mipSrc from './shaders/mip.wgsl'
import { loadFieldTable } from './bake.ts'
import { AZ, ELEV, ELEV_DQ, ELEV_Q0, GEOM_PX, TILE_M, TILE_PX, mipCount } from './field.ts'
import type { Experiment, ExperimentContext, FrameInfo, ViewTargets } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * DEFERRED GRASS FIELD — a canopy that exists only as the answer to a
 * per-pixel ray query.
 *
 * Rasterized per frame: ONE fullscreen triangle (1 primitive, 1 draw call).
 * There is no plant geometry anywhere in the frame — no card, quad, ribbon,
 * prism or shell, for any plant, at any distance. Every blade you see is the
 * baked table's answer to "what does this eye ray meet on its way into the
 * canopy", put back onto the ray in closed form.
 *
 * Cost is bounded by screen pixels, not by plants: 560k plants and 134M plants
 * issue exactly the same GPU work (the stand only ever appears in the bake).
 */
export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx
  const table = await loadFieldTable(ctx, (note) => console.info(`[${ctx.id}] ${note}`))

  // --- table upload -------------------------------------------------------
  // The table is ONE composite of every species in the stand (a ray does not
  // care which species it hits), so its bytes are split evenly across the
  // stand's species for the budget meter — see NOTES.md.
  const speciesIds = [...new Set(ctx.stand.species.map((e) => e.species))]
  const slabSpecies = (i: number): string | undefined => speciesIds[i % speciesIds.length]

  const surfTex: GPUTexture[] = []
  const geomTex: GPUTexture[] = []
  for (let e = 0; e < ELEV; e++) {
    const species = slabSpecies(e)
    const mk = (label: string, px: number, format: GPUTextureFormat): GPUTexture =>
      ctx.res.createTexture(
        {
          label: `${ctx.id}/${label}`,
          size: [px, px, AZ],
          dimension: '3d',
          format,
          mipLevelCount: mipCount(px),
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
        },
        { ...(species !== undefined && { species }), tag: 'ray-answer-table' },
      )
    const s = mk(`surf-e${e}`, TILE_PX, 'rgba8unorm')
    const g = mk(`geom-e${e}`, GEOM_PX, 'rgba8snorm')
    device.queue.writeTexture(
      { texture: s },
      table.surf[e]!,
      { bytesPerRow: TILE_PX * 4, rowsPerImage: TILE_PX },
      [TILE_PX, TILE_PX, AZ],
    )
    device.queue.writeTexture(
      { texture: g },
      table.geom[e]!,
      { bytesPerRow: GEOM_PX * 4, rowsPerImage: GEOM_PX },
      [GEOM_PX, GEOM_PX, AZ],
    )
    surfTex.push(s)
    geomTex.push(g)
  }

  // --- mip chain (once, at load) ------------------------------------------
  {
    const module = ctx.shaders.module(mipSrc)
    const mkLayout = (format: GPUTextureFormat, binding: number): GPUBindGroupLayout =>
      device.createBindGroupLayout({
        label: `${ctx.id}/mip-${format}`,
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '3d' } },
          {
            binding,
            visibility: GPUShaderStage.COMPUTE,
            storageTexture: { access: 'write-only', format, viewDimension: '3d' },
          },
        ],
      })
    const layoutU = mkLayout('rgba8unorm', 1)
    const layoutS = mkLayout('rgba8snorm', 2)
    const mkPipe = (bgl: GPUBindGroupLayout, entryPoint: string): GPUComputePipeline =>
      device.createComputePipeline({
        label: `${ctx.id}/${entryPoint}`,
        layout: device.createPipelineLayout({ label: `${ctx.id}/${entryPoint}`, bindGroupLayouts: [bgl] }),
        compute: { module, entryPoint },
      })
    const pipeU = mkPipe(layoutU, 'mip_unorm')
    const pipeS = mkPipe(layoutS, 'mip_snorm')

    const enc = device.createCommandEncoder({ label: `${ctx.id}/mipgen` })
    const pass = enc.beginComputePass({ label: `${ctx.id}/mipgen` })
    const chain = (tex: GPUTexture, px: number, snorm: boolean): void => {
      const levels = mipCount(px)
      pass.setPipeline(snorm ? pipeS : pipeU)
      for (let level = 1; level < levels; level++) {
        const w = Math.max(1, px >> level)
        const dz = Math.max(1, AZ >> level)
        const bg = device.createBindGroup({
          label: `${ctx.id}/mip-${level}`,
          layout: snorm ? layoutS : layoutU,
          entries: [
            { binding: 0, resource: tex.createView({ dimension: '3d', baseMipLevel: level - 1, mipLevelCount: 1 }) },
            {
              binding: snorm ? 2 : 1,
              resource: tex.createView({ dimension: '3d', baseMipLevel: level, mipLevelCount: 1 }),
            },
          ],
        })
        pass.setBindGroup(0, bg)
        pass.dispatchWorkgroups(Math.ceil(w / 4), Math.ceil(w / 4), Math.ceil(dz / 4))
      }
    }
    for (let e = 0; e < ELEV; e++) {
      chain(surfTex[e]!, TILE_PX, false)
      chain(geomTex[e]!, GEOM_PX, true)
    }
    pass.end()
    device.queue.submit([enc.finish()])
  }

  // --- runtime pass -------------------------------------------------------
  const sampler = device.createSampler({
    label: `${ctx.id}/field`,
    addressModeU: 'repeat',
    addressModeV: 'repeat',
    addressModeW: 'repeat',
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
  })

  const uniforms = new Float32Array(24)
  const paramsBuffer = ctx.res.createBuffer(
    { label: `${ctx.id}/params`, size: uniforms.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'params' },
  )

  const texEntry = (binding: number): GPUBindGroupLayoutEntry => ({
    binding,
    visibility: GPUShaderStage.FRAGMENT,
    texture: { sampleType: 'float', viewDimension: '3d' },
  })
  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/field-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', minBindingSize: 96 } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ...Array.from({ length: ELEV }, (_, i) => texEntry(2 + i)),
      ...Array.from({ length: ELEV }, (_, i) => texEntry(2 + ELEV + i)),
      { binding: 2 + 2 * ELEV, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
    ],
  })

  const bindGroups = new Map<GPUTexture, GPUBindGroup>()
  const bindGroupFor = (depth: GPUTexture): GPUBindGroup => {
    let bg = bindGroups.get(depth)
    if (!bg) {
      bg = device.createBindGroup({
        label: `${ctx.id}/field-bg`,
        layout: bgl,
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: sampler },
          ...surfTex.map((t, i) => ({ binding: 2 + i, resource: t.createView({ dimension: '3d' }) })),
          ...geomTex.map((t, i) => ({ binding: 2 + ELEV + i, resource: t.createView({ dimension: '3d' }) })),
          { binding: 2 + 2 * ELEV, resource: depth.createView({ aspect: 'depth-only' }) },
        ],
      })
      bindGroups.set(depth, bg)
    }
    return bg
  }

  let pipeline!: GPURenderPipeline
  const build = (): void => {
    const module = ctx.shaders.module(fieldSrc)
    pipeline = device.createRenderPipeline({
      label: `${ctx.id}/grass-field`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/grass-field`,
        bindGroupLayouts: [ctx.frame.layout, bgl],
      }),
      vertex: { module, entryPoint: 'vs_fullscreen' },
      fragment: {
        module,
        entryPoint: 'fs_main',
        targets: [
          {
            format: ctx.colorFormat,
            // Coverage IS alpha: a partially covered pixel composites honestly
            // over whatever the base pass left behind it (soil or sky). No
            // dithering anywhere, and distant grass is an averaged canopy.
            blend: {
              color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-list' },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  // Density-weighted mean sway of the stand: one composite table can only be
  // sheared by one amount, so this is the honest single number to shear it by.
  const meanSway = ((): number => {
    let num = 0
    let den = 0
    for (const e of ctx.stand.species) {
      num += e.sway * e.density
      den += e.density
    }
    return den > 0 ? num / den : 0.6
  })()

  return {
    update(_frame: FrameInfo): void {
      const p = ctx.params
      uniforms.set([table.canopyH, TILE_M, TILE_PX, ctx.stand.radius], 0)
      uniforms.set([ELEV_Q0, ELEV_DQ, ELEV - 1, AZ], 4)
      uniforms.set([p.covThresh, p.aoRate, p.aoFloor, p.detailAmp], 8)
      uniforms.set([p.maxDist, meanSway, p.lodBias, p.edgeSharp], 12)
      uniforms.set([p.detailFreq, p.windScale, p.macroAmp, p.elevLerp ? 1 : 0], 16)
      uniforms.set([p.alignDepth, p.reliefAmp, p.farShade, 0], 20)
      device.queue.writeBuffer(paramsBuffer, 0, uniforms)
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const pass = ctx.timing.renderPass(enc, 'grass-field', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      pass.setBindGroup(1, bindGroupFor(targets.depthTexture))
      pass.draw(3)
      pass.end()
    },

    resize(): void {
      bindGroups.clear()
    },

    dispose(): void {
      unsubscribe()
      bindGroups.clear()
    },
  }
}
