import canopySrc from './shaders/canopy.wgsl'
import mipgenSrc from './shaders/mipgen.wgsl'
import { loadEntryTable } from './bake.ts'
import { LAYERS, MIP_LEVELS, TABLE_U, type CanopyTable } from './layout.ts'
import type { Experiment, ExperimentContext, FrameInfo, ViewTargets } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * RAYCAST CANOPY VOLUME — runtime.
 *
 * Rasterized geometry: ONE camera-centred polar shell conformal to the terrain
 * (144 spokes x ~140 rings, 1-2 instances, ~120k triangles) and nothing else.
 * No card, quad, ribbon or prism exists anywhere in this renderer; the plants
 * are purely the answer to a per-pixel ray query against the baked table, so
 * the primitive count is identical for 20 plants and for 134 million.
 */

const SPOKES = 144 // must match NS in canopy.wgsl
const RING_GROWTH = 1.055
const RING_MIN = 0.1
/** Carrier A sits this far above the tallest canopy (covers the sky fringe). */
const CARRIER_A_MARGIN = 0.6
/** Carrier B sits this far below the eye when the camera is inside the canopy. */
const CARRIER_B_DROP = 0.04
/** The entry plane is kept below carrier B by this much (see NOTES). */
const START_MARGIN = 0.1

/** Entry = tile/shade/meanq/meanq4 (4 vec4). Deliberately tiny: the fragment
 * shader indexes it per pixel, and a fat struct would be copied per access. */
const ENTRY_FLOATS = 16
const TAIL_FLOATS = 16
const UNIFORM_FLOATS = ENTRY_FLOATS * 3 + TAIL_FLOATS
/** 3 entries x 32 height bins x 2 clusters x vec4 — its own binding. */
const PALETTE_FLOATS = 3 * 32 * 2 * 4

interface EntryGpu {
  table: CanopyTable
  texture: GPUTexture
  speciesId: string
}

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx
  const note = (msg: string): void => console.info(`[${ctx.id}] ${msg}`)

  const standEntries = ctx.stand.species.slice(0, 3)
  if (ctx.stand.species.length > 3) {
    note(`stand has ${ctx.stand.species.length} entries; this renderer binds the first 3`)
  }

  // ---- baked ray-answer tables --------------------------------------------
  const entries: EntryGpu[] = []
  for (let i = 0; i < standEntries.length; i++) {
    const entry = standEntries[i]!
    const table = await loadEntryTable(
      ctx,
      { entry, entryIndex: i, scatter: ctx.scene.scatter, seed: ctx.seed },
      note,
    )
    const texture = ctx.res.createTexture(
      {
        label: `${ctx.id}/ray-answer-${i}-${entry.species}`,
        size: [TABLE_U, TABLE_U, LAYERS],
        dimension: '2d',
        format: 'rgba8unorm',
        mipLevelCount: MIP_LEVELS,
        usage:
          GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
      },
      { species: entry.species, tag: 'ray-answer-table' },
    )
    device.queue.writeTexture(
      { texture },
      table.data,
      { bytesPerRow: TABLE_U * 4, rowsPerImage: TABLE_U },
      [TABLE_U, TABLE_U, LAYERS],
    )
    entries.push({ table, texture, speciesId: entry.species })
    note(
      `${entry.species}: ${table.plants} plants / ${table.period.toFixed(2)}m tile, ` +
        `H ${table.height.toFixed(2)}m, texel ${((table.period / TABLE_U) * 100).toFixed(2)}cm`,
    )
  }
  buildMips(ctx, entries)

  // A 1x1 stand-in keeps the bind group complete when the stand has <3 entries.
  const dummy = ctx.res.createTexture(
    {
      label: `${ctx.id}/unused-table`,
      size: [1, 1, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    },
    { tag: 'placeholder' },
  )
  device.queue.writeTexture({ texture: dummy }, new Uint8Array([255, 0, 128, 128]), { bytesPerRow: 4 }, [1, 1, 1])

  // ---- carrier shell rings -------------------------------------------------
  // Capped: past ~420m the canopy is fog, so the shell's extent (and therefore
  // its triangle count) stops growing with the stand — 134M plants rasterize
  // the same shell as 20.
  const rMax = Math.min(ctx.stand.radius * 1.45 + 6, 420)
  const radii: number[] = [0]
  let r = RING_MIN
  while (r < rMax) {
    radii.push(r)
    r *= RING_GROWTH
  }
  radii.push(rMax)
  const rings = radii.length - 1
  const vertexCount = SPOKES * rings * 6
  const ringBuf = ctx.res.createBuffer(
    {
      label: `${ctx.id}/ring-radii`,
      size: radii.length * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    },
    { tag: 'carrier-rings' },
  )
  device.queue.writeBuffer(ringBuf, 0, new Float32Array(radii))
  note(
    `carrier shell: ${SPOKES} spokes x ${rings} rings = ${(vertexCount / 3).toLocaleString()} triangles/instance`,
  )

  // ---- uniforms ------------------------------------------------------------
  const uniforms = new Float32Array(UNIFORM_FLOATS)
  const uniformBuf = ctx.res.createBuffer(
    {
      label: `${ctx.id}/params`,
      size: uniforms.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    },
    { tag: 'params' },
  )
  const paletteBuf = ctx.res.createBuffer(
    {
      label: `${ctx.id}/palette`,
      size: PALETTE_FLOATS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    },
    { tag: 'palette' },
  )
  {
    const pal = new Float32Array(PALETTE_FLOATS)
    entries.forEach((e, i) => pal.set(e.table.palette, i * 32 * 2 * 4))
    device.queue.writeBuffer(paletteBuf, 0, pal)
  }

  const sampler = device.createSampler({
    label: `${ctx.id}/table-sampler`,
    addressModeU: 'repeat',
    addressModeV: 'repeat',
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    // The lookup footprint is extremely anisotropic at grazing angles (the
    // entry point sweeps along the ray); anisotropy keeps the across-ray detail
    // while averaging along it.
    maxAnisotropy: 8,
  })

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/bindings`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: '2d-array' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: '2d-array' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: '2d-array' } },
      { binding: 5, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 6, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    ],
  })
  const tableView = (i: number): GPUTextureView =>
    (entries[i]?.texture ?? dummy).createView({ dimension: '2d-array' })
  const bindGroup = device.createBindGroup({
    label: `${ctx.id}/bg`,
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: sampler },
      { binding: 2, resource: tableView(0) },
      { binding: 3, resource: tableView(1) },
      { binding: 4, resource: tableView(2) },
      { binding: 5, resource: { buffer: ringBuf } },
      { binding: 6, resource: { buffer: paletteBuf } },
    ],
  })

  let pipeline!: GPURenderPipeline
  const build = (): void => {
    const module = ctx.shaders.module(canopySrc)
    pipeline = device.createRenderPipeline({
      label: `${ctx.id}/canopy`,
      layout: device.createPipelineLayout({ label: ctx.id, bindGroupLayouts: [ctx.frame.layout, bgl] }),
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: ctx.colorFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  // ---- static uniform block ------------------------------------------------
  const writeStatic = (): void => {
    for (let i = 0; i < 3; i++) {
      const base = i * ENTRY_FLOATS
      const e = entries[i]
      if (!e) {
        uniforms.fill(0, base, base + ENTRY_FLOATS)
        continue
      }
      const t = e.table
      uniforms.set([t.period, t.height, t.psi, 1], base)
      uniforms.set([standEntries[i]!.sway, 1 - ctx.params.aoStrength, 1.4, 0], base + 4)
      uniforms.set([t.meanQ[0]!, t.meanQ[1]!, t.meanQ[2]!, t.meanQ[3]!], base + 8)
      uniforms.set([t.meanQ[4]!, 0, 0, 0], base + 12)
    }
    device.queue.writeBuffer(uniformBuf, 0, uniforms, 0, ENTRY_FLOATS * 3)
  }
  writeStatic()

  const maxCanopy = entries.reduce((m, e) => Math.max(m, e.table.height), 0.5)
  const tail = new Float32Array(TAIL_FLOATS)
  let instanceCount = 1

  return {
    update(frame: FrameInfo): void {
      const pose = frame.camera.pose
      const camLocal = pose.y - ctx.scene.terrain.height(pose.x, pose.z)
      const carrierA = maxCanopy + CARRIER_A_MARGIN
      const carrierB = Math.max(0.05, camLocal - CARRIER_B_DROP)
      instanceCount = camLocal < carrierA ? 2 : 1
      const pixel = (2 * Math.tan(((pose.fov * Math.PI) / 180) * 0.5)) / Math.max(ctx.size().height, 1)

      tail.set([standEntries.length, ctx.stand.radius, ctx.params.alphaRef, ctx.params.lodBias], 0)
      tail.set([pixel, ctx.params.nearFade, ctx.params.detail, ctx.params.tintVar], 4)
      tail.set([carrierA, carrierB, START_MARGIN, ctx.params.windScale], 8)
      tail.set([ctx.params.bandBlend ? 1 : 0, ctx.params.shearFix, camLocal, maxCanopy + 0.35], 12)
      device.queue.writeBuffer(uniformBuf, ENTRY_FLOATS * 3 * 4, tail)
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const pass = ctx.timing.renderPass(enc, 'canopy-rays', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      pass.setBindGroup(1, bindGroup)
      pass.draw(vertexCount, instanceCount)
      pass.end()
    },

    onParamsChanged(keys: ReadonlySet<string>): void {
      if (keys.has('aoStrength')) writeStatic()
    },

    dispose(): void {
      unsubscribe()
    },
  }
}

/** Coverage-weighted mip chain over the table array (one dispatch per level). */
function buildMips(ctx: Pick<ExperimentContext, 'id' | 'device' | 'shaders'>, entries: EntryGpu[]): void {
  const { device } = ctx
  const module = ctx.shaders.module(mipgenSrc)
  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/mipgen-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { viewDimension: '2d-array' } },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba8unorm', viewDimension: '2d-array' },
      },
    ],
  })
  const pipeline = device.createComputePipeline({
    label: `${ctx.id}/mipgen`,
    layout: device.createPipelineLayout({ label: `${ctx.id}/mipgen-pl`, bindGroupLayouts: [bgl] }),
    compute: { module, entryPoint: 'reduce' },
  })
  const enc = device.createCommandEncoder({ label: `${ctx.id}/mipgen-enc` })
  for (const e of entries) {
    for (let level = 1; level < MIP_LEVELS; level++) {
      const w = Math.max(1, TABLE_U >> level)
      const bg = device.createBindGroup({
        label: `${ctx.id}/mipgen-bg-${level}`,
        layout: bgl,
        entries: [
          {
            binding: 0,
            resource: e.texture.createView({
              dimension: '2d-array',
              baseMipLevel: level - 1,
              mipLevelCount: 1,
            }),
          },
          {
            binding: 1,
            resource: e.texture.createView({
              dimension: '2d-array',
              baseMipLevel: level,
              mipLevelCount: 1,
            }),
          },
        ],
      })
      // One pass per level: level N is read by level N+1, so they must be
      // separate usage scopes.
      const pass = enc.beginComputePass({ label: `${ctx.id}/mipgen-${level}` })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bg)
      pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(w / 8), LAYERS)
      pass.end()
    }
  }
  device.queue.submit([enc.finish()])
}
