import cullSrc from './shaders/cull.wgsl'
import relightSrc from './shaders/relight.wgsl'
import { ATLAS, MIPS, levelPx, loadSpeciesRelief, type ReliefAtlas } from './bake.ts'
import {
  SCATTER_CELL_SIZE,
  SCATTER_MAX_DENSITY,
  SCATTER_MAX_PER_CELL,
  type Experiment,
  type ExperimentContext,
  type FrameInfo,
  type ViewTargets,
} from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Relief-lit cards. Startup: each species' relief atlas (8 side views + 1
 * canopy view; albedo+coverage and normal/heightfield/canopy-occlusion, with a
 * per-tile mip chain) is loaded from mesh/baked / OPFS or baked in-browser from
 * the raw mesh, then uploaded level by level.
 *
 * Per frame: one compute pass evaluates the shared scatter over a camera-
 * centered cell region, frustum-culls, and compacts survivors into FOUR
 * distance rings, each with its own indirect draw. The rings are then submitted
 * near->far so early-z discards the deep layers of a grazing view instead of
 * shading them — the overdraw budget that pays for the relief taps. Per-frame
 * cost is O(visible region), independent of the stand's plant count.
 */

const CELL = SCATTER_CELL_SIZE
const REGION_MAX = 128 // keep equal to the manifest's regionRadius max
const INFO_FLOATS = 64
const N_RINGS = 4
/**
 * Ring boundaries (m) == LOD tier boundaries, fixed at init so ring capacities
 * never reallocate and the tier can be an override constant in the shader.
 * Tuned by A/B ratio against the billboard baseline at cam=grazing: the relief
 * solve is only legible while a plant is more than ~40 px tall, and the
 * self-shadow features (~10 cm) only while it is more than ~100 px.
 */
const RING_R = [7, 24, 55, REGION_MAX]
const VERTS_PER_PLANT = 12

interface SpeciesGpu {
  atlas: ReliefAtlas
  albedoTex: GPUTexture
  geomTex: GPUTexture
}

interface EntryGpu {
  speciesId: string
  gpu: SpeciesGpu
  ringBase: number[]
  ringCap: number[]
  infoBuffer: GPUBuffer
  indirectBuffer: GPUBuffer
  cullBindGroup: GPUBindGroup
  drawBindGroup: GPUBindGroup
  slotsPerFrame: number
}

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx

  const sampler = device.createSampler({
    label: `${ctx.id}/atlas-sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    maxAnisotropy: 4,
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  })

  // --- species atlases (sequential: the poa bake transiently needs ~400MB) --
  const speciesGpu = new Map<string, SpeciesGpu>()
  for (const entry of ctx.stand.species) {
    if (speciesGpu.has(entry.species)) continue
    const atlas = await loadSpeciesRelief(ctx, entry.species)
    speciesGpu.set(entry.species, uploadAtlas(ctx, entry.species, atlas))
  }

  const cullBgl = device.createBindGroupLayout({
    label: `${ctx.id}/cull-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  })
  const drawBgl = device.createBindGroupLayout({
    label: `${ctx.id}/draw-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  const entries: EntryGpu[] = ctx.stand.species.map((standEntry, entryIndex) => {
    const gpu = speciesGpu.get(standEntry.species)!
    const density = Math.min(standEntry.density, SCATTER_MAX_DENSITY)
    // Ring capacity from the annulus area at the LARGEST region radius, so the
    // regionRadius slider never reallocates and rings never overflow.
    const ringCap: number[] = []
    const ringBase: number[] = []
    let cursor = 0
    for (let ring = 0; ring < N_RINGS; ring++) {
      const rIn = ring === 0 ? 0 : RING_R[ring - 1]!
      const rOut = RING_R[ring]!
      const cap = Math.ceil(Math.PI * (rOut * rOut - rIn * rIn) * density * 1.15) + 512
      ringBase.push(cursor)
      ringCap.push(cap)
      cursor += cap
    }
    const infoBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/info-${entryIndex}`,
        size: INFO_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'entry-info' },
    )
    const instBuffer = ctx.res.createBuffer(
      { label: `${ctx.id}/instances-${entryIndex}`, size: cursor * 16, usage: GPUBufferUsage.STORAGE },
      { species: standEntry.species, tag: 'culled-instances' },
    )
    const indirectBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/indirect-${entryIndex}`,
        size: N_RINGS * 16,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'indirect-args' },
    )
    const cullBindGroup = device.createBindGroup({
      label: `${ctx.id}/cull-bg-${entryIndex}`,
      layout: cullBgl,
      entries: [
        { binding: 0, resource: { buffer: infoBuffer } },
        { binding: 1, resource: { buffer: instBuffer } },
        { binding: 2, resource: { buffer: indirectBuffer } },
      ],
    })
    const drawBindGroup = device.createBindGroup({
      label: `${ctx.id}/draw-bg-${entryIndex}`,
      layout: drawBgl,
      entries: [
        { binding: 0, resource: { buffer: infoBuffer } },
        { binding: 1, resource: { buffer: instBuffer } },
        { binding: 2, resource: gpu.albedoTex.createView() },
        { binding: 3, resource: gpu.geomTex.createView() },
        { binding: 4, resource: sampler },
      ],
    })
    return {
      speciesId: standEntry.species,
      gpu,
      ringBase,
      ringCap,
      infoBuffer,
      indirectBuffer,
      cullBindGroup,
      drawBindGroup,
      slotsPerFrame: 0,
    }
  })

  let cullPipeline!: GPUComputePipeline
  let cardsLayout!: GPUPipelineLayout
  // One pipeline per distance ring, specialized on LOD_TIER: a far card's shader
  // does not contain the relief/shadow code at all. Keyed by the two other
  // override constants so a param flip just rebuilds four small pipelines.
  const pipeCache = new Map<string, GPURenderPipeline[]>()
  let ringPipes: GPURenderPipeline[] = []
  let pipeKey = ''
  const buildRingPipes = (steps: number, showLod: boolean): GPURenderPipeline[] => {
    const key = `${steps}|${showLod ? 1 : 0}`
    const hit = pipeCache.get(key)
    if (hit) return hit
    const module = ctx.shaders.module(relightSrc)
    const pipes = Array.from({ length: N_RINGS }, (_, tier) => {
      // Only the nearest ring gets the Newton refinement of the relief solve;
      // past ~7 m the extra probe is a tap nobody can see.
      const consts = { LOD_TIER: tier, RELIEF_STEPS: tier === 0 ? steps : 1, SHOW_LOD: showLod ? 1 : 0 }
      return device.createRenderPipeline({
        label: `${ctx.id}/relief-cards-t${tier}`,
        layout: cardsLayout,
        vertex: { module, entryPoint: 'vs_main', constants: consts },
        fragment: {
          module,
          entryPoint: 'fs_main',
          constants: consts,
          targets: [{ format: ctx.colorFormat }],
        },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
      })
    })
    pipeCache.set(key, pipes)
    return pipes
  }
  const build = (): void => {
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/cull-pl`,
        bindGroupLayouts: [ctx.frame.layout, cullBgl],
      }),
      compute: { module: ctx.shaders.module(cullSrc), entryPoint: 'cs_cull' },
    })
    cardsLayout = device.createPipelineLayout({
      label: `${ctx.id}/cards-pl`,
      bindGroupLayouts: [ctx.frame.layout, drawBgl],
    })
    pipeCache.clear()
    pipeKey = ''
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const cellMin = Math.floor(-ctx.stand.radius / CELL)
  const cellMax = Math.floor(ctx.stand.radius / CELL)
  const info = new Float32Array(INFO_FLOATS)
  const planes = new Float32Array(24)
  const indirectReset = new Uint32Array(N_RINGS * 4)
  for (let ring = 0; ring < N_RINGS; ring++) {
    indirectReset[ring * 4] = VERTS_PER_PLANT
    indirectReset[ring * 4 + 2] = ring * VERTS_PER_PLANT // firstVertex encodes the ring
  }

  return {
    update(frame: FrameInfo): void {
      const R = Math.min(ctx.params.regionRadius, REGION_MAX)
      const cam = frame.camera.pose
      // Region cell rect, clamped to the stand's cell range on the CPU: cells
      // outside the stand hold nothing, so they must not be dispatched.
      const x0 = Math.max(cellMin, Math.floor((cam.x - R) / CELL))
      const z0 = Math.max(cellMin, Math.floor((cam.z - R) / CELL))
      const x1 = Math.min(cellMax, Math.floor((cam.x + R) / CELL))
      const z1 = Math.min(cellMax, Math.floor((cam.z + R) / CELL))
      const sideX = Math.max(0, x1 - x0 + 1)
      const sideZ = Math.max(0, z1 - z0 + 1)
      frustumPlanes(frame.camera.viewProj, planes)

      const key = `${ctx.params.reliefSteps}|${ctx.params.showLod ? 1 : 0}`
      if (key !== pipeKey) {
        ringPipes = buildRingPipes(ctx.params.reliefSteps, ctx.params.showLod)
        pipeKey = key
      }

      entries.forEach((entry, entryIndex) => {
        const a = entry.gpu.atlas
        info.set(planes, 0)
        info[24] = x0
        info[25] = z0
        info[26] = sideX
        info[27] = sideZ
        info[28] = ctx.seed
        info[29] = entryIndex
        info[30] = R
        info[31] = Math.max(a.rXZ, (a.y1 - a.y0) / 2) * 1.05
        info.set(entry.ringBase, 32)
        info.set(entry.ringCap, 36)
        info[40] = Math.min(RING_R[0]!, R)
        info[41] = Math.min(RING_R[1]!, R)
        info[42] = Math.min(RING_R[2]!, R)
        info[43] = R
        info[44] = a.y0
        info[45] = a.y1
        info[46] = a.rXZ
        info[47] = a.cx
        info[48] = a.cz
        info[49] = ctx.params.alphaRef
        info[50] = ctx.params.topPlane
        info[51] = ctx.params.relief
        info[52] = ctx.params.reliefClamp
        info[53] = ctx.params.selfShadow
        info[54] = ctx.params.shadowStep
        info[55] = ctx.params.ao
        info[56] = ctx.params.translucency
        info[57] = 4 / ATLAS // tile_inset: 4 texels of the source tile
        info[58] = ctx.params.mirrorVariants ? 1 : 0
        info[59] = ctx.params.contactSkirt
        info[60] = ctx.params.topCard ? 1 : 0
        device.queue.writeBuffer(entry.infoBuffer, 0, info)
        device.queue.writeBuffer(entry.indirectBuffer, 0, indirectReset)
        entry.slotsPerFrame = sideX * sideZ * SCATTER_MAX_PER_CELL
      })
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const cull = ctx.timing.computePass(enc, 'cull')
      cull.setPipeline(cullPipeline)
      cull.setBindGroup(0, ctx.frame.bindGroup)
      for (const entry of entries) {
        if (entry.slotsPerFrame === 0) continue // camera outside the stand
        cull.setBindGroup(1, entry.cullBindGroup)
        cull.dispatchWorkgroups(Math.ceil(entry.slotsPerFrame / 64))
      }
      cull.end()

      const pass = ctx.timing.renderPass(enc, 'relief-cards', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setBindGroup(0, ctx.frame.bindGroup)
      // Near ring first: every filled near card is depth that rejects the far
      // rings before their fragments ever reach the relief taps. Each ring runs
      // its own tier-specialized pipeline.
      for (let ring = 0; ring < N_RINGS; ring++) {
        pass.setPipeline(ringPipes[ring]!)
        for (const entry of entries) {
          pass.setBindGroup(1, entry.drawBindGroup)
          pass.drawIndirect(entry.indirectBuffer, ring * 16)
        }
      }
      pass.end()
    },

    dispose(): void {
      unsubscribe()
      // GPU resources are reclaimed by the harness via ctx.res.
    },
  }
}

/** Upload both atlas planes, mip level by mip level (the chain is baked in). */
function uploadAtlas(ctx: ExperimentContext<typeof PARAMS>, speciesId: string, atlas: ReliefAtlas): SpeciesGpu {
  const { device } = ctx
  const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
  const mkTex = (label: string): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/${speciesId}/${label}`,
        size: [ATLAS, ATLAS],
        format: 'rgba8unorm',
        mipLevelCount: MIPS,
        usage,
      },
      { species: speciesId, tag: `atlas-${label}` },
    )
  const albedoTex = mkTex('albedo')
  const geomTex = mkTex('geom')

  let offset = 0
  for (let level = 0; level < MIPS; level++) {
    const px = levelPx(level)
    const bytes = px * px * 4
    device.queue.writeTexture(
      { texture: albedoTex, mipLevel: level },
      atlas.albedo.subarray(offset, offset + bytes),
      { bytesPerRow: px * 4, rowsPerImage: px },
      [px, px],
    )
    device.queue.writeTexture(
      { texture: geomTex, mipLevel: level },
      atlas.geom.subarray(offset, offset + bytes),
      { bytesPerRow: px * 4, rowsPerImage: px },
      [px, px],
    )
    offset += bytes
  }
  return { atlas, albedoTex, geomTex }
}

/** Gribb–Hartmann frustum planes from a column-major view-proj matrix. */
function frustumPlanes(m: Float32Array | number[], out: Float32Array): void {
  const row = (r: number): [number, number, number, number] => [m[r]!, m[4 + r]!, m[8 + r]!, m[12 + r]!]
  const r0 = row(0)
  const r1 = row(1)
  const r2 = row(2)
  const r3 = row(3)
  const list: [number, number, number, number][] = [
    [r3[0] + r0[0], r3[1] + r0[1], r3[2] + r0[2], r3[3] + r0[3]], // left
    [r3[0] - r0[0], r3[1] - r0[1], r3[2] - r0[2], r3[3] - r0[3]], // right
    [r3[0] + r1[0], r3[1] + r1[1], r3[2] + r1[2], r3[3] + r1[3]], // bottom
    [r3[0] - r1[0], r3[1] - r1[1], r3[2] - r1[2], r3[3] - r1[3]], // top
    [r2[0], r2[1], r2[2], r2[3]], // near (WebGPU z >= 0)
    [r3[0] - r2[0], r3[1] - r2[1], r3[2] - r2[2], r3[3] - r2[3]], // far
  ]
  list.forEach((p, i) => {
    const len = Math.hypot(p[0], p[1], p[2]) || 1
    out[i * 4] = p[0] / len
    out[i * 4 + 1] = p[1] / len
    out[i * 4 + 2] = p[2] / len
    out[i * 4 + 3] = p[3] / len
  })
}
