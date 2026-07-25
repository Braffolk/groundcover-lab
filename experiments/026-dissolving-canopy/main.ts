import cullSrc from './shaders/cull.wgsl'
import tuftsSrc from './shaders/tufts.wgsl'
import shellSrc from './shaders/shell.wgsl'
import mipgenSrc from './shaders/mipgen.wgsl'
import {
  ALB_H,
  ALB_MIPS,
  ALB_W,
  COV_H,
  COV_MIPS,
  COV_W,
  NRM_H,
  NRM_MIPS,
  NRM_W,
  loadSpeciesBake,
  type CanopyBake,
} from './bake.ts'
import { CANOPY_MIPS, CANOPY_TEX, CANOPY_TILE_M, compositeCanopy } from './canopy.ts'
import {
  SCATTER_CELL_SIZE,
  SCATTER_MAX_PER_CELL,
  type Experiment,
  type ExperimentContext,
  type FrameInfo,
  type ViewTargets,
} from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * A canopy that dissolves from individual plants into one continuous surface.
 *
 * Startup: per species, a node atlas (whole plant + its four xz quadrants, 8
 * azimuths each; full-res coverage, half-res colour, quarter-res normals) is
 * loaded from mesh/baked / OPFS or baked in-browser from the raw GCMESH1 mesh.
 * The active stand's species mix is then composited into ONE tileable canopy
 * texture out of the baked top-down captures (canopy.ts).
 *
 * Per frame:
 *   1. `cull` — the shared scatter over a camera-centered cell region, with the
 *      dissolve field rejecting every plant the shell has already taken over,
 *      compacting survivors into two radial bands (near = 4 sub-tuft splats,
 *      far = 1 whole-plant splat) of one instance buffer + indirect args.
 *   2. `tufts` — near band across all species first (coarse front-to-back), then
 *      the far band. Hard alpha test, depth write, 3 taps.
 *   3. `shell` — a bufferless two-level snapped grid whose coverage threshold
 *      starts above 1.0 near the camera (nothing rasterized there) and falls to
 *      the far-field floor, so distant plants cost O(area), not O(plants).
 *
 * Per-frame cost is O(visible region), independent of the stand's plant count,
 * and collapses with distance AND with view elevation: looking down, the splat
 * set empties out entirely and only the surface remains.
 */

const CELL = SCATTER_CELL_SIZE
const REGION_MAX = 112 // must stay >= the manifest's dissolveGrazing max
const TUFT_MAX = 36 // must stay >= the manifest's tuftRadius max
const INFO_FLOATS = 84
const SHELL_FLOATS = 68
const SHELL_G = 96 // cells per side, per level
const SHELL_SPACING: readonly [number, number] = [0.75, 3]
const SHELL_EDGE_FADE = 14
const NEAR_VERTS = 24 // 4 quadrant splats
const FAR_VERTS = 6 // 1 whole-plant splat

interface SpeciesGpu {
  bake: CanopyBake
  cov: GPUTexture
  albedo: GPUTexture
  normal: GPUTexture
}

interface EntryGpu {
  gpu: SpeciesGpu
  cap0: number
  cap1: number
  infoBuffer: GPUBuffer
  argsBuffer: GPUBuffer
  cullBindGroup: GPUBindGroup
  nearBindGroup: GPUBindGroup
  farBindGroup: GPUBindGroup
  slotsPerFrame: number
}

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx

  const tuftSampler = device.createSampler({
    label: `${ctx.id}/tuft-sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    maxAnisotropy: 4,
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  })
  const canopySampler = device.createSampler({
    label: `${ctx.id}/canopy-sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    maxAnisotropy: 8,
    addressModeU: 'repeat',
    addressModeV: 'repeat',
  })

  // --- baked node atlases (sequential: poa transiently needs a few hundred MB)
  const bakes = new Map<string, CanopyBake>()
  for (const entry of ctx.stand.species) {
    if (bakes.has(entry.species)) continue
    bakes.set(entry.species, await loadSpeciesBake(ctx, entry.species))
  }

  const mip = new MipGen(ctx)
  const speciesGpu = new Map<string, SpeciesGpu>()
  const uploadEnc = device.createCommandEncoder({ label: `${ctx.id}/upload` })
  for (const [speciesId, bake] of bakes) {
    speciesGpu.set(speciesId, uploadNodeAtlas(ctx, mip, uploadEnc, speciesId, bake))
  }

  // --- the stand's single tileable canopy texture ---------------------------
  const tile = compositeCanopy(ctx.stand, bakes, ctx.seed)
  console.info(
    `[${ctx.id}] canopy tile ${CANOPY_TEX}px/${CANOPY_TILE_M}m: surface=${tile.baseHeight.toFixed(2)}m ` +
      `top=${tile.heightMax.toFixed(2)}m coverage=${(tile.meanCoverage * 100).toFixed(0)}% (top-down)`,
  )
  const mkCanopy = (label: string): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/${label}`,
        size: [CANOPY_TEX, CANOPY_TEX],
        format: 'rgba8unorm',
        mipLevelCount: CANOPY_MIPS,
        usage:
          GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
      },
      { tag: 'canopy-shell' },
    )
  const canopyAlb = mkCanopy('canopy-albedo')
  const canopyAux = mkCanopy('canopy-aux')
  const canopyLayout = { bytesPerRow: CANOPY_TEX * 4, rowsPerImage: CANOPY_TEX }
  device.queue.writeTexture({ texture: canopyAlb }, tile.albedo, canopyLayout, [CANOPY_TEX, CANOPY_TEX])
  device.queue.writeTexture({ texture: canopyAux }, tile.aux, canopyLayout, [CANOPY_TEX, CANOPY_TEX])
  mip.run(uploadEnc, canopyAlb, CANOPY_MIPS, 'fs_albedo', 'rgba8unorm')
  mip.run(uploadEnc, canopyAux, CANOPY_MIPS, 'fs_aux', 'rgba8unorm')
  device.queue.submit([uploadEnc.finish()])

  // --- layouts / pipelines --------------------------------------------------
  const cullBgl = device.createBindGroupLayout({
    label: `${ctx.id}/cull-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  })
  const tuftBgl = device.createBindGroupLayout({
    label: `${ctx.id}/tuft-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })
  const shellBgl = device.createBindGroupLayout({
    label: `${ctx.id}/shell-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  const shellInfoBuffer = ctx.res.createBuffer(
    { label: `${ctx.id}/shell-info`, size: SHELL_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'shell-info' },
  )
  const shellBindGroup = device.createBindGroup({
    label: `${ctx.id}/shell-bg`,
    layout: shellBgl,
    entries: [
      { binding: 0, resource: { buffer: shellInfoBuffer } },
      { binding: 1, resource: canopyAlb.createView() },
      { binding: 2, resource: canopyAux.createView() },
      { binding: 3, resource: canopySampler },
    ],
  })

  const entries: EntryGpu[] = ctx.stand.species.map((standEntry, entryIndex) => {
    const gpu = speciesGpu.get(standEntry.species)!
    const density = Math.min(standEntry.density, 8)
    const cap0 = align16(Math.ceil(Math.PI * TUFT_MAX * TUFT_MAX * density * 1.08) + 1024)
    const cap1 = align16(Math.ceil(Math.PI * REGION_MAX * REGION_MAX * density * 1.08) + 2048)
    const infoBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/info-${entryIndex}`,
        size: INFO_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'entry-info' },
    )
    const instBuffer = ctx.res.createBuffer(
      { label: `${ctx.id}/instances-${entryIndex}`, size: (cap0 + cap1) * 16, usage: GPUBufferUsage.STORAGE },
      { species: standEntry.species, tag: 'culled-instances' },
    )
    const argsBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/args-${entryIndex}`,
        size: 32,
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
        { binding: 2, resource: { buffer: argsBuffer } },
      ],
    })
    // WebGPU forbids a non-zero firstInstance in indirect draws without an
    // optional feature, so each band binds its own view of the instance buffer.
    const drawBindGroup = (label: string, offset: number, size: number): GPUBindGroup =>
      device.createBindGroup({
        label,
        layout: tuftBgl,
        entries: [
          { binding: 0, resource: { buffer: infoBuffer } },
          { binding: 1, resource: { buffer: instBuffer, offset, size } },
          { binding: 2, resource: gpu.cov.createView() },
          { binding: 3, resource: gpu.albedo.createView() },
          { binding: 4, resource: gpu.normal.createView() },
          { binding: 5, resource: tuftSampler },
        ],
      })
    return {
      gpu,
      cap0,
      cap1,
      infoBuffer,
      argsBuffer,
      cullBindGroup,
      nearBindGroup: drawBindGroup(`${ctx.id}/near-bg-${entryIndex}`, 0, cap0 * 16),
      farBindGroup: drawBindGroup(`${ctx.id}/far-bg-${entryIndex}`, cap0 * 16, cap1 * 16),
      slotsPerFrame: 0,
    }
  })

  let cullPipeline!: GPUComputePipeline
  let tuftPipeline!: GPURenderPipeline
  let shellPipeline!: GPURenderPipeline
  const build = (): void => {
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/cull-pl`, bindGroupLayouts: [ctx.frame.layout, cullBgl] }),
      compute: { module: ctx.shaders.module(cullSrc), entryPoint: 'cs_cull' },
    })
    const mkRender = (label: string, src: typeof tuftsSrc, bgl: GPUBindGroupLayout): GPURenderPipeline =>
      device.createRenderPipeline({
        label: `${ctx.id}/${label}`,
        layout: device.createPipelineLayout({
          label: `${ctx.id}/${label}-pl`,
          bindGroupLayouts: [ctx.frame.layout, bgl],
        }),
        vertex: { module: ctx.shaders.module(src), entryPoint: 'vs_main' },
        fragment: { module: ctx.shaders.module(src), entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
      })
    tuftPipeline = mkRender('tufts', tuftsSrc, tuftBgl)
    shellPipeline = mkRender('shell', shellSrc, shellBgl)
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const cellMin = Math.floor(-ctx.stand.radius / CELL)
  const cellMax = Math.floor(ctx.stand.radius / CELL)
  const standHalf = Math.min(ctx.stand.radius, ctx.scene.terrain.desc.size / 2)
  const info = new Float32Array(INFO_FLOATS)
  const shellInfo = new Float32Array(SHELL_FLOATS)
  const planes = new Float32Array(24)
  const argsReset = new Uint32Array(8)
  argsReset[0] = NEAR_VERTS
  argsReset[4] = FAR_VERTS

  return {
    update(frame: FrameInfo): void {
      const p = ctx.params
      const regionR = Math.min(REGION_MAX, p.dissolveGrazing + 1)
      const tuftR = Math.min(TUFT_MAX, p.tuftRadius)
      const cam = frame.camera.pose
      frustumPlanes(frame.camera.viewProj, planes)

      // Region cell rect, clamped to the stand's cell range on the CPU: cells
      // outside the stand hold nothing, so they must not be dispatched.
      const x0 = Math.max(cellMin, Math.floor((cam.x - regionR) / CELL))
      const z0 = Math.max(cellMin, Math.floor((cam.z - regionR) / CELL))
      const x1 = Math.min(cellMax, Math.floor((cam.x + regionR) / CELL))
      const z1 = Math.min(cellMax, Math.floor((cam.z + regionR) / CELL))
      const sideX = Math.max(0, x1 - x0 + 1)
      const sideZ = Math.max(0, z1 - z0 + 1)

      entries.forEach((entry, entryIndex) => {
        const b = entry.gpu.bake
        info.set(planes, 0)
        b.nodes.forEach((n, i) => {
          info.set([n.cx, n.cy, n.cz, n.halfW], 24 + i * 8)
          info.set([n.y0, n.y1, n.azOffset, 0], 28 + i * 8)
        })
        info.set([x0, z0, sideX, sideZ], 64)
        info.set([ctx.seed, entryIndex, regionR, tuftR], 68)
        info.set([entry.cap0, entry.cap1, p.alphaRef, p.groundShade], 72)
        info.set([p.dissolveGrazing, p.dissolveOverhead, b.plantH, b.rootHalfW], 76)
        info.set([Math.max(b.rootHalfW, b.plantH * 0.5) * 1.05, 0, 0, 0], 80)
        device.queue.writeBuffer(entry.infoBuffer, 0, info)
        device.queue.writeBuffer(entry.argsBuffer, 0, argsReset)
        entry.slotsPerFrame = sideX * sideZ * SCATTER_MAX_PER_CELL
      })

      // Shell: two camera-snapped grid levels; the coarse one skips whatever the
      // fine one covers, overlapping by one of its own cells so the seam between
      // levels can never crack.
      const s0 = SHELL_SPACING[0]
      const s1 = SHELL_SPACING[1]
      const snap = (v: number, s: number): number => Math.floor(v / (2 * s)) * (2 * s) - (SHELL_G / 2) * s
      const o0x = snap(cam.x, s0)
      const o0z = snap(cam.z, s0)
      shellInfo.set(planes, 0)
      shellInfo.set([o0x, o0z, s0, 0], 24)
      shellInfo.set([snap(cam.x, s1), snap(cam.z, s1), s1, 0], 28)
      shellInfo.set([1e9, 1e9, -1e9, -1e9], 32) // level 0 never skips
      shellInfo.set([o0x + s0, o0z + s0, o0x + SHELL_G * s0 - s0, o0z + SHELL_G * s0 - s0], 36)
      shellInfo.set([standHalf, tile.baseHeight, CANOPY_TILE_M, p.shellRelief], 40)
      shellInfo.set([p.shellFloor, p.dissolveGrazing, p.dissolveOverhead, p.groundShade], 44)
      shellInfo.set([SHELL_G * SHELL_G, SHELL_G, tile.meanSway, SHELL_EDGE_FADE], 48)
      shellInfo.set([...tile.sideTopColor, tile.meanLuma], 52)
      shellInfo.set([...tile.sideLowColor, tile.heightMax], 56)
      shellInfo.set([tile.hueGain, 0, 0, 0], 60)
      device.queue.writeBuffer(shellInfoBuffer, 0, shellInfo)
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const p = ctx.params
      if (p.tufts) {
        const cull = ctx.timing.computePass(enc, 'cull')
        cull.setPipeline(cullPipeline)
        cull.setBindGroup(0, ctx.frame.bindGroup)
        for (const entry of entries) {
          if (entry.slotsPerFrame === 0) continue // camera outside the stand
          cull.setBindGroup(1, entry.cullBindGroup)
          cull.dispatchWorkgroups(Math.ceil(entry.slotsPerFrame / 64))
        }
        cull.end()
      }

      // Splats and shell share ONE render pass, drawn strictly front to back:
      // near band, far band, then the surface that only exists behind them.
      // Depth is written by every one of them, so early-z does the culling the
      // ordering sets up (and one pass instead of two keeps the pass overhead
      // comparable to a single-primitive renderer).
      const pass = ctx.timing.renderPass(enc, 'canopy', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setBindGroup(0, ctx.frame.bindGroup)
      if (p.tufts) {
        pass.setPipeline(tuftPipeline)
        for (const entry of entries) {
          pass.setBindGroup(1, entry.nearBindGroup)
          pass.drawIndirect(entry.argsBuffer, 0)
        }
        for (const entry of entries) {
          pass.setBindGroup(1, entry.farBindGroup)
          pass.drawIndirect(entry.argsBuffer, 16)
        }
      }
      if (p.shell) {
        pass.setPipeline(shellPipeline)
        pass.setBindGroup(1, shellBindGroup)
        pass.draw(6, SHELL_G * SHELL_G * SHELL_SPACING.length)
      }
      pass.end()
    },

    dispose(): void {
      unsubscribe()
      // GPU resources are reclaimed by the harness via ctx.res.
    },
  }
}

function align16(n: number): number {
  return Math.ceil(n / 16) * 16
}

/** Upload one species' node atlas (coverage / albedo / normals) and its mips. */
function uploadNodeAtlas(
  ctx: ExperimentContext<typeof PARAMS>,
  mip: MipGen,
  enc: GPUCommandEncoder,
  speciesId: string,
  bake: CanopyBake,
): SpeciesGpu {
  const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST
  const mk = (label: string, w: number, h: number, format: GPUTextureFormat, mips: number): GPUTexture =>
    ctx.res.createTexture(
      { label: `${ctx.id}/${speciesId}/${label}`, size: [w, h], format, mipLevelCount: mips, usage },
      { species: speciesId, tag: `node-${label}` },
    )
  const cov = mk('coverage', COV_W, COV_H, 'r8unorm', COV_MIPS)
  const albedo = mk('albedo', ALB_W, ALB_H, 'rgba8unorm', ALB_MIPS)
  const normal = mk('normal', NRM_W, NRM_H, 'rg8unorm', NRM_MIPS)
  const q = ctx.device.queue
  q.writeTexture({ texture: cov }, bake.coverage, { bytesPerRow: COV_W, rowsPerImage: COV_H }, [COV_W, COV_H])
  q.writeTexture({ texture: albedo }, bake.albedo, { bytesPerRow: ALB_W * 4, rowsPerImage: ALB_H }, [ALB_W, ALB_H])
  q.writeTexture({ texture: normal }, bake.normal, { bytesPerRow: NRM_W * 2, rowsPerImage: NRM_H }, [NRM_W, NRM_H])
  mip.run(enc, cov, COV_MIPS, 'fs_cov', 'r8unorm')
  mip.run(enc, albedo, ALB_MIPS, 'fs_albedo', 'rgba8unorm')
  mip.run(enc, normal, NRM_MIPS, 'fs_oct', 'rg8unorm')
  return { bake, cov, albedo, normal }
}

/** GPU mip generation, one full-screen pass per level (init only). */
class MipGen {
  private bgl: GPUBindGroupLayout
  private layout: GPUPipelineLayout
  private pipelines = new Map<string, GPURenderPipeline>()

  constructor(private ctx: ExperimentContext<typeof PARAMS>) {
    this.bgl = ctx.device.createBindGroupLayout({
      label: `${ctx.id}/mipgen-bgl`,
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      ],
    })
    this.layout = ctx.device.createPipelineLayout({
      label: `${ctx.id}/mipgen-pl`,
      bindGroupLayouts: [this.bgl],
    })
  }

  run(enc: GPUCommandEncoder, tex: GPUTexture, levels: number, entryPoint: string, format: GPUTextureFormat): void {
    const { device } = this.ctx
    const key = `${entryPoint}/${format}`
    let pipeline = this.pipelines.get(key)
    if (!pipeline) {
      const module = this.ctx.shaders.module(mipgenSrc)
      pipeline = device.createRenderPipeline({
        label: `${this.ctx.id}/mipgen-${key}`,
        layout: this.layout,
        vertex: { module, entryPoint: 'vs' },
        fragment: { module, entryPoint, targets: [{ format }] },
        primitive: { topology: 'triangle-list' },
      })
      this.pipelines.set(key, pipeline)
    }
    for (let level = 1; level < levels; level++) {
      const bg = device.createBindGroup({
        label: `${this.ctx.id}/mipgen-bg-${level}`,
        layout: this.bgl,
        entries: [{ binding: 0, resource: tex.createView({ baseMipLevel: level - 1, mipLevelCount: 1 }) }],
      })
      const pass = enc.beginRenderPass({
        label: `${this.ctx.id}/mipgen-${level}`,
        colorAttachments: [
          { view: tex.createView({ baseMipLevel: level, mipLevelCount: 1 }), loadOp: 'clear', storeOp: 'store' },
        ],
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bg)
      pass.draw(3)
      pass.end()
    }
  }
}

/** Gribb–Hartmann frustum planes from a column-major view-proj matrix. */
function frustumPlanes(m: Float32Array | number[], out: Float32Array): void {
  const row = (r: number): [number, number, number, number] => [m[r]!, m[4 + r]!, m[8 + r]!, m[12 + r]!]
  const r0 = row(0)
  const r1 = row(1)
  const r2 = row(2)
  const r3 = row(3)
  const p: [number, number, number, number][] = [
    [r3[0] + r0[0], r3[1] + r0[1], r3[2] + r0[2], r3[3] + r0[3]], // left
    [r3[0] - r0[0], r3[1] - r0[1], r3[2] - r0[2], r3[3] - r0[3]], // right
    [r3[0] + r1[0], r3[1] + r1[1], r3[2] + r1[2], r3[3] + r1[3]], // bottom
    [r3[0] - r1[0], r3[1] - r1[1], r3[2] - r1[2], r3[3] - r1[3]], // top
    [r2[0], r2[1], r2[2], r2[3]], // near (WebGPU z >= 0)
    [r3[0] - r2[0], r3[1] - r2[1], r3[2] - r2[2], r3[3] - r2[3]], // far
  ]
  p.forEach((pl, i) => {
    const len = Math.hypot(pl[0], pl[1], pl[2]) || 1
    out[i * 4] = pl[0] / len
    out[i * 4 + 1] = pl[1] / len
    out[i * 4 + 2] = pl[2] / len
    out[i * 4 + 3] = pl[3] / len
  })
}
