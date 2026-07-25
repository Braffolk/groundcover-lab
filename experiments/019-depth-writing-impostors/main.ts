import cullSrc from './shaders/cull.wgsl'
import impostorSrc from './shaders/impostor.wgsl'
import { MIP_LEVELS, N_LAYERS, TILE, loadSpeciesViews, mipTexels, type ViewSet } from './bake.ts'
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
 * True-depth impostors. Startup: each species' baked view set (17 orthographic
 * layers — 10 azimuths on the horizon, 6 at 40 degrees, plus a top view —
 * carrying albedo + coverage, octahedral normals, SIGNED DEPTH and baked canopy
 * occlusion) is loaded from mesh/baked / OPFS or baked in-browser from the raw
 * mesh, and uploaded as two mipped texture arrays.
 *
 * Per frame: one compute pass per stand entry evaluates the shared scatter over
 * a camera-centered cell region, frustum-culls, and compacts survivors into
 * fixed distance SHELLS inside one instance buffer. The shells are drawn near
 * to far, so every shell's hard-alpha-tested depth writes reject the shells
 * behind it before they shade. Each plant is a single screen-aligned quad; the
 * fragment shader inverts the image->screen map with one coarse depth tap,
 * shades the warped texel, and in the nearest shells rebuilds the true 3D
 * surface point and writes it as frag_depth. Per-frame cost is O(visible
 * region), independent of the stand's plant count.
 */

const CELL = SCATTER_CELL_SIZE
const REGION_MAX = 128 // keep equal to the manifest's regionRadius max
const INFO_FLOATS = 96
/**
 * Outer radius (m) of each distance shell. Fixed, not param-derived, so the
 * per-shell instance capacities are allocated once and the LOD params only
 * choose which pipeline each shell draws with.
 */
const SHELL_R = [6, 11, 18, 28, 45, 75, REGION_MAX]
const N_SHELLS = SHELL_R.length
/** Instance-buffer capacity share of each shell's full annulus area. */
const SHELL_FILL = [1.15, 1.15, 1.15, 0.9, 0.75, 0.6, 0.5]
const SHELL_SLACK = [512, 512, 512, 1024, 1024, 2048, 4096]
/** 16B per instance; sub-range bindings need 256B alignment. */
const ALIGN = 16

interface SpeciesGpu {
  set: ViewSet
  albedoTex: GPUTexture
  geoTex: GPUTexture
  viewBuffer: GPUBuffer
}

interface EntryGpu {
  speciesId: string
  gpu: SpeciesGpu
  /** Per shell: element base into the shared instance buffer, and capacity. */
  bases: number[]
  caps: number[]
  infoBuffer: GPUBuffer
  indirectBuffer: GPUBuffer
  cullBindGroup: GPUBindGroup
  drawBindGroups: GPUBindGroup[]
  slotsPerFrame: number
}

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx

  const sampler = device.createSampler({
    label: `${ctx.id}/atlas-sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  })

  // Sequential: the poa bake transiently needs several hundred MB.
  const speciesGpu = new Map<string, SpeciesGpu>()
  for (const entry of ctx.stand.species) {
    if (speciesGpu.has(entry.species)) continue
    const set = await loadSpeciesViews(ctx, entry.species)
    speciesGpu.set(entry.species, upload(ctx, entry.species, set))
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
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  const entries: EntryGpu[] = ctx.stand.species.map((standEntry, entryIndex) => {
    const gpu = speciesGpu.get(standEntry.species)!
    const density = Math.min(standEntry.density, 8)
    // Per-shell capacity from its annulus area. The frustum keeps at most a
    // ~90-degree horizontal wedge, so the outer shells are sized well below
    // their full-annulus bound; a clamp only ever drops a few of the most
    // distant plants, it never breaks anything.
    const caps: number[] = []
    const bases: number[] = []
    let total = 0
    for (let i = 0; i < N_SHELLS; i++) {
      const r0 = i === 0 ? 0 : SHELL_R[i - 1]!
      const area = Math.PI * (SHELL_R[i]! ** 2 - r0 * r0)
      const cap = Math.ceil((area * density * SHELL_FILL[i]! + SHELL_SLACK[i]!) / ALIGN) * ALIGN
      caps.push(cap)
      bases.push(total)
      total += cap
    }
    const instBuffer = ctx.res.createBuffer(
      { label: `${ctx.id}/instances-${entryIndex}`, size: total * 16, usage: GPUBufferUsage.STORAGE },
      { species: standEntry.species, tag: 'culled-shells' },
    )
    const infoBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/info-${entryIndex}`,
        size: INFO_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'entry-info' },
    )
    const indirectBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/indirect-${entryIndex}`,
        size: 8 * 16, // 8 slots: matches the cull shader's atomic array
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
    const albedoView = gpu.albedoTex.createView({ dimension: '2d-array' })
    const geoView = gpu.geoTex.createView({ dimension: '2d-array' })
    // One bind group per shell: the same buffer bound at the shell's own
    // offset, so the vertex shader indexes from 0 and no draw needs the
    // (optional) indirect-first-instance feature.
    const drawBindGroups = caps.map((cap, i) =>
      device.createBindGroup({
        label: `${ctx.id}/draw-bg-${entryIndex}-${i}`,
        layout: drawBgl,
        entries: [
          { binding: 0, resource: { buffer: infoBuffer } },
          { binding: 1, resource: { buffer: instBuffer, offset: bases[i]! * 16, size: cap * 16 } },
          { binding: 2, resource: { buffer: gpu.viewBuffer } },
          { binding: 3, resource: albedoView },
          { binding: 4, resource: geoView },
          { binding: 5, resource: sampler },
        ],
      }),
    )
    return {
      speciesId: standEntry.species,
      gpu,
      bases,
      caps,
      infoBuffer,
      indirectBuffer,
      cullBindGroup,
      drawBindGroups,
      slotsPerFrame: 0,
    }
  })

  let cullPipeline!: GPUComputePipeline
  const drawPipelines = new Map<string, GPURenderPipeline>()
  const build = (): void => {
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/cull-pl`, bindGroupLayouts: [ctx.frame.layout, cullBgl] }),
      compute: { module: ctx.shaders.module(cullSrc), entryPoint: 'cs_cull' },
    })
    const module = ctx.shaders.module(impostorSrc)
    const layout = device.createPipelineLayout({
      label: `${ctx.id}/draw-pl`,
      bindGroupLayouts: [ctx.frame.layout, drawBgl],
    })
    drawPipelines.clear()
    for (const fs of ['fs_near', 'fs_mid', 'fs_far']) {
      drawPipelines.set(
        fs,
        device.createRenderPipeline({
          label: `${ctx.id}/${fs}`,
          layout,
          vertex: { module, entryPoint: 'vs_main' },
          fragment: { module, entryPoint: fs, targets: [{ format: ctx.colorFormat }] },
          primitive: { topology: 'triangle-list', cullMode: 'none' },
          depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
        }),
      )
    }
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const cellMin = Math.floor(-ctx.stand.radius / CELL)
  const cellMax = Math.floor(ctx.stand.radius / CELL)
  const info = new Float32Array(INFO_FLOATS)
  const planes = new Float32Array(24)
  const indirectReset = new Uint32Array(8 * 4)
  for (let i = 0; i < N_SHELLS; i++) indirectReset[i * 4] = 6
  /** Fragment entry point each shell draws with, refreshed when params move. */
  const shellPipe: GPURenderPipeline[] = []

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

      // A shell writes true per-pixel depth if it starts inside depthDist, and
      // runs the depth warp if it starts inside warpDist. The warp then fades
      // to nothing exactly at the last warped shell's outer radius, so the
      // no-warp shells begin with a silhouette that already matches.
      let warpEnd = SHELL_R[0]!
      shellPipe.length = 0
      for (let i = 0; i < N_SHELLS; i++) {
        const inner = i === 0 ? 0 : SHELL_R[i - 1]!
        if (inner < ctx.params.depthDist) {
          shellPipe.push(drawPipelines.get('fs_near')!)
          warpEnd = SHELL_R[i]!
        } else if (inner < ctx.params.warpDist) {
          shellPipe.push(drawPipelines.get('fs_mid')!)
          warpEnd = SHELL_R[i]!
        } else {
          shellPipe.push(drawPipelines.get('fs_far')!)
        }
      }

      entries.forEach((entry, entryIndex) => {
        const s = entry.gpu.set
        info.set(planes, 0)
        for (let i = 0; i < 8; i++) {
          const o = 24 + i * 4
          info[o] = i < N_SHELLS ? SHELL_R[i]! : 0
          info[o + 1] = i < N_SHELLS ? entry.bases[i]! : 0
          info[o + 2] = i < N_SHELLS ? entry.caps[i]! : 0
          info[o + 3] = 0
        }
        info[56] = x0
        info[57] = z0
        info[58] = sideX
        info[59] = sideZ
        info[60] = ctx.seed
        info[61] = entryIndex
        info[62] = R
        info[63] = N_SHELLS
        info[64] = s.cx
        info[65] = s.cy
        info[66] = s.cz
        info[67] = s.y0
        info[68] = s.y1
        for (let i = 0; i < 4; i++) {
          info[69 + i] = s.rowElev[i] ?? 0
          info[73 + i] = s.rowAz[i] ?? 1
          info[77 + i] = s.rowOffset[i] ?? 0
        }
        info[81] = s.nRows
        info[82] = s.topLayer
        info[83] = ctx.params.alphaRef
        info[84] = Math.hypot(s.hx, s.hy, s.hz)
        info[85] = 0.012
        info[86] = ctx.params.aoStrength
        info[87] = ctx.params.warp
        info[88] = warpEnd * 0.72
        info[89] = warpEnd
        info[90] = TILE
        info[91] = MIP_LEVELS - 1
        info[92] = ctx.params.warpBlur
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

      const pass = ctx.timing.renderPass(enc, 'impostors', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setBindGroup(0, ctx.frame.bindGroup)
      // Near to far: each shell lays down hard-edged depth that rejects the
      // shells behind it before they ever reach the texture units.
      for (let i = 0; i < N_SHELLS; i++) {
        pass.setPipeline(shellPipe[i]!)
        for (const entry of entries) {
          pass.setBindGroup(1, entry.drawBindGroups[i]!)
          pass.drawIndirect(entry.indirectBuffer, i * 16)
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

/** Upload the baked mip chains into two 2d-array textures. */
function upload(ctx: ExperimentContext<typeof PARAMS>, speciesId: string, set: ViewSet): SpeciesGpu {
  const { device } = ctx
  const mk = (tag: string): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/${speciesId}/${tag}`,
        size: [TILE, TILE, N_LAYERS],
        format: 'rgba8unorm',
        mipLevelCount: MIP_LEVELS,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      },
      { species: speciesId, tag },
    )
  const albedoTex = mk('view-albedo')
  const geoTex = mk('view-geo')

  let texelOffset = 0
  let size = TILE
  for (let level = 0; level < MIP_LEVELS; level++) {
    const layout = { offset: texelOffset * 4, bytesPerRow: size * 4, rowsPerImage: size }
    device.queue.writeTexture({ texture: albedoTex, mipLevel: level }, set.albedo, layout, [size, size, N_LAYERS])
    device.queue.writeTexture({ texture: geoTex, mipLevel: level }, set.geo, layout, [size, size, N_LAYERS])
    texelOffset += size * size * N_LAYERS
    size = Math.max(1, size >> 1)
  }
  if (texelOffset !== mipTexels(TILE, MIP_LEVELS) * N_LAYERS) {
    console.warn(`[${ctx.id}] mip chain size mismatch for ${speciesId}`)
  }

  const viewBuffer = ctx.res.createBuffer(
    {
      label: `${ctx.id}/${speciesId}/view-table`,
      size: set.viewTable.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    },
    { species: speciesId, tag: 'view-table' },
  )
  device.queue.writeBuffer(viewBuffer, 0, set.viewTable)

  return { set, albedoTex, geoTex, viewBuffer }
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
