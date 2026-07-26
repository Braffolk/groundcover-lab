import carpetSrc from './shaders/carpet.wgsl'
import cullSrc from './shaders/cull.wgsl'
import impostorSrc from './shaders/impostor.wgsl'
import { MIP_LEVELS, N_LAYERS, TILE, TOP_LAYER, levelOffsetTexels, loadSpeciesViews, mipTexels, type ViewSet } from './bake.ts'
import {
  SCATTER_CELL_SIZE,
  SCATTER_MAX_DENSITY,
  speciesById,
  standEntrySlots,
  type Experiment,
  type ExperimentContext,
  type FrameInfo,
  type StandSpecies,
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
const INFO_FLOATS = 112
/** Cull workgroup width — slot counts are padded up to this. */
const CULL_WG = 64
/**
 * Shell area (m^2) below which a carpet's instance capacity is sized for the
 * FULL grid rate rather than the entry's expected share of the wetness axis:
 * a small shell can easily sit entirely inside one zone, where the entry owns
 * every node, while a large annulus averages over many zones.
 */
const CARPET_FULL_RATE_AREA = 800
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
  /** Layer index of the straight-down view in the UPLOADED texture. */
  topLayer: number
}

/**
 * Everything a mat tile needs, all constant. A carpet species is a periodic
 * community tile, so the runtime only ever samples the straight-down view, and
 * only the tile's own square of it.
 */
interface CarpetConst {
  /** footprint_m * carpet scale — one grid step, the quad's exact size. */
  tileWorld: number
  /** Height of the quad plane above the ground (the capture centre). */
  quadY: number
  /** World metres spanned by the full [-0.5, 0.5] depth code. */
  reliefM: number
  tLo: [number, number]
  tSpan: [number, number]
  tPerM: [number, number]
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
  /** Candidate slots per cell, padded to the cull workgroup width. */
  slotsPerCell: number
  /** Non-null for a mat species (stand entry with carpetDiv > 0). */
  carpet: CarpetConst | null
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

  // A species drawn only as a mat never samples anything but the straight-down
  // view: the other 16 layers are pure dead weight (10.1 of 10.7 MiB per
  // plane), so they are not uploaded at all.
  const matOnly = (id: string): boolean =>
    ctx.stand.species.some((e) => e.species === id && (e.carpetDiv ?? 0) > 0) &&
    !ctx.stand.species.some((e) => e.species === id && (e.carpetDiv ?? 0) <= 0)

  // Sequential: the poa bake transiently needs several hundred MB.
  const speciesGpu = new Map<string, SpeciesGpu>()
  for (const entry of ctx.stand.species) {
    if (speciesGpu.has(entry.species)) continue
    const set = await loadSpeciesViews(ctx, entry.species)
    speciesGpu.set(entry.species, upload(ctx, entry.species, set, matOnly(entry.species)))
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
    const carpet = carpetConstants(ctx, standEntry, gpu.set)
    // Slots to EVALUATE per cell is not the same number as instances to store:
    // a carpet has carpet_div^2 of them (484 for the bog moss), and visiting
    // only SCATTER_MAX_PER_CELL of those renders a quarter of the mat.
    const slotsPerCell = Math.ceil(standEntrySlots(standEntry) / CULL_WG) * CULL_WG
    // Expected survivors per m^2. A carpet ignores `density` entirely — its
    // cover comes from the grid — and its wetness interval is a hard spatial
    // partition, so a small shell can sit wholly inside one zone at the full
    // grid rate while a large annulus averages over many zones.
    const band = standEntry.wetWidth && standEntry.wetWidth > 0 ? Math.min(1, standEntry.wetWidth) : 1
    const gridRate = carpet ? standEntrySlots(standEntry) / (CELL * CELL) : Math.min(standEntry.density, SCATTER_MAX_DENSITY)
    const bandRate = carpet ? gridRate * Math.min(1, band * 1.15) : gridRate
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
      const perM2 = area < CARPET_FULL_RATE_AREA ? gridRate : bandRate
      const cap = Math.ceil((area * perM2 * SHELL_FILL[i]! + SHELL_SLACK[i]!) / ALIGN) * ALIGN
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
      slotsPerCell,
      carpet,
    }
  })

  const upright = entries.filter((e) => e.carpet === null)
  const mats = entries.filter((e) => e.carpet !== null)

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
    // Mat species draw a ground-parallel, terrain-conformed tile instead of a
    // screen-aligned card — a different shape, so a different shader.
    const carpetModule = ctx.shaders.module(carpetSrc)
    for (const fs of ['fs_carpet_near', 'fs_carpet_mid', 'fs_carpet_far']) {
      drawPipelines.set(
        fs,
        device.createRenderPipeline({
          label: `${ctx.id}/${fs}`,
          layout,
          vertex: { module: carpetModule, entryPoint: 'vs_carpet' },
          fragment: { module: carpetModule, entryPoint: fs, targets: [{ format: ctx.colorFormat }] },
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
  const carpetShellPipe: GPURenderPipeline[] = []

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
      carpetShellPipe.length = 0
      for (let i = 0; i < N_SHELLS; i++) {
        const inner = i === 0 ? 0 : SHELL_R[i - 1]!
        if (inner < ctx.params.depthDist) {
          shellPipe.push(drawPipelines.get('fs_near')!)
          carpetShellPipe.push(drawPipelines.get('fs_carpet_near')!)
          warpEnd = SHELL_R[i]!
        } else if (inner < ctx.params.warpDist) {
          shellPipe.push(drawPipelines.get('fs_mid')!)
          carpetShellPipe.push(drawPipelines.get('fs_carpet_mid')!)
          warpEnd = SHELL_R[i]!
        } else {
          shellPipe.push(drawPipelines.get('fs_far')!)
          carpetShellPipe.push(drawPipelines.get('fs_carpet_far')!)
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
        info[82] = entry.gpu.topLayer
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
        info[93] = entry.slotsPerCell
        const c = entry.carpet
        info[94] = c ? c.tileWorld : 0
        info[95] = c ? c.quadY : 0
        info[96] = c ? c.reliefM : 0
        info[97] = c ? c.tLo[0] : 0
        info[98] = c ? c.tLo[1] : 0
        info[99] = c ? c.tSpan[0] : 1
        info[100] = c ? c.tSpan[1] : 1
        info[101] = c ? c.tPerM[0] : 0
        info[102] = c ? c.tPerM[1] : 0
        info[103] = ctx.params.carpetAlphaRef
        info[104] = entry.gpu.topLayer
        info[105] = 0.004
        device.queue.writeBuffer(entry.infoBuffer, 0, info)
        device.queue.writeBuffer(entry.indirectBuffer, 0, indirectReset)
        entry.slotsPerFrame = sideX * sideZ * entry.slotsPerCell
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
        // Cards first, then mats: two shapes, so two pipelines per shell —
        // grouped so the switch happens twice, not once per entry.
        if (upright.length > 0) {
          pass.setPipeline(shellPipe[i]!)
          for (const entry of upright) {
            pass.setBindGroup(1, entry.drawBindGroups[i]!)
            pass.drawIndirect(entry.indirectBuffer, i * 16)
          }
        }
        if (mats.length > 0) {
          pass.setPipeline(carpetShellPipe[i]!)
          for (const entry of mats) {
            pass.setBindGroup(1, entry.drawBindGroups[i]!)
            pass.drawIndirect(entry.indirectBuffer, i * 16)
          }
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

/**
 * Upload the baked mip chains into two 2d-array textures. `matOnly` uploads
 * ONLY the straight-down layer: a species drawn as a carpet never samples an
 * azimuth view, so the other 16 layers would be 20 MiB of dead VRAM.
 */
function upload(
  ctx: ExperimentContext<typeof PARAMS>,
  speciesId: string,
  set: ViewSet,
  matOnly: boolean,
): SpeciesGpu {
  const { device } = ctx
  const layers = matOnly ? 1 : N_LAYERS
  const srcLayer = matOnly ? TOP_LAYER : 0
  const mk = (tag: string): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/${speciesId}/${tag}`,
        size: [TILE, TILE, layers],
        format: 'rgba8unorm',
        mipLevelCount: MIP_LEVELS,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      },
      { species: speciesId, tag },
    )
  const albedoTex = mk('view-albedo')
  const geoTex = mk('view-geo')

  let size = TILE
  for (let level = 0; level < MIP_LEVELS; level++) {
    // The chain is level-major then layer-major, so one layer of one level is
    // a contiguous run either way.
    const texelOffset = levelOffsetTexels(TILE, level, N_LAYERS) + size * size * srcLayer
    const layout = { offset: texelOffset * 4, bytesPerRow: size * 4, rowsPerImage: size }
    device.queue.writeTexture({ texture: albedoTex, mipLevel: level }, set.albedo, layout, [size, size, layers])
    device.queue.writeTexture({ texture: geoTex, mipLevel: level }, set.geo, layout, [size, size, layers])
    size = Math.max(1, size >> 1)
  }
  if (levelOffsetTexels(TILE, MIP_LEVELS, N_LAYERS) !== mipTexels(TILE, MIP_LEVELS) * N_LAYERS) {
    console.warn(`[${ctx.id}] mip chain size mismatch for ${speciesId}`)
  }

  const table = matOnly ? set.viewTable.slice(TOP_LAYER * 16, (TOP_LAYER + 1) * 16) : set.viewTable
  const viewBuffer = ctx.res.createBuffer(
    {
      label: `${ctx.id}/${speciesId}/view-table`,
      size: table.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    },
    { species: speciesId, tag: 'view-table' },
  )
  device.queue.writeBuffer(viewBuffer, 0, table)

  return { set, albedoTex, geoTex, viewBuffer, topLayer: matOnly ? 0 : TOP_LAYER }
}

/**
 * Constants for a mat species, all derived from the bake and the stand. Null
 * for an ordinary scattered entry.
 *
 * The straight-down view has R along +X and U along -Z, so the map from a mesh
 * point (px, pz) to a texcoord is separable — one tile period is a rectangle in
 * texture space, and the runtime needs nothing but its origin and extent.
 */
function carpetConstants(
  ctx: ExperimentContext<typeof PARAMS>,
  standEntry: StandSpecies,
  set: ViewSet,
): CarpetConst | null {
  const div = standEntry.carpetDiv ?? 0
  if (div <= 0) return null
  const tileM = speciesById(standEntry.species).tileM
  if (tileM === undefined || tileM <= 0) {
    throw new Error(`[${ctx.id}] carpet entry "${standEntry.species}" has no periodic tileM`)
  }
  // The constant scale a carpet tile is placed at, i.e. one tile exactly fills
  // its grid step. Recomputed rather than read from the entry, because
  // standEntry.scaleMin still holds the stand's placeholder and the harness
  // does not export carpetScale() (see NOTES.md, interface feedback).
  const scale = CELL / div / tileM
  const vt = set.viewTable
  const o = TOP_LAYER * 16
  const rx = vt[o]!
  const rz = vt[o + 2]!
  const eu = vt[o + 3]!
  const ux = vt[o + 4]!
  const uz = vt[o + 6]!
  const ev = vt[o + 7]!
  const ew = vt[o + 11]!
  // t = (0.5 + (p-C).R / 2eu, 0.5 - (p-C).U / 2ev), evaluated at the tile
  // square's (0, 0) corner — every current source mesh puts its periodic tile
  // origin at the mesh origin.
  const tLoX = 0.5 + (-set.cx * rx - set.cz * rz) / (2 * eu)
  const tLoY = 0.5 - (-set.cx * ux - set.cz * uz) / (2 * ev)
  const tSpanX = (tileM * rx) / (2 * eu)
  const tSpanY = (-tileM * uz) / (2 * ev)
  const skew = Math.max(Math.abs(rz), Math.abs(ux))
  if (skew > 1e-4) console.warn(`[${ctx.id}] top view is not axis-aligned (skew ${skew}) — carpet uv will be wrong`)
  const tileWorld = tileM * scale
  return {
    tileWorld,
    quadY: set.cy * scale,
    reliefM: 2 * ew * scale,
    tLo: [tLoX, tLoY],
    tSpan: [tSpanX, tSpanY],
    tPerM: [tSpanX / tileWorld, tSpanY / tileWorld],
  }
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
