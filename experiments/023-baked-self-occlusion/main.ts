import cullSrc from './shaders/cull.wgsl'
import cardsSrc from './shaders/cards.wgsl'
import mipgenSrc from './shaders/mipgen.wgsl'
import { ATLAS_A, ATLAS_B, GRID, TILE_A, TILE_B, TOP_TILE, loadSpeciesAtlas, type SelfOccAtlas } from './bake.ts'
import {
  SCATTER_CELL_SIZE,
  SCATTER_MAX_PER_CELL,
  speciesById,
  type Experiment,
  type ExperimentContext,
  type FrameInfo,
  type ViewTargets,
} from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Baked self-occlusion cards. Startup: per species a 5x5 atlas of baked views
 * (8 azimuths x 3 elevations + top) is loaded from mesh/baked / OPFS or baked
 * in-browser from the raw GCMESH1 mesh, uploaded and mipped — an albedo +
 * coverage atlas and a geometry atlas holding, per texel, the surface depth
 * behind the tile's near plane, the fraction of sky it can see, and a shading
 * normal leaned toward its openness direction.
 *
 * Per frame: one compute pass evaluates the shared scatter over a
 * camera-centered cell region, frustum-culls, and compacts survivors into TWO
 * indirect draws — near plants get the depth-warped card (3 taps), far plants
 * get the same card flat (2 taps). One view-aligned quad per plant either way.
 * Per-frame cost is O(visible region), independent of the stand's plant count.
 *
 * CARPET SPECIES (stand `carpetDiv > 0`, e.g. the bog Sphagnum) take a
 * different shape, because a camera-facing card cannot be a mat: one
 * GROUND-PARALLEL, tile-sized quad per tile, conformed to the terrain per
 * vertex, textured from the straight-down baked view only, and given its
 * cushion relief by the same per-texel depth field — used here as a classic
 * one-step parallax offset in the tile's tangent plane rather than as a warp
 * along the eye ray. Only 1 of the 25 baked views is ever sampled, so a carpet
 * species uploads that single tile instead of the whole atlas: 0.9MB instead of
 * 20.8MB, which is where the budget belongs for a plant that has no silhouette.
 */

const CELL = SCATTER_CELL_SIZE
const REGION_MAX = 112 // keep equal to the manifest's regionRadius max
const NEAR_SPLIT_MAX = 44 // keep equal to the manifest's nearSplit max
const MIPS_A = Math.floor(Math.log2(ATLAS_A)) + 1
const MIPS_B = Math.floor(Math.log2(ATLAS_B)) + 1
const INFO_FLOATS = 56
/**
 * Height of a carpet tile's quad, as a fraction of the baked capture height —
 * where the source mesh puts its mean capitulum apex, i.e. the surface the
 * straight-down view actually shows. The parallax offset is measured from this
 * plane, so it moves the imagery both ways (capitula above it, hollows below)
 * instead of only sinking.
 */
const CARPET_PLANE = 0.74

interface SpeciesGpu {
  atlas: SelfOccAtlas
  albedoTex: GPUTexture
  geomTex: GPUTexture
  tileBuffer: GPUBuffer
  /** Only the straight-down tile was uploaded (mat species). */
  carpet: boolean
  /** (scale.xy, bias.xy) mapping tile-local [0,1]² into that tile. */
  carpetUv: [number, number, number, number]
}

interface EntryGpu {
  gpu: SpeciesGpu
  isCarpet: boolean
  capNear: number
  capFar: number
  infoBuffer: GPUBuffer
  argsNear: GPUBuffer
  argsFar: GPUBuffer
  cullBindGroup: GPUBindGroup
  drawNear: GPUBindGroup
  drawFar: GPUBindGroup
  /** Candidate slots per cell the cull must EVALUATE (carpetDiv², or 128). */
  enumSlots: number
  /** (scale.xy, bias.xy) mapping tile-local [0,1]² into the top view's box. */
  carpetUv: [number, number, number, number]
  slotsPerFrame: number
}


export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx

  const mkSampler = (name: string, aniso: number): GPUSampler =>
    device.createSampler({
      label: `${ctx.id}/${name}`,
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
      maxAnisotropy: aniso,
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    })
  // Coverage/albedo is the sharp channel and earns anisotropy at grazing
  // angles; the geometry atlas is a third of that resolution and does not.
  const albedoSampler = mkSampler('albedo-sampler', 4)
  const geomSampler = mkSampler('geom-sampler', 1)

  // --- species atlases (sequential: the poa bake is transiently heavy) -------
  // A species laid out as a carpet only ever samples the straight-down view, so
  // it uploads that one tile rather than all 25.
  const carpetSpecies = new Set(
    ctx.stand.species.filter((e) => (e.carpetDiv ?? 0) > 0).map((e) => e.species),
  )
  const speciesGpu = new Map<string, SpeciesGpu>()
  for (const entry of ctx.stand.species) {
    if (speciesGpu.has(entry.species)) continue
    const atlas = await loadSpeciesAtlas(ctx, entry.species)
    const tileM = speciesById(entry.species).tileM ?? 0
    speciesGpu.set(entry.species, uploadAtlas(ctx, entry.species, atlas, carpetSpecies.has(entry.species), tileM))
  }

  const cullBgl = device.createBindGroupLayout({
    label: `${ctx.id}/cull-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  })
  const drawBgl = device.createBindGroupLayout({
    label: `${ctx.id}/draw-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      { binding: 6, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  const entries: EntryGpu[] = ctx.stand.species.map((standEntry, entryIndex) => {
    const gpu = speciesGpu.get(standEntry.species)!
    const carpetDiv = standEntry.carpetDiv ?? 0
    const isCarpet = carpetDiv > 0
    // TWO DIFFERENT NUMBERS, and conflating them silently drops plants:
    //  * enumSlots — candidate slots per cell the cull must EVALUATE. A carpet
    //    has carpetDiv² of them (484 at life size), deliberately over the
    //    128-slot scatter budget; clamping to 128 renders ~26% of the mat as
    //    horizontal bands with bare peat between them.
    //  * perM2 — how many are expected to SURVIVE, which is all the instance
    //    buffers need to hold. For a zone-partitioned carpet that is roughly
    //    the entry's `wetWidth` share of the grid, not the whole grid.
    const enumSlots = isCarpet ? carpetDiv * carpetDiv : SCATTER_MAX_PER_CELL
    const band = Math.min(1, standEntry.wetWidth ?? 1)
    const perM2 = isCarpet
      ? ((carpetDiv * carpetDiv) / (CELL * CELL)) * Math.min(1, band * 1.25)
      : Math.min(standEntry.density, 8)
    // Region area x density plus slack. The scatter's count over a region this
    // size has a relative spread well under 1%, so 6% + a flat 2k is ample.
    const capNear = Math.ceil(Math.PI * NEAR_SPLIT_MAX ** 2 * perM2 * 1.2) + 1024
    const capFar = Math.ceil(Math.PI * REGION_MAX ** 2 * perM2 * 1.06) + 2048
    const infoBuffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/info-${entryIndex}`,
        size: INFO_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      },
      { species: standEntry.species, tag: 'entry-info' },
    )
    const mkInst = (name: string, cap: number): GPUBuffer =>
      ctx.res.createBuffer(
        { label: `${ctx.id}/${name}-${entryIndex}`, size: cap * 16, usage: GPUBufferUsage.STORAGE },
        { species: standEntry.species, tag: 'culled-instances' },
      )
    const instNear = mkInst('inst-near', capNear)
    const instFar = mkInst('inst-far', capFar)
    const mkArgs = (name: string): GPUBuffer =>
      ctx.res.createBuffer(
        {
          label: `${ctx.id}/${name}-${entryIndex}`,
          size: 16,
          usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        },
        { species: standEntry.species, tag: 'indirect-args' },
      )
    const argsNear = mkArgs('args-near')
    const argsFar = mkArgs('args-far')
    const mkDraw = (name: string, inst: GPUBuffer): GPUBindGroup =>
      device.createBindGroup({
        label: `${ctx.id}/${name}-${entryIndex}`,
        layout: drawBgl,
        entries: [
          { binding: 0, resource: { buffer: infoBuffer } },
          { binding: 1, resource: { buffer: inst } },
          { binding: 2, resource: { buffer: gpu.tileBuffer } },
          { binding: 3, resource: gpu.albedoTex.createView() },
          { binding: 4, resource: gpu.geomTex.createView() },
          { binding: 5, resource: albedoSampler },
          { binding: 6, resource: geomSampler },
        ],
      })
    return {
      gpu,
      isCarpet,
      capNear,
      capFar,
      infoBuffer,
      argsNear,
      argsFar,
      cullBindGroup: device.createBindGroup({
        label: `${ctx.id}/cull-bg-${entryIndex}`,
        layout: cullBgl,
        entries: [
          { binding: 0, resource: { buffer: infoBuffer } },
          { binding: 1, resource: { buffer: instNear } },
          { binding: 2, resource: { buffer: instFar } },
          { binding: 3, resource: { buffer: argsNear } },
          { binding: 4, resource: { buffer: argsFar } },
        ],
      }),
      drawNear: mkDraw('draw-near', instNear),
      drawFar: mkDraw('draw-far', instFar),
      enumSlots,
      carpetUv: gpu.carpetUv,
      slotsPerFrame: 0,
    }
  })

  let cullPipeline!: GPUComputePipeline
  let nearPipeline!: GPURenderPipeline
  let farPipeline!: GPURenderPipeline
  let carpetNearPipeline!: GPURenderPipeline
  let carpetFarPipeline!: GPURenderPipeline
  const build = (): void => {
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/cull-pl`,
        bindGroupLayouts: [ctx.frame.layout, cullBgl],
      }),
      compute: { module: ctx.shaders.module(cullSrc), entryPoint: 'cs_cull' },
    })
    const module = ctx.shaders.module(cardsSrc)
    const layout = device.createPipelineLayout({
      label: `${ctx.id}/cards-pl`,
      bindGroupLayouts: [ctx.frame.layout, drawBgl],
    })
    const mkCards = (name: string, vs: string, fs: string): GPURenderPipeline =>
      device.createRenderPipeline({
        label: `${ctx.id}/${name}`,
        layout,
        vertex: { module, entryPoint: vs },
        fragment: { module, entryPoint: fs, targets: [{ format: ctx.colorFormat }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
      })
    nearPipeline = mkCards('cards-near', 'vs_near', 'fs_near')
    farPipeline = mkCards('cards-far', 'vs_far', 'fs_far')
    carpetNearPipeline = mkCards('carpet-near', 'vs_carpet', 'fs_carpet_near')
    carpetFarPipeline = mkCards('carpet-far', 'vs_carpet', 'fs_carpet_far')
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const cellMin = Math.floor(-ctx.stand.radius / CELL)
  const cellMax = Math.floor(ctx.stand.radius / CELL)
  const info = new Float32Array(INFO_FLOATS)
  const planes = new Float32Array(24)
  const argsReset = new Uint32Array([6, 0, 0, 0])

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

      const split = Math.min(ctx.params.nearSplit, NEAR_SPLIT_MAX)
      entries.forEach((entry, entryIndex) => {
        const a = entry.gpu.atlas
        info.set(planes, 0)
        info.set([x0, z0, sideX, sideZ], 24)
        info.set([ctx.seed, entryIndex, R, split], 28)
        info.set([entry.capNear, entry.capFar, ctx.params.alphaRef, a.sphereR], 32)
        info.set([a.aabbC[0], a.aabbC[1], a.aabbC[2], a.aabbC[1] + a.aabbH[1]], 36)
        info.set([a.aabbH[0], a.aabbH[1], a.aabbH[2], 0], 40)
        info.set([ctx.params.occlusion, ctx.params.transmission, ctx.params.canopyDepth, ctx.params.parallax], 44)
        info.set([entry.enumSlots, CARPET_PLANE * (a.aabbC[1] + a.aabbH[1]), entry.isCarpet ? 1 : 0, 0], 48)
        info.set(entry.carpetUv, 52)
        device.queue.writeBuffer(entry.infoBuffer, 0, info)
        device.queue.writeBuffer(entry.argsNear, 0, argsReset)
        device.queue.writeBuffer(entry.argsFar, 0, argsReset)
        entry.slotsPerFrame = sideX * sideZ * entry.enumSlots
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

      const pass = ctx.timing.renderPass(enc, 'cards', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setBindGroup(0, ctx.frame.bindGroup)
      // Near first: coarse front-to-back so early-z rejects the far tier.
      // Carpets take their own pipelines (ground-parallel quad, tangent-space
      // parallax) but the same near/far split and the same instance lists.
      const drawGroup = (pipeline: GPURenderPipeline, carpet: boolean, near: boolean): void => {
        let bound = false
        for (const entry of entries) {
          if (entry.isCarpet !== carpet) continue
          if (!bound) {
            pass.setPipeline(pipeline)
            bound = true
          }
          pass.setBindGroup(1, near ? entry.drawNear : entry.drawFar)
          pass.drawIndirect(near ? entry.argsNear : entry.argsFar, 0)
        }
      }
      drawGroup(nearPipeline, false, true)
      drawGroup(carpetNearPipeline, true, true)
      drawGroup(farPipeline, false, false)
      drawGroup(carpetFarPipeline, true, false)
      pass.end()
    },

    dispose(): void {
      unsubscribe()
      // GPU resources are reclaimed by the harness via ctx.res.
    },
  }
}

/** Copy one tile out of a square atlas of GRID x GRID tiles. */
function cropTile(
  src: Uint8Array,
  atlasSize: number,
  tileSize: number,
  tileIndex: number,
): Uint8Array<ArrayBuffer> {
  const col = tileIndex % GRID
  const row = Math.floor(tileIndex / GRID)
  const out = new Uint8Array(tileSize * tileSize * 4)
  for (let y = 0; y < tileSize; y++) {
    const from = ((row * tileSize + y) * atlasSize + col * tileSize) * 4
    out.set(src.subarray(from, from + tileSize * 4), y * tileSize * 4)
  }
  return out
}

/**
 * Make a carpet species' straight-down view PERIODIC, at load time.
 *
 * The bake renders one mesh instance, and a community tile overflows its own
 * period (0.24m of Sphagnum inside a 0.18m step). So the captured square is a
 * tile plus its own overhang, and the overhang of the *neighbouring* tiles —
 * which in the real mat grows back into this square — was never drawn. Show
 * such a square as one quad and every tile edge is short of coverage: the mat
 * gets a visible lattice of thin seams, which is exactly what it did.
 *
 * The fix needs no rebake, because everything missing is present elsewhere in
 * the same image: the mesh is periodic, so the neighbour's overhang at p is the
 * mesh's own overhang at p ± period. Compositing each texel with its lattice
 * partners (the more-covered sample wins — a top-down view of a mat sees the
 * fuller surface) both fills the deficit and makes the result exactly periodic,
 * so the sampling window can then be ANY window one period wide and its edges
 * match by construction. Coverage on the bog tile goes 0.71 -> ~0.79 and the
 * seams disappear.
 *
 * The normal channel is smoothed in the same pass. Sphagnum leaflets are far
 * below one geometry texel (1.6mm), so the bake's point-picked normal — right
 * for a grass blade, which spans many texels — is per-texel noise here, and it
 * turns the sun-visibility cone into salt and pepper. Averaging the decoded
 * vectors over 5x5 is safe in this one case because a straight-down capture of
 * a mat has every normal in the upper hemisphere (the usual "octahedral normals
 * are not mip-averageable" trap needs opposing front/back faces to bite), and
 * capitula are ~15 texels across, so their shape survives.
 */
function periodicCarpetTile(atlas: SelfOccAtlas, tileM: number): {
  albedo: Uint8Array<ArrayBuffer>
  geom: Uint8Array<ArrayBuffer>
  uv: [number, number, number, number]
} {
  const o = TOP_TILE * 16
  const boxCx = atlas.tileTable[o + 3]!
  const boxCy = atlas.tileTable[o + 7]!
  const hx = atlas.tileTable[o + 12]!
  const hy = atlas.tileTable[o + 13]!
  const albedo = cropTile(atlas.albedo, ATLAS_A, TILE_A, TOP_TILE)
  const geom = cropTile(atlas.geom, ATLAS_B, TILE_B, TOP_TILE)
  const ratio = TILE_A / TILE_B // exactly 3
  // Period in texels, rounded on the COARSE grid so both atlases share a
  // lattice and stay aligned.
  const px = Math.round((tileM / (2 * hx)) * TILE_B)
  const py = Math.round((tileM / (2 * hy)) * TILE_B)

  // Coverage key at geometry resolution: the mean albedo alpha of the block.
  const cover = new Float32Array(TILE_B * TILE_B)
  for (let y = 0; y < TILE_B; y++) {
    for (let x = 0; x < TILE_B; x++) {
      let s = 0
      for (let j = 0; j < ratio; j++) {
        for (let i = 0; i < ratio; i++) s += albedo[((y * ratio + j) * TILE_A + x * ratio + i) * 4 + 3]!
      }
      cover[y * TILE_B + x] = s / (ratio * ratio)
    }
  }

  /** Pick the lattice partner with the most coverage, per texel. */
  const composite = (
    src: Uint8Array,
    size: number,
    perX: number,
    perY: number,
    key: (x: number, y: number) => number,
  ): Uint8Array<ArrayBuffer> => {
    const out = new Uint8Array(size * size * 4)
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        let bx = x
        let by = y
        let best = key(x, y)
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue
            const sx = x + dx * perX
            const sy = y + dy * perY
            if (sx < 0 || sy < 0 || sx >= size || sy >= size) continue
            const k = key(sx, sy)
            if (k > best) {
              best = k
              bx = sx
              by = sy
            }
          }
        }
        const from = (by * size + bx) * 4
        out.set(src.subarray(from, from + 4), (y * size + x) * 4)
      }
    }
    return out
  }

  const albedoOut = composite(albedo, TILE_A, px * ratio, py * ratio, (x, y) => albedo[(y * TILE_A + x) * 4 + 3]!)
  const geomOut = composite(geom, TILE_B, px, py, (x, y) => cover[y * TILE_B + x]!)

  // 5x5 vector average of the shading normals (channels r,g = octahedral).
  // Mirrors oct_decode/oct_encode in cards.wgsl / bake_resolve.wgsl exactly.
  const dec = (e0: number, e1: number): [number, number, number] => {
    const eu = e0 / 127.5 - 1
    const ev = e1 / 127.5 - 1
    const y = 1 - Math.abs(eu) - Math.abs(ev)
    const x = y < 0 ? (1 - Math.abs(ev)) * (eu >= 0 ? 1 : -1) : eu
    const z = y < 0 ? (1 - Math.abs(eu)) * (ev >= 0 ? 1 : -1) : ev
    const l = Math.hypot(x, y, z) || 1
    return [x / l, y / l, z / l]
  }
  const smoothed = geomOut.slice()
  const R = 2
  for (let y = 0; y < TILE_B; y++) {
    for (let x = 0; x < TILE_B; x++) {
      let sx = 0
      let sy = 0
      let sz = 0
      for (let j = -R; j <= R; j++) {
        for (let i = -R; i <= R; i++) {
          const cx = Math.min(TILE_B - 1, Math.max(0, x + i))
          const cy = Math.min(TILE_B - 1, Math.max(0, y + j))
          const t = (cy * TILE_B + cx) * 4
          const n = dec(geomOut[t]!, geomOut[t + 1]!)
          sx += n[0]
          sy += n[1]
          sz += n[2]
        }
      }
      const s = Math.abs(sx) + Math.abs(sy) + Math.abs(sz) || 1
      // Upper hemisphere only (a straight-down capture of a mat), so no fold.
      const u = sy >= 0 ? sx / s : (1 - Math.abs(sz / s)) * (sx >= 0 ? 1 : -1)
      const v = sy >= 0 ? sz / s : (1 - Math.abs(sx / s)) * (sz >= 0 ? 1 : -1)
      const t = (y * TILE_B + x) * 4
      smoothed[t] = Math.round(Math.min(255, Math.max(0, (u * 0.5 + 0.5) * 255)))
      smoothed[t + 1] = Math.round(Math.min(255, Math.max(0, (v * 0.5 + 0.5) * 255)))
    }
  }

  return {
    albedo: albedoOut,
    geom: smoothed,
    // Sampling window = exactly one composited period, centred on the box.
    uv: [
      (px * ratio) / TILE_A,
      (py * ratio) / TILE_A,
      0.5 - boxCx / (2 * hx),
      0.5 + boxCy / (2 * hy),
    ],
  }
}

/**
 * Upload both atlases plus the tile table, and build the mip chains.
 *
 * A CARPET species uploads only the straight-down tile. It is the only view a
 * ground-parallel mat can ever sample, so the other 24 would be 20MB of dead
 * VRAM — and cropping also gives the mip chain a tile of its own instead of one
 * shared across the 5x5 grid, so distant tiles never blend in a neighbouring
 * view's imagery.
 */
function uploadAtlas(
  ctx: ExperimentContext<typeof PARAMS>,
  speciesId: string,
  atlas: SelfOccAtlas,
  carpet: boolean,
  tileM: number,
): SpeciesGpu {
  const { device } = ctx
  const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST
  const sizeA = carpet ? TILE_A : ATLAS_A
  const sizeB = carpet ? TILE_B : ATLAS_B
  const mipsA = carpet ? Math.floor(Math.log2(TILE_A)) + 1 : MIPS_A
  const mipsB = carpet ? Math.floor(Math.log2(TILE_B)) + 1 : MIPS_B
  const albedoTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/albedo`,
      size: [sizeA, sizeA],
      format: 'rgba8unorm',
      mipLevelCount: mipsA,
      usage,
    },
    { species: speciesId, tag: 'view-albedo' },
  )
  const geomTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/${speciesId}/geom`,
      size: [sizeB, sizeB],
      format: 'rgba8unorm',
      mipLevelCount: mipsB,
      usage,
    },
    { species: speciesId, tag: 'view-geometry' },
  )
  const tile = carpet ? periodicCarpetTile(atlas, tileM) : null
  device.queue.writeTexture(
    { texture: albedoTex },
    tile ? tile.albedo : atlas.albedo,
    { bytesPerRow: sizeA * 4, rowsPerImage: sizeA },
    [sizeA, sizeA],
  )
  device.queue.writeTexture(
    { texture: geomTex },
    tile ? tile.geom : atlas.geom,
    { bytesPerRow: sizeB * 4, rowsPerImage: sizeB },
    [sizeB, sizeB],
  )
  const tileBuffer = ctx.res.createBuffer(
    {
      label: `${ctx.id}/${speciesId}/tiles`,
      size: 100 * 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    },
    { species: speciesId, tag: 'tile-table' },
  )
  device.queue.writeBuffer(tileBuffer, 0, atlas.tileTable)

  const module = ctx.shaders.module(mipgenSrc)
  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/mipgen-bgl`,
    entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } }],
  })
  const layout = device.createPipelineLayout({ label: `${ctx.id}/mipgen-pl`, bindGroupLayouts: [bgl] })
  const mkPipeline = (entryPoint: string): GPURenderPipeline =>
    device.createRenderPipeline({
      label: `${ctx.id}/mipgen-${entryPoint}`,
      layout,
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint, targets: [{ format: 'rgba8unorm' }] },
      primitive: { topology: 'triangle-list' },
    })
  const albedoPipe = mkPipeline('fs_albedo')
  const geomPipe = mkPipeline('fs_geom')

  const enc = device.createCommandEncoder({ label: `${ctx.id}/mipgen` })
  const genMips = (tex: GPUTexture, levels: number, pipeline: GPURenderPipeline): void => {
    for (let level = 1; level < levels; level++) {
      const bg = device.createBindGroup({
        label: `${ctx.id}/mipgen-bg-${level}`,
        layout: bgl,
        entries: [{ binding: 0, resource: tex.createView({ baseMipLevel: level - 1, mipLevelCount: 1 }) }],
      })
      const pass = enc.beginRenderPass({
        label: `${ctx.id}/mipgen-${level}`,
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
  genMips(albedoTex, mipsA, albedoPipe)
  genMips(geomTex, mipsB, geomPipe)
  device.queue.submit([enc.finish()])

  return { atlas, albedoTex, geomTex, tileBuffer, carpet, carpetUv: tile?.uv ?? [1, 1, 0.5, 0.5] }
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
