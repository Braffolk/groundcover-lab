import shaderSrc from './shaders/qlf.wgsl'
import { ATLAS, loadOrBakeSpecies, PROFILE_CARPET, PROFILE_PLANT, type QlfBaked } from './bake.ts'
import {
  speciesById,
  standEntrySlots,
  type Experiment,
  type ExperimentContext,
  type FrameInfo,
  type ViewTargets,
} from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Factored quantized light field renderer. Per species the 4D light field is
 * stored factored: a hemi-oct grid of 8-bit ortho DEPTH views (geometry, all
 * the view-dependence) plus view-independent 3D albedo/normal volumes
 * (radiance). Every plant is a single camera-facing PROXY card whose fragments
 * reconstruct the eye ray's hit point from the 4 nearest depth views, shade it
 * from the volumes and write its true depth — the card is a window, not the
 * surface. Placement is the scatter WGSL twin over a bounded camera-centred
 * region: per-frame cost never depends on the stand's total plant count.
 *
 * Carpet entries (Sphagnum, `carpetDiv > 0`) render the stand's mat exactly as
 * given — same grid, same 90-degree yaws, same constant scale — but get their
 * own bake profile (fewer views, 2x the texels per view), a tile-centred
 * anchor, ground-tilted frames, slab-hugging proxies and depth-gradient
 * normals. See NOTES.md.
 */

interface EntryGpu {
  entryIndex: number
  speciesId: string
  baked: QlfBaked
  /** carpet_div^2 for a mat, else the scatter budget. NEVER the global budget. */
  slots: number
  isCarpet: boolean
  infoBuffer: GPUBuffer
  bindGroup: GPUBindGroup
}

const CELL = 4 // must equal SCATTER_CELL_SIZE
/**
 * Alpha reference for carpet tiles instead of the `coverage` param: a mat is a
 * closed surface and must not dissolve into speckle with distance, where a
 * 0.9mm texel is far below one pixel and a single point sample decides a whole
 * tile. Low reference = more solid, depth-writing coverage (the opposite of
 * dithering), while genuinely empty texels between cushions still open.
 */
const CARPET_COV = 0.15
/**
 * Height of the mat's mean capitulum apex as a fraction of the tile's top: the
 * three Sphagnum manifests all put canopy.capitulumApexMeanH / tile.topH at
 * 0.739-0.754, so this is the surface a collapsed far-field tile presents.
 */
const CARPET_TOP = 0.74

/**
 * Six inward normalized frustum planes of `viewProj`, for the cell-level reject
 * in the vertex stage. Computed on the CPU because it is per-frame constant, so
 * the shader pays one dot product per plane and no square roots.
 */
function frustumPlanes(vp: Float32Array, out: Float32Array, at: number): void {
  const row = (r: number): number[] => [vp[r]!, vp[4 + r]!, vp[8 + r]!, vp[12 + r]!]
  const r0 = row(0)
  const r1 = row(1)
  const r2 = row(2)
  const r3 = row(3)
  const comb = (a: number[], b: number[], s: number): number[] => [
    a[0]! + s * b[0]!,
    a[1]! + s * b[1]!,
    a[2]! + s * b[2]!,
    a[3]! + s * b[3]!,
  ]
  // WebGPU clip space: 0 <= z <= w, so near is row2 and far is row3 - row2.
  const planes = [comb(r3, r0, 1), comb(r3, r0, -1), comb(r3, r1, 1), comb(r3, r1, -1), r2, comb(r3, r2, -1)]
  planes.forEach((pl, i) => {
    const l = Math.hypot(pl[0]!, pl[1]!, pl[2]!) || 1
    out[at + i * 4] = pl[0]! / l
    out[at + i * 4 + 1] = pl[1]! / l
    out[at + i * 4 + 2] = pl[2]! / l
    out[at + i * 4 + 3] = pl[3]! / l
  })
}

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx

  const sampler = device.createSampler({
    label: `${ctx.id}/samp`,
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
    addressModeW: 'clamp-to-edge',
  })

  // One light field per (species, profile): OPFS cache -> committed artifact ->
  // fresh bake (which is then committed). The Sphagnum tiles are 19.8M tris, so
  // a fresh bake costs real time and must never be paid twice.
  const speciesCache = new Map<string, Promise<QlfBaked>>()
  const loadSpecies = (speciesId: string, isCarpet: boolean): Promise<QlfBaked> => {
    const profile = isCarpet ? PROFILE_CARPET : PROFILE_PLANT
    const cacheKey = `${speciesId}/${profile.key}`
    let p = speciesCache.get(cacheKey)
    if (!p) {
      p = loadOrBakeSpecies(ctx, speciesId, profile, () => ctx.meshes.load(speciesById(speciesId).meshId))
      speciesCache.set(cacheKey, p)
    }
    return p
  }

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '3d' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '3d' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })

  // Bake sequentially — the species share the GPU during startup and parallel
  // bakes would just interleave submits.
  const entries: EntryGpu[] = []
  for (const [entryIndex, e] of ctx.stand.species.entries()) {
    const isCarpet = (e.carpetDiv ?? 0) > 0
    const baked = await loadSpecies(e.species, isCarpet)
    const infoBuffer = ctx.res.createBuffer(
      { label: `${ctx.id}/info-${entryIndex}`, size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
      { species: e.species, tag: 'qlf-info' },
    )
    const bindGroup = device.createBindGroup({
      label: `${ctx.id}/bg-${entryIndex}`,
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: infoBuffer } },
        { binding: 1, resource: baked.depthAtlas.createView() },
        { binding: 2, resource: baked.albedoVol.createView() },
        { binding: 3, resource: baked.normalVol.createView() },
        { binding: 4, resource: sampler },
      ],
    })
    entries.push({
      entryIndex,
      speciesId: e.species,
      baked,
      slots: standEntrySlots(e),
      isCarpet,
      infoBuffer,
      bindGroup,
    })
  }

  let pipeline!: GPURenderPipeline
  const build = (): void => {
    const module = ctx.shaders.module(shaderSrc)
    pipeline = device.createRenderPipeline({
      label: `${ctx.id}/qlf-cards`,
      layout: device.createPipelineLayout({ label: ctx.id, bindGroupLayouts: [ctx.frame.layout, bgl] }),
      vertex: { module, entryPoint: 'vs_main' },
      fragment: { module, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  const info = new Float32Array(64)
  let side = 1

  return {
    update(frame: FrameInfo): void {
      const R = ctx.params.regionRadius
      const cam = frame.camera.pose
      const originCellX = Math.floor((cam.x - R) / CELL)
      const originCellZ = Math.floor((cam.z - R) / CELL)
      side = Math.max(1, Math.ceil((2 * R) / CELL) + 1)
      frustumPlanes(frame.camera.viewProj as Float32Array, info, 36)

      for (const entry of entries) {
        const b = entry.baked
        const e = ctx.stand.species[entry.entryIndex]!
        const rwMax = b.radius * Math.max(e.scaleMax, e.scaleMin)
        info[0] = b.center[0]; info[1] = b.center[1]; info[2] = b.center[2]; info[3] = b.radius
        info[4] = b.bmin[0]; info[5] = b.bmin[1]; info[6] = b.bmin[2]; info[7] = b.grid
        info[8] = b.bmax[0]; info[9] = b.bmax[1]; info[10] = b.bmax[2]; info[11] = b.tile
        info[12] = ATLAS; info[13] = entry.entryIndex; info[14] = originCellX; info[15] = originCellZ
        info[16] = side; info[17] = ctx.seed; info[18] = R
        info[19] = entry.isCarpet ? CARPET_COV : ctx.params.coverage
        info[20] = ctx.params.refine ? 1 : 0; info[21] = b.heightM
        info[22] = ctx.params.showViewGrid ? 1 : 0
        info[23] = entry.isCarpet ? ctx.params.carpetRelief : 0
        info[24] = b.halfU[0]; info[25] = b.halfU[1]; info[26] = b.halfU[2]
        // Half a cell + the widest card + wind slack: the cell-level reject must
        // never clip a plant whose card still reaches the screen.
        info[27] = CELL * 0.5 + rwMax + 0.6
        // Collapse slab for a carpet, in unit-sphere local space: the periodic
        // tile square (centre + half size) and the mean capitulum height. Beyond
        // FAR0..FAR1 the light field is sampled far below one texel per pixel
        // and can only alias, so the mat becomes what it actually is at that
        // range: a closed slab of cushion tops.
        const f2 = (e.carpetDiv ? (speciesById(e.species).tileM ?? 0) : 0) * 0.5
        info[28] = (f2 - b.center[0]) / b.radius
        info[29] = (f2 - b.center[2]) / b.radius
        info[30] = f2 / b.radius
        info[31] = (CARPET_TOP * b.bmax[1] - b.center[1]) / b.radius
        info[32] = (0 - b.center[1]) / b.radius
        info[33] = entry.isCarpet ? ctx.params.collapseNear : 1e9
        info[34] = entry.isCarpet ? ctx.params.collapseFar : 2e9
        device.queue.writeBuffer(entry.infoBuffer, 0, info)
      }
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const pass = ctx.timing.renderPass(enc, 'qlf-cards', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      for (const entry of entries) {
        pass.setBindGroup(1, entry.bindGroup)
        // Slots per cell are per ENTRY: 484 for the bog's life-size Sphagnum
        // carpet, 128 for a scattered species.
        pass.draw(6, side * side * entry.slots)
      }
      pass.end()
    },

    dispose(): void {
      unsubscribe()
      // GPU resources are destroyed by the harness via ctx.res.
    },
  }
}
