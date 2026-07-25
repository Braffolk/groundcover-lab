import cullSrc from './shaders/cull.wgsl'
import tuftsSrc from './shaders/tufts.wgsl'
import carpetSrc from './shaders/carpet.wgsl'
import shellSrc from './shaders/shell.wgsl'
import mipgenSrc from './shaders/mipgen.wgsl'
import {
  ALB_H,
  ALB_MIPS,
  ALB_W,
  COV_H,
  COV_MIPS,
  COV_W,
  CT_MIPS,
  CT_PX,
  NRM_H,
  NRM_MIPS,
  NRM_W,
  loadCarpetTile,
  loadSpeciesBake,
  type CanopyBake,
  type CarpetTileBake,
} from './bake.ts'
import { CANOPY_MIPS, CANOPY_TEX, CANOPY_TILE_M, compositeCanopy } from './canopy.ts'
import {
  SCATTER_CELL_SIZE,
  standEntrySlots,
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
const INFO_FLOATS = 88
const SHELL_FLOATS = 68
const SHELL_G = 96 // cells per side, per level
const SHELL_SPACING: readonly [number, number] = [0.75, 3]
const SHELL_EDGE_FADE = 14
const NEAR_VERTS = 24 // 4 quadrant splats
const FAR_VERTS = 6 // 1 whole-plant splat
const CARPET_VERTS = 6 // 1 ground-parallel tile quad
/**
 * Alpha reference for a CARPET, instead of the params' grass reference. A mat
 * is a closed surface, so its tiles must not dissolve with distance: a tile's
 * coverage is ~80% up close, but the mip chain pulls it toward the tile mean
 * and at a grass reference whole distant tiles fail the test, punching holes in
 * ground that is fully covered. Low reference = the mat stays a depth-writing
 * occluder while the genuine gaps down to the peat still open.
 */
const CARPET_ALPHA_REF = 0.06
/**
 * Parallax offset cap (m). Must stay below the lateral size of the cushion
 * mounds the offset is read from (~15mm) or a grazing ray combs the tile into
 * streaks instead of sliding it.
 */
const CARPET_PARALLAX_MAX = 0.008

interface SpeciesGpu {
  bake: CanopyBake
  /** Node atlas — absent for a species that only ever appears as a carpet. */
  cov?: GPUTexture
  albedo?: GPUTexture
  normal?: GPUTexture
  /** Carpet tile capture — present only for carpet species. */
  tile?: CarpetTileBake
  tileAlbedo?: GPUTexture
  tileAux?: GPUTexture
}

interface EntryGpu {
  gpu: SpeciesGpu
  carpet: boolean
  /** Constant tile scale of a carpet entry (the stand's own, recomputed). */
  carpetScale: number
  cap0: number
  cap1: number
  infoBuffer: GPUBuffer
  argsBuffer: GPUBuffer
  argsReset: Uint32Array<ArrayBuffer>
  cullBindGroup: GPUBindGroup
  /** Splat entries: near / far band views of the instance buffer. */
  nearBindGroup?: GPUBindGroup
  farBindGroup?: GPUBindGroup
  /** Carpet entries: the tile textures over segment 0. */
  carpetBindGroup?: GPUBindGroup
  slots: number
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

  const carpetSampler = device.createSampler({
    label: `${ctx.id}/carpet-sampler`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    maxAnisotropy: 8,
    // The tile capture is genuinely periodic (baked with the mesh's 3x3 wrap),
    // so repeat is the correct wrap for both filtering and parallax overshoot.
    addressModeU: 'repeat',
    addressModeV: 'repeat',
  })

  // A species that appears ONLY as a carpet in this stand never draws a splat,
  // so its 14.8 MiB node atlas (5 nodes x 8 azimuths of SIDE views) is pure
  // dead VRAM: it is a 0.09m cushion, and the side view is the one thing a mat
  // does not show. Those species get the carpet tile capture instead, and the
  // node atlas stays on the CPU only — compositeCanopy still needs its top view
  // for the shell.
  const carpetOnly = new Set<string>()
  for (const entry of ctx.stand.species) {
    if ((entry.carpetDiv ?? 0) > 0) carpetOnly.add(entry.species)
  }
  for (const entry of ctx.stand.species) {
    if (!((entry.carpetDiv ?? 0) > 0)) carpetOnly.delete(entry.species)
  }

  // --- baked node atlases (sequential: poa transiently needs a few hundred MB)
  const bakes = new Map<string, CanopyBake>()
  const tiles = new Map<string, CarpetTileBake>()
  for (const entry of ctx.stand.species) {
    if (!bakes.has(entry.species)) bakes.set(entry.species, await loadSpeciesBake(ctx, entry.species))
    if (carpetOnly.has(entry.species) && !tiles.has(entry.species)) {
      tiles.set(entry.species, await loadCarpetTile(ctx, entry.species))
    }
  }

  const mip = new MipGen(ctx)
  const speciesGpu = new Map<string, SpeciesGpu>()
  const uploadEnc = device.createCommandEncoder({ label: `${ctx.id}/upload` })
  for (const [speciesId, bake] of bakes) {
    const tile = tiles.get(speciesId)
    speciesGpu.set(
      speciesId,
      tile
        ? uploadCarpetTile(ctx, mip, uploadEnc, speciesId, bake, tile)
        : uploadNodeAtlas(ctx, mip, uploadEnc, speciesId, bake),
    )
  }

  // --- the stand's single tileable canopy texture ---------------------------
  const tile = compositeCanopy(ctx.stand, bakes, ctx.seed, tiles)
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
  const carpetBgl = device.createBindGroupLayout({
    label: `${ctx.id}/carpet-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
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
    const carpet = (standEntry.carpetDiv ?? 0) > 0
    const density = Math.min(standEntry.density, 8)
    // TWO DIFFERENT NUMBERS, and conflating them silently drops plants:
    //  * `slots` is how many candidate slots the cull must EVALUATE per cell —
    //    carpet_div^2 (484) for a carpet, the scatter budget (128) otherwise.
    //  * the capacities are how many are expected to SURVIVE. For a carpet the
    //    three moss states PARTITION the wetness axis, so each entry claims
    //    about `wetWidth` of the nodes; sizing for all 484 would waste 3x.
    const slots = standEntrySlots(standEntry)
    const band = Math.min(1, standEntry.wetWidth ?? 1)
    // A carpet is one quad at every distance: everything lands in segment 0.
    // Its region is the dissolve region, but never bigger than the stand — the
    // cull's cell rect is clamped to the stand's cells, and outside them the
    // scatter is empty.
    const regionArea = Math.min(Math.PI * REGION_MAX * REGION_MAX, (2 * ctx.stand.radius) ** 2)
    const cap0 = carpet
      ? align16(Math.ceil((regionArea * slots * band * 1.25) / (CELL * CELL)) + 1024)
      : align16(Math.ceil(Math.PI * TUFT_MAX * TUFT_MAX * density * 1.08) + 1024)
    const cap1 = carpet ? 16 : align16(Math.ceil(Math.PI * REGION_MAX * REGION_MAX * density * 1.08) + 2048)
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
          { binding: 2, resource: gpu.cov!.createView() },
          { binding: 3, resource: gpu.albedo!.createView() },
          { binding: 4, resource: gpu.normal!.createView() },
          { binding: 5, resource: tuftSampler },
        ],
      })
    const argsReset = new Uint32Array(8)
    argsReset[0] = carpet ? CARPET_VERTS : NEAR_VERTS
    argsReset[4] = FAR_VERTS
    return {
      gpu,
      carpet,
      // Same rule the stand uses (carpetScale in src/scene/stands.ts): a tile
      // exactly fills its grid step, which is what makes the mat seamless.
      carpetScale: carpet && gpu.tile ? CELL / (standEntry.carpetDiv ?? 1) / gpu.tile.tileM : 1,
      cap0,
      cap1,
      infoBuffer,
      argsBuffer,
      argsReset,
      cullBindGroup,
      nearBindGroup: carpet ? undefined : drawBindGroup(`${ctx.id}/near-bg-${entryIndex}`, 0, cap0 * 16),
      farBindGroup: carpet ? undefined : drawBindGroup(`${ctx.id}/far-bg-${entryIndex}`, cap0 * 16, cap1 * 16),
      carpetBindGroup: carpet
        ? device.createBindGroup({
            label: `${ctx.id}/carpet-bg-${entryIndex}`,
            layout: carpetBgl,
            entries: [
              { binding: 0, resource: { buffer: infoBuffer } },
              { binding: 1, resource: { buffer: instBuffer, offset: 0, size: cap0 * 16 } },
              { binding: 2, resource: gpu.tileAlbedo!.createView() },
              { binding: 3, resource: gpu.tileAux!.createView() },
              { binding: 4, resource: carpetSampler },
            ],
          })
        : undefined,
      slots,
      slotsPerFrame: 0,
    }
  })

  const anyCarpet = entries.some((e) => e.carpet)
  let cullPipeline!: GPUComputePipeline
  let tuftPipeline!: GPURenderPipeline
  let carpetPipeline: GPURenderPipeline | undefined
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
    carpetPipeline = anyCarpet ? mkRender('carpet', carpetSrc, carpetBgl) : undefined
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
        const ct = entry.gpu.tile
        info.set(planes, 0)
        b.nodes.forEach((n, i) => {
          info.set([n.cx, n.cy, n.cz, n.halfW], 24 + i * 8)
          info.set([n.y0, n.y1, n.azOffset, 0], 28 + i * 8)
        })
        info.set([x0, z0, sideX, sideZ], 64)
        info.set([ctx.seed, entryIndex, regionR, tuftR], 68)
        // A carpet uses its OWN alpha reference everywhere the grass one would
        // appear (cull rejection, vertex-stage skip, fragment test).
        info.set([entry.cap0, entry.cap1, entry.carpet ? CARPET_ALPHA_REF : p.alphaRef, p.groundShade], 72)
        info.set([p.dissolveGrazing, p.dissolveOverhead, b.plantH, b.rootHalfW], 76)
        info.set([Math.max(b.rootHalfW, b.plantH * 0.5) * 1.05, entry.slots, 0, 0], 80)
        if (ct) {
          // Carpet geometry, pre-multiplied by the mat's constant scale so the
          // shader never has to reconstruct it.
          const s = entry.carpetScale
          info.set([ct.tileM * s, CARPET_PARALLAX_MAX], 82)
          info.set([ct.planeFrac, (ct.y1 - ct.y0) * s, ct.y0 * s, p.carpetRelief], 84)
        }
        device.queue.writeBuffer(entry.infoBuffer, 0, info)
        device.queue.writeBuffer(entry.argsBuffer, 0, entry.argsReset)
        entry.slotsPerFrame = sideX * sideZ * entry.slots
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
        // Carpets first: a mat is a closed, depth-writing ground surface, so
        // drawing it before the splats lets early-z reject the parts of the
        // sparse grass that are below it.
        if (carpetPipeline) {
          pass.setPipeline(carpetPipeline)
          for (const entry of entries) {
            if (!entry.carpetBindGroup) continue
            pass.setBindGroup(1, entry.carpetBindGroup)
            pass.drawIndirect(entry.argsBuffer, 0)
          }
        }
        pass.setPipeline(tuftPipeline)
        for (const entry of entries) {
          if (!entry.nearBindGroup) continue
          pass.setBindGroup(1, entry.nearBindGroup)
          pass.drawIndirect(entry.argsBuffer, 0)
        }
        for (const entry of entries) {
          if (!entry.farBindGroup) continue
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

/**
 * Upload one carpet species: the seamless tile capture only. Its node atlas
 * (5 nodes x 8 azimuths of SIDE views, 14.8 MiB) is deliberately NOT uploaded —
 * a mat never draws a splat, and side views of a 0.09m cushion are the least
 * useful bytes in the budget. The CPU-side node bake is still kept, because the
 * shell's canopy composite reads its straight-down capture.
 */
function uploadCarpetTile(
  ctx: ExperimentContext<typeof PARAMS>,
  mip: MipGen,
  enc: GPUCommandEncoder,
  speciesId: string,
  bake: CanopyBake,
  tile: CarpetTileBake,
): SpeciesGpu {
  const mk = (label: string): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/${speciesId}/${label}`,
        size: [CT_PX, CT_PX],
        format: 'rgba8unorm',
        mipLevelCount: CT_MIPS,
        usage:
          GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
      },
      { species: speciesId, tag: `carpet-${label}` },
    )
  const tileAlbedo = mk('tile-albedo')
  const tileAux = mk('tile-aux')
  const layout = { bytesPerRow: CT_PX * 4, rowsPerImage: CT_PX }
  ctx.device.queue.writeTexture({ texture: tileAlbedo }, tile.albedo, layout, [CT_PX, CT_PX])
  ctx.device.queue.writeTexture({ texture: tileAux }, tile.aux, layout, [CT_PX, CT_PX])
  mip.run(enc, tileAlbedo, CT_MIPS, 'fs_albedo', 'rgba8unorm')
  mip.run(enc, tileAux, CT_MIPS, 'fs_aux_ao', 'rgba8unorm')
  console.info(
    `[${ctx.id}] carpet tile ${speciesId}: ${CT_PX}px / ${tile.tileM}m ` +
      `(${Math.round(CT_PX / tile.tileM)} texels/m), coverage ${(tile.meanCov * 100).toFixed(0)}%, ` +
      `surface at ${(tile.planeFrac * 100).toFixed(0)}% of ${(tile.y1 - tile.y0).toFixed(3)}m`,
  )
  return { bake, tile, tileAlbedo, tileAux }
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
