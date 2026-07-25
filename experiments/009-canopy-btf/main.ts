import shaderSrc from './shaders/canopy.wgsl'
import { ATLAS_PX, LEVELS, loadOrBakeSpecies, type BtfArtifact } from './bake.ts'
import { SCATTER_CELL_SIZE, type Experiment, type ExperimentContext, type FrameInfo, type ViewTargets } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Canopy BTF field — the stand as a statistical texture.
 *
 * Per species, a baked hemi-oct grid of ground-parameterised community-tile
 * captures (see bake.ts). Per frame, ONE draw: a camera-centred
 * terrain-following grid rasterised as ground + canopy-top layers; each
 * fragment reconstructs the aggregate stand appearance at its view ray's
 * ground hit from precomputed lookups. Nothing scales with plant count — the
 * per-frame cost is bounded by screen pixels + a fixed proxy grid.
 */

const GRID_CELLS = 128
const SPACING = 2
/**
 * Entry slots in the uniform + the fragment's entry loop. The bog stand has
 * FIVE entries (three zoned Sphagnum states + calamagrostis + a trace of poa),
 * so a 4-entry limit silently dropped one species. 6 = the whole catalog.
 */
const MAX_ENTRIES = 6
/** Texture slots in the bind group — one per UNIQUE species of the stand. */
const MAX_SLOTS = 6
/** Order must match the `inspect` branch in shaders/canopy.wgsl. */
const INSPECT_MODES = ['off', 'bin', 'lod', 'layer'] as const
/** Densities the community tiles are treated as representing (plants/m²). */
const REF_DENSITY: Record<string, number> = {
  'calamagrostis-canescens': 3,
  'elymus-repens': 2.5,
  'poa-pratensis': 3,
}

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx
  const note = (s: string): void => console.log(`[${ctx.id}] ${s}`)

  const entries = ctx.stand.species.slice(0, MAX_ENTRIES)
  if (ctx.stand.species.length > MAX_ENTRIES) {
    console.warn(`[${ctx.id}] stand has ${ctx.stand.species.length} entries; rendering the first ${MAX_ENTRIES}`)
  }

  // --- load or bake one BTF artifact per unique species ---------------------
  const unique = [...new Set(entries.map((e) => e.species))]
  const artifacts = new Map<string, BtfArtifact>()
  const textures = new Map<string, GPUTexture>()
  for (const sid of unique) {
    const art = await loadOrBakeSpecies(ctx, sid, note)
    artifacts.set(sid, art)
    const tex = ctx.res.createTexture(
      {
        label: `${ctx.id}/${sid}/btf`,
        size: [ATLAS_PX, ATLAS_PX, 2],
        format: 'rgba8unorm',
        mipLevelCount: LEVELS,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      },
      { species: sid, tag: 'btf-atlas' },
    )
    for (let l = 0; l < LEVELS; l++) {
      const px = ATLAS_PX >> l
      device.queue.writeTexture(
        { texture: tex, mipLevel: l, origin: [0, 0, 0] },
        art.levelsA[l]!,
        { bytesPerRow: px * 4, rowsPerImage: px },
        [px, px, 1],
      )
      device.queue.writeTexture(
        { texture: tex, mipLevel: l, origin: [0, 0, 1] },
        art.levelsB[l]!,
        { bytesPerRow: px * 4, rowsPerImage: px },
        [px, px, 1],
      )
    }
    textures.set(sid, tex)
  }

  // Texture slot per UNIQUE SPECIES OF THIS STAND, not per catalog index: the
  // catalog now holds 6 species (moss is index 3-5) while only 3 slots existed,
  // so every Sphagnum entry fell through to slot 0 and sampled the
  // CALAMAGROSTIS atlas — the moss rendered as 18cm-periodic grass streaks.
  // Unused slots alias slot 0 (never sampled: the entry loop only touches
  // species the stand actually contains).
  const fallback = textures.get(unique[0]!)!
  const slotOf = new Map<string, number>(unique.slice(0, MAX_SLOTS).map((sid, i) => [sid, i]))
  const slotTex: GPUTexture[] = Array.from({ length: MAX_SLOTS }, (_, slot) => {
    for (const [sid, s] of slotOf) if (s === slot) return textures.get(sid)!
    return fallback
  })

  const sampler = device.createSampler({
    label: `${ctx.id}/samp`,
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    lodMaxClamp: LEVELS - 1,
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  })

  // 16 header + 6*12 entries + 32*4 bin dirs (entries at float 16, dirs at 88).
  const ENTRY_BASE = 16
  const DIRS_BASE = ENTRY_BASE + MAX_ENTRIES * 12
  const UNIFORM_FLOATS = DIRS_BASE + 32 * 4
  const uniformBuf = ctx.res.createBuffer(
    { label: `${ctx.id}/uniforms`, size: UNIFORM_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'params' },
  )

  const bgl = device.createBindGroupLayout({
    label: `${ctx.id}/bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ...slotTex.map((_, i) => ({
        binding: 1 + i,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: 'float' as GPUTextureSampleType, viewDimension: '2d-array' as GPUTextureViewDimension },
      })),
      { binding: 1 + MAX_SLOTS, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  })
  const bindGroup = device.createBindGroup({
    label: `${ctx.id}/bg`,
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      ...slotTex.map((tex, i) => ({ binding: 1 + i, resource: tex.createView({ dimension: '2d-array' as const }) })),
      { binding: 1 + MAX_SLOTS, resource: sampler },
    ],
  })

  let pipeline!: GPURenderPipeline
  const build = (): void => {
    const module = ctx.shaders.module(shaderSrc)
    pipeline = device.createRenderPipeline({
      label: `${ctx.id}/canopy`,
      layout: device.createPipelineLayout({ label: ctx.id, bindGroupLayouts: [ctx.frame.layout, bgl] }),
      vertex: { module, entryPoint: 'vs_main' },
      fragment: { module, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  // Everything but the camera snap and the params depends only on stand +
  // baked meta, so it is computed and uploaded ONCE here; the per-frame
  // writeBuffer below only touches the 12-float header.
  const uniforms = new Float32Array(UNIFORM_FLOATS)
  // Bin directions are identical across species by construction.
  uniforms.set(artifacts.get(unique[0]!)!.meta.dirs, DIRS_BASE)
  uniforms[2] = SPACING
  uniforms[3] = GRID_CELLS
  uniforms[4] = ctx.stand.radius
  uniforms[12] = ctx.seed // scatter seed — the wetness field is seed-dependent
  let shellTop = 0.4
  entries.forEach((e, i) => {
    const art = artifacts.get(e.species)!
    const div = e.carpetDiv ?? 0
    // CARPET: the entry's own scaleMin/scaleMax are placeholders the harness
    // overrides (stands.ts carpetScale(), mirrored into stand_table.scale_min
    // but NOT into ctx.stand) — 1.7-2.5 for the bog moss, where the real scale
    // is 1.01. The defining quantity is the grid step: div tiles per 4m cell,
    // each tile exactly filling its step. So the tile PERIOD is the step, and
    // the implied uniform scale is step/tileSize. Never height_scale, and never
    // the raw scale range.
    const step = div > 0 ? SCATTER_CELL_SIZE / div : 0
    const sMean = div > 0 ? step / art.meta.tileSize : (e.scaleMin + e.scaleMax) / 2
    const sMax = div > 0 ? sMean : e.scaleMax
    shellTop = Math.max(shellTop, art.meta.topH * sMax)
    const o = ENTRY_BASE + i * 12
    uniforms[o] = art.meta.tileSize * sMean
    uniforms[o + 1] = art.meta.topH * sMean
    // Carpet grid step (m), 0 for a scattered entry. The fragment shader uses
    // it to find the mat node under the pixel and ask which zoned state owns it.
    uniforms[o + 2] = step
    uniforms[o + 3] = e.sway
    // A carpet's coverage comes from its grid, not from `density` (which is 8,
    // i.e. meaningless for a mat) — so it is exactly 1.
    uniforms[o + 4] = div > 0 ? 1 : Math.min(1.3, e.density / (REF_DENSITY[e.species] ?? 3))
    uniforms[o + 5] = slotOf.get(e.species) ?? 0
    uniforms[o + 6] = art.meta.avgColor[0]
    uniforms[o + 7] = art.meta.avgColor[1]
    uniforms[o + 8] = art.meta.avgColor[2]
    // How much this species lies into the ground (stand's slope_align; 1 for a
    // carpet). Rotates the baked aggregate normal into the terrain frame.
    uniforms[o + 9] = e.slopeAlign ?? (div > 0 ? 1 : 0)
    // Habitat band. Carpet: the half-open wetness interval [lo, hi) that claims
    // a grid node. Scattered: (centre, width) of the soft acceptance falloff.
    // wet_b == 0 means "no band" for both.
    const wc = e.wetCenter ?? 0
    const ww = e.wetWidth ?? 0
    uniforms[o + 10] = div > 0 ? wc - ww / 2 : wc
    uniforms[o + 11] = div > 0 ? wc + ww / 2 : ww
  })
  uniforms[5] = shellTop + 0.05
  uniforms[6] = entries.length
  device.queue.writeBuffer(uniformBuf, 0, uniforms)

  const HEADER_FLOATS = 12

  return {
    update(frame: FrameInfo): void {
      const cam = frame.camera.pose
      uniforms[0] = Math.floor(cam.x / SPACING) * SPACING
      uniforms[1] = Math.floor(cam.z / SPACING) * SPACING
      uniforms[7] = ctx.params.taps === '4' ? 4 : ctx.params.taps === '1' ? 1 : 0
      uniforms[8] = ctx.params.windAmount
      uniforms[9] = ctx.params.coverage
      uniforms[10] = ctx.params.macroTint ? 1 : 0
      uniforms[11] = INSPECT_MODES.indexOf(ctx.params.inspect)
      device.queue.writeBuffer(uniformBuf, 0, uniforms, 0, HEADER_FLOATS)
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      const pass = ctx.timing.renderPass(enc, 'canopy-btf', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: {
          view: targets.depthView,
          depthLoadOp: 'load',
          depthStoreOp: 'store',
        },
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      pass.setBindGroup(1, bindGroup)
      pass.draw(GRID_CELLS * GRID_CELLS * 6, ctx.params.topShell ? 2 : 1)
      pass.end()
    },

    dispose(): void {
      unsubscribe()
      // GPU resources are reclaimed by the harness via ctx.res.
    },
  }
}
