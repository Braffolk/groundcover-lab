import slicesSrc from './shaders/slices.wgsl'
import farshellSrc from './shaders/farshell.wgsl'
import { SCATTER_CELL_SIZE, commitBake, speciesById } from '@harness'
import type { Aabb2, Experiment, ExperimentContext, FrameInfo, StandSpecies, ViewTargets } from '@harness'
import {
  BAKE_VERSION,
  TOP_RES,
  bakeVolume,
  buildAuxTexture,
  buildMips2D,
  buildMips3D,
  buildTopView,
  isVolume,
  parseVolume,
  serializeVolume,
  type Volume,
} from './bake.ts'
import type { PARAMS } from './manifest.ts'

/**
 * Axis-locked slice-stack volume impostors + far canopy shell.
 *
 * Every plant of the active stand is one baked voxel volume (premultiplied
 * albedo/coverage + mean normal + baked sky visibility), drawn as a stack of
 * K planar slices. The stack axis snaps per plant to the local axis best
 * aligned with the view — horizontal shells from above, vertical curtains at
 * grazing — so the classic shell-texturing grazing collapse cannot happen.
 * K shrinks with distance (kNear -> kMid -> 1) with the slab opacity
 * renormalized and a volume mip integrating the fatter slabs. Beyond the
 * camera-centered plant region the whole meadow collapses into ONE
 * terrain-draped canopy layer built from baked top-down composites.
 *
 * Per-frame cost is bounded by the region around the camera and the screen —
 * never by the stand's total plant count.
 */

const MIPS_3D = 3
/**
 * Carpet species get a deeper 3D mip chain. A 0.18m Sphagnum tile is 2-5px
 * beyond ~20m, i.e. ~1/40 of its 138-voxel width, so mip 2 (4x, 6mm voxels) is
 * still a massive undersample and the distant mat sparkles. Two more levels cost
 * 0.2% of the volume's bytes. Kept off the grasses so the `default` stand keeps
 * its exact behaviour (mip_max is per species, in the species uniform).
 */
const MIPS_3D_CARPET = 5
const MIPS_2D = 6
const REBUILD_DIST = 6

/**
 * The constant tile scale of a carpet entry (grid step / periodic tile size) —
 * every tile of the species shares it, which is what keeps the mat a lattice.
 * The harness computes the same value into `stand_table[i].scale_min`, but does
 * not export its `carpetScale()` helper, and the TS scatter mirror reports the
 * stand's placeholder scaleMin instead. Null for ordinary scattered entries.
 */
function carpetTileScale(entry: StandSpecies): number | null {
  if (!entry.carpetDiv || entry.carpetDiv <= 0) return null
  const tileM = speciesById(entry.species).tileM
  return tileM ? SCATTER_CELL_SIZE / entry.carpetDiv / tileM : null
}

/**
 * Validating artifact flow: committed file -> OPFS cache -> bake in-browser.
 * We cannot use @harness bakedArtifact directly because the dev server's SPA
 * fallback answers missing /mesh/baked/ files with index.html (HTTP 200),
 * which would be cached as the artifact. Every path here magic-checks the
 * bytes; fresh bakes are committed back to mesh/baked/<id>/ when possible.
 */
async function loadValidatedArtifact(
  expId: string,
  key: string,
  bake: () => Promise<ArrayBuffer>,
): Promise<ArrayBuffer> {
  const committed = await fetch(`/mesh/baked/${expId}/${key}.bin`).catch(() => null)
  if (committed?.ok) {
    const buf = await committed.arrayBuffer()
    if (isVolume(buf)) return buf
  }
  const opfsName = `${expId}__${key}.bin`
  try {
    const root = await navigator.storage.getDirectory()
    const handle = await root.getFileHandle(opfsName)
    const buf = await (await handle.getFile()).arrayBuffer()
    if (isVolume(buf)) return buf
  } catch {
    // cache miss
  }
  const data = await bake()
  try {
    const root = await navigator.storage.getDirectory()
    const handle = await root.getFileHandle(opfsName, { create: true })
    const writable = await handle.createWritable()
    await writable.write(data)
    await writable.close()
  } catch (err) {
    console.warn(`[${expId}] OPFS cache write failed:`, err)
  }
  commitBake(expId, key, data).catch((err: unknown) => {
    console.warn(`[${expId}] commitBake failed (cache-only):`, err)
  })
  return data
}

interface SpeciesVol {
  id: string
  volume: Volume
  bindGroup: GPUBindGroup
  topView: GPUTextureView
}

interface EntryState {
  speciesId: string
  buffer: GPUBuffer
  bindGroup: GPUBindGroup
  scratch: Float32Array<ArrayBuffer>
  cap: number
  counts: [number, number, number]
  offsets: [number, number, number]
}

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx
  const stand = ctx.stand

  // -------------------------------------------------------------------------
  // Bind group layouts
  // -------------------------------------------------------------------------
  const speciesBGL = device.createBindGroupLayout({
    label: `${ctx.id}/species`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: '3d' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: '3d' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    ],
  })
  const instBGL = device.createBindGroupLayout({
    label: `${ctx.id}/instances`,
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }],
  })
  const bandBGL = device.createBindGroupLayout({
    label: `${ctx.id}/band`,
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
  })
  const shellBGL = device.createBindGroupLayout({
    label: `${ctx.id}/shell`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      { binding: 3, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    ],
  })

  // -------------------------------------------------------------------------
  // Bake (or load) one voxel volume per species used by the stand
  // -------------------------------------------------------------------------
  const uniqueSpecies = [...new Set(stand.species.map((e) => e.species))]
  const carpetSpecies = new Set(stand.species.filter((e) => (e.carpetDiv ?? 0) > 0).map((e) => e.species))
  const volumes = new Map<string, SpeciesVol>()

  for (const spId of uniqueSpecies) {
    const isCarpet = carpetSpecies.has(spId)
    const mips3d = isCarpet ? MIPS_3D_CARPET : MIPS_3D
    const meshId = speciesById(spId).meshId
    const key = `${spId}-v${BAKE_VERSION}`
    const data = await loadValidatedArtifact(ctx.id, key, async () => {
      console.log(`[${ctx.id}] baking ${spId} — fetching source mesh…`)
      const mesh = await ctx.meshes.load(meshId)
      const t0 = performance.now()
      const vol = bakeVolume(mesh, ctx.meshes.info(meshId).tileSize)
      console.log(
        `[${ctx.id}] baked ${spId}: ${vol.header.nx}x${vol.header.ny}x${vol.header.nz} in ${(performance.now() - t0).toFixed(0)}ms`,
      )
      return serializeVolume(vol)
    })
    const volume = parseVolume(data)
    const { nx, ny, nz } = volume.header

    const make3D = (label: string, base: Uint8Array): GPUTexture => {
      const tex = ctx.res.createTexture(
        {
          label: `${ctx.id}/${spId}/${label}`,
          dimension: '3d',
          size: [nx, ny, nz],
          format: 'rgba8unorm',
          mipLevelCount: mips3d,
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        },
        { species: spId, tag: 'voxel-volume' },
      )
      device.queue.writeTexture(
        { texture: tex, mipLevel: 0 },
        base as BufferSource,
        { bytesPerRow: nx * 4, rowsPerImage: ny },
        { width: nx, height: ny, depthOrArrayLayers: nz },
      )
      // Rounded mip averages only for carpets — see buildMips3D().
      buildMips3D(nx, ny, nz, base, mips3d, isCarpet).forEach((m, i) => {
        device.queue.writeTexture(
          { texture: tex, mipLevel: i + 1 },
          m.data as BufferSource,
          { bytesPerRow: m.nx * 4, rowsPerImage: m.ny },
          { width: m.nx, height: m.ny, depthOrArrayLayers: m.nz },
        )
      })
      return tex
    }
    const texColor = make3D('vox-color', volume.tex0)
    // Transcoded from the stored oct pair so linear/mip filtering of the
    // 99%-empty volume yields the occupancy-weighted mean normal — see
    // buildAuxTexture().
    const texAux = make3D('vox-aux', buildAuxTexture(volume))

    // Far-shell top view (2D, mipped, tileable over the species' period).
    const top = buildTopView(volume)
    const topTex = ctx.res.createTexture(
      {
        label: `${ctx.id}/${spId}/topview`,
        size: [TOP_RES, TOP_RES],
        format: 'rgba8unorm',
        mipLevelCount: MIPS_2D,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      },
      { species: spId, tag: 'topview' },
    )
    device.queue.writeTexture({ texture: topTex }, top as BufferSource, { bytesPerRow: TOP_RES * 4 }, [TOP_RES, TOP_RES])
    buildMips2D(TOP_RES, top, MIPS_2D).forEach((m, i) => {
      device.queue.writeTexture(
        { texture: topTex, mipLevel: i + 1 },
        m.data as BufferSource,
        { bytesPerRow: m.res * 4 },
        [m.res, m.res],
      )
    })

    const uniform = ctx.res.createBuffer(
      { label: `${ctx.id}/${spId}/uniform`, size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
      { species: spId },
    )
    const h = volume.header
    device.queue.writeBuffer(
      uniform,
      0,
      new Float32Array([h.lmin[0], h.lmin[1], h.lmin[2], h.voxel, h.lsize[0], h.lsize[1], h.lsize[2], mips3d - 1]),
    )

    volumes.set(spId, {
      id: spId,
      volume,
      topView: topTex.createView(),
      bindGroup: device.createBindGroup({
        label: `${ctx.id}/${spId}`,
        layout: speciesBGL,
        entries: [
          { binding: 0, resource: texColor.createView() },
          { binding: 1, resource: texAux.createView() },
          { binding: 2, resource: { buffer: uniform } },
        ],
      }),
    })
  }

  // Dummy 1x1 view for catalog species absent from this stand (shell slots).
  const dummyTex = ctx.res.createTexture(
    {
      label: `${ctx.id}/dummy`,
      size: [1, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    },
    { tag: 'dummy' },
  )
  device.queue.writeTexture({ texture: dummyTex }, new Uint8Array(4), { bytesPerRow: 4 }, [1, 1])
  const dummyView = dummyTex.createView()

  // -------------------------------------------------------------------------
  // Per-entry instance buffers (camera-region bounded, rebuilt on movement)
  // -------------------------------------------------------------------------
  const entries: EntryState[] = stand.species.map((e, entryIndex) => {
    // Size for the region actually in use (rRegion), not the param's max —
    // the rebuild path already grows the buffer if the region is widened.
    // The 1.15 slack also covers the 4 m cell rounding of scatter.region().
    const half = Math.min(ctx.params.rRegion, stand.radius)
    const area = 4 * half * half
    const cap = Math.ceil(area * Math.min(e.density, 8) * 1.15) + 256
    const buffer = ctx.res.createBuffer(
      {
        label: `${ctx.id}/instances/${entryIndex}-${e.species}`,
        size: cap * 8 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      },
      { species: e.species, tag: 'region-instances' },
    )
    return {
      speciesId: e.species,
      buffer,
      bindGroup: device.createBindGroup({
        label: `${ctx.id}/inst/${entryIndex}`,
        layout: instBGL,
        entries: [{ binding: 0, resource: { buffer } }],
      }),
      scratch: new Float32Array(cap * 8),
      cap,
      counts: [0, 0, 0] as [number, number, number],
      offsets: [0, 0, 0] as [number, number, number],
    }
  })

  // -------------------------------------------------------------------------
  // Uniforms: 3 LOD bands + far shell
  // -------------------------------------------------------------------------
  const bandBuffers: GPUBuffer[] = []
  const bandGroups: GPUBindGroup[] = []
  for (let b = 0; b < 3; b++) {
    const buf = ctx.res.createBuffer(
      { label: `${ctx.id}/band${b}`, size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
      { tag: 'uniforms' },
    )
    bandBuffers.push(buf)
    bandGroups.push(
      device.createBindGroup({
        label: `${ctx.id}/band${b}`,
        layout: bandBGL,
        entries: [{ binding: 0, resource: { buffer: buf } }],
      }),
    )
  }

  const shellBuffer = ctx.res.createBuffer(
    { label: `${ctx.id}/shell`, size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'uniforms' },
  )

  // ---------------------------------------------------------------------------
  // Far-shell slots: the THREE HEAVIEST-COVERING SPECIES OF THIS STAND.
  //
  // It used to be "the first three species of the catalog", which was the same
  // thing while the catalog had three. With the Sphagnum states appended at
  // indices 3-5 the bog stand's far field was built from calamagrostis + poa
  // (0.5-1.4 plants/m² of emergent grass) and drew the mat's replacement 0.6m
  // ABOVE the ground — a floating grass plateau over a moss bog.
  //
  // Coverage for a carpet entry comes from its GRID (carpet_div² tiles per 4m
  // cell, times the share of the wetness axis it claims), never from `density`,
  // which is 8 for every carpet entry regardless.
  // ---------------------------------------------------------------------------
  interface ShellSlot {
    speciesId: string
    weight: number
    /** World-space tiling period of the top view (m). */
    period: number
    /** Canopy height the shell should be draped at (m above terrain). */
    canopy: number
    catalogIndex: number
  }
  const slotBySpecies = new Map<string, ShellSlot>()
  for (const e of stand.species) {
    const sp = speciesById(e.species)
    const carpetScale = carpetTileScale(e)
    const scale = carpetScale ?? (e.scaleMin + e.scaleMax) / 2
    const band = e.wetWidth !== undefined && e.wetWidth > 0 ? Math.min(1, e.wetWidth) : 1
    const weight = carpetScale
      ? ((e.carpetDiv ?? 1) ** 2 / SCATTER_CELL_SIZE ** 2) * band
      : Math.min(e.density, 8) * band
    const existing = slotBySpecies.get(e.species)
    if (existing) {
      existing.weight += weight
      continue
    }
    const tile = volumes.get(e.species)?.volume.header.tile[0] ?? 0.5
    slotBySpecies.set(e.species, {
      speciesId: e.species,
      weight,
      period: Math.max(tile * scale, 0.05),
      canopy: sp.heightScale * scale * 0.55,
      catalogIndex: sp.index,
    })
  }
  const shellSlots = [...slotBySpecies.values()]
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 3)
    // Catalog order among the chosen three, so a three-species stand keeps the
    // exact slot assignment (and therefore the exact mottle-noise salts) it had.
    .sort((a, b) => a.catalogIndex - b.catalogIndex)

  const shellGroup = device.createBindGroup({
    label: `${ctx.id}/shell`,
    layout: shellBGL,
    entries: [
      { binding: 0, resource: volumes.get(shellSlots[0]?.speciesId ?? '')?.topView ?? dummyView },
      { binding: 1, resource: volumes.get(shellSlots[1]?.speciesId ?? '')?.topView ?? dummyView },
      { binding: 2, resource: volumes.get(shellSlots[2]?.speciesId ?? '')?.topView ?? dummyView },
      { binding: 3, resource: { buffer: shellBuffer } },
    ],
  })

  const shellWeights = [0, 0, 0]
  const shellInvTiles = [1, 1, 1]
  let avgH = 0.5
  {
    let hSum = 0
    let wSum = 0
    shellSlots.forEach((slot, i) => {
      shellWeights[i] = slot.weight
      shellInvTiles[i] = 1 / slot.period
      hSum += slot.weight * slot.canopy
      wSum += slot.weight
    })
    if (wSum > 0) avgH = hSum / wSum
  }

  // -------------------------------------------------------------------------
  // Pipelines
  // -------------------------------------------------------------------------
  let slicesPipeline!: GPURenderPipeline
  let shellPipeline!: GPURenderPipeline
  const build = (): void => {
    const slicesModule = ctx.shaders.module(slicesSrc)
    slicesPipeline = device.createRenderPipeline({
      label: `${ctx.id}/slices`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/slices`,
        bindGroupLayouts: [ctx.frame.layout, speciesBGL, instBGL, bandBGL],
      }),
      vertex: { module: slicesModule, entryPoint: 'vs_main' },
      fragment: { module: slicesModule, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
    const shellModule = ctx.shaders.module(farshellSrc)
    shellPipeline = device.createRenderPipeline({
      label: `${ctx.id}/farshell`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/farshell`,
        bindGroupLayouts: [ctx.frame.layout, shellBGL],
      }),
      vertex: { module: shellModule, entryPoint: 'vs_main' },
      fragment: { module: shellModule, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  // -------------------------------------------------------------------------
  // Region rebuild (CPU, only on camera movement — never per frame)
  // -------------------------------------------------------------------------
  let lastX = Number.NaN
  let lastZ = Number.NaN
  let rebuildDirty = true

  const rebuild = (camX: number, camY: number, camZ: number): void => {
    const half = ctx.params.rRegion
    const aabb: Aabb2 = {
      minX: Math.max(-stand.radius, camX - half),
      minZ: Math.max(-stand.radius, camZ - half),
      maxX: Math.min(stand.radius, camX + half),
      maxZ: Math.min(stand.radius, camZ + half),
    }
    const rNear2 = ctx.params.rNear * ctx.params.rNear
    const rMid2 = ctx.params.rMid * ctx.params.rMid

    for (let entryIndex = 0; entryIndex < entries.length; entryIndex++) {
      const st = entries[entryIndex]!
      const pts = aabb.minX < aabb.maxX && aabb.minZ < aabb.maxZ ? ctx.scene.scatter.region(entryIndex, aabb) : []
      // Counting sort into the scratch: count per band, then scatter straight
      // into the band's slot. No intermediate per-band point arrays.
      let nNear = 0
      let nMid = 0
      for (const pt of pts) {
        const dx = pt.x - camX
        const dy = pt.y - camY
        const dz = pt.z - camZ
        const d2 = dx * dx + dy * dy + dz * dz
        if (d2 < rNear2) nNear++
        else if (d2 < rMid2) nMid++
      }
      if (pts.length > st.cap) {
        st.cap = Math.ceil(pts.length * 1.3)
        st.buffer.destroy()
        st.buffer = ctx.res.createBuffer(
          {
            label: `${ctx.id}/instances/${entryIndex}-${st.speciesId}`,
            size: st.cap * 8 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          },
          { species: st.speciesId, tag: 'region-instances' },
        )
        st.bindGroup = device.createBindGroup({
          label: `${ctx.id}/inst/${entryIndex}`,
          layout: instBGL,
          entries: [{ binding: 0, resource: { buffer: st.buffer } }],
        })
        st.scratch = new Float32Array(st.cap * 8)
      }
      st.counts = [nNear, nMid, pts.length - nNear - nMid]
      st.offsets = [0, nNear, nNear + nMid]
      let oNear = 0
      let oMid = nNear * 8
      let oFar = (nNear + nMid) * 8
      const s = st.scratch
      for (const pt of pts) {
        const dx = pt.x - camX
        const dy = pt.y - camY
        const dz = pt.z - camZ
        const d2 = dx * dx + dy * dy + dz * dz
        let o: number
        if (d2 < rNear2) {
          o = oNear
          oNear += 8
        } else if (d2 < rMid2) {
          o = oMid
          oMid += 8
        } else {
          o = oFar
          oFar += 8
        }
        s[o] = pt.x
        s[o + 1] = pt.y
        s[o + 2] = pt.z
        s[o + 3] = pt.yaw
        s[o + 4] = pt.scale
        s[o + 5] = pt.entry
        s[o + 6] = pt.phase
        s[o + 7] = 0
      }
      if (pts.length > 0) device.queue.writeBuffer(st.buffer, 0, s, 0, pts.length * 8)
    }
    lastX = camX
    lastZ = camZ
    rebuildDirty = false
  }

  // -------------------------------------------------------------------------
  // Frame loop
  // -------------------------------------------------------------------------
  const bandData = new Float32Array(16)
  const shellData = new Float32Array(16)
  /** Slices per plant per band — kept in lockstep with the band uniforms. */
  const bandK = [0, 0, 1]
  // Near slices under-integrate on purpose: up close real grass is
  // see-through; full slab compensation there reads as solid blobs.
  const sigmaScale = [0.5, 0.85, 1]
  // The band/shell uniforms depend ONLY on params + fov + viewport height —
  // never on time or camera pose. Rewrite them when those change, not per frame.
  let uniformsDirty = true
  let lastMpp1 = -1

  return {
    update(frame: FrameInfo): void {
      const p = ctx.params
      const cam = frame.camera.pose
      const dx = cam.x - lastX
      const dz = cam.z - lastZ
      if (rebuildDirty || !Number.isFinite(lastX) || dx * dx + dz * dz > REBUILD_DIST * REBUILD_DIST) {
        rebuild(cam.x, cam.y, cam.z)
      }

      const { height } = ctx.size()
      const mpp1 = (2 * Math.tan((cam.fov * Math.PI) / 360)) / Math.max(height, 1)
      if (!uniformsDirty && mpp1 === lastMpp1) return
      uniformsDirty = false
      lastMpp1 = mpp1
      bandK[0] = p.kNear
      bandK[1] = p.kMid
      for (let b = 0; b < 3; b++) {
        bandData[0] = bandK[b]!
        bandData[1] = p.alphaThresh
        bandData[2] = p.sigma * sigmaScale[b]!
        bandData[3] = p.axisBiasY
        bandData[4] = b === 2 ? p.rRegion - 14 : 0
        bandData[5] = b === 2 ? p.rRegion - 6 : 0
        bandData[6] = 0.22
        bandData[7] = 0.5
        bandData[8] = p.lodBias
        bandData[9] = p.debugAxis ? 1 : 0
        bandData[10] = mpp1
        bandData[11] = p.carpetAlpha
        bandData[12] = p.carpetOcc
        device.queue.writeBuffer(bandBuffers[b]!, 0, bandData)
      }
      shellData[0] = shellWeights[0]!
      shellData[1] = shellWeights[1]!
      shellData[2] = shellWeights[2]!
      shellData[3] = stand.radius
      shellData[4] = shellInvTiles[0]!
      shellData[5] = shellInvTiles[1]!
      shellData[6] = shellInvTiles[2]!
      shellData[7] = avgH
      shellData[8] = p.rRegion - 20 // inner_start
      shellData[9] = p.rRegion - 6 // inner_full
      shellData[10] = Math.max(6, p.rRegion - 22) // r0
      shellData[11] = 900 // r1
      shellData[12] = (TOP_RES * 2 * Math.tan((cam.fov * Math.PI) / 360)) / Math.max(height, 1)
      shellData[13] = MIPS_2D - 1
      device.queue.writeBuffer(shellBuffer, 0, shellData)
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      // Slices first: their depth lets early-z kill most far-shell fragments.
      const pass = ctx.timing.renderPass(enc, 'slice-stacks', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setPipeline(slicesPipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      for (let b = 0; b < 3; b++) {
        pass.setBindGroup(3, bandGroups[b]!)
        const k = bandK[b]!
        for (const st of entries) {
          const count = st.counts[b as 0 | 1 | 2]
          if (count === 0) continue
          const vol = volumes.get(st.speciesId)
          if (!vol) continue
          pass.setBindGroup(1, vol.bindGroup)
          pass.setBindGroup(2, st.bindGroup)
          pass.draw(6, count * k, 0, st.offsets[b as 0 | 1 | 2] * k)
        }
      }
      pass.end()

      if (ctx.params.farShell) {
        const shellPass = ctx.timing.renderPass(enc, 'far-shell', {
          colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
          depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
        })
        shellPass.setPipeline(shellPipeline)
        shellPass.setBindGroup(0, ctx.frame.bindGroup)
        shellPass.setBindGroup(1, shellGroup)
        shellPass.draw(48 * 96 * 6)
        shellPass.end()
      }
    },

    onParamsChanged(keys: ReadonlySet<string>): void {
      if (keys.has('rNear') || keys.has('rMid') || keys.has('rRegion')) rebuildDirty = true
      uniformsDirty = true
    },

    dispose(): void {
      unsubscribe()
      // GPU resources are destroyed by the harness via ctx.res.
    },
  }
}
