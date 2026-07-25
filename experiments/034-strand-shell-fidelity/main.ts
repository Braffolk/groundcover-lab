import bladesSrc from './shaders/blades.wgsl'
import farshellSrc from './shaders/farshell.wgsl'
import { SCATTER_CELL_SIZE, SCATTER_MAX_PER_CELL, SPECIES, assetUrl, commitBake, speciesById } from '@harness'
import type { Experiment, ExperimentContext, FrameInfo, ViewTargets } from '@harness'
import {
  BAKE_VERSION,
  BLADES,
  STATIONS,
  bakeBladeField,
  isField,
  parseField,
  serializeField,
  toF16,
  type BladeField,
} from './bake.ts'
import type { PARAMS } from './manifest.ts'

/**
 * Blade library shell — 013-displacement-shell's structure with its fidelity
 * put back.
 *
 * Kept from 013 (the parts that made it fast and artifact-free): one
 * GPU-procedural cull pass over a camera-centred scatter region, distance
 * rings drawn by drawIndexedIndirect, continuous count LOD with a fractional
 * marginal row, hard opaque depth-writing geometry, and a single
 * terrain-conformal canopy shell past the region.
 *
 * Replaced (the part the owner called "simplifies the geometry WAY too much"):
 *  - the bake traces REAL blades tip-downwards with a blade-scale gather and a
 *    per-station PCA frame, instead of 96 centroid bundles with an RMS width;
 *  - a ribbon is built in the BLADE's plane, not facing the camera, so the
 *    silhouette changes with view direction and blades occlude each other;
 *  - near rings carry a geometric keel (3 verts across) and a cross-blade
 *    shading normal, so a blade reads as a channelled 3D surface;
 *  - widths are calibrated at bake time against the source mesh's own
 *    silhouette area, so a plant is as dense as the real thing;
 *  - occlusion is measured neighbour density, not a fake height curve.
 */

const RING_COUNT = 5
const RING_STATIONS = [16, 12, 9, 6, 4] as const
/**
 * Vertices across a ribbon: 3 = real keeled cross-section, 2 = flat strip.
 * Only ring 0 gets the keel — past ~2 m the bulge is well under a pixel.
 */
const RING_LAT = [3, 2, 2, 2, 2] as const
/**
 * Row-count multiplier per ring. Coverage is conserved by the width boost, so
 * shedding rows in the far rings trades many thin strands for fewer wide ones
 * at distances where a whole plant is a smudge. This is what buys ring 0 its
 * 320-blade near field inside the same frame budget.
 */
const ROW_GAIN = [1, 1, 0.85, 0.7, 0.5] as const
/**
 * Upper bound of each ring's outer radius over the whole param space
 * (r_k = rFull^(1-k/4) * rOuter^(k/4), rFull <= 3, rOuter <= 44). Used only to
 * size instance capacity, so a plant can never be dropped for lack of room.
 */
const RING_R_MAX = [3, 5.9, 11.5, 22.6, 44] as const
const SHELL_SEGMENTS = 72
const SHELL_ROWS = 20
const GLOBALS_BYTES = 256

function cellsAcross(radius: number): number {
  return 2 * Math.ceil((radius + 3) / SCATTER_CELL_SIZE) + 1
}

/**
 * Validating artifact flow: committed file -> OPFS cache -> bake in-browser.
 * The dev server's SPA fallback answers missing baked files with index.html at
 * HTTP 200, so every path magic-checks the bytes before trusting them.
 */
async function loadValidatedArtifact(
  expId: string,
  key: string,
  bake: () => Promise<ArrayBuffer>,
): Promise<ArrayBuffer> {
  const committed = await fetch(assetUrl(`/mesh/baked/${expId}/${key}.bin`)).catch(() => null)
  if (committed?.ok) {
    const buf = await committed.arrayBuffer()
    if (isField(buf)) return buf
  }
  const opfsName = `${expId}__${key}.bin`
  try {
    const root = await navigator.storage.getDirectory()
    const handle = await root.getFileHandle(opfsName)
    const buf = await (await handle.getFile()).arrayBuffer()
    if (isField(buf)) return buf
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

export async function create(ctx: ExperimentContext<typeof PARAMS>): Promise<Experiment> {
  const { device } = ctx
  const stand = ctx.stand
  const entryCount = stand.species.length

  // -------------------------------------------------------------------------
  // Bake (or load) one blade library per species used by the stand
  // -------------------------------------------------------------------------
  const uniqueSpecies = [...new Set(stand.species.map((e) => e.species))]
  const fields = new Map<string, BladeField>()
  for (const spId of uniqueSpecies) {
    const meshId = speciesById(spId).meshId
    const key = `${spId}-v${BAKE_VERSION}`
    const data = await loadValidatedArtifact(ctx.id, key, async () => {
      console.log(`[${ctx.id}] baking ${spId} — fetching source mesh…`)
      const mesh = await ctx.meshes.load(meshId)
      const t0 = performance.now()
      const baked = bakeBladeField(mesh, ctx.meshes.info(meshId).tileSize)
      console.log(
        `[${ctx.id}] baked ${spId} in ${(performance.now() - t0).toFixed(0)}ms ` +
          `(widthCalib ${baked.widthCalib.toFixed(2)}x, topH ${baked.topH.toFixed(2)}m)`,
      )
      return serializeField(baked)
    })
    const field = parseField(data)
    fields.set(spId, field)
    console.log(`[${ctx.id}] ${spId}: widthCalib ${field.widthCalib.toFixed(2)}x`)
  }

  // -------------------------------------------------------------------------
  // Blade library textures: one rgba16float layer per species-catalog index
  // -------------------------------------------------------------------------
  const layerCount = SPECIES.length
  const makeFieldTex = (name: string): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/${name}`,
        size: [STATIONS, BLADES, layerCount],
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      },
      { tag: 'blade-library' },
    )
  const posTex = makeFieldTex('lib-pos')
  const norTex = makeFieldTex('lib-nor')
  const colTex = makeFieldTex('lib-col')

  for (const spId of uniqueSpecies) {
    const field = fields.get(spId)!
    const layer = speciesById(spId).index
    const texel = STATIONS * BLADES
    const pos = new Uint16Array(texel * 4)
    const nor = new Uint16Array(texel * 4)
    const col = new Uint16Array(texel * 4)
    for (let i = 0; i < texel; i++) {
      const src = i * 12
      for (let c = 0; c < 4; c++) {
        pos[i * 4 + c] = toF16(field.data[src + c]!)
        nor[i * 4 + c] = toF16(field.data[src + 4 + c]!)
        col[i * 4 + c] = toF16(field.data[src + 8 + c]!)
      }
    }
    const layout = { bytesPerRow: STATIONS * 8, rowsPerImage: BLADES }
    const extent: [number, number, number] = [STATIONS, BLADES, 1]
    device.queue.writeTexture({ texture: posTex, origin: [0, 0, layer] }, pos, layout, extent)
    device.queue.writeTexture({ texture: norTex, origin: [0, 0, layer] }, nor, layout, extent)
    device.queue.writeTexture({ texture: colTex, origin: [0, 0, layer] }, col, layout, extent)
  }

  // -------------------------------------------------------------------------
  // Buffers: ONE instance buffer partitioned into per-ring sub-ranges
  // -------------------------------------------------------------------------
  const ringCaps = RING_R_MAX.map((r) => cellsAcross(r) ** 2 * SCATTER_MAX_PER_CELL * entryCount)
  const ringBases: number[] = []
  let totalInstances = 0
  for (const cap of ringCaps) {
    ringBases.push(totalInstances)
    totalInstances += cap
  }
  const instanceBuf = ctx.res.createBuffer(
    { label: `${ctx.id}/instances`, size: totalInstances * 16, usage: GPUBufferUsage.STORAGE },
    { tag: 'ring-instances' },
  )
  const indirectBuf = ctx.res.createBuffer(
    {
      label: `${ctx.id}/indirect`,
      size: RING_COUNT * 5 * 4,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    },
    { tag: 'indirect' },
  )
  const globalsBuf = ctx.res.createBuffer(
    { label: `${ctx.id}/globals`, size: GLOBALS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'globals' },
  )
  const ringInfoBufs = RING_STATIONS.map((st, i) => {
    const buf = ctx.res.createBuffer(
      { label: `${ctx.id}/ring${i}-info`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
      { tag: 'ring-info' },
    )
    const info = new ArrayBuffer(16)
    new Uint32Array(info, 0, 3).set([i, st, RING_LAT[i]!])
    new Float32Array(info, 12, 1)[0] = ROW_GAIN[i]!
    device.queue.writeBuffer(buf, 0, info)
    return buf
  })

  // Patch topology per ring: BLADES ribbons of `st` stations, `lat` verts wide.
  const indexBufs = RING_STATIONS.map((st, i) => {
    const lat = RING_LAT[i]!
    const idx = new Uint16Array(BLADES * (st - 1) * (lat - 1) * 6)
    let k = 0
    for (let b = 0; b < BLADES; b++) {
      const base = b * st * lat
      for (let s = 0; s + 1 < st; s++) {
        for (let l = 0; l + 1 < lat; l++) {
          const v00 = base + s * lat + l
          const v01 = v00 + 1
          const v10 = v00 + lat
          const v11 = v10 + 1
          idx[k++] = v00
          idx[k++] = v10
          idx[k++] = v01
          idx[k++] = v01
          idx[k++] = v10
          idx[k++] = v11
        }
      }
    }
    const buf = ctx.res.createBuffer(
      {
        label: `${ctx.id}/ring${i}-indices`,
        size: idx.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      },
      { tag: 'topology' },
    )
    device.queue.writeBuffer(buf, 0, idx)
    return buf
  })

  // Far-shell annulus: (angle, rowIndex) verts; radius = shell_in * g^row in VS.
  const shellVerts = new Float32Array(SHELL_SEGMENTS * SHELL_ROWS * 2)
  for (let r = 0; r < SHELL_ROWS; r++) {
    for (let a = 0; a < SHELL_SEGMENTS; a++) {
      const i = (r * SHELL_SEGMENTS + a) * 2
      shellVerts[i] = (a / SHELL_SEGMENTS) * Math.PI * 2
      shellVerts[i + 1] = r
    }
  }
  const shellIdx = new Uint16Array((SHELL_ROWS - 1) * SHELL_SEGMENTS * 6)
  let sk = 0
  for (let r = 0; r + 1 < SHELL_ROWS; r++) {
    for (let a = 0; a < SHELL_SEGMENTS; a++) {
      const a1 = (a + 1) % SHELL_SEGMENTS
      const v00 = r * SHELL_SEGMENTS + a
      const v01 = r * SHELL_SEGMENTS + a1
      const v10 = (r + 1) * SHELL_SEGMENTS + a
      const v11 = (r + 1) * SHELL_SEGMENTS + a1
      shellIdx[sk++] = v00
      shellIdx[sk++] = v10
      shellIdx[sk++] = v01
      shellIdx[sk++] = v01
      shellIdx[sk++] = v10
      shellIdx[sk++] = v11
    }
  }
  const shellVB = ctx.res.createBuffer(
    {
      label: `${ctx.id}/shell-verts`,
      size: shellVerts.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    },
    { tag: 'shell-mesh' },
  )
  const shellIB = ctx.res.createBuffer(
    {
      label: `${ctx.id}/shell-indices`,
      size: shellIdx.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    },
    { tag: 'shell-mesh' },
  )
  device.queue.writeBuffer(shellVB, 0, shellVerts)
  device.queue.writeBuffer(shellIB, 0, shellIdx)

  // -------------------------------------------------------------------------
  // Bind groups + pipelines
  // -------------------------------------------------------------------------
  const cullBGL = device.createBindGroupLayout({
    label: `${ctx.id}/cull`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  })
  const drawBGL = device.createBindGroupLayout({
    label: `${ctx.id}/blades`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 5, visibility: GPUShaderStage.VERTEX, texture: { viewDimension: '2d-array' } },
      { binding: 6, visibility: GPUShaderStage.VERTEX, texture: { viewDimension: '2d-array' } },
      { binding: 7, visibility: GPUShaderStage.VERTEX, texture: { viewDimension: '2d-array' } },
    ],
  })
  const shellBGL = device.createBindGroupLayout({
    label: `${ctx.id}/shell`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    ],
  })

  const cullBG = device.createBindGroup({
    label: `${ctx.id}/cull`,
    layout: cullBGL,
    entries: [
      { binding: 0, resource: { buffer: globalsBuf } },
      { binding: 1, resource: { buffer: instanceBuf } },
      { binding: 2, resource: { buffer: indirectBuf } },
    ],
  })
  const posView = posTex.createView({ dimension: '2d-array' })
  const norView = norTex.createView({ dimension: '2d-array' })
  const colView = colTex.createView({ dimension: '2d-array' })
  const drawBGs = ringInfoBufs.map((info, i) =>
    device.createBindGroup({
      label: `${ctx.id}/blades-ring${i}`,
      layout: drawBGL,
      entries: [
        { binding: 0, resource: { buffer: globalsBuf } },
        { binding: 3, resource: { buffer: instanceBuf } },
        { binding: 4, resource: { buffer: info } },
        { binding: 5, resource: posView },
        { binding: 6, resource: norView },
        { binding: 7, resource: colView },
      ],
    }),
  )
  const shellBG = device.createBindGroup({
    label: `${ctx.id}/shell`,
    layout: shellBGL,
    entries: [{ binding: 0, resource: { buffer: globalsBuf } }],
  })

  let cullPipeline!: GPUComputePipeline
  let bladePipeline!: GPURenderPipeline
  let shellPipeline!: GPURenderPipeline
  const build = (): void => {
    const bladeModule = ctx.shaders.module(bladesSrc)
    const shellModule = ctx.shaders.module(farshellSrc)
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/cull`, bindGroupLayouts: [ctx.frame.layout, cullBGL] }),
      compute: { module: bladeModule, entryPoint: 'cs_cull' },
    })
    bladePipeline = device.createRenderPipeline({
      label: `${ctx.id}/blades`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/blades`,
        bindGroupLayouts: [ctx.frame.layout, drawBGL],
      }),
      vertex: { module: bladeModule, entryPoint: 'vs_main' },
      fragment: { module: bladeModule, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
    shellPipeline = device.createRenderPipeline({
      label: `${ctx.id}/shell`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/shell`, bindGroupLayouts: [ctx.frame.layout, shellBGL] }),
      vertex: {
        module: shellModule,
        entryPoint: 'vs_main',
        buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }] }],
      },
      fragment: {
        module: shellModule,
        entryPoint: 'fs_main',
        targets: [
          {
            format: ctx.colorFormat,
            blend: {
              color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: ctx.depthFormat, depthCompare: 'less', depthWriteEnabled: true },
    })
  }
  build()
  const unsubscribe = ctx.shaders.onReload(build)

  // -------------------------------------------------------------------------
  // Per-frame state
  // -------------------------------------------------------------------------
  const globalsData = new ArrayBuffer(GLOBALS_BYTES)
  const gI32 = new Int32Array(globalsData)
  const gU32 = new Uint32Array(globalsData)
  const gF32 = new Float32Array(globalsData)
  const indirectData = new Uint32Array(RING_COUNT * 5)
  let regionArea = 0

  // Stand-mixed shell constants (canopy height, bob, per-species density).
  const density = [0, 0, 0, 0]
  let shellHNom = 0
  let shellBob = 0
  {
    let dSum = 0
    for (const e of stand.species) {
      const sp = speciesById(e.species)
      const field = fields.get(e.species)!
      const meanScale = (e.scaleMin + e.scaleMax) / 2
      density[sp.index]! += e.density
      shellHNom += e.density * field.topH * meanScale
      shellBob += e.density * e.sway
      dSum += e.density
    }
    shellHNom = dSum > 0 ? shellHNom / dSum : 0.6
    shellBob = dSum > 0 ? shellBob / dSum : 0.5
  }

  const standCellMin = Math.floor(-stand.radius / SCATTER_CELL_SIZE)
  const standCellMax = Math.floor(stand.radius / SCATTER_CELL_SIZE)

  const writeFrameData = (frame: FrameInfo): void => {
    const prm = ctx.params
    const half = Math.ceil((prm.rOuter + 3) / SCATTER_CELL_SIZE)
    const camCX = Math.floor(frame.camera.pose.x / SCATTER_CELL_SIZE)
    const camCZ = Math.floor(frame.camera.pose.z / SCATTER_CELL_SIZE)
    const minX = Math.max(camCX - half, standCellMin)
    const minZ = Math.max(camCZ - half, standCellMin)
    const maxX = Math.min(camCX + half, standCellMax)
    const maxZ = Math.min(camCZ + half, standCellMax)
    const dimX = Math.max(maxX - minX + 1, 0)
    const dimZ = Math.max(maxZ - minZ + 1, 0)
    regionArea = dimX * dimZ

    // Ring radii are a geometric progression from rFull to rOuter, so the
    // blade-count law rows = c_rows/d steps by a constant factor per ring.
    const rFull = Math.min(prm.rFull, prm.rOuter * 0.5)
    const ratio = Math.pow(prm.rOuter / rFull, 1 / (RING_COUNT - 1))
    const cRows = prm.blades * rFull

    gI32[0] = minX
    gI32[1] = minZ
    gI32[2] = dimX
    gI32[3] = dimZ
    gU32[4] = ctx.seed >>> 0
    gF32[5] = prm.blades
    gF32[6] = BLADES
    gF32[7] = STATIONS
    gF32[8] = cRows
    gF32[9] = prm.rOuter
    gF32[10] = prm.widthScale
    gF32[11] = prm.coverPow
    gF32[12] = prm.keel
    gF32[13] = prm.curl
    gF32[14] = Math.max(prm.orientFar * 0.35, 1)
    gF32[15] = prm.orientFar
    gF32[16] = prm.minPx
    gF32[17] = prm.aoMin
    gF32[18] = prm.upBias
    gF32[19] = prm.rOuter - 8
    gF32[20] = shellHNom * prm.shellHeight
    gF32[21] = stand.radius
    gF32[22] = shellBob
    gF32[23] = prm.debugRings ? 1 : 0
    gF32[24] = prm.bendAmp
    // gF32[25..27] padding

    for (let k = 0; k < RING_COUNT; k++) {
      const rOut = k === RING_COUNT - 1 ? prm.rOuter : rFull * Math.pow(ratio, k)
      const rows = Math.max(1, Math.min(prm.blades, Math.ceil((cRows * ROW_GAIN[k]!) / rOut)))
      const o = 28 + k * 4
      gF32[o] = rOut
      gF32[o + 1] = ringBases[k]!
      gF32[o + 2] = ringCaps[k]!
      gF32[o + 3] = rows
      indirectData[k * 5] = rows * (RING_STATIONS[k]! - 1) * (RING_LAT[k]! - 1) * 6
      indirectData[k * 5 + 1] = 0
      indirectData[k * 5 + 2] = 0
      indirectData[k * 5 + 3] = 0
      indirectData[k * 5 + 4] = 0
    }
    for (const sp of SPECIES) {
      const field = fields.get(sp.id)
      const o = 48 + sp.index * 4
      if (field) {
        gF32[o] = field.canopy[0]
        gF32[o + 1] = field.canopy[1]
        gF32[o + 2] = field.canopy[2]
        gF32[o + 3] = field.topH
      }
    }
    gF32[60] = density[0]!
    gF32[61] = density[1]!
    gF32[62] = density[2]!
    gF32[63] = density[3]!
    device.queue.writeBuffer(globalsBuf, 0, globalsData)
    device.queue.writeBuffer(indirectBuf, 0, indirectData)
  }

  return {
    update(frame: FrameInfo): void {
      writeFrameData(frame)
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      if (regionArea > 0) {
        const cull = ctx.timing.computePass(enc, 'cull')
        cull.setPipeline(cullPipeline)
        cull.setBindGroup(0, ctx.frame.bindGroup)
        cull.setBindGroup(1, cullBG)
        cull.dispatchWorkgroups(Math.ceil(SCATTER_MAX_PER_CELL / 64), regionArea, entryCount)
        cull.end()
      }

      // ONE render pass for blades + shell: identical attachments, no
      // dependency. Splitting them would cost a colour+depth store and reload
      // (a tile flush on a TBDR GPU) for nothing.
      const pass = ctx.timing.renderPass(enc, 'blades+shell', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setPipeline(bladePipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      // Front to back: near rings first, so every later ring is depth-rejected
      // wherever a nearer blade already wrote depth.
      for (let i = 0; i < RING_COUNT; i++) {
        pass.setBindGroup(1, drawBGs[i]!)
        pass.setIndexBuffer(indexBufs[i]!, 'uint16')
        pass.drawIndexedIndirect(indirectBuf, i * 20)
      }
      pass.setPipeline(shellPipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      pass.setBindGroup(1, shellBG)
      pass.setVertexBuffer(0, shellVB)
      pass.setIndexBuffer(shellIB, 'uint16')
      pass.drawIndexed(shellIdx.length)
      pass.end()
    },

    dispose(): void {
      unsubscribe()
      // Buffers/textures are destroyed by the harness via ctx.res.
    },
  }
}
