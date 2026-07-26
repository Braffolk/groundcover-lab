import strandsSrc from './shaders/strands.wgsl'
import farshellSrc from './shaders/farshell.wgsl'
import { SCATTER_CELL_SIZE, SCATTER_MAX_PER_CELL, SPECIES, commitBake, speciesById } from '@harness'
import type { Experiment, ExperimentContext, FrameInfo, StandSpecies, ViewTargets } from '@harness'
import {
  BAKE_VERSION,
  GRID_BAKE_VERSION,
  GRID_EXT_FACTOR,
  GRID_LINES,
  GRID_N,
  STATIONS,
  STRANDS,
  bakeCarpetGrid,
  bakeStrandField,
  isField,
  isGrid,
  parseField,
  parseGrid,
  serializeField,
  serializeGrid,
  toF16,
  type GridField,
  type StrandField,
} from './bake.ts'
import type { PARAMS } from './manifest.ts'

/**
 * Strand displacement shell.
 *
 * The only geometry that ever exists is one flat sheet: identical little
 * patches, one per plant of the stand, whose vertices are thrown into place by
 * baked vector-displacement field textures. Placement is fully GPU-procedural —
 * a per-frame compute pass walks the scatter cells of a camera-centered region
 * (the bit-identical WGSL twin of the harness scatter), frustum-culls, and
 * appends survivors to distance buckets with decreasing patch topology.
 *
 * Two field flavours, because a plant and a mat are not the same shape:
 *  - SCATTERED species (grasses): 96 strand ribbons x 16 stations, camera-facing,
 *    LOD by continuous strand count. Three distance rings.
 *  - CARPET species (Sphagnum, stand carpetDiv > 0): a 65x65 top-shell
 *    displacement GRID per tile, drawn as a watertight terrain-conforming sheet,
 *    LOD by integer texel stride. Six stride buckets.
 *
 * Beyond the region the meadow IS a single terrain-conformal canopy shell
 * (camera-centered annulus at baked canopy color/height). Per-frame cost is
 * bounded by region + screen, never by plant count.
 */

const RING_T = [12, 8, 5] as const
const RING_R_MAX = [12, 24, 56] as const // must match manifest param maxima
/** Carpet LOD buckets: texel stride 1,2,4,...,32 over the baked grid. */
const CARPET_LEVELS = 6
const CARPET_OUTER_MAX = 20 // must match the carpetOuter param maximum
const DRAWS = 3 + CARPET_LEVELS
const SHELL_SEGMENTS = 72
/**
 * Far-shell annulus rows. A mat stand hands over to the shell within metres, so
 * the shell has to resolve the terrain it is draped on much better than it needs
 * to at 32 m on a grass stand — with too few rows it linearly interpolates over
 * a whole hill and the terrain pokes through it.
 */
const SHELL_ROWS_SCATTER = 20
const SHELL_ROWS_CARPET = 40
const GLOBALS_BYTES = 336

function cellsAcross(radius: number): number {
  return 2 * Math.ceil((radius + 3) / SCATTER_CELL_SIZE) + 1
}

const isCarpet = (e: StandSpecies): boolean => (e.carpetDiv ?? 0) > 0

/** Indices for `rows` independent 2-wide strips of `cols` columns each. */
function stripIndices(rows: number, cols: number): Uint16Array<ArrayBuffer> {
  const idx = new Uint16Array(rows * (cols - 1) * 6)
  let k = 0
  for (let r = 0; r < rows; r++) {
    const base = r * cols * 2
    for (let c = 0; c < cols - 1; c++) {
      const v00 = base + c * 2
      idx[k++] = v00
      idx[k++] = v00 + 2
      idx[k++] = v00 + 1
      idx[k++] = v00 + 1
      idx[k++] = v00 + 2
      idx[k++] = v00 + 3
    }
  }
  return idx
}

/**
 * Validating artifact flow: committed file -> OPFS cache -> bake in-browser.
 * (The dev server SPA-fallback answers missing baked files with index.html,
 * so every path magic-checks the bytes; same pattern as 003.)
 */
async function loadValidatedArtifact(
  expId: string,
  key: string,
  valid: (buf: ArrayBuffer | null) => boolean,
  bake: () => Promise<ArrayBuffer>,
): Promise<ArrayBuffer> {
  const committed = await fetch(`/mesh/baked/${expId}/${key}.bin`).catch(() => null)
  if (committed?.ok) {
    const buf = await committed.arrayBuffer()
    if (valid(buf)) return buf
  }
  const opfsName = `${expId}__${key}.bin`
  try {
    const root = await navigator.storage.getDirectory()
    const handle = await root.getFileHandle(opfsName)
    const buf = await (await handle.getFile()).arrayBuffer()
    if (valid(buf)) return buf
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
  const carpetEntries = stand.species.filter(isCarpet)
  const scatterEntries = stand.species.filter((e) => !isCarpet(e))
  const hasCarpet = carpetEntries.length > 0

  // -------------------------------------------------------------------------
  // Bake (or load) one field per species — strand field for scattered species,
  // top-shell displacement grid for carpet species.
  // -------------------------------------------------------------------------
  const strandIds = [...new Set(scatterEntries.map((e) => e.species))]
  const gridIds = [...new Set(carpetEntries.map((e) => e.species))]
  const fields = new Map<string, StrandField>()
  const grids = new Map<string, GridField>()
  for (const spId of strandIds) {
    const meshId = speciesById(spId).meshId
    const key = `${spId}-v${BAKE_VERSION}`
    const data = await loadValidatedArtifact(ctx.id, key, isField, async () => {
      console.log(`[${ctx.id}] baking strands ${spId} — fetching source mesh…`)
      const mesh = await ctx.meshes.load(meshId)
      const t0 = performance.now()
      const field = bakeStrandField(mesh, ctx.meshes.info(meshId).tileSize)
      console.log(`[${ctx.id}] baked ${spId} in ${(performance.now() - t0).toFixed(0)}ms`)
      return serializeField(field)
    })
    fields.set(spId, parseField(data))
  }
  for (const spId of gridIds) {
    const meshId = speciesById(spId).meshId
    const key = `${spId}-grid-v${GRID_BAKE_VERSION}`
    const data = await loadValidatedArtifact(ctx.id, key, isGrid, async () => {
      console.log(`[${ctx.id}] baking carpet grid ${spId} — fetching source mesh…`)
      const mesh = await ctx.meshes.load(meshId)
      const tile = ctx.meshes.info(meshId).tileSize
      if (!tile) throw new Error(`${spId} is a carpet species but its mesh has no periodic tile`)
      const t0 = performance.now()
      const grid = bakeCarpetGrid(mesh, tile)
      console.log(`[${ctx.id}] baked ${spId} grid in ${(performance.now() - t0).toFixed(0)}ms`)
      return serializeGrid(grid)
    })
    grids.set(spId, parseGrid(data))
  }

  // -------------------------------------------------------------------------
  // Field textures: one rgba16float layer per species-catalog index. Two sets,
  // because the two field flavours have different shapes.
  // -------------------------------------------------------------------------
  const layerCount = SPECIES.length
  const makeFieldTex = (name: string, w: number, h: number, tag: string): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/${name}`,
        size: [w, h, layerCount],
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      },
      { tag },
    )
  const posTex = makeFieldTex('field-pos', STATIONS, STRANDS, 'strand-field')
  const norTex = makeFieldTex('field-nor', STATIONS, STRANDS, 'strand-field')
  const colTex = makeFieldTex('field-col', STATIONS, STRANDS, 'strand-field')
  const gridW = hasCarpet ? GRID_LINES : 1
  const gPosTex = makeFieldTex('grid-pos', gridW, gridW, 'carpet-grid')
  const gNorTex = makeFieldTex('grid-nor', gridW, gridW, 'carpet-grid')
  const gColTex = makeFieldTex('grid-col', gridW, gridW, 'carpet-grid')

  /** Split interleaved [pos|nor|col] float triples into three f16 layers. */
  const uploadLayer = (
    data: Float32Array,
    texels: number,
    w: number,
    h: number,
    layer: number,
    dst: [GPUTexture, GPUTexture, GPUTexture],
  ): void => {
    const bufs = [new Uint16Array(texels * 4), new Uint16Array(texels * 4), new Uint16Array(texels * 4)]
    for (let i = 0; i < texels; i++) {
      const src = i * 12
      for (let c = 0; c < 4; c++) {
        bufs[0]![i * 4 + c] = toF16(data[src + c]!)
        bufs[1]![i * 4 + c] = toF16(data[src + 4 + c]!)
        bufs[2]![i * 4 + c] = toF16(data[src + 8 + c]!)
      }
    }
    const layout = { bytesPerRow: w * 8, rowsPerImage: h }
    const extent: [number, number, number] = [w, h, 1]
    for (let k = 0; k < 3; k++) {
      device.queue.writeTexture({ texture: dst[k]!, origin: [0, 0, layer] }, bufs[k]!, layout, extent)
    }
  }
  for (const spId of strandIds) {
    const field = fields.get(spId)!
    uploadLayer(field.data, STATIONS * STRANDS, STATIONS, STRANDS, speciesById(spId).index, [posTex, norTex, colTex])
  }
  for (const spId of gridIds) {
    const grid = grids.get(spId)!
    uploadLayer(grid.data, GRID_LINES * GRID_LINES, GRID_LINES, GRID_LINES, speciesById(spId).index, [
      gPosTex,
      gNorTex,
      gColTex,
    ])
  }

  // -------------------------------------------------------------------------
  // Buffers
  // -------------------------------------------------------------------------
  // Ring capacity is the number of instances expected to SURVIVE, which for
  // scattered species is density/8 of the slots — NOT the slot count itself.
  const scatterPerCell = scatterEntries.reduce(
    (a, e) => a + SCATTER_MAX_PER_CELL * (Math.min(e.density, 8) / 8),
    0,
  )
  const ringCaps = RING_R_MAX.map((r) =>
    scatterPerCell > 0 ? Math.ceil(cellsAcross(r) ** 2 * scatterPerCell * 1.15) + 2048 : 16,
  )
  const instanceBufs = ringCaps.map((cap, i) =>
    ctx.res.createBuffer(
      { label: `${ctx.id}/ring${i}-instances`, size: cap * 16, usage: GPUBufferUsage.STORAGE },
      { tag: 'ring-instances' },
    ),
  )

  // Carpet: a mat's entries PARTITION the wetness axis, so across all carpet
  // entries exactly carpetDiv^2 nodes per cell survive. Buckets share one buffer
  // (six storage bindings would blow the 8-per-stage limit) at 256B-aligned
  // slices; a full bucket promotes to the next coarser one instead of dropping
  // tiles, so capacity can never punch holes in the mat.
  const carpetDiv = Math.max(0, ...carpetEntries.map((e) => e.carpetDiv ?? 0))
  const carpetPerCell = hasCarpet ? carpetDiv ** 2 * 1.06 + 16 : 0
  const carpetCaps = Array.from({ length: CARPET_LEVELS }, (_, l) => {
    if (!hasCarpet) return 16
    const radius = Math.min(CARPET_OUTER_MAX, 0.9 * 2 ** l)
    return Math.ceil((cellsAcross(radius) ** 2 * carpetPerCell) / 16) * 16
  })
  const carpetBases: number[] = []
  let carpetTotal = 0
  for (const cap of carpetCaps) {
    carpetBases.push(carpetTotal)
    carpetTotal += cap
  }
  const carpetBuf = ctx.res.createBuffer(
    { label: `${ctx.id}/carpet-instances`, size: carpetTotal * 16, usage: GPUBufferUsage.STORAGE },
    { tag: 'carpet-instances' },
  )

  const indirectBuf = ctx.res.createBuffer(
    {
      label: `${ctx.id}/indirect`,
      size: DRAWS * 5 * 4,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    },
    { tag: 'indirect' },
  )
  const globalsBuf = ctx.res.createBuffer(
    { label: `${ctx.id}/globals`, size: GLOBALS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'globals' },
  )
  // Per-draw constants: strand rings carry their station count, carpet buckets
  // their texel stride and band count.
  const drawInfoBufs = Array.from({ length: DRAWS }, (_, i) => {
    const buf = ctx.res.createBuffer(
      { label: `${ctx.id}/draw${i}-info`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
      { tag: 'draw-info' },
    )
    const stride = i < 3 ? 0 : 1 << (i - 3)
    const bands = i < 3 ? 0 : GRID_N >> (i - 3)
    device.queue.writeBuffer(buf, 0, new Uint32Array([i < 3 ? RING_T[i]! : 0, i, stride, bands]))
    return buf
  })

  // Topology: strand rings are STRANDS strips of t stations; carpet buckets are
  // (bands) strips of (bands+1) columns — a full lattice over the baked grid.
  const indexBufs = Array.from({ length: DRAWS }, (_, i) => {
    const idx =
      i < 3 ? stripIndices(STRANDS, RING_T[i]!) : stripIndices(GRID_N >> (i - 3), (GRID_N >> (i - 3)) + 1)
    const buf = ctx.res.createBuffer(
      {
        label: `${ctx.id}/draw${i}-indices`,
        size: Math.max(idx.byteLength, 4),
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      },
      { tag: 'topology' },
    )
    device.queue.writeBuffer(buf, 0, idx)
    return buf
  })

  // Far-shell annulus: (angle, rowIndex) verts; radius = shell_r0 * g^row in VS.
  const shellRows = hasCarpet ? SHELL_ROWS_CARPET : SHELL_ROWS_SCATTER
  const shellVerts = new Float32Array(SHELL_SEGMENTS * shellRows * 2)
  for (let r = 0; r < shellRows; r++) {
    for (let a = 0; a < SHELL_SEGMENTS; a++) {
      const i = (r * SHELL_SEGMENTS + a) * 2
      shellVerts[i] = (a / SHELL_SEGMENTS) * Math.PI * 2
      shellVerts[i + 1] = r
    }
  }
  const shellIdx = new Uint16Array((shellRows - 1) * SHELL_SEGMENTS * 6)
  let sk = 0
  for (let r = 0; r < shellRows - 1; r++) {
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
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  })
  const drawBGL = device.createBindGroupLayout({
    label: `${ctx.id}/draw`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 5, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 6, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 7, visibility: GPUShaderStage.VERTEX, texture: { viewDimension: '2d-array' } },
      { binding: 8, visibility: GPUShaderStage.VERTEX, texture: { viewDimension: '2d-array' } },
      { binding: 9, visibility: GPUShaderStage.VERTEX, texture: { viewDimension: '2d-array' } },
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
      { binding: 1, resource: { buffer: instanceBufs[0]! } },
      { binding: 2, resource: { buffer: instanceBufs[1]! } },
      { binding: 3, resource: { buffer: instanceBufs[2]! } },
      { binding: 4, resource: { buffer: indirectBuf } },
      { binding: 10, resource: { buffer: carpetBuf } },
    ],
  })
  const strandViews = [posTex, norTex, colTex].map((t) => t.createView({ dimension: '2d-array' }))
  const gridViews = [gPosTex, gNorTex, gColTex].map((t) => t.createView({ dimension: '2d-array' }))
  const drawBGs = Array.from({ length: DRAWS }, (_, i) => {
    const views = i < 3 ? strandViews : gridViews
    const instances =
      i < 3
        ? { buffer: instanceBufs[i]! }
        : { buffer: carpetBuf, offset: carpetBases[i - 3]! * 16, size: carpetCaps[i - 3]! * 16 }
    return device.createBindGroup({
      label: `${ctx.id}/draw${i}`,
      layout: drawBGL,
      entries: [
        { binding: 0, resource: { buffer: globalsBuf } },
        { binding: 5, resource: instances },
        { binding: 6, resource: { buffer: drawInfoBufs[i]! } },
        { binding: 7, resource: views[0]! },
        { binding: 8, resource: views[1]! },
        { binding: 9, resource: views[2]! },
      ],
    })
  })
  const shellBG = device.createBindGroup({
    label: `${ctx.id}/shell`,
    layout: shellBGL,
    entries: [{ binding: 0, resource: { buffer: globalsBuf } }],
  })

  let cullPipeline!: GPUComputePipeline
  let cullCarpetPipeline!: GPUComputePipeline
  let strandPipeline!: GPURenderPipeline
  let carpetPipeline!: GPURenderPipeline
  let shellPipeline!: GPURenderPipeline
  const build = (): void => {
    const strandModule = ctx.shaders.module(strandsSrc)
    const shellModule = ctx.shaders.module(farshellSrc)
    const cullLayout = device.createPipelineLayout({
      label: `${ctx.id}/cull`,
      bindGroupLayouts: [ctx.frame.layout, cullBGL],
    })
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: cullLayout,
      compute: { module: strandModule, entryPoint: 'cs_cull' },
    })
    cullCarpetPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull-carpet`,
      layout: cullLayout,
      compute: { module: strandModule, entryPoint: 'cs_cull_carpet' },
    })
    const drawLayout = device.createPipelineLayout({
      label: `${ctx.id}/draw`,
      bindGroupLayouts: [ctx.frame.layout, drawBGL],
    })
    const target = { format: ctx.colorFormat }
    const depth = { format: ctx.depthFormat, depthCompare: 'less' as const, depthWriteEnabled: true }
    strandPipeline = device.createRenderPipeline({
      label: `${ctx.id}/strands`,
      layout: drawLayout,
      vertex: { module: strandModule, entryPoint: 'vs_main' },
      fragment: { module: strandModule, entryPoint: 'fs_main', targets: [target] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: depth,
    })
    carpetPipeline = device.createRenderPipeline({
      label: `${ctx.id}/carpet`,
      layout: drawLayout,
      vertex: { module: strandModule, entryPoint: 'vs_carpet' },
      fragment: { module: strandModule, entryPoint: 'fs_carpet', targets: [target] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: depth,
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
      depthStencil: depth,
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
  const indirectData = new Uint32Array(DRAWS * 5)
  let regionArea = 0 // dims.x * dims.y, for the cull dispatch
  let carpetArea = 0

  // Stand-mixed shell constants. Coverage weight, not raw `density`: a carpet
  // entry's density is meaningless (its cover comes from the grid), so weight it
  // by tiles per m² times its wetness band.
  const density = new Array<number>(SPECIES.length).fill(0)
  let shellHCarpet = 0
  let shellWCarpet = 0
  let shellHScatter = 0
  let shellWScatter = 0
  let shellBob = 0
  {
    let wSum = 0
    for (const e of stand.species) {
      const sp = speciesById(e.species)
      const band = Math.min(1, e.wetWidth === undefined || e.wetWidth <= 0 ? 1 : e.wetWidth)
      if (isCarpet(e)) {
        const grid = grids.get(e.species)!
        const scale = SCATTER_CELL_SIZE / (e.carpetDiv ?? 1) / (sp.tileM ?? 1)
        const w = ((e.carpetDiv ?? 1) ** 2 / SCATTER_CELL_SIZE ** 2) * band
        density[sp.index]! += w
        // A mat's "canopy" is its own surface, and it must not be scaled down by
        // the grass-oriented shellHeight param or the shell sinks into the soil.
        shellHCarpet += w * grid.topH * scale
        shellWCarpet += w
        wSum += w
      } else {
        const field = fields.get(e.species)!
        const meanScale = (e.scaleMin + e.scaleMax) / 2
        const w = Math.min(e.density, 8) * band
        density[sp.index]! += w
        shellHScatter += w * field.topH * meanScale
        shellWScatter += w
        shellBob += w * e.sway
        wSum += w
      }
    }
    shellBob = wSum > 0 ? shellBob / wSum : 0.5
  }
  const canopyOf = (index: number): [number, number, number, number] => {
    for (const spId of gridIds) {
      if (speciesById(spId).index === index) {
        const g = grids.get(spId)!
        return [g.canopy[0], g.canopy[1], g.canopy[2], g.topH]
      }
    }
    for (const spId of strandIds) {
      if (speciesById(spId).index === index) {
        const f = fields.get(spId)!
        return [f.canopy[0], f.canopy[1], f.canopy[2], f.topH]
      }
    }
    return [0, 0, 0, 0]
  }

  const standCellMin = Math.floor(-stand.radius / SCATTER_CELL_SIZE)
  const standCellMax = Math.floor(stand.radius / SCATTER_CELL_SIZE)

  const regionCells = (camX: number, camZ: number, radius: number): [number, number, number, number] => {
    const half = Math.ceil((radius + 3) / SCATTER_CELL_SIZE)
    const camCX = Math.floor(camX / SCATTER_CELL_SIZE)
    const camCZ = Math.floor(camZ / SCATTER_CELL_SIZE)
    const minX = Math.max(camCX - half, standCellMin)
    const minZ = Math.max(camCZ - half, standCellMin)
    const maxX = Math.min(camCX + half, standCellMax)
    const maxZ = Math.min(camCZ + half, standCellMax)
    return [minX, minZ, Math.max(maxX - minX + 1, 0), Math.max(maxZ - minZ + 1, 0)]
  }

  const writeFrameData = (frame: FrameInfo): void => {
    const p = ctx.params
    const cam = frame.camera.pose
    const [minX, minZ, dimX, dimZ] = regionCells(cam.x, cam.z, p.rOuter)
    regionArea = scatterEntries.length > 0 ? dimX * dimZ : 0
    const carpetOuter = Math.min(p.carpetOuter, CARPET_OUTER_MAX)
    const [cMinX, cMinZ, cDimX, cDimZ] = regionCells(cam.x, cam.z, carpetOuter)
    carpetArea = hasCarpet ? cDimX * cDimZ : 0

    gI32[0] = minX
    gI32[1] = minZ
    gI32[2] = dimX
    gI32[3] = dimZ
    gI32[4] = cMinX
    gI32[5] = cMinZ
    gI32[6] = cDimX
    gI32[7] = cDimZ
    gU32[8] = ctx.seed >>> 0
    gF32[9] = p.strands
    gF32[10] = p.r0
    gF32[11] = p.r1
    gF32[12] = p.rOuter
    gF32[13] = p.widthScale
    gF32[14] = STRANDS
    gF32[15] = STATIONS
    // The shell has to become opaque exactly where the geometry it replaces
    // stops: a carpet's geometry stops at carpetOuter, long before rOuter.
    const shellEdge = hasCarpet ? Math.min(p.rOuter, carpetOuter) : p.rOuter
    const shellRamp = hasCarpet ? Math.min(6, carpetOuter * 0.5) : 8
    gF32[16] = shellEdge - shellRamp
    const shellH =
      (shellHCarpet + shellHScatter * p.shellHeight) / Math.max(shellWCarpet + shellWScatter, 1e-5)
    gF32[17] = Math.max(shellH - (shellWCarpet > shellWScatter ? 0.03 : 0), 0.01)
    gF32[18] = stand.radius
    gF32[19] = p.debugRings ? 1 : 0 // ring_debug (our param, NOT frame.debug_mode)
    gF32[20] = shellBob
    gF32[21] = shellRamp
    gF32[22] = carpetOuter
    gF32[23] = p.carpetPx
    gF32[24] = hasCarpet ? 1 : 0
    gU32[25] = entryCount
    // Annulus geometry: the ramp threshold is a 3D distance on carpet stands, so
    // the innermost HORIZONTAL row sits where that distance is first reached —
    // right under a high camera, out at shell_in for an eye-height one.
    const dy = hasCarpet ? cam.y - ctx.scene.terrain.height(cam.x, cam.z) : 0
    const inner = gF32[16]!
    const r0Geom = Math.max(0.4, Math.sqrt(Math.max(inner * inner - dy * dy, 0)))
    gF32[26] = r0Geom
    gF32[27] = Math.min(
      1.7,
      Math.max(hasCarpet ? 1.13 : 1.285, ((stand.radius * 3) / r0Geom) ** (1 / (shellRows - 1))),
    )
    for (let i = 0; i < 3; i++) gU32[28 + i] = ringCaps[i]!
    for (let l = 0; l < CARPET_LEVELS; l++) {
      gU32[31 + l] = carpetCaps[l]!
      gU32[40 + l] = carpetBases[l]!
    }
    for (const sp of SPECIES) {
      const c = canopyOf(sp.index)
      const o = 48 + sp.index * 4
      gF32[o] = c[0]
      gF32[o + 1] = c[1]
      gF32[o + 2] = c[2]
      gF32[o + 3] = c[3]
      gF32[72 + sp.index] = density[sp.index]!
    }
    gF32[80] = hasCarpet ? 0.69 : 0.92
    device.queue.writeBuffer(globalsBuf, 0, globalsData)

    // Indirect args. Strand rings: index counts follow the continuous strand
    // LOD. Carpet buckets: the whole lattice of the bucket's stride, no waste —
    // every vertex of a carpet draw is live geometry.
    const s0 = p.strands
    const sRing = [
      s0,
      Math.max(1, Math.min(s0, Math.ceil((s0 * p.r0) / p.r1))),
      Math.max(1, Math.min(s0, Math.ceil((s0 * p.r0) / p.rOuter))),
    ]
    for (let i = 0; i < DRAWS; i++) {
      const bands = i < 3 ? 0 : GRID_N >> (i - 3)
      indirectData[i * 5] = i < 3 ? sRing[i]! * (RING_T[i]! - 1) * 6 : bands * bands * 6
      indirectData[i * 5 + 1] = 0
      indirectData[i * 5 + 2] = 0
      indirectData[i * 5 + 3] = 0
      indirectData[i * 5 + 4] = 0
    }
    device.queue.writeBuffer(indirectBuf, 0, indirectData)
  }

  return {
    update(frame: FrameInfo): void {
      writeFrameData(frame)
    },

    encode(enc: GPUCommandEncoder, _frame: FrameInfo, targets: ViewTargets): void {
      if (regionArea > 0 || carpetArea > 0) {
        const cull = ctx.timing.computePass(enc, 'cull')
        cull.setBindGroup(0, ctx.frame.bindGroup)
        cull.setBindGroup(1, cullBG)
        if (regionArea > 0) {
          cull.setPipeline(cullPipeline)
          cull.dispatchWorkgroups(Math.ceil(SCATTER_MAX_PER_CELL / 64), regionArea, entryCount)
        }
        if (carpetArea > 0) {
          // EVERY carpet slot must be evaluated: carpetDiv² (484), not the
          // 128-slot scatter budget.
          cull.setPipeline(cullCarpetPipeline)
          cull.dispatchWorkgroups(Math.ceil(carpetDiv ** 2 / 64), carpetArea, entryCount)
        }
        cull.end()
      }

      // ONE render pass for everything: identical attachments and load/store
      // ops, no dependency between the draws. Two passes would cost a full
      // colour+depth store and reload (tile flush on TBDR) for nothing.
      const pass = ctx.timing.renderPass(enc, 'strands+shell', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setBindGroup(0, ctx.frame.bindGroup)
      if (scatterEntries.length > 0) {
        pass.setPipeline(strandPipeline)
        for (let i = 0; i < 3; i++) {
          pass.setBindGroup(1, drawBGs[i]!)
          pass.setIndexBuffer(indexBufs[i]!, 'uint16')
          pass.drawIndexedIndirect(indirectBuf, i * 20)
        }
      }
      if (hasCarpet) {
        pass.setPipeline(carpetPipeline)
        for (let i = 3; i < DRAWS; i++) {
          pass.setBindGroup(1, drawBGs[i]!)
          pass.setIndexBuffer(indexBufs[i]!, 'uint16')
          pass.drawIndexedIndirect(indirectBuf, i * 20)
        }
      }
      // Shell last: it is alpha-blended over whatever the plants left, and
      // depth-tests against them so occluded shell pixels are rejected.
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
