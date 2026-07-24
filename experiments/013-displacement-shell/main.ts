import strandsSrc from './shaders/strands.wgsl'
import farshellSrc from './shaders/farshell.wgsl'
import { SCATTER_CELL_SIZE, SCATTER_MAX_PER_CELL, SPECIES, commitBake, speciesById } from '@harness'
import type { Experiment, ExperimentContext, FrameInfo, ViewTargets } from '@harness'
import {
  BAKE_VERSION,
  STATIONS,
  STRANDS,
  bakeStrandField,
  isField,
  parseField,
  serializeField,
  toF16,
  type StrandField,
} from './bake.ts'
import type { PARAMS } from './manifest.ts'

/**
 * Strand displacement shell.
 *
 * The only geometry that ever exists is one flat sheet: identical little
 * ribbon patches, one per plant of the stand, whose vertices are thrown into
 * place by baked "strand vector-displacement field" textures (offset + width
 * + normal + color per strand station). Placement is fully GPU-procedural —
 * a per-frame compute pass walks the scatter cells of a camera-centered
 * region (the bit-identical WGSL twin of the harness scatter), frustum-culls,
 * and appends survivors to three distance rings with decreasing patch
 * topology. Strand-count LOD is continuous (s0 * r0 / d) so ring handoffs
 * never pop. Beyond the region the meadow IS a single terrain-conformal
 * canopy shell (camera-centered annulus at baked canopy color/height).
 * Per-frame cost is bounded by region + screen, never by plant count.
 */

const RING_T = [12, 8, 5] as const
const RING_R_MAX = [12, 24, 56] as const // must match manifest param maxima
const SHELL_SEGMENTS = 72
const SHELL_ROWS = 20
const GLOBALS_BYTES = 144

function cellsAcross(radius: number): number {
  return 2 * Math.ceil((radius + 3) / SCATTER_CELL_SIZE) + 1
}

/**
 * Validating artifact flow: committed file -> OPFS cache -> bake in-browser.
 * (The dev server SPA-fallback answers missing baked files with index.html,
 * so every path magic-checks the bytes; same pattern as 003.)
 */
async function loadValidatedArtifact(
  expId: string,
  key: string,
  bake: () => Promise<ArrayBuffer>,
): Promise<ArrayBuffer> {
  const committed = await fetch(`/mesh/baked/${expId}/${key}.bin`).catch(() => null)
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
  // Bake (or load) one strand field per species used by the stand
  // -------------------------------------------------------------------------
  const uniqueSpecies = [...new Set(stand.species.map((e) => e.species))]
  const fields = new Map<string, StrandField>()
  for (const spId of uniqueSpecies) {
    const meshId = speciesById(spId).meshId
    const key = `${spId}-v${BAKE_VERSION}`
    const data = await loadValidatedArtifact(ctx.id, key, async () => {
      console.log(`[${ctx.id}] baking ${spId} — fetching source mesh…`)
      const mesh = await ctx.meshes.load(meshId)
      const t0 = performance.now()
      const field = bakeStrandField(mesh, ctx.meshes.info(meshId).tileSize)
      console.log(`[${ctx.id}] baked ${spId} in ${(performance.now() - t0).toFixed(0)}ms`)
      return serializeField(field)
    })
    fields.set(spId, parseField(data))
  }

  // -------------------------------------------------------------------------
  // Strand field textures: one rgba16float layer per species-catalog index
  // -------------------------------------------------------------------------
  const layerCount = SPECIES.length
  const makeFieldTex = (name: string): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/${name}`,
        size: [STATIONS, STRANDS, layerCount],
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      },
      { tag: 'strand-field' },
    )
  const posTex = makeFieldTex('field-pos')
  const norTex = makeFieldTex('field-nor')
  const colTex = makeFieldTex('field-col')

  for (const spId of uniqueSpecies) {
    const field = fields.get(spId)!
    const layer = speciesById(spId).index
    const texel = STATIONS * STRANDS
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
    const layout = { bytesPerRow: STATIONS * 8, rowsPerImage: STRANDS }
    const extent: [number, number, number] = [STATIONS, STRANDS, 1]
    device.queue.writeTexture({ texture: posTex, origin: [0, 0, layer] }, pos, layout, extent)
    device.queue.writeTexture({ texture: norTex, origin: [0, 0, layer] }, nor, layout, extent)
    device.queue.writeTexture({ texture: colTex, origin: [0, 0, layer] }, col, layout, extent)
  }

  // -------------------------------------------------------------------------
  // Buffers
  // -------------------------------------------------------------------------
  const caps = RING_R_MAX.map((r) => cellsAcross(r) ** 2 * SCATTER_MAX_PER_CELL * entryCount)
  const instanceBufs = caps.map((cap, i) =>
    ctx.res.createBuffer(
      { label: `${ctx.id}/ring${i}-instances`, size: cap * 16, usage: GPUBufferUsage.STORAGE },
      { tag: 'ring-instances' },
    ),
  )
  const indirectBuf = ctx.res.createBuffer(
    {
      label: `${ctx.id}/indirect`,
      size: 15 * 4,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    },
    { tag: 'indirect' },
  )
  const globalsBuf = ctx.res.createBuffer(
    { label: `${ctx.id}/globals`, size: GLOBALS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
    { tag: 'globals' },
  )
  const ringInfoBufs = RING_T.map((t, i) => {
    const buf = ctx.res.createBuffer(
      { label: `${ctx.id}/ring${i}-info`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST },
      { tag: 'ring-info' },
    )
    device.queue.writeBuffer(buf, 0, new Uint32Array([t, i, 0, 0]))
    return buf
  })

  // Patch topology per ring: STRANDS ribbon strips of t stations, 2 verts wide.
  const indexBufs = RING_T.map((t, i) => {
    const idx = new Uint16Array(STRANDS * (t - 1) * 6)
    let k = 0
    for (let s = 0; s < STRANDS; s++) {
      const base = s * t * 2
      for (let st = 0; st < t - 1; st++) {
        const v00 = base + st * 2
        idx[k++] = v00
        idx[k++] = v00 + 2
        idx[k++] = v00 + 1
        idx[k++] = v00 + 1
        idx[k++] = v00 + 2
        idx[k++] = v00 + 3
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
  for (let r = 0; r < SHELL_ROWS - 1; r++) {
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
    ],
  })
  const strandBGL = device.createBindGroupLayout({
    label: `${ctx.id}/strands`,
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
    ],
  })
  const posView = posTex.createView({ dimension: '2d-array' })
  const norView = norTex.createView({ dimension: '2d-array' })
  const colView = colTex.createView({ dimension: '2d-array' })
  const strandBGs = instanceBufs.map((buf, i) =>
    device.createBindGroup({
      label: `${ctx.id}/strands-ring${i}`,
      layout: strandBGL,
      entries: [
        { binding: 0, resource: { buffer: globalsBuf } },
        { binding: 5, resource: { buffer: buf } },
        { binding: 6, resource: { buffer: ringInfoBufs[i]! } },
        { binding: 7, resource: posView },
        { binding: 8, resource: norView },
        { binding: 9, resource: colView },
      ],
    }),
  )
  const shellBG = device.createBindGroup({
    label: `${ctx.id}/shell`,
    layout: shellBGL,
    entries: [{ binding: 0, resource: { buffer: globalsBuf } }],
  })

  let cullPipeline!: GPUComputePipeline
  let strandPipeline!: GPURenderPipeline
  let shellPipeline!: GPURenderPipeline
  const build = (): void => {
    const strandModule = ctx.shaders.module(strandsSrc)
    const shellModule = ctx.shaders.module(farshellSrc)
    cullPipeline = device.createComputePipeline({
      label: `${ctx.id}/cull`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/cull`, bindGroupLayouts: [ctx.frame.layout, cullBGL] }),
      compute: { module: strandModule, entryPoint: 'cs_cull' },
    })
    strandPipeline = device.createRenderPipeline({
      label: `${ctx.id}/strands`,
      layout: device.createPipelineLayout({
        label: `${ctx.id}/strands`,
        bindGroupLayouts: [ctx.frame.layout, strandBGL],
      }),
      vertex: { module: strandModule, entryPoint: 'vs_main' },
      fragment: { module: strandModule, entryPoint: 'fs_main', targets: [{ format: ctx.colorFormat }] },
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
  const indirectData = new Uint32Array(15)
  let regionArea = 0 // dims.x * dims.y, for the cull dispatch

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
    const p = ctx.params
    const half = Math.ceil((p.rOuter + 3) / SCATTER_CELL_SIZE)
    const camCX = Math.floor(frame.camera.pose.x / SCATTER_CELL_SIZE)
    const camCZ = Math.floor(frame.camera.pose.z / SCATTER_CELL_SIZE)
    const minX = Math.max(camCX - half, standCellMin)
    const minZ = Math.max(camCZ - half, standCellMin)
    const maxX = Math.min(camCX + half, standCellMax)
    const maxZ = Math.min(camCZ + half, standCellMax)
    const dimX = Math.max(maxX - minX + 1, 0)
    const dimZ = Math.max(maxZ - minZ + 1, 0)
    regionArea = dimX * dimZ

    gI32[0] = minX
    gI32[1] = minZ
    gI32[2] = dimX
    gI32[3] = dimZ
    gU32[4] = ctx.seed >>> 0
    gF32[5] = p.strands
    gF32[6] = p.r0
    gF32[7] = p.r1
    gF32[8] = p.rOuter
    gF32[9] = p.widthScale
    gF32[10] = STRANDS
    gF32[11] = STATIONS
    gF32[12] = p.rOuter - 8
    gF32[13] = shellHNom * p.shellHeight
    gF32[14] = stand.radius
    gF32[15] = p.debugRings ? 1 : 0 // ring_debug (our param, NOT frame.debug_mode)
    gF32[16] = shellBob
    gU32[17] = caps[0]!
    gU32[18] = caps[1]!
    gU32[19] = caps[2]!
    for (const sp of SPECIES) {
      const field = fields.get(sp.id)
      const o = 20 + sp.index * 4
      if (field) {
        gF32[o] = field.canopy[0]
        gF32[o + 1] = field.canopy[1]
        gF32[o + 2] = field.canopy[2]
        gF32[o + 3] = field.topH
      }
    }
    gF32[32] = density[0]!
    gF32[33] = density[1]!
    gF32[34] = density[2]!
    gF32[35] = density[3]!
    device.queue.writeBuffer(globalsBuf, 0, globalsData)

    // Indirect args: per-ring index counts follow the continuous strand LOD.
    const s0 = p.strands
    const sRing = [
      s0,
      Math.max(1, Math.min(s0, Math.ceil((s0 * p.r0) / p.r1))),
      Math.max(1, Math.min(s0, Math.ceil((s0 * p.r0) / p.rOuter))),
    ]
    for (let i = 0; i < 3; i++) {
      indirectData[i * 5] = sRing[i]! * (RING_T[i]! - 1) * 6
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
      if (regionArea > 0) {
        const cull = ctx.timing.computePass(enc, 'cull')
        cull.setPipeline(cullPipeline)
        cull.setBindGroup(0, ctx.frame.bindGroup)
        cull.setBindGroup(1, cullBG)
        cull.dispatchWorkgroups(Math.ceil(SCATTER_MAX_PER_CELL / 64), regionArea, entryCount)
        cull.end()
      }

      // ONE render pass for strands + far shell: identical attachments and
      // load/store ops, no dependency between them. Two passes would cost a
      // full colour+depth store and reload (tile flush on TBDR) for nothing.
      const pass = ctx.timing.renderPass(enc, 'strands+shell', {
        colorAttachments: [{ view: targets.colorView, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: targets.depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
      })
      pass.setPipeline(strandPipeline)
      pass.setBindGroup(0, ctx.frame.bindGroup)
      for (let i = 0; i < 3; i++) {
        pass.setBindGroup(1, strandBGs[i]!)
        pass.setIndexBuffer(indexBufs[i]!, 'uint16')
        pass.drawIndexedIndirect(indirectBuf, i * 20)
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
