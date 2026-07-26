import captureShaderSrc from './shaders/capture.wgsl'
import fitShaderSrc from './shaders/fit.wgsl'
import horizonShaderSrc from './shaders/horizon.wgsl'
import carpetMipShaderSrc from './shaders/carpet_mip.wgsl'
import type { ExperimentContext, GcMesh } from '@harness'
import type { PARAMS } from './manifest.ts'

/**
 * Bake a species into Fourier coefficient textures, entirely on the GPU.
 *
 * UPRIGHT species (the grasses). For each of E elevation rings, the source mesh
 * is rendered orthographically from V azimuths into a V*TILE x TILE ring atlas
 * (albedo+coverage, mesh-frame normal + relative depth). A compute pass then
 * projects, per card texel, the angular functions onto a truncated Fourier basis
 * and writes the packed coefficients straight into one layer of four persistent
 * rgba8 array textures.
 *
 * CARPET species (the bog's Sphagnum, `carpetDiv > 0`). A mat gets a completely
 * different — and much smaller — bake: ONE top-down orthographic capture of the
 * tile's own square, replicated over a 3x3 periodic lattice so it is exactly
 * seamless, at 512^2 (0.35 mm/texel) with a mip chain, plus a compute pass that
 * fits the per-texel HORIZON as a Fourier series in azimuth (horizon.wgsl).
 * The ring/azimuth appearance atlas is skipped entirely: for a 0.18 m cushion
 * the appearance is high-frequency in azimuth (parallax exceeds a tile period at
 * grazing) so the fit blurs it to mush, while the horizon is smooth in azimuth
 * and is what actually carries the cushion's 3D reading. That is also 10x less
 * bake work — 9 mesh draws instead of 96 for a 19.8 M triangle mesh.
 *
 * Nothing is ever read back to the CPU; a species bake is one queue submit.
 */

// Must match fit.wgsl (V, TILE) and card.wgsl (RINGS, PHI_MIN/MAX).
export const V_AZIMUTHS = 24
export const TILE = 256
export const RINGS = 4
export const PHI_MIN = 0.13962634 //  8 deg
export const PHI_MAX = 1.43116999 // 82 deg

// Must match horizon.wgsl (TEX) — power of two, mipped down to 1x1.
export const CARPET_TEX = 512
export const CARPET_MIPS = 10
/**
 * Where the runtime quad sits, as a fraction of the tile's top height: the mean
 * capitulum apex, i.e. the surface the top-down capture actually shows. All
 * three Sphagnum states put it at 0.739-0.754 of `tile.topH` in their mesh
 * manifests, so one constant covers them (the mesh header carries bounds only).
 */
export const CARPET_PLANE_FRAC = 0.742

export interface BakedCarpet {
  /** rgb = coverage-weighted albedo, a = coverage. Mipped. */
  albedo: GPUTexture
  /** rg = cushion normal xz (biased), b = sky visibility, a = height / topH. Mipped. */
  relief: GPUTexture
  /** Horizon fit: r = DC, g/b = order-1 harmonics (biased), a reserved. Mipped. */
  horizon: GPUTexture
  /** Periodic tile size (m) the capture covers — equals stand footprint_m. */
  tileM: number
  /** Height (m, scale 1) of the tile's top, i.e. the height channel's unit. */
  topH: number
  /** Height (m, scale 1) above the tile base where the runtime quad sits. */
  planeH: number
}

export interface BakedSpecies {
  coeffA: GPUTexture
  coeffB: GPUTexture
  coeffC: GPUTexture
  coeffD: GPUTexture
  center: [number, number, number]
  radius: number
  /**
   * Card crop, in bounding-radius units. The baked tile spans ±radius on both
   * card axes, but the mesh bbox can only project into
   *   |x| ≤ extXZ,  |y| ≤ cos(phi)·extY + sin(phi)·extXZ
   * of it (`right` is horizontal, `up` tilts by the view elevation). Outside
   * that the fitted coverage is exactly zero, so the runtime card is cropped
   * to it — pure overdraw removal, identical image.
   */
  extXZ: number
  extY: number
}

type V3 = [number, number, number]
const cross = (a: V3, b: V3): V3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
]
const norm = (a: V3): V3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1
  return [a[0] / l, a[1] / l, a[2] / l]
}

/** Which representation(s) this species is actually drawn with in the stand. */
export interface BakeNeed {
  upright: boolean
  carpet: boolean
  /** Periodic tile size (m) — required when `carpet` is set. */
  tileM: number
}

export interface BakedSet {
  upright: BakedSpecies | null
  carpet: BakedCarpet | null
}

export function bakeSpecies(
  ctx: ExperimentContext<typeof PARAMS>,
  mesh: GcMesh,
  speciesId: string,
  need: BakeNeed,
): BakedSet {
  const { device } = ctx
  /** Extra transient resources destroyed right after the single submit. */
  const trash: GPUBuffer[] = []
  const bmin = mesh.header.boundsMin
  const bmax = mesh.header.boundsMax
  const center: V3 = [(bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2, (bmin[2] + bmax[2]) / 2]
  const radius = 0.5 * Math.hypot(bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2])
  const invR = 1 / radius
  const extXZ = (0.5 * Math.hypot(bmax[0] - bmin[0], bmax[2] - bmin[2])) / radius
  const extY = (0.5 * (bmax[1] - bmin[1])) / radius

  // --- persistent coefficient array textures (the whole species footprint) --
  const coeffTex = (tag: string): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/${speciesId}/${tag}`,
        size: [TILE, TILE, RINGS],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      },
      { species: speciesId, tag },
    )
  const coeffA = need.upright ? coeffTex('fourier-a') : null
  const coeffB = need.upright ? coeffTex('fourier-b') : null
  const coeffC = need.upright ? coeffTex('fourier-c') : null
  const coeffD = need.upright ? coeffTex('fourier-d') : null

  // --- persistent carpet textures (512^2 + mips, ~1.4 MB each) --------------
  const carpetTex = (tag: string): GPUTexture =>
    ctx.res.createTexture(
      {
        label: `${ctx.id}/${speciesId}/${tag}`,
        size: [CARPET_TEX, CARPET_TEX],
        mipLevelCount: CARPET_MIPS,
        format: 'rgba8unorm',
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.RENDER_ATTACHMENT,
      },
      { species: speciesId, tag },
    )
  const carpetAlbedo = need.carpet ? carpetTex('carpet-albedo') : null
  const carpetRelief = need.carpet ? carpetTex('carpet-relief') : null
  const carpetHorizon = need.carpet ? carpetTex('carpet-horizon') : null

  // --- transient bake resources (destroyed right after submit) --------------
  const vbuf = device.createBuffer({
    label: `${ctx.id}/bake-verts`,
    size: mesh.vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(vbuf, 0, mesh.vertices)
  const indices = mesh.indices()
  const ibuf = device.createBuffer({
    label: `${ctx.id}/bake-idx`,
    size: indices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(ibuf, 0, indices)

  const atlasW = V_AZIMUTHS * TILE
  const capTex = (tag: string, w: number, h: number, format: GPUTextureFormat): GPUTexture =>
    device.createTexture({
      label: `${ctx.id}/${tag}`,
      size: [w, h],
      format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
  const capColor = need.upright ? capTex('bake-color', atlasW, TILE, 'rgba8unorm') : null
  const capNormal = need.upright ? capTex('bake-normal', atlasW, TILE, 'rgba8unorm') : null
  const capDepth = need.upright ? capTex('bake-depth', atlasW, TILE, 'depth32float') : null
  const carpetDepth = need.carpet ? capTex('carpet-depth', CARPET_TEX, CARPET_TEX, 'depth32float') : null
  // The carpet capture's second target (mesh normal + height) is consumed by the
  // relief/horizon pass and thrown away: the leaf normals in it are useless for
  // shading a mat (see horizon.wgsl) and only its height channel survives.
  // rgba16float, not rgba8: the horizon sweep differentiates this height field
  // over single texels, and 8-bit steps (0.36 mm over a 0.35 mm texel) are a
  // slope of 1.0 — i.e. pure noise. See horizon.wgsl.
  const carpetHeight = need.carpet ? capTex('carpet-height', CARPET_TEX, CARPET_TEX, 'rgba16float') : null

  // Per-view uniforms: 112B of data padded to 256B for dynamic offsets. View
  // `nViews` is the carpet's single top-down capture.
  const STRIDE = 256
  const nViews = RINGS * V_AZIMUTHS
  const uni = device.createBuffer({
    label: `${ctx.id}/bake-uni`,
    size: STRIDE * (nViews + 1),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const scratch = new ArrayBuffer(STRIDE * (nViews + 1))
  const fv = new Float32Array(scratch)
  const writeView = (
    slot: number,
    c: V3,
    invRadius: number,
    right: V3,
    up: V3,
    fwd: V3,
    period: number,
    depthScale: number,
    depthBias: number,
  ): void => {
    const o = (slot * STRIDE) / 4
    fv[o + 0] = c[0]; fv[o + 1] = c[1]; fv[o + 2] = c[2]; fv[o + 3] = invRadius
    fv[o + 4] = right[0]; fv[o + 5] = right[1]; fv[o + 6] = right[2]
    fv[o + 8] = up[0]; fv[o + 9] = up[1]; fv[o + 10] = up[2]
    fv[o + 12] = fwd[0]; fv[o + 13] = fwd[1]; fv[o + 14] = fwd[2]
    fv[o + 16] = bmin[0]; fv[o + 17] = bmin[1]; fv[o + 18] = bmin[2]
    fv[o + 20] = bmax[0]; fv[o + 21] = bmax[1]; fv[o + 22] = bmax[2]
    fv[o + 24] = period; fv[o + 25] = depthScale; fv[o + 26] = depthBias
  }
  for (let e = 0; e < RINGS; e++) {
    const phi = PHI_MIN + (e / (RINGS - 1)) * (PHI_MAX - PHI_MIN)
    for (let j = 0; j < V_AZIMUTHS; j++) {
      // MUST match fit.wgsl (theta_j) and card.wgsl (basis construction).
      const theta = (2 * Math.PI * j) / V_AZIMUTHS
      const dir: V3 = [Math.cos(theta) * Math.cos(phi), Math.sin(phi), Math.sin(theta) * Math.cos(phi)]
      const right = norm(cross([0, 1, 0], dir))
      const up = cross(dir, right) as V3
      // depth_scale/bias reproduce the historical clamp(d * 0.5 + 0.5) exactly.
      writeView(e * V_AZIMUTHS + j, center, invR, right, up, dir, 0, 0.5 * invR, 0.5)
    }
  }
  // Carpet view: straight down over the tile square [0, tileM]^2 of the mesh
  // frame (tile origin is (0,0) for every source mesh). cross(right, up) = fwd,
  // the same right-handed convention as the ring views, so the image is not
  // mirrored: u = x / tileM, v = z / tileM. The depth channel becomes absolute
  // height above the tile base, normalised by the tile's top height.
  const topH = bmax[1]
  if (need.carpet) {
    const cCenter: V3 = [need.tileM / 2, topH / 2, need.tileM / 2]
    writeView(
      nViews,
      cCenter,
      2 / need.tileM,
      [1, 0, 0],
      [0, 0, -1],
      [0, 1, 0],
      need.tileM,
      1 / topH,
      cCenter[1] / topH,
    )
  }
  device.queue.writeBuffer(uni, 0, scratch)

  // --- capture pipeline -----------------------------------------------------
  const capBgl = device.createBindGroupLayout({
    label: `${ctx.id}/bake-bgl`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 112 },
      },
    ],
  })
  const capBg = device.createBindGroup({
    label: `${ctx.id}/bake-bg`,
    layout: capBgl,
    entries: [{ binding: 0, resource: { buffer: uni, size: 112 } }],
  })
  const capModule = ctx.shaders.module(captureShaderSrc)
  const capLayout = device.createPipelineLayout({ label: `${ctx.id}/bake-pl`, bindGroupLayouts: [capBgl] })
  const capPipelineFor = (secondTarget: GPUTextureFormat): GPURenderPipeline =>
    device.createRenderPipeline({
      label: `${ctx.id}/bake-capture-${secondTarget}`,
      layout: capLayout,
      vertex: {
        module: capModule,
        entryPoint: 'vs',
        buffers: [
          {
            arrayStride: 16,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'uint16x4' },
              { shaderLocation: 1, offset: 8, format: 'uint16x4' },
            ],
          },
        ],
      },
      fragment: {
        module: capModule,
        entryPoint: 'fs',
        targets: [{ format: 'rgba8unorm' }, { format: secondTarget }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthCompare: 'less', depthWriteEnabled: true },
    })
  const capPipeline = capPipelineFor('rgba8unorm')

  // --- fit pipeline (upright only) ------------------------------------------
  const fitBgl = device.createBindGroupLayout({
    label: `${ctx.id}/fit-bgl`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'rgba8unorm' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'rgba8unorm' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'rgba8unorm' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'rgba8unorm' } },
    ],
  })
  const fitPipeline = device.createComputePipeline({
    label: `${ctx.id}/fit`,
    layout: device.createPipelineLayout({ label: `${ctx.id}/fit-pl`, bindGroupLayouts: [fitBgl] }),
    compute: { module: ctx.shaders.module(fitShaderSrc), entryPoint: 'cs' },
  })
  const layerView = (tex: GPUTexture, e: number): GPUTextureView =>
    tex.createView({ dimension: '2d', baseArrayLayer: e, arrayLayerCount: 1 })
  const fitBindGroups =
    need.upright && capColor && capNormal && coeffA && coeffB && coeffC && coeffD
      ? Array.from({ length: RINGS }, (_, e) =>
          device.createBindGroup({
            label: `${ctx.id}/fit-bg-${e}`,
            layout: fitBgl,
            entries: [
              { binding: 0, resource: capColor.createView() },
              { binding: 1, resource: capNormal.createView() },
              { binding: 2, resource: layerView(coeffA, e) },
              { binding: 3, resource: layerView(coeffB, e) },
              { binding: 4, resource: layerView(coeffC, e) },
              { binding: 5, resource: layerView(coeffD, e) },
            ],
          }),
        )
      : []

  const enc = device.createCommandEncoder({ label: `${ctx.id}/bake-${speciesId}` })

  // --- upright: (capture ring -> fit ring) x RINGS ---------------------------
  if (need.upright && capColor && capNormal && capDepth) {
    for (let e = 0; e < RINGS; e++) {
      const pass = enc.beginRenderPass({
        label: `${ctx.id}/capture-ring-${e}`,
        colorAttachments: [
          { view: capColor.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
          {
            view: capNormal.createView(),
            clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 0.5 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
        depthStencilAttachment: {
          view: capDepth.createView(),
          depthClearValue: 1,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
      })
      pass.setPipeline(capPipeline)
      pass.setVertexBuffer(0, vbuf)
      pass.setIndexBuffer(ibuf, 'uint32')
      for (let j = 0; j < V_AZIMUTHS; j++) {
        pass.setViewport(j * TILE, 0, TILE, TILE, 0, 1)
        pass.setScissorRect(j * TILE, 0, TILE, TILE)
        pass.setBindGroup(0, capBg, [(e * V_AZIMUTHS + j) * STRIDE])
        pass.drawIndexed(indices.length)
      }
      pass.end()

      const fit = enc.beginComputePass({ label: `${ctx.id}/fit-ring-${e}` })
      fit.setPipeline(fitPipeline)
      fit.setBindGroup(0, fitBindGroups[e]!)
      fit.dispatchWorkgroups(TILE / 8, TILE / 8)
      fit.end()
    }
  }

  // --- carpet: top-down periodic capture -> relief/horizon fit -> mips -------
  if (need.carpet && carpetAlbedo && carpetRelief && carpetHorizon && carpetDepth && carpetHeight) {
    const mip0 = (tex: GPUTexture): GPUTextureView =>
      tex.createView({ baseMipLevel: 0, mipLevelCount: 1 })

    const pass = enc.beginRenderPass({
      label: `${ctx.id}/carpet-capture`,
      colorAttachments: [
        { view: mip0(carpetAlbedo), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
        // Cleared to zero height: the sane neutral for a texel with no geometry
        // over it (bare peat), so the relief pass sees a flat floor there.
        {
          view: carpetHeight.createView(),
          clearValue: { r: 0.5, g: 1, b: 0.5, a: 0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: carpetDepth.createView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    })
    pass.setPipeline(capPipelineFor('rgba16float'))
    pass.setVertexBuffer(0, vbuf)
    pass.setIndexBuffer(ibuf, 'uint32')
    pass.setBindGroup(0, capBg, [nViews * STRIDE])
    // 9 instances = the 3x3 periodic lattice, so overhanging geometry wraps.
    pass.drawIndexed(indices.length, 9)
    pass.end()

    // Relief (cushion normal, sky visibility, height) + horizon fit, both from
    // the captured height field.
    const hzBgl = device.createBindGroupLayout({
      label: `${ctx.id}/hz-bgl`,
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'rgba8unorm' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'rgba8unorm' } },
      ],
    })
    const hzUni = device.createBuffer({
      label: `${ctx.id}/hz-uni`,
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(hzUni, 0, new Float32Array([need.tileM / CARPET_TEX, topH, 0, 0]))
    const hzPipeline = device.createComputePipeline({
      label: `${ctx.id}/horizon`,
      layout: device.createPipelineLayout({ label: `${ctx.id}/hz-pl`, bindGroupLayouts: [hzBgl] }),
      compute: { module: ctx.shaders.module(horizonShaderSrc), entryPoint: 'cs' },
    })
    const hzPass = enc.beginComputePass({ label: `${ctx.id}/carpet-horizon` })
    hzPass.setPipeline(hzPipeline)
    hzPass.setBindGroup(
      0,
      device.createBindGroup({
        label: `${ctx.id}/hz-bg`,
        layout: hzBgl,
        entries: [
          { binding: 0, resource: { buffer: hzUni } },
          { binding: 1, resource: carpetHeight.createView() },
          { binding: 2, resource: mip0(carpetRelief) },
          { binding: 3, resource: mip0(carpetHorizon) },
        ],
      }),
    )
    hzPass.dispatchWorkgroups(CARPET_TEX / 8, CARPET_TEX / 8)
    hzPass.end()

    // Mip chain. Albedo first at every level: the data textures are weighted by
    // its coverage at the SOURCE level, so it must already be written.
    const mipBgl = device.createBindGroupLayout({
      label: `${ctx.id}/mip-bgl`,
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'rgba8unorm' } },
      ],
    })
    const mipModule = ctx.shaders.module(carpetMipShaderSrc)
    const mipLayout = device.createPipelineLayout({ label: `${ctx.id}/mip-pl`, bindGroupLayouts: [mipBgl] })
    const mipPipe = (entryPoint: string): GPUComputePipeline =>
      device.createComputePipeline({ label: `${ctx.id}/mip-${entryPoint}`, layout: mipLayout, compute: { module: mipModule, entryPoint } })
    const albedoMip = mipPipe('cs_albedo')
    const dataMip = mipPipe('cs_data')
    const level = (tex: GPUTexture, l: number): GPUTextureView =>
      tex.createView({ baseMipLevel: l, mipLevelCount: 1 })
    for (let l = 1; l < CARPET_MIPS; l++) {
      const n = Math.max(1, CARPET_TEX >> l)
      const groups = Math.ceil(n / 8)
      const run = (label: string, pipeline: GPUComputePipeline, src: GPUTexture, dst: GPUTexture): void => {
        const p = enc.beginComputePass({ label: `${ctx.id}/${label}-${l}` })
        p.setPipeline(pipeline)
        p.setBindGroup(
          0,
          device.createBindGroup({
            label: `${ctx.id}/${label}-bg-${l}`,
            layout: mipBgl,
            entries: [
              { binding: 0, resource: level(src, l - 1) },
              { binding: 1, resource: level(carpetAlbedo, l - 1) },
              { binding: 2, resource: level(dst, l) },
            ],
          }),
        )
        p.dispatchWorkgroups(groups, groups)
        p.end()
      }
      run('mip-albedo', albedoMip, carpetAlbedo, carpetAlbedo)
      run('mip-relief', dataMip, carpetRelief, carpetRelief)
      run('mip-horizon', dataMip, carpetHorizon, carpetHorizon)
    }
    trash.push(hzUni)
  }

  device.queue.submit([enc.finish()])

  // Safe immediately after submit — WebGPU defers actual release.
  for (const r of [vbuf, ibuf, uni, capColor, capNormal, capDepth, carpetDepth, carpetHeight, ...trash]) {
    r?.destroy()
  }

  return {
    upright:
      coeffA && coeffB && coeffC && coeffD
        ? { coeffA, coeffB, coeffC, coeffD, center, radius, extXZ, extY }
        : null,
    carpet:
      carpetAlbedo && carpetRelief && carpetHorizon
        ? {
            albedo: carpetAlbedo,
            relief: carpetRelief,
            horizon: carpetHorizon,
            tileM: need.tileM,
            topH,
            planeH: topH * CARPET_PLANE_FRAC,
          }
        : null,
  }
}
