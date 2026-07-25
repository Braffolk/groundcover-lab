import studioSrc from '../scene/studio.wgsl'
import { createFullscreenPipeline } from '../gpu/screenPass.ts'
import type { ShaderRegistry } from '../gpu/shaders.ts'
import type { PassTimer } from '../gpu/timing.ts'
import { loadUserBookmarks } from '../scene/bookmarks.ts'
import type { CameraPose } from '../scene/camera.ts'
import type { HeightfieldDims } from '../scene/frameUbo.ts'
import { PreviewGeometry, type MaterialContextExt, type PreviewObjectId } from '../scene/preview.ts'
import { CameraSpline } from '../scene/spline.ts'
import type { Stage, StageFactory, StageInit, StageView } from './stage.ts'

/**
 * The MATERIAL stage: one preview object on a neutral studio backdrop.
 *
 * It is the material counterpart of `standStage` — the harness owns the scene,
 * the experiment owns only how a surface is shaded. Two things are worth
 * knowing before changing anything here.
 *
 * 1. THE @group(0) LAYOUT IS FROZEN (see FRAME_LAYOUT_ENTRIES in
 *    src/scene/frameUbo.ts). A material scene has no terrain and no stand, but
 *    it still supplies binding 1 and binding 3 — as a 4x4 heightmap of zeroes
 *    and a single neutral stand row, under 1 KB together. That is a deliberate
 *    trade: it buys `light_surface()` and `debug_shade()` working VERBATIM in
 *    a material shader, on the same sun, ambient and debug selector as every
 *    groundcover renderer, which is the only reason a material and a renderer
 *    can be compared at all. A second bind group layout would fork
 *    frame.wgsl / lighting.wgsl / debug.wgsl, whose include mechanism has no
 *    conditionals.
 *
 * 2. THE BASE PASS DOES NOT DRAW THE OBJECT. It clears colour to the backdrop
 *    and clears depth, and stops. The experiment draws the preview geometry
 *    itself, because a silhouette-POM material has to be free to discard
 *    fragments and to write its own depth; a deferred split where the harness
 *    laid down the geometry would forbid both.
 */

/** The material-specific slice of ExperimentContext (declared in preview.ts). */
export type { MaterialContextExt }

const ORIGIN: [number, number, number] = [0, 0, 0]

/** Pose looking from `eye` at the world origin. */
function lookingAtOrigin(eye: [number, number, number], fov: number): CameraPose {
  const len = Math.hypot(eye[0], eye[1], eye[2]) || 1
  const d: [number, number, number] = [-eye[0] / len, -eye[1] / len, -eye[2] / len]
  return {
    x: eye[0],
    y: eye[1],
    z: eye[2],
    // poseForward = (sin(yaw)cos(pitch), sin(pitch), -cos(yaw)cos(pitch))
    yaw: Math.atan2(d[0], -d[2]),
    pitch: Math.asin(Math.max(-1, Math.min(1, d[1]))),
    fov,
  }
}

/** Pose from `eye`, aimed at `at` — for the tangential silhouette view. */
function lookingAt(eye: [number, number, number], at: [number, number, number], fov: number): CameraPose {
  const v: [number, number, number] = [at[0] - eye[0], at[1] - eye[1], at[2] - eye[2]]
  const len = Math.hypot(v[0], v[1], v[2]) || 1
  return {
    x: eye[0],
    y: eye[1],
    z: eye[2],
    yaw: Math.atan2(v[0] / len, -v[2] / len),
    pitch: Math.asin(Math.max(-1, Math.min(1, v[1] / len))),
    fov,
  }
}

/**
 * Camera bookmarks for judging a MATERIAL (keys 1-4, `cam=`). Every preview
 * object is centred on the origin and roughly 1 m across, so one set frames all
 * four. These names are the reliable way to reach a viewpoint: a URL
 * `cam=x,y,z,...` pose is ABSOLUTE and says nothing about where the object is.
 */
export function materialBookmarks(): Record<string, CameraPose> {
  return {
    // The default read: three-quarter, slightly above, moderate lens. Framed
    // for the LARGEST preview object (the edge-on cube, 1.41 m across its
    // diagonal), so switching object never pushes the subject out of frame.
    'three-quarter': lookingAtOrigin([1.59, 1.18, 1.89], 38),
    // Very close — the object overfills the frame, so texel density, tiling
    // seams and normal-map detail are all at their most unforgiving.
    macro: lookingAtOrigin([0.36, 0.3, 0.62], 38),
    // ~4 degrees off the surface of the ground-parallel plane. Where
    // impostor-ish and parallax methods fall apart, and where a flat map
    // stops being able to pretend it has thickness.
    grazing: lookingAtOrigin([0.55, 0.115, 1.55], 44),
    // Aimed at the LIMB, not the centre: a tangential telephoto view where the
    // object's outline sits mid-frame against the backdrop. The only place a
    // silhouette-POM material's whole point is visible.
    silhouette: lookingAt([0, -0.03, 1.3], [0.6, -0.03, 0], 30),
  }
}

/** Bench camera paths for a material (`spline=`). */
export function materialSplines(): Record<string, CameraSpline> {
  const orbit: CameraPose[] = []
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * Math.PI * 2
    orbit.push(lookingAtOrigin([Math.sin(a) * 1.75, 0.72, Math.cos(a) * 1.75], 38))
  }
  const macroPass: CameraPose[] = [
    lookingAtOrigin([1.5, 1.1, 1.7], 38),
    lookingAtOrigin([0.75, 0.55, 0.9], 38),
    lookingAtOrigin([0.36, 0.3, 0.62], 38),
    lookingAtOrigin([0.55, 0.115, 1.55], 44),
  ]
  return {
    'orbit-object': new CameraSpline(orbit, true),
    'macro-pass': new CameraSpline(macroPass, false),
  }
}

/** The default bench spline for a material — see `orbit-low` for a stand. */
export const MATERIAL_DEFAULT_SPLINE = 'orbit-object'

/**
 * A material stage previewing `object`. Which object is shown can be changed
 * live afterwards through `MaterialStage.previewObject`.
 */
export function materialStage(object: PreviewObjectId): StageFactory {
  return (init) => new MaterialStage(init, object)
}

export function isMaterialStage(stage: Stage): stage is MaterialStage {
  return stage.kind === 'material'
}

export class MaterialStage implements Stage {
  readonly kind = 'material'
  readonly heightmapView: GPUTextureView
  readonly standBuffer: GPUBuffer
  readonly terrainDims: HeightfieldDims
  readonly defaultPose: CameraPose
  readonly cameraMode = 'orbit'
  /** Orbit rotation pivots on the object, not on a point in front of the eye. */
  readonly orbitPivot = ORIGIN
  readonly geometry: PreviewGeometry

  private shaders: ShaderRegistry
  private frameLayout: GPUBindGroupLayout
  private colorFormat: GPUTextureFormat
  private depthFormat: GPUTextureFormat
  private device: GPUDevice
  private backdrop!: GPURenderPipeline
  private ext: MaterialContextExt
  private cams: Record<string, CameraPose>
  private paths: Record<string, CameraSpline>

  constructor(init: StageInit, object: PreviewObjectId) {
    this.device = init.device
    this.shaders = init.shaders
    this.frameLayout = init.frameLayout
    this.colorFormat = init.colorFormat
    this.depthFormat = init.depthFormat

    this.geometry = new PreviewGeometry(init.scope, init.queue, object)
    this.ext = { preview: this.geometry }

    // --- the two frozen-layout stand-ins ---------------------------------
    // 4x4 rgba16float of zeroes: terrain_height() reads 0 everywhere and
    // terrain_normal() decodes to straight up, which is a coherent answer for
    // "there is no terrain here" rather than NaNs. 128 bytes.
    const heightmap = init.scope.createTexture(
      {
        label: 'material/dummy-heightmap',
        size: [4, 4],
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      },
      { tag: 'stage-stub' },
    )
    init.queue.writeTexture({ texture: heightmap }, new Uint16Array(4 * 4 * 4), { bytesPerRow: 4 * 8 }, [4, 4])
    this.heightmapView = heightmap.createView()
    this.terrainDims = { size: 1, heightScale: 0, resolution: 4 }

    // One neutral StandEntry (12 floats, 48 bytes) so `stand_table[0]` is a
    // legal read. No material should index it, but the binding must exist and
    // an experiment that does read it gets unit scale rather than garbage.
    this.standBuffer = init.scope.createBuffer(
      { label: 'material/dummy-stand', size: 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST },
      { tag: 'stage-stub' },
    )
    init.queue.writeBuffer(
      this.standBuffer,
      0,
      // density, scaleMin, scaleMax, sway, heightScale, speciesIndex,
      // wetCenter, wetWidth, carpetDiv, footprintM, slopeAlign, pad
      new Float32Array([0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0]),
    )

    // One record, shared with LabApp and every view's context — the runner
    // adds a manifest's `cams` and Shift+N adds user bookmarks by mutating it,
    // exactly as StandStage does.
    this.cams = { ...materialBookmarks(), ...loadUserBookmarks() }
    this.paths = materialSplines()
    this.defaultPose = this.cams['three-quarter']!

    this.rebuild()
  }

  /** Which object is previewed. Assign to switch live — no rebuild needed. */
  get previewObject(): PreviewObjectId {
    return this.geometry.active
  }

  set previewObject(id: PreviewObjectId) {
    this.geometry.active = id
  }

  bookmarks(): Record<string, CameraPose> {
    return this.cams
  }

  splines(): Record<string, CameraSpline> {
    return this.paths
  }

  contextFor(_view: StageView): MaterialContextExt {
    // Every view on the page previews the SAME object off the same buffers —
    // that is what makes a material A/B a comparison of shading only.
    return this.ext
  }

  encodeBase(
    enc: GPUCommandEncoder,
    timer: PassTimer,
    frameBindGroup: GPUBindGroup,
    colorView: GPUTextureView,
    depthView: GPUTextureView,
  ): void {
    const pass = timer.renderPass(enc, 'base', {
      colorAttachments: [
        { view: colorView, loadOp: 'clear', clearValue: { r: 0.13, g: 0.13, b: 0.13, a: 1 }, storeOp: 'store' },
      ],
      depthStencilAttachment: { view: depthView, depthLoadOp: 'clear', depthClearValue: 1, depthStoreOp: 'store' },
    })
    pass.setBindGroup(0, frameBindGroup)
    pass.setPipeline(this.backdrop)
    pass.draw(3)
    pass.end()
  }

  rebuild(): void {
    this.backdrop = createFullscreenPipeline(this.device, this.shaders, studioSrc, {
      label: 'material/backdrop',
      format: this.colorFormat,
      bindGroupLayouts: [this.frameLayout],
      // Same contract as the sky: far-plane z with less-equal and no depth
      // write, so the preview object's own depth still wins everywhere.
      depth: { format: this.depthFormat, depthCompare: 'less-equal', depthWriteEnabled: false },
    })
  }

  dispose(): void {
    // Everything went through the harness-owned 'scene' scope, which LabApp
    // disposes; pipelines have no explicit destructor.
  }
}
