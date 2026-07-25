import type { VramScope } from '../gpu/resources.ts'
import type { ShaderRegistry } from '../gpu/shaders.ts'
import type { PassTimer } from '../gpu/timing.ts'
import { BasePass } from '../scene/basePass.ts'
import { loadUserBookmarks, standardBookmarks } from '../scene/bookmarks.ts'
import type { CameraPose } from '../scene/camera.ts'
import type { HeightfieldDims } from '../scene/frameUbo.ts'
import { Scatter } from '../scene/scatter.ts'
import { SPECIES } from '../scene/species.ts'
import { builtinSplines, type CameraSpline } from '../scene/spline.ts'
import { createStandBuffer, type Stand } from '../scene/stands.ts'
import { Terrain } from '../scene/terrain.ts'
import { WIND_DEFAULTS } from '../scene/wind.ts'
import type { MaterialContextExt } from './materialStage.ts'
import type { SceneServices, StandContextExt } from './registry.ts'

/**
 * A Stage is WHAT the lab draws around an experiment: the world the shared
 * @group(0) describes, the base pass that fills the color/depth targets before
 * an experiment appends its own passes, and the camera setup that world wants.
 * `LabApp` owns everything else — pass timer, VRAM tracking, the same-frame A/B
 * compositor with one shared camera, HUD, params/URL state, capture, debug
 * views — and works with any stage.
 *
 * Two stage kinds exist, one per experiment kind (`manifest.kind` picks it):
 *   `stand`    the groundcover scene — terrain, sky, the active stand's scatter
 *   `material` one preview object on a studio backdrop (src/harness/materialStage.ts)
 *
 * TWO INVARIANTS A NEW STAGE MUST NOT BREAK:
 *
 * 1. The @group(0) LAYOUT is frozen (see FRAME_LAYOUT_ENTRIES in
 *    src/scene/frameUbo.ts). A stage supplies the CONTENTS of bindings 1 and 3
 *    — a heightmap view and a stand-table buffer — and nothing else. A stage
 *    with no terrain supplies a dummy 4x4 heightmap and a one-row stand buffer
 *    rather than propose a second layout; MaterialStage does exactly that, and
 *    the payoff is that light_surface() and debug_shade() work verbatim in a
 *    material shader.
 * 2. A stage does NOT own the base-pass copy to view B. It only encodes the
 *    base pass ONCE, into view 0; `LabApp.frame()` copies colour AND depth into
 *    view 1, and that copy is the A/B fairness invariant.
 */

export interface StageInit {
  device: GPUDevice
  queue: GPUQueue
  /** Harness-owned scope ('scene') — everything the stage allocates goes here. */
  scope: VramScope
  shaders: ShaderRegistry
  /** THE frozen @group(0) layout; build base-pass pipelines against it. */
  frameLayout: GPUBindGroupLayout
  colorFormat: GPUTextureFormat
  depthFormat: GPUTextureFormat
  /** Report long setup work (0..1) — the runner shows it as a status line. */
  onProgress?: (fraction: number, note?: string) => void
}

/** The three slots a page can have: solo runner, or the A/B pair. */
export type StageView = 'solo' | 'A' | 'B'

/**
 * The stand-specific slice of ExperimentContext (declared in registry.ts, next
 * to SceneServices). Merged in by LabApp, so an experiment sees `ctx.scene` /
 * `ctx.stand` exactly as it always has.
 */
export type { StandContextExt }

/**
 * Whatever a stage merges into ExperimentContext. One member per stage kind;
 * an experiment picks its own via `ExperimentContext<PARAMS, MaterialContextExt>`,
 * and the default is the stand flavour so every pre-existing experiment's
 * `ExperimentContext<typeof PARAMS>` keeps meaning exactly what it meant.
 */
export type StageContextExt = StandContextExt | MaterialContextExt

export interface Stage {
  readonly kind: 'stand' | 'material'
  /** Contents of @group(0) @binding(1). */
  readonly heightmapView: GPUTextureView
  /** Contents of @group(0) @binding(3). */
  readonly standBuffer: GPUBuffer
  /** Heightfield geometry the Frame struct carries. */
  readonly terrainDims: HeightfieldDims
  /** Pose the camera starts at when the URL/caller names none. */
  readonly defaultPose: CameraPose
  readonly cameraMode: 'fly' | 'orbit'
  /**
   * Fixed world point orbit rotation pivots about. Set it when the stage has
   * ONE subject (a material preview object); leave it undefined for a world
   * you fly through, where the pivot is a point in front of the eye.
   */
  readonly orbitPivot?: readonly [number, number, number]
  /** Named poses (keys 1-4, `cam=`); the returned record is live-mutable. */
  bookmarks(): Record<string, CameraPose>
  /** Named bench camera paths (`spline=`). */
  splines(): Record<string, CameraSpline>
  /** Per-view extension merged into ExperimentContext. */
  contextFor(view: StageView): StageContextExt
  /** Clear + fill the targets. Encoded ONCE, into view 0 — never into B. */
  encodeBase(
    enc: GPUCommandEncoder,
    timer: PassTimer,
    frameBindGroup: GPUBindGroup,
    colorView: GPUTextureView,
    depthView: GPUTextureView,
  ): void
  /** Shader hot reload — recreate pipelines. */
  rebuild(): void
  dispose(): void
}

export type StageFactory = (init: StageInit) => Stage

/**
 * The groundcover stage: FBM terrain + the active stand's deterministic
 * scatter, with the harness-owned terrain+sky base pass. Stand + seed fully
 * determine every plant instance; experiments are pure renderers of it.
 */
export function standStage(stand: Stand, seed: number): StageFactory {
  return (init) => new StandStage(init, stand, seed)
}

class StandStage implements Stage {
  readonly kind = 'stand'
  readonly heightmapView: GPUTextureView
  readonly standBuffer: GPUBuffer
  readonly terrainDims: HeightfieldDims
  readonly defaultPose: CameraPose
  readonly cameraMode = 'fly'

  private basePass: BasePass
  private scene: SceneServices

  constructor(
    init: StageInit,
    private stand: Stand,
    seed: number,
  ) {
    const terrain = Terrain.generate(init.scope, init.queue)
    this.standBuffer = createStandBuffer(init.scope, init.queue, stand)
    this.heightmapView = terrain.texture.createView()
    this.terrainDims = terrain.desc
    this.basePass = new BasePass(init.device, init.shaders, init.frameLayout, init.colorFormat, init.depthFormat)
    this.scene = {
      terrain,
      scatter: new Scatter(terrain, seed, stand),
      wind: WIND_DEFAULTS,
      species: SPECIES,
      // One record, shared with LabApp and with every view's ctx.scene: the
      // runner adds a manifest's `cams` and the Shift+N hotkey adds user
      // bookmarks by mutating it, and both must be visible everywhere.
      bookmarks: { ...standardBookmarks(terrain), ...loadUserBookmarks() },
      splines: builtinSplines(terrain),
    }
    this.defaultPose = this.scene.bookmarks['grazing']!
  }

  bookmarks(): Record<string, CameraPose> {
    return this.scene.bookmarks
  }

  splines(): Record<string, CameraSpline> {
    return this.scene.splines
  }

  contextFor(_view: StageView): StandContextExt {
    // Every view on a page renders the SAME stand + seed off the same terrain —
    // that is what makes an A/B honest — so all three slots get one scene.
    return { scene: this.scene, stand: this.stand }
  }

  encodeBase(
    enc: GPUCommandEncoder,
    timer: PassTimer,
    frameBindGroup: GPUBindGroup,
    colorView: GPUTextureView,
    depthView: GPUTextureView,
  ): void {
    this.basePass.encode(enc, timer, frameBindGroup, colorView, depthView)
  }

  rebuild(): void {
    this.basePass.rebuild()
  }

  dispose(): void {
    // Nothing to do: every allocation went through the harness-owned 'scene'
    // scope, which LabApp disposes. Pipelines have no explicit destructor.
  }
}
