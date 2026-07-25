import type { CameraPose } from '../scene/camera.ts'
import type { CameraMatrices } from '../scene/camera.ts'
import type { CameraSpline } from '../scene/spline.ts'
import type { MaterialContextExt, PreviewObjectId } from '../scene/preview.ts'
import type { Scatter } from '../scene/scatter.ts'
import type { SpeciesDesc } from '../scene/species.ts'
import type { Stand } from '../scene/stands.ts'
import type { Terrain } from '../scene/terrain.ts'
import type { WindParams } from '../scene/wind.ts'
import type { VramScope } from '../gpu/resources.ts'
import type { ScopedTimer } from '../gpu/timing.ts'
import type { ShaderRegistry } from '../gpu/shaders.ts'
import type { MeshCatalog } from '../mesh/catalog.ts'
import { assetUrl } from '../util/paths.ts'
import type { ParamSchema, ParamValues } from './params.ts'

/** Bump ONLY on breaking changes to the experiment contract. */
export const HARNESS_API = 1

export type ExperimentStatus = 'idea' | 'wip' | 'working' | 'best' | 'abandoned' | 'reference'

/**
 * What sort of experiment this is. The kind selects the Stage (what scene the
 * harness builds around it) and which rules in CLAUDE.md bind the author.
 * Additive: an absent `kind` means 'renderer', so every pre-existing manifest
 * keeps working untouched.
 */
export type ExperimentKind = 'renderer' | 'material'

export interface ExperimentManifest<S extends ParamSchema = ParamSchema, X = StandContextExt> {
  /**
   * Unique id. For a top-level experiment this is the directory name; for a
   * nested one (materials/mosses/sphagnum/003-nd-fuzz) it is the LAST path
   * segment. The id — not the path — is what ratings, bench results and
   * goldens are keyed by, so a taxonomy reshuffle never orphans them.
   */
  id: string
  title: string
  description: string
  status: ExperimentStatus
  /** Defaults to 'renderer'. */
  kind?: ExperimentKind
  harnessApi: number
  /**
   * Species ids rendered — each gets a 25MB VRAM budget row in the HUD.
   *
   * A MATERIAL has no species, and nothing here is renamed for it: a material
   * lists its own experiment id and tags its `ctx.res` allocations with the
   * same string, so the existing budget bar keeps working unchanged and one
   * row means "this material's maps". (25MB was calibrated for a plant
   * species' baked representation, not for a measured 2048² map set — see the
   * VRAM section of a material's NOTES.md before reading the bar as a verdict.)
   */
  species: string[]
  params: S
  /** Optional extra camera bookmarks (serialized poses). */
  cams?: Record<string, string>
  /**
   * MATERIAL ONLY — which preview object to open on (`sphere` by default). The
   * URL `obj=` wins over it, and the toolbar picker changes it live.
   */
  previewObject?: PreviewObjectId
  /** Bump to invalidate this experiment's bake caches. */
  bakeVersion?: number
  load: () => Promise<ExperimentModule<S, X>>
}

export interface ExperimentModule<S extends ParamSchema = ParamSchema, X = StandContextExt> {
  create(ctx: ExperimentContext<S, X>): Experiment | Promise<Experiment>
}

export interface SceneServices {
  terrain: Terrain
  scatter: Scatter
  wind: WindParams
  species: readonly SpeciesDesc[]
  bookmarks: Record<string, CameraPose>
  splines: Record<string, CameraSpline>
}

/**
 * What the `stand` stage merges into an experiment's context — declared here,
 * beside SceneServices, so registry.ts never has to import from stage.ts.
 */
export interface StandContextExt {
  scene: SceneServices
  /**
   * The active stand — THE definition of what grows (species, densities,
   * scales, sway, region). Render exactly its plants via ctx.scene.scatter
   * (or the WGSL twin); experiments never define placement themselves.
   */
  stand: Stand
}

export interface FrameInfo {
  frameIndex: number
  time: number
  dt: number
  /** True in bench/diff/capture — derive ALL animation from `time` only. */
  deterministic: boolean
  camera: { pose: Readonly<CameraPose> } & CameraMatrices
  wind: WindParams
}

export interface ViewTargets {
  /** Base pass (terrain+sky) already drawn — use loadOp 'load'. */
  colorView: GPUTextureView
  /** Shared scene depth — loadOp 'load'; sample it via depthReadOnly passes. */
  depthView: GPUTextureView
  depthTexture: GPUTexture
  width: number
  height: number
  view: 'solo' | 'A' | 'B'
}

/**
 * Everything an experiment gets regardless of kind. The stage-specific half
 * (`ctx.scene`/`ctx.stand` for a renderer, `ctx.preview` for a material) is
 * merged in on top — see `ExperimentContext`.
 */
export interface ExperimentContextCore<S extends ParamSchema = ParamSchema> {
  /** This experiment's id (directory name) — use it in labels and bake keys. */
  id: string
  device: GPUDevice
  colorFormat: GPUTextureFormat
  depthFormat: GPUTextureFormat
  /** Shared @group(0) — bind `frame.bindGroup`, include src/wgsl/frame.wgsl. */
  frame: { layout: GPUBindGroupLayout; bindGroup: GPUBindGroup }
  /**
   * VRAM-tracked allocator scoped to this experiment. Pass `{ species }` — a
   * plant species id for a renderer, the experiment's own id for a material
   * (see `ExperimentManifest.species`).
   */
  res: VramScope
  shaders: ShaderRegistry
  /** Pass factory — all passes are auto-timed and labeled per view. */
  timing: ScopedTimer
  /** Live values; mutated by the harness UI. See onParamsChanged. */
  params: ParamValues<S>
  /** Placement seed (URL `seed`). Feed it to scatter / the shared hash. */
  seed: number
  meshes: MeshCatalog
  size(): { width: number; height: number }
  /**
   * Report long setup work (fraction 0..1, plus a short note) while `create()`
   * is still running — the runner shows it as a status line. A multi-minute
   * bake otherwise blocks the page with nothing on screen. Pass it straight
   * through as `BakeContext.onProgress`. A no-op where the host provides none.
   */
  progress(fraction: number, note?: string): void
}

/**
 * The context an experiment's `create()` receives: the core above plus the
 * active stage's extension.
 *
 * `X` defaults to `StandContextExt`, so every renderer keeps writing
 * `ExperimentContext<typeof PARAMS>` and keeps getting `ctx.scene` and
 * `ctx.stand`. A material writes
 * `ExperimentContext<typeof PARAMS, MaterialContextExt>` and gets `ctx.preview`
 * instead — and, correctly, no `ctx.stand`, because there is no stand.
 */
export type ExperimentContext<
  S extends ParamSchema = ParamSchema,
  X = StandContextExt,
> = ExperimentContextCore<S> & X

export interface Experiment {
  /** CPU work + uniform uploads. Called once per frame before encoding. */
  update?(frame: FrameInfo): void
  /** Append compute/render passes via ctx.timing. Base pass is already encoded. */
  encode(enc: GPUCommandEncoder, frame: FrameInfo, targets: ViewTargets): void
  resize?(width: number, height: number): void
  onParamsChanged?(keys: ReadonlySet<string>): void
  dispose(): void
}

/**
 * Declare a RENDERER (the default kind). `X` exists so the same helper can
 * declare any kind explicitly, but a material should use `defineMaterial`,
 * which infers the context for you.
 */
export function defineExperiment<S extends ParamSchema, X = StandContextExt>(
  manifest: ExperimentManifest<S, X>,
): ExperimentManifest<S, X> {
  return manifest
}

/**
 * Declare a MATERIAL. Sets `kind: 'material'` (so it cannot be forgotten, and
 * the route therefore always builds the material stage) and types `create`'s
 * context as core + `ctx.preview` — with no `ctx.stand`, because there is no
 * stand.
 */
export function defineMaterial<S extends ParamSchema>(
  manifest: Omit<ExperimentManifest<S, MaterialContextExt>, 'kind'>,
): ExperimentManifest<S, MaterialContextExt> {
  return { ...manifest, kind: 'material' }
}

// ---------------------------------------------------------------------------
// Discovery — import.meta.glob over experiment manifests. No central registry
// file: dropping a folder into experiments/ is the entire registration, so
// parallel agents never touch a shared file. A broken manifest becomes an
// error entry (greyed card), never a crash.
// ---------------------------------------------------------------------------

export interface RegistryEntry {
  id: string
  /**
   * Repo-relative directory, e.g. `experiments/001-billboard-smoke` or
   * `experiments/materials/mosses/sphagnum/003-nd-fuzz`. Every per-experiment
   * asset URL derives from THIS, not from the id — the two stopped being the
   * same thing when experiments gained a hierarchy.
   */
  dir: string
  /**
   * Path segments between `experiments/` and the experiment's own directory,
   * e.g. `['materials', 'mosses', 'sphagnum']`. Empty for a top-level
   * experiment. This is the grouping the browser will render as a tree.
   */
  taxonomy: readonly string[]
  kind: ExperimentKind
  manifest: ExperimentManifest | null
  /** Present when the manifest failed to load or is incompatible. */
  error?: string
  thumbnailUrl?: string
}

// Nested so a hierarchy (materials/<class>/<subject>/<attempt>) is discoverable.
// Directories starting with `_` are skipped at ANY depth, which is what keeps
// `_template` out.
const manifestLoaders = import.meta.glob('/experiments/**/manifest.ts')

/** Path segments between `experiments/` and `manifest.ts`. */
function segmentsOf(path: string): string[] {
  return path.split('/').slice(2, -1)
}

export async function discoverExperiments(): Promise<RegistryEntry[]> {
  const entries: RegistryEntry[] = []
  for (const [path, loader] of Object.entries(manifestLoaders)) {
    const segments = segmentsOf(path)
    if (segments.length === 0 || segments.some((s) => s.startsWith('_'))) continue
    const id = segments[segments.length - 1]!
    const dir = `experiments/${segments.join('/')}`
    const entry: RegistryEntry = {
      id,
      dir,
      taxonomy: segments.slice(0, -1),
      kind: 'renderer',
      manifest: null,
      // Conventional URL, not a glob import — the dev server serves it directly,
      // so fresh captures appear without rebundling (cards fall back on 404).
      thumbnailUrl: assetUrl(`/${dir}/thumbnail.png`),
    }
    entries.push(entry)
    try {
      const mod = (await loader()) as { default?: ExperimentManifest }
      const manifest = mod.default
      if (!manifest) throw new Error('manifest.ts has no default export')
      if (manifest.id !== id) throw new Error(`manifest id "${manifest.id}" != directory name "${id}"`)
      if (manifest.harnessApi !== HARNESS_API) {
        throw new Error(`built for harness API ${manifest.harnessApi}, current is ${HARNESS_API}`)
      }
      entry.manifest = manifest
      entry.kind = manifest.kind ?? 'renderer'
    } catch (err) {
      entry.error = err instanceof Error ? err.message : String(err)
    }
  }
  const duplicates = new Map<string, string[]>()
  for (const e of entries) duplicates.set(e.id, [...(duplicates.get(e.id) ?? []), e.dir])
  for (const [id, dirs] of duplicates) {
    if (dirs.length < 2) continue
    // Ids key ratings, bench results and goldens, so a collision silently mixes
    // two experiments' history. Surface it as a broken entry rather than a merge.
    for (const e of entries.filter((x) => x.id === id)) {
      e.manifest = null
      e.error = `duplicate experiment id "${id}" in ${dirs.join(' and ')} — ids must be unique across the whole tree`
    }
  }
  return entries.sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }))
}

export async function findExperiment(id: string): Promise<RegistryEntry> {
  const all = await discoverExperiments()
  const entry = all.find((e) => e.id === id)
  if (!entry) throw new Error(`unknown experiment "${id}"`)
  if (entry.error) throw new Error(`experiment "${id}": ${entry.error}`)
  return entry
}
