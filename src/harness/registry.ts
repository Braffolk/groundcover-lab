import type { CameraPose } from '../scene/camera.ts'
import type { CameraMatrices } from '../scene/camera.ts'
import type { CameraSpline } from '../scene/spline.ts'
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

export interface ExperimentManifest<S extends ParamSchema = ParamSchema> {
  /** Must equal the directory name under experiments/. */
  id: string
  title: string
  description: string
  status: ExperimentStatus
  harnessApi: number
  /** Species ids rendered — each gets a 25MB VRAM budget row. */
  species: string[]
  params: S
  /** Optional extra camera bookmarks (serialized poses). */
  cams?: Record<string, string>
  /** Bump to invalidate this experiment's bake caches. */
  bakeVersion?: number
  load: () => Promise<ExperimentModule<S>>
}

export interface ExperimentModule<S extends ParamSchema = ParamSchema> {
  create(ctx: ExperimentContext<S>): Experiment | Promise<Experiment>
}

export interface SceneServices {
  terrain: Terrain
  scatter: Scatter
  wind: WindParams
  species: readonly SpeciesDesc[]
  bookmarks: Record<string, CameraPose>
  splines: Record<string, CameraSpline>
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

export interface ExperimentContext<S extends ParamSchema = ParamSchema> {
  /** This experiment's id (directory name) — use it in labels and bake keys. */
  id: string
  device: GPUDevice
  colorFormat: GPUTextureFormat
  depthFormat: GPUTextureFormat
  scene: SceneServices
  /** Shared @group(0) — bind `frame.bindGroup`, include src/wgsl/frame.wgsl. */
  frame: { layout: GPUBindGroupLayout; bindGroup: GPUBindGroup }
  /** VRAM-tracked allocator scoped to this experiment. Pass { species }. */
  res: VramScope
  shaders: ShaderRegistry
  /** Pass factory — all passes are auto-timed and labeled per view. */
  timing: ScopedTimer
  /** Live values; mutated by the harness UI. See onParamsChanged. */
  params: ParamValues<S>
  /** Placement seed (URL `seed`). Feed it to scatter / the shared hash. */
  seed: number
  /**
   * The active stand — THE definition of what grows (species, densities,
   * scales, sway, region). Render exactly its plants via ctx.scene.scatter
   * (or the WGSL twin); experiments never define placement themselves.
   */
  stand: Stand
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

export interface Experiment {
  /** CPU work + uniform uploads. Called once per frame before encoding. */
  update?(frame: FrameInfo): void
  /** Append compute/render passes via ctx.timing. Base pass is already encoded. */
  encode(enc: GPUCommandEncoder, frame: FrameInfo, targets: ViewTargets): void
  resize?(width: number, height: number): void
  onParamsChanged?(keys: ReadonlySet<string>): void
  dispose(): void
}

export function defineExperiment<S extends ParamSchema>(manifest: ExperimentManifest<S>): ExperimentManifest<S> {
  return manifest
}

// ---------------------------------------------------------------------------
// Discovery — import.meta.glob over experiment manifests. No central registry
// file: dropping a folder into experiments/ is the entire registration, so
// parallel agents never touch a shared file. A broken manifest becomes an
// error entry (greyed card), never a crash.
// ---------------------------------------------------------------------------

export interface RegistryEntry {
  id: string
  manifest: ExperimentManifest | null
  /** Present when the manifest failed to load or is incompatible. */
  error?: string
  thumbnailUrl?: string
}

const manifestLoaders = import.meta.glob('/experiments/*/manifest.ts')

function dirOf(path: string): string {
  return path.split('/')[2]!
}

export async function discoverExperiments(): Promise<RegistryEntry[]> {
  const entries: RegistryEntry[] = []
  for (const [path, loader] of Object.entries(manifestLoaders)) {
    const id = dirOf(path)
    if (id.startsWith('_')) continue
    // Conventional URL, not a glob import — the dev server serves it directly,
    // so fresh captures appear without rebundling (cards fall back on 404).
    const entry: RegistryEntry = { id, manifest: null, thumbnailUrl: assetUrl(`/experiments/${id}/thumbnail.png`) }
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
    } catch (err) {
      entry.error = err instanceof Error ? err.message : String(err)
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
