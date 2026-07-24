/**
 * THE public harness API. Experiments import ONLY from '@harness' (this
 * file) plus their own files — everything else under src/ is internal and
 * may change freely. Breaking changes here bump HARNESS_API in registry.ts.
 */

export {
  HARNESS_API,
  defineExperiment,
  type Experiment,
  type ExperimentContext,
  type ExperimentManifest,
  type ExperimentModule,
  type ExperimentStatus,
  type FrameInfo,
  type SceneServices,
  type ViewTargets,
} from './registry.ts'

export {
  p,
  paramDefaults,
  paramsHash,
  type BooleanParam,
  type EnumParam,
  type NumberParam,
  type ParamDef,
  type ParamSchema,
  type ParamValues,
} from './params.ts'

export { asU32, hash2, hash3, hash4, hashF32, pcg } from '../scene/hash.ts'
export {
  Scatter,
  SCATTER_CELL_SIZE,
  SCATTER_INSTANCE_FLOATS,
  SCATTER_MAX_DENSITY,
  SCATTER_MAX_PER_CELL,
  type Aabb2,
  type ScatterPoint,
} from '../scene/scatter.ts'
export { SPECIES, speciesById, type SpeciesDesc } from '../scene/species.ts'
export { STANDS, standById, standPlantCounts, type Stand, type StandSpecies } from '../scene/stands.ts'
export { Terrain } from '../scene/terrain.ts'
export { windGust, windSway, type WindParams } from '../scene/wind.ts'
export { parsePose, serializePose, type CameraPose } from '../scene/camera.ts'
export { CameraSpline } from '../scene/spline.ts'

export { SPECIES_BUDGET_BYTES, type VramAttr, type VramScope } from '../gpu/resources.ts'
export { ScopedTimer } from '../gpu/timing.ts'
export type { ShaderRegistry, WgslSource } from '../gpu/shaders.ts'
export { createFullscreenPipeline } from '../gpu/screenPass.ts'

export { GcMesh, parseGcMesh, type GcMeshHeader } from '../mesh/gcmesh.ts'
export { MeshCatalog, type MeshInfo } from '../mesh/catalog.ts'

export { bakedArtifact, commitBake, type BakeContext } from '../bake/io.ts'

/**
 * Deploy-awareness. `assetUrl('/mesh/baked/...')` is the correct way to build
 * a runtime asset URL (identity in dev, base-prefixed in a static build);
 * HAS_DEV_SINK is false on static deployments, where nothing can be written
 * back into the repo.
 */
export { assetUrl } from '../util/paths.ts'
export { HAS_DEV_SINK, RAW_MESHES_AVAILABLE } from '../util/env.ts'
