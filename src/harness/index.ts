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
export { DEBUG_VIEW_MODES, type DebugViewMode } from './debug.ts'
export { SPECIES, speciesById, type SpeciesDesc } from '../scene/species.ts'
export { STANDS, standById, standEntrySlots, standPlantCounts, type Stand, type StandSpecies } from '../scene/stands.ts'
export { Terrain } from '../scene/terrain.ts'
export { windGust, windSway, type WindParams } from '../scene/wind.ts'
export { parsePose, serializePose, type CameraPose } from '../scene/camera.ts'
export { CameraSpline } from '../scene/spline.ts'

export { SPECIES_BUDGET_BYTES, type VramAttr, type VramScope } from '../gpu/resources.ts'
export { ScopedTimer } from '../gpu/timing.ts'
export type { ShaderRegistry, WgslSource } from '../gpu/shaders.ts'
export { createFullscreenPipeline } from '../gpu/screenPass.ts'

/**
 * Texel semantics. ONE declaration produces BOTH the mip filter and the WGSL
 * decoder, so they cannot disagree — which is what makes the two documented
 * mip bugs (un-premultiplying already-normalised colour, box-filtering
 * octahedral normals) unrepresentable rather than merely documented. Do not
 * hand-roll a mipgen shader or a `rgb / a` in a fragment shader; declare the
 * semantics and let `generateMips` + `wgslDecoder` agree.
 */
export {
  checkMipConvention,
  createTexture2D,
  dilateWrap,
  generateMips,
  mipLevelsFor,
  octDecode,
  octEncode,
  readTexture,
  uploadImageBitmap,
  wgslDecoder,
  wgslOctahedral,
  type DilateOptions,
  type MipCheckLevel,
  type MipCheckReport,
  type MipPlan,
  type NonMippableEncoding,
  type ReadTextureOptions,
  type ReadTextureResult,
  type TexelSemantics,
  type Texture2DDesc,
  type TextureEncoding,
} from '../gpu/texture.ts'

/**
 * PNG loading/encoding, including the 16-bit path the browser cannot do:
 * canvas cannot ENCODE 16-bit PNG and `createImageBitmap` silently TRUNCATES
 * one to 8 bits, so 16-bit maps go through the hand-written codec here.
 * Every loader guards the dev server's SPA fallback (index.html at HTTP 200)
 * by content-type AND the PNG magic bytes.
 */
export {
  assertPngBytes,
  bakedImage,
  decodePng,
  encodePng,
  fetchPngBytes,
  hasPngMagic,
  loadImage16Texture,
  loadImageTexture,
  readPngHeader,
  readTextureRgba8,
  textureFromPng16,
  type BakedImageContext,
  type EncodePngInput,
  type LoadImage16Options,
  type LoadImageTextureOptions,
  type PngChannels,
  type PngHeader,
  type PngImage,
  type ReadRgba8Options,
} from '../gpu/image.ts'

export { GcMesh, parseGcMesh, type GcMeshHeader } from '../mesh/gcmesh.ts'
export { MeshCatalog, type MeshInfo } from '../mesh/catalog.ts'

export { bakedArtifact, commitBake, type BakeContext, type BakeExt } from '../bake/io.ts'

/**
 * Deploy-awareness. `assetUrl('/mesh/baked/...')` is the correct way to build
 * a runtime asset URL (identity in dev, base-prefixed in a static build);
 * HAS_DEV_SINK is false on static deployments, where nothing can be written
 * back into the repo.
 */
export { assetUrl } from '../util/paths.ts'
export { HAS_DEV_SINK, RAW_MESHES_AVAILABLE } from '../util/env.ts'
