/**
 * THE public harness API. Experiments import ONLY from '@harness' (this
 * file) plus their own files — everything else under src/ is internal and
 * may change freely. Breaking changes here bump HARNESS_API in registry.ts.
 */

export {
  HARNESS_API,
  defineExperiment,
  defineMaterial,
  type Experiment,
  type ExperimentContext,
  type ExperimentContextCore,
  type ExperimentKind,
  type ExperimentManifest,
  type ExperimentModule,
  type ExperimentStatus,
  type FrameInfo,
  type SceneServices,
  type StandContextExt,
  type ViewTargets,
} from './registry.ts'

/**
 * MATERIALS. A material experiment is previewed on harness-owned geometry, so
 * its context carries `ctx.preview` where a renderer's carries `ctx.scene` /
 * `ctx.stand`; declare it as
 * `ExperimentContext<typeof PARAMS, MaterialContextExt>`.
 *
 * It never authors a preview mesh — two materials on two slightly different
 * spheres turn every A/B into a comparison of geometry. Draw `ctx.preview.mesh`
 * with `PREVIEW_VERTEX_LAYOUT`; the uv/tangent contract is documented in full
 * at the top of src/scene/preview.ts.
 */
export {
  DEFAULT_PREVIEW_OBJECT,
  PREVIEW_OBJECTS,
  PREVIEW_VERTEX_FLOATS,
  PREVIEW_VERTEX_LAYOUT,
  PREVIEW_VERTEX_STRIDE,
  isPreviewObjectId,
  type MaterialContextExt,
  type PreviewMesh,
  type PreviewObjectId,
  type PreviewScene,
} from '../scene/preview.ts'

/**
 * THE MATERIAL GRAPH. A material's CHANNEL DATA is a declarative typed graph
 * (`MaterialGraph`) that the runtime resolves, executes, validates and bakes;
 * its BEHAVIOUR — the view-uv stage and the BRDF — is authored WGSL with the
 * fixed signatures in src/wgsl/material.wgsl.
 *
 * `createMaterialExperiment(ctx, def)` IS the material: it builds the graph,
 * loads or materializes every texture, generates the WGSL (routing the
 * fragment through `debug_shade` so a material cannot skip the debug views),
 * validates it and returns an `Experiment`.
 *
 * Node kinds are FROZEN at five — image, procedural, filter, combine,
 * variantBlend. `meshCapture` is deliberately absent: the source meshes it
 * would capture were deleted and images are the data source now.
 */
export {
  CHANNEL_COMPONENTS,
  DEFERRED_NODE_KINDS,
  MATERIALIZE_FORMAT,
  MATERIAL_FIRST_TEXTURE_BINDING,
  MATERIAL_SAMPLER_BINDING,
  MATERIAL_UNIFORM_BINDING,
  allMaterialReports,
  bakeKeyFor,
  createMaterialExperiment,
  dependenciesOf,
  formatIssues,
  generateMainModule,
  generateMaterializeModule,
  hashText,
  materialReport,
  paramsUsedBy,
  resolveGraph,
  subtreeOf,
  tilesPerMetre,
  uniformLayout,
  validateGeneratedMain,
  validateMaterial,
  wgslIdent,
  wgslParamName,
  writeUniforms,
  type ChannelSpec,
  type ChannelType,
  type CombineNode,
  type CombineOp,
  type FilterNode,
  type FilterOp,
  type GeneratedModule,
  type ImageNode,
  type IssueLevel,
  type MaterialDef,
  type MaterialGraph,
  type MaterialIssue,
  type MaterialNode,
  type MaterialNodeKind,
  type MaterialReport,
  type MaterializationRecord,
  type NodeMode,
  type ProceduralNode,
  type ResolvedGraph,
  type ResolvedNode,
  type Scalar,
  type TextureSource,
  type UvTransform,
  type ValidationReport,
  type VariantBlendNode,
  type ViewUvSpec,
} from '../material/index.ts'

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
  wgslEncoder,
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
