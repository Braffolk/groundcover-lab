/**
 * The material graph system — declaration types, resolution, WGSL codegen,
 * validation and the runtime that turns a `MaterialDef` into an `Experiment`.
 *
 * Re-exported through `@harness`; a material experiment imports from there.
 */

export {
  CHANNEL_COMPONENTS,
  DEFERRED_NODE_KINDS,
  formatIssues,
  type ChannelSpec,
  type ChannelType,
  type CombineNode,
  type CombineOp,
  type FilterNode,
  type FilterOp,
  type ImageNode,
  type IssueLevel,
  type MaterialDef,
  type MaterialGraph,
  type MaterialIssue,
  type MaterialNode,
  type MaterialNodeKind,
  type NodeBase,
  type ProceduralNode,
  type Scalar,
  type UvTransform,
  type ValidationReport,
  type VariantBlendNode,
  type ViewUvSpec,
} from './types.ts'

export {
  MATERIAL_FIRST_TEXTURE_BINDING,
  MATERIAL_SAMPLER_BINDING,
  MATERIAL_UNIFORM_BINDING,
  dependenciesOf,
  resolveGraph,
  viewUvDependencies,
  type NodeMode,
  type ResolvedGraph,
  type ResolvedNode,
  type TextureSource,
} from './resolve.ts'

export {
  generateMainModule,
  generateMaterializeModule,
  hashText,
  paramsUsedBy,
  subtreeOf,
  uniformLayout,
  wgslIdent,
  wgslParamName,
  writeUniforms,
  type GeneratedModule,
  type MaterializeModule,
} from './codegen.ts'

export { STORAGE_FORMATS, validateCompilation, validateGeneratedMain, validateMaterial } from './validate.ts'

export { MATERIALIZE_FORMAT, bakeKeyFor } from './materialize.ts'

export {
  allMaterialReports,
  materialReport,
  type MaterialReport,
  type MaterializationRecord,
} from './report.ts'

export { createMaterialExperiment, tilesPerMetre } from './runtime.ts'
