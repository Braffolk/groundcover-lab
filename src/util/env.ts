/**
 * Build-environment flags.
 *
 * The lab has two lives:
 *   - the dev server: a full read/write lab, where the dev-sink middleware
 *     (tools/vite-plugin-dev-sink.ts) writes thumbnails, goldens, ratings,
 *     bakes and bench results straight into the repo;
 *   - a static deployment (S3/CloudFront/Pages): a read-only mirror where
 *     none of those endpoints exist.
 *
 * Everything that would write into the repo must be gated on HAS_DEV_SINK so
 * a static build never even issues the request (a 404/SPA-fallback response
 * would otherwise show up as a bogus error toast).
 */

/** Injected by tools/vite-plugin-static-deploy.ts — always true in dev. */
declare const __GC_RAW_MESHES__: boolean

/** The /__bench, /__bench/list, /__thumb, /__bake, /__rating endpoints exist. */
export const HAS_DEV_SINK: boolean = import.meta.env.DEV

/**
 * Raw GCMESH1 `.bin` source meshes are reachable. True in dev; in a static
 * build only when it was made with GC_DEPLOY_MESHES=1 (they are ~357MB, so
 * they are left out by default and the mesh inspector says so).
 */
export const RAW_MESHES_AVAILABLE: boolean = __GC_RAW_MESHES__
