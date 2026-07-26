/**
 * Turning the lab's generated material shader into a STANDALONE one.
 *
 * ---------------------------------------------------------------------------
 * WHAT IS ACTUALLY LAB-SPECIFIC IN A GENERATED MATERIAL
 * ---------------------------------------------------------------------------
 *
 * Exactly one file: `src/wgsl/frame.wgsl`. It declares the frozen @group(0) —
 * the frame uniform, a terrain heightmap, a linear sampler and the stand table
 * — and nothing else in the generated module depends on the lab at all.
 *
 * So the port is a SUBSTITUTION, not a rewrite: strip the inlined frame.wgsl
 * block out of the generated text and put a small documented host struct in its
 * place, keeping the name `frame` so every reference in the generated code AND
 * in the material's own authored WGSL resolves unchanged.
 *
 * Everything else is kept VERBATIM, deliberately:
 *
 *  - `light_surface()` (lighting.wgsl) is inlined, because a material like
 *    003-nd-fuzz is DEFINED by splitting it in half — it calls `light_surface`
 *    and then subtracts the sun term back out so cavity AO occludes only the
 *    indirect half. Dropping it and substituting "some diffuse model" would
 *    change the material's appearance while looking like a port.
 *  - `apply_fog()` / `sky_horizon_color()` come along with it. A material that
 *    declared `fog: true` calls them; one that did not simply carries two
 *    unused functions.
 *  - `debug_shade()` (debug.wgsl) is inlined, because the generator routes
 *    every fragment through it and there is no code path that skips it. It
 *    costs one host float (`debug_mode`, 0 = off) and buys the consumer the
 *    same albedo / normals / lighting / coverage / depth views the lab has.
 *
 * The one thing that CANNOT be carried: the generated vertex stage multiplies
 * `frame.view_proj` by the vertex position directly, because the lab's preview
 * objects are authored in world space and have no model matrix. A consumer with
 * a transform hierarchy must fold the model matrix into `view_proj` and
 * pre-transform positions/normals/tangents, or edit `vs_main`. Said out loud in
 * the bundle's README rather than papered over.
 */

/** Byte size of `MatHost` — 4 x vec4 of matrix plus four 16-byte rows. */
export const MAT_HOST_SIZE = 128

export interface HostField {
  name: string
  /** Byte offset inside `MatHost`. */
  offset: number
  type: 'mat4x4f' | 'vec3f' | 'f32'
  doc: string
}

/**
 * The whole host contract, as data — the manifest publishes this so a consumer
 * can fill the buffer without parsing WGSL.
 */
export const MAT_HOST_FIELDS: readonly HostField[] = [
  { name: 'view_proj', offset: 0, type: 'mat4x4f', doc: 'projection * view, column-major, WebGPU clip space (z in [0,1])' },
  { name: 'camera_pos', offset: 64, type: 'vec3f', doc: 'world-space eye position' },
  { name: 'time', offset: 76, type: 'f32', doc: 'seconds; only read by a material that animates' },
  { name: 'sun_dir', offset: 80, type: 'vec3f', doc: 'UNIT vector pointing FROM the surface TOWARD the sun' },
  { name: 'debug_mode', offset: 92, type: 'f32', doc: '0 off · 1 albedo · 2 normals · 3 lighting · 4 coverage · 5 depth' },
  { name: 'sun_color', offset: 96, type: 'vec3f', doc: 'linear radiance of the sun term (the lab uses 1.15, 1.02, 0.82)' },
  { name: '_host_pad0', offset: 108, type: 'f32', doc: 'padding — vec3f is 16-byte aligned in a uniform' },
  { name: 'ambient', offset: 112, type: 'vec3f', doc: 'linear ambient/hemisphere radiance (the lab uses 0.21, 0.25, 0.32)' },
  { name: '_host_pad1', offset: 124, type: 'f32', doc: 'padding' },
]

/**
 * The replacement for frame.wgsl. It keeps the variable name `frame` on purpose
 * — that is what makes the substitution textual and total, with no rewriting of
 * either generated or material-authored code.
 */
export const PORTABLE_HOST_WGSL = `// ===========================================================================
// THE HOST CONTRACT — everything this shader needs from your engine.
//
// This material was exported from the groundcover lab. Its generated code and
// its authored BRDF both reference a module-scope uniform called \`frame\`; in
// the lab that is a 352-byte struct carrying a terrain heightmap, a wind state
// and a stand table beside the camera. Here it is the ${MAT_HOST_SIZE}-byte struct below,
// and NOTHING ELSE from the lab's bind-group layout survives.
//
// Fill one uniform buffer of ${MAT_HOST_SIZE} bytes and bind it at @group(0) @binding(0).
// See material.json -> hostBindGroup for the byte offsets.
//
// NOTE ON THE VERTEX STAGE: vs_main multiplies view_proj by the vertex
// position directly — the geometry it was authored against lives in world
// space and has no model matrix. If yours does, fold it into view_proj and
// transform position/normal/tangent on the CPU, or edit vs_main.
// ===========================================================================

struct MatHost {
  /// projection * view. Column-major, WebGPU clip space (z in [0, 1]).
  view_proj: mat4x4f,
  /// World-space eye position. Used for the view vector and for fog/depth.
  camera_pos: vec3f,
  /// Seconds. Only read by a material that animates.
  time: f32,
  /// UNIT vector pointing FROM the surface TOWARD the sun.
  sun_dir: vec3f,
  /// Debug view selector: 0 off, 1 albedo, 2 normals, 3 lighting, 4 coverage,
  /// 5 depth. Set 0 for normal rendering — see debug_shade() below.
  debug_mode: f32,
  /// Linear radiance of the sun term.
  sun_color: vec3f,
  _host_pad0: f32,
  /// Linear ambient / hemisphere radiance.
  ambient: vec3f,
  _host_pad1: f32,
}

@group(0) @binding(0) var<uniform> frame: MatHost;
`

const BEGIN = (ref: string): string => `// -- begin #include "${ref}"`
const END = (ref: string): string => `// -- end #include "${ref}"`
const ALREADY = (ref: string): string => `// #include "${ref}" (already included)`

/**
 * Remove one inlined `#include` block from expanded WGSL.
 *
 * `tools/vite-plugin-wgsl.ts` wraps every inlined file in
 * `// -- begin #include "<ref>"` / `// -- end #include "<ref>"` markers and
 * replaces a repeat include with `// #include "<ref>" (already included)`. That
 * makes the expansion reversible per file, which is the whole reason this
 * function can exist — and it throws rather than guessing if the markers are
 * not there, because silently exporting a shader that still declares the lab's
 * terrain bindings would fail at the consumer's pipeline layout, far from here.
 */
export function stripInclude(code: string, ref: string): string {
  const lines = code.split('\n')
  const begin = lines.findIndex((l) => l.trim() === BEGIN(ref))
  const end = lines.findIndex((l) => l.trim() === END(ref))
  if (begin < 0 || end < begin) {
    throw new Error(
      `stripInclude: no inlined "${ref}" block in the generated WGSL. The material export substitutes the lab's ` +
        'frame uniform for a portable one by removing that block, and it relies on the include markers ' +
        'tools/vite-plugin-wgsl.ts emits. If the plugin stopped emitting them, this substitution has to change too.',
    )
  }
  const kept = [...lines.slice(0, begin), ...lines.slice(end + 1)]
  return kept
    .map((l) => (l.trim() === ALREADY(ref) ? `// (${ref} replaced by the portable MatHost above)` : l))
    .join('\n')
}

/**
 * Remove one graph node's generated code — its `n_<ident>_eval` function and,
 * if it had one, its texture declaration.
 *
 * WHY THIS EXISTS. Codegen emits a function for every REACHABLE node, and
 * reachability there does not stop at a materialized node: when `ao-soft` is a
 * baked filter over `ao-map`, both get emitted and both get bound, but the
 * fragment shader only ever samples the baked result. `ao-map` is declared,
 * bound, and never read.
 *
 * In the lab that is merely wasteful. In a bundle it is a trap, and one that
 * cost two rounds of this test to pin down:
 *
 *   1. `layout: 'auto'` derives a bind group layout from STATICALLY USED
 *      resources only, so the unsampled texture is absent from the layout while
 *      the manifest still asks the consumer to bind it. The bind group is
 *      invalid, the draw is dropped, and the exported POM/materialized variant
 *      of 002-graph-maps rendered a COMPLETELY BLACK frame — 99.96% of pixels
 *      differing from the lab.
 *   2. Removing only the declaration does not help: the dead reader function
 *      still names the identifier, and WGSL resolves names in dead code too
 *      ("unresolved value 't_ao_map'"). The function has to go with it.
 *
 * So the rule is the complete one: drop every node the fragment shader does not
 * call, which by construction leaves nothing dangling — a surviving function
 * only calls nodes that are themselves reachable from a channel.
 */
export function dropDeadNode(code: string, ident: string): string {
  let out = code.replace(new RegExp(`^@group\\(1\\) @binding\\(\\d+\\) var t_${ident}: texture_2d<f32>;\\n`, 'm'), '')
  const lines = out.split('\n')
  const start = lines.findIndex((l) => l.startsWith(`fn n_${ident}_eval(`))
  if (start >= 0) {
    // Generated node functions are flat: the first line that is exactly '}'
    // closes this one.
    let end = start
    while (end < lines.length && lines[end] !== '}') end++
    lines.splice(
      start,
      end - start + 1,
      `// (n_${ident}_eval and its texture binding removed: nothing the fragment shader calls reaches`,
      '//  this node — its only consumer was baked into another texture. Kept out so that an explicit',
      "//  bind group layout and `layout: 'auto'` agree about what has to be bound.)",
    )
    out = lines.join('\n')
  }
  return out
}

/**
 * The standalone shader: the lab's generated module with frame.wgsl swapped for
 * `PORTABLE_HOST_WGSL`, plus a provenance header.
 */
export function portableMaterialWgsl(
  generated: string,
  meta: { id: string; moduleId: string; deadNodes?: readonly string[] },
): string {
  let body = stripInclude(generated, 'src/wgsl/frame.wgsl')
  for (const ident of meta.deadNodes ?? []) body = dropDeadNode(body, ident)
  return `// GENERATED — do not hand-edit; re-export from the material inspector instead.
//
// Material "${meta.id}" (lab module ${meta.moduleId}), exported as a standalone
// WGSL module. Everything below is the shader the lab actually runs, with ONE
// substitution: the lab's frozen @group(0) (frame uniform + terrain heightmap +
// stand table) has been replaced by the small MatHost struct that follows.
//
// The shared lighting model (light_surface / apply_fog) and the debug views
// (debug_shade) are INLINED rather than dropped: a material can be defined by
// how it splits light_surface apart, so substituting a different diffuse term
// would change its look while looking like a port.
//
// Entry points: vs_main (vertex), fs_main (fragment).
// Bind groups:  @group(0) = the host contract, @group(1) = this material.

${PORTABLE_HOST_WGSL}
${body}`
}
