#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"

// ===========================================================================
// THE MATERIAL ABI — the fixed structs a graph-authored material is compiled
// against. Everything here is HAND-WRITTEN and stable; the generator pastes
// its declarations, node functions and entry points around it.
//
// Do NOT `#include` this file (or lighting/debug/frame) from a material's own
// surface/shade WGSL: the generator concatenates all of them into one module
// and the include mechanism has no idea it already emitted them. The validator
// rejects a stage source that redefines a shared symbol, with that message.
// ===========================================================================

/// Everything the surface knows about ITSELF, before any channel is read.
///
/// `geo_n` is the GEOMETRY normal and is kept separate from the shading normal
/// on purpose: ND moss's light wrap and its AO-Fresnel are *defined* by using
/// the geometry normal where a normal map would be the obvious choice, and a
/// contract that only carries the shading normal makes that material
/// inexpressible.
///
/// `ddx`/`ddy` are the derivatives of the UNPERTURBED material uv, always. A
/// parallax stage offsets `uv` per fragment, which makes the implicit
/// derivative the HEIGHTMAP's gradient rather than the surface's; left to pick
/// its own mip the sampler jumps to near the top of the chain at grazing
/// angles and the far field turns into flat paint. Carrying them here means a
/// node cannot accidentally take the implicit ones.
struct Geo {
  /// World-space position of the fragment.
  world: vec3f,
  /// GEOMETRY normal (interpolated vertex normal), unit.
  geo_n: vec3f,
  /// Surface tangent along +u, re-orthogonalised against geo_n.
  tangent: vec3f,
  /// cross(geo_n, tangent) * handedness, along +v.
  bitangent: vec3f,
  /// Unit vector from the fragment TOWARD the camera.
  view_ws: vec3f,
  /// Mesh uv, in METRES OF SURFACE (see src/scene/preview.ts). Unperturbed.
  uv0: vec2f,
  /// Material uv (uv0 through the uv transform), in TILE units. The view-uv
  /// stage may offset this; `uv0` never moves.
  uv: vec2f,
  /// Derivatives of the UNPERTURBED material uv.
  ddx: vec2f,
  ddy: vec2f,
  /// The metre rectangle the mesh's uv actually covers. A displacing method
  /// tests its offset uv against THIS: leaving it means the ray left the
  /// patch, as opposed to merely leaving one tile.
  uv_min: vec2f,
  uv_max: vec2f,
  /// The uv transform: tiles per metre, and rotation in radians. A silhouette
  /// POM has to invert these to bring its material-uv offset back into metres
  /// before testing it against uv_min/uv_max — see mat_uv_to_surface.
  uv_scale: vec2f,
  uv_rot: f32,
}

/// What every graph node is evaluated against. Deliberately smaller than Geo:
/// a node that a material wants to MATERIALIZE into a texture is evaluated
/// off-screen, where `world`, `geo_n` and `view_ws` have no meaningful value.
/// The validator refuses to materialize a node whose subtree reads them.
struct EvalCtx {
  /// Material uv, in tile units.
  uv: vec2f,
  /// Unperturbed derivatives of `uv`.
  ddx: vec2f,
  ddy: vec2f,
  /// NOT AVAILABLE in a materialized node (zeroed there).
  world: vec3f,
  geo_n: vec3f,
  view_ws: vec3f,
}

fn mat_eval_ctx(g: Geo) -> EvalCtx {
  var c: EvalCtx;
  c.uv = g.uv;
  c.ddx = g.ddx;
  c.ddy = g.ddy;
  c.world = g.world;
  c.geo_n = g.geo_n;
  c.view_ws = g.view_ws;
  return c;
}

/// The EvalCtx a materialize pass builds from the DESTINATION texel: uv at the
/// texel centre, gradients of exactly one texel, and no view/world at all.
fn mat_eval_ctx_texel(texel: vec2u, dims: vec2u) -> EvalCtx {
  let d = vec2f(dims);
  var c: EvalCtx;
  c.uv = (vec2f(texel) + vec2f(0.5)) / d;
  c.ddx = vec2f(1.0 / d.x, 0.0);
  c.ddy = vec2f(0.0, 1.0 / d.y);
  c.world = vec3f(0.0);
  c.geo_n = vec3f(0.0, 1.0, 0.0);
  c.view_ws = vec3f(0.0, 1.0, 0.0);
  return c;
}

/// What the view-uv stage returns. `clip` is the SPOM escape: the march left
/// the mesh's uv rectangle, so this fragment is not on the surface at all.
struct ViewUv {
  uv: vec2f,
  clip: bool,
  /// Self-shadow along the light, 1 = unshadowed. 1.0 when not computed.
  shadow: f32,
  /// Depth to write, in [0,1] clip space. Only used when `pdo` is on.
  depth: f32,
  /// True when `depth` was actually computed.
  has_depth: bool,
};

/// The open surface description stage 4 assembles. A material fills the slots
/// it has and leaves the rest at their defaults (mat_surface_default).
struct Surface {
  albedo: vec3f,
  /// Shading normal, world space, unit.
  normal: vec3f,
  /// GEOMETRY normal — kept beside `normal` deliberately (see Geo).
  geo_normal: vec3f,
  /// Ambient occlusion / cavity, 1 = unoccluded.
  ao: f32,
  /// Height above the base plane, in the material's own units.
  height: f32,
  roughness: f32,
  /// Fragment coverage, 1 = opaque. Reported to debug_shade.
  opacity: f32,
  emissive: vec3f,
  sheen: vec3f,
  /// Colour of light that came back OUT of the material's interior — the tint
  /// for wrapped/subsurface diffuse and for any bounce term added back where
  /// occlusion goes dark. Distinct from `sheen` (the surface-fibre tint at the
  /// tips) and from `emissive` (light the material generates by itself): a
  /// fuzzy material needs BOTH tints, and the first port of the Uncharted 4
  /// moss had to smuggle this one through `emissive`, which was a lie in the
  /// shader that nobody reading `Surface` could have caught.
  subsurface: vec3f,
  thickness: f32,
};

fn mat_surface_default(g: Geo) -> Surface {
  var s: Surface;
  s.albedo = vec3f(0.5);
  s.normal = g.geo_n;
  s.geo_normal = g.geo_n;
  s.ao = 1.0;
  s.height = 0.0;
  s.roughness = 1.0;
  s.opacity = 1.0;
  s.emissive = vec3f(0.0);
  s.sheen = vec3f(0.0);
  s.subsurface = vec3f(0.0);
  s.thickness = 0.0;
  return s;
}

/// What stage 5 returns. The generator routes exactly these four values into
/// debug_shade(), which is why a graph material cannot skip the debug views.
struct Shaded {
  color: vec3f,
  albedo: vec3f,
  normal: vec3f,
  coverage: f32,
};

fn mat_shaded_default(s: Surface) -> Shaded {
  var sh: Shaded;
  sh.color = vec3f(0.0);
  sh.albedo = s.albedo;
  sh.normal = s.normal;
  sh.coverage = s.opacity;
  return sh;
}

// ---------------------------------------------------------------------------
// The uv transform, and its inverse
// ---------------------------------------------------------------------------

fn mat_rot2(v: vec2f, a: f32) -> vec2f {
  let c = cos(a);
  let s = sin(a);
  return vec2f(v.x * c - v.y * s, v.x * s + v.y * c);
}

/// Mesh uv (metres) -> material uv (tile units).
fn mat_uv_forward(uv0: vec2f, scale: vec2f, rot: f32) -> vec2f {
  return mat_rot2(uv0, rot) * scale;
}

/// A DELTA in material uv -> the same delta in mesh uv (metres). This is the
/// inverse a silhouette POM needs: it marches in tile units, then has to ask
/// whether the offset left the MESH's uv rectangle, which is in metres.
fn mat_uv_to_surface(d_uv: vec2f, scale: vec2f, rot: f32) -> vec2f {
  return mat_rot2(d_uv / max(scale, vec2f(1.0e-6)), -rot);
}

/// True when `uv0` is outside the mesh's uv rectangle.
fn mat_uv_outside(uv0: vec2f, uv_min: vec2f, uv_max: vec2f) -> bool {
  return any(uv0 < uv_min) || any(uv0 > uv_max);
}

/// The view direction expressed in the material's uv frame: xy is the
/// horizontal travel per unit of DOWNWARD depth, z is the cosine against the
/// geometry normal. Used by every parallax/POM variant.
fn mat_view_tangent(g: Geo) -> vec3f {
  let v = g.view_ws;
  let t = g.tangent;
  let b = g.bitangent;
  let n = g.geo_n;
  // View vector in tangent space, then into material-uv units.
  let vt = vec3f(dot(v, t), dot(v, b), dot(v, n));
  return vec3f(mat_uv_forward(vt.xy, g.uv_scale, g.uv_rot), vt.z);
}
