#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"

// NOTE: main.ts PREPENDS three generated decoders to this source before
// compiling it — albedo_map_*, normal_map_* and ao_map_*, emitted by
// wgslDecoder() from the very same TexelSemantics declarations that
// generateMips() built the mip chains from. That is the point of the pairing:
// the filter and the decoder cannot disagree, so neither of the two documented
// mip bugs (un-premultiplying already-normalised colour, box-filtering
// octahedral normals) is expressible here.

struct MapParams {
  /**
   * TILES PER METRE. The preview uv arrives in metres of surface (see
   * src/scene/preview.ts), so this is one isotropic number written into both
   * lanes — not a per-axis fit to the object's parametrisation.
   */
  uv_repeat: vec2f,
  normal_strength: f32,
  ao_strength: f32,
}

@group(1) @binding(0) var<uniform> maps: MapParams;
@group(1) @binding(1) var map_sampler: sampler;
@group(1) @binding(2) var albedo_tex: texture_2d<f32>;
@group(1) @binding(3) var normal_tex: texture_2d<f32>;
@group(1) @binding(4) var ao_tex: texture_2d<f32>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) tangent: vec3f,
  @location(3) bitangent: vec3f,
  @location(4) uv: vec2f,
}

// Vertex layout is PREVIEW_VERTEX_LAYOUT (src/scene/preview.ts). The preview
// objects live in world space already, so there is no model matrix: the
// tangent frame arrives world-space and needs no transform either.
@vertex
fn vs_main(
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) tangent: vec4f,
  @location(3) uv: vec2f,
) -> VOut {
  var out: VOut;
  out.pos = frame.view_proj * vec4f(position, 1.0);
  out.world = position;
  out.normal = normal;
  out.tangent = tangent.xyz;
  // tangent.w is the handedness the geometry builder measured from its own
  // analytic dp/dv — never a hand-typed constant.
  out.bitangent = cross(normal, tangent.xyz) * tangent.w;
  out.uv = uv;
  return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let n_geo = normalize(in.normal);
  // Re-orthogonalise after interpolation; across a curved patch the
  // interpolated T drifts out of the tangent plane.
  let t = normalize(in.tangent - n_geo * dot(n_geo, in.tangent));
  let b = normalize(in.bitangent - n_geo * dot(n_geo, in.bitangent));

  let uv = in.uv * maps.uv_repeat;
  // Explicit gradients of the SURFACE uv. Nothing offsets the uv here, so
  // these equal the implicit ones — but this is the signature a parallax
  // material must use, and passing them from the start means the first
  // material that does offset the uv cannot forget.
  let ddx_uv = dpdx(uv);
  let ddy_uv = dpdy(uv);

  // Straight LINEAR colour. The map is measured, not authored: its bytes are
  // the linear albedo (mean 0.304, 0.586, 0.147 == measured.json meanColor),
  // so it is loaded as rgba8unorm and there is no sRGB decode anywhere.
  let albedo = albedo_map_read(albedo_tex, map_sampler, uv, ddx_uv, ddy_uv).rgb;

  // The measured normal map is MESH-FRAME, not the usual tangent-space
  // convention: rgb = mesh (x, y, z), upper hemisphere, and the bake laid the
  // tile out with +u along mesh +x and +v along mesh +z. So the axis order is
  // (T, N, B) — the map's GREEN channel is the surface normal, not its blue.
  // Reading it as (T, B, N) tilts every fragment ~90 degrees and looks, at a
  // glance, merely "wrong-ish".
  let m = normal_map_read(normal_tex, map_sampler, uv, ddx_uv, ddy_uv).xyz;
  let n_map = normalize(t * m.x + n_geo * m.y + b * m.z);
  let n = normalize(mix(n_geo, n_map, maps.normal_strength));

  let ao = ao_map_read(ao_tex, map_sampler, uv, ddx_uv, ddy_uv).r;
  let occlusion = mix(1.0, ao, maps.ao_strength);

  // Occlusion multiplies the whole light term, which keeps debug_shade's
  // LIGHTING view exact: shaded / albedo is (sun + ambient*hemi) * occlusion
  // and nothing else.
  let shaded = light_surface(albedo, n, in.world) * occlusion;

  // No fog: a studio backdrop has no atmosphere, and at ~1 m the shared fog
  // term is 1.6e-5 anyway — applying it would only tint the object toward a
  // sky colour that is not in this scene.
  // Coverage is honestly 1.0: flat maps on an opaque surface cover every
  // fragment they are asked about. A silhouette material is where that stops
  // being true.
  return vec4f(debug_shade(shaded, albedo, n, 1.0, in.world), 1.0);
}
