#include "src/wgsl/scatter.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./carpet_common.wgsl"

// Carpet (mat) rendering for a periodic community tile — the bog's Sphagnum.
//
// A cushion 0.18 m across and 0.07-0.09 m tall is not a specimen and not a flat
// surface: it is intricate, low, and periodic. The LDI's 4 side captures see it
// at 20 degrees of elevation, where it is a sliver, so the carpet path throws
// them away and uses ONLY the top capture — which for this shape is the good
// asset: 0.84 mm/texel albedo + coverage + per-texel normal + per-texel capture
// DEPTH, i.e. a displacement map of the cushion surface.
//
// Per tile: one ground-parallel quad exactly one grid step across, terrain
// conformed at its (shared) corners, and per fragment the baked depth is turned
// into a real world height that goes into frag_depth. That is the whole point
// of using an LDI on a mat rather than a billboard: the mat gets thickness —
// capitula occlude each other and interpenetrate their neighbours' cushions
// through the ordinary depth test, instead of being a texture painted on a
// plane. A one-tap parallax reprojection (sample depth, shift uv by the height
// difference along the view ray, resample) puts the texels near their true
// screen position at oblique angles.
//
// NEAR (< tune.x): 2 quads — peel layer 0 (cushion surface) and layer 1 (~2 cm
// inside), so the gaps between capitula show interior moss instead of peat.
// Per-texel frag_depth.
// FAR: 1 quad, layer 0, rasterizer depth only, so early-z survives where the
// mat covers the whole screen. Mip-filtered coverage plus a carpet-specific
// (low) alpha reference keeps the distant mat a solid occluder — the opposite
// of dithering it away.

@group(1) @binding(0) var<uniform> uni: CarpetU;
@group(1) @binding(1) var<storage, read> instances: array<u32>;
@group(1) @binding(2) var carpet_albedo: texture_2d_array<f32>;
@group(1) @binding(3) var carpet_aux: texture_2d_array<f32>;
@group(1) @binding(4) var carpet_samp: sampler;

struct COut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,                   // card-plane point, terrain conformed
  @location(1) uv: vec2f,                      // tile uv (periodic, repeat-addressed)
  @location(2) up_ws: vec3f,                   // ground up, interpolated from the corners
  @location(3) @interpolate(flat) fdat: vec4f, // fade, layer, yaw, plane height (m)
}

fn degenerate() -> COut {
  var out: COut;
  out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
  return out;
}

fn build_tile(ii: u32, layer: u32, corner_i: u32) -> COut {
  let t = carpet_unpack(instances[ii], vec2i(i32(uni.region.x), i32(uni.region.y)), uni.grid.x);

  // Region-edge fade only. NO camera-inside fade: a mat you are standing on
  // must not open a hole under you. Measured from the tile centre, never per
  // vertex — a per-vertex fade would emit some corners at the clip point and
  // rasterize the quad as a sliver metres long.
  let d_xz = distance(t.node, frame.camera_pos.xz);
  let fade = 1.0 - smoothstep(uni.region.w * 0.88, uni.region.w * 0.995, d_xz);
  if (fade < 0.01) { return degenerate(); }

  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  let c = corners[corner_i];

  // The quad is the tile's own footprint: half width = half the grid step (times
  // the uniform overscale, default 1). Rotation is the scatter's 90-degree
  // quadrant and the scale is the same for every tile of the species, so the
  // lattice invariant holds: neighbours still agree.
  let xz = t.node + carpet_rot(c * uni.geom.y, t.yaw);

  // Terrain fitting, ladder rung 3 (per vertex). Neighbouring tiles SHARE
  // corner positions (the card is exactly one grid step), so a per-vertex fit
  // is the only rung that keeps the whole mat C0-continuous; any rung that
  // fits one plane per tile cracks at every tile edge. terrain_sample returns
  // the height AND (nx, nz) in one bilinear fetch, so the shading basis is
  // free.
  let g = terrain_sample(xz);
  var plane_y = uni.planes.x;
  if (layer == 1u) { plane_y = uni.planes.y; }

  var out: COut;
  out.world = vec3f(xz.x, g.x + plane_y, xz.y);
  out.pos = frame.view_proj * vec4f(out.world, 1.0);
  out.uv = 0.5 + c * 0.5 * uni.grid.z;
  out.up_ws = vec3f(g.y, sqrt(max(1.0 - g.y * g.y - g.z * g.z, 0.0)), g.z);
  out.fdat = vec4f(fade, f32(layer), t.yaw, plane_y);
  return out;
}

@vertex
fn vs_carpet(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> COut {
  return build_tile(ii, vi / 6u, vi % 6u);
}

/// Far tiles: layer 0 only, and the descending compaction of carpet_cull is
/// mirrored here so the indirect draw can keep firstInstance = 0.
@vertex
fn vs_carpet_far(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> COut {
  return build_tile(u32(uni.ids.z) - 1u - ii, 0u, vi % 6u);
}

fn oct_decode_yup(e: vec2f) -> vec3f {
  let y = 1.0 - abs(e.x) - abs(e.y);
  var x = e.x;
  var z = e.y;
  if (y < 0.0) {
    x = (1.0 - abs(e.y)) * sign(e.x);
    z = (1.0 - abs(e.x)) * sign(e.y);
  }
  return normalize(vec3f(x, y, z));
}

// Filtering note: every tap is an IMPLICIT-LOD textureSample, and the sampler
// runs anisotropic. A mat is the worst case for isotropic minification — a
// ground-parallel quad seen at grazing has a screen-space uv derivative tens of
// times larger along the view direction than across it, so a max-axis explicit
// LOD (which is what this shader did first) walks straight to the 1x1 mip and
// paints the whole near field one flat mustard colour. Measured: the entire
// foreground went featureless at cam=grazing. Hence: no textureSampleLevel
// anywhere, no branch around a sample (the parallax shift is a `select`, so
// control flow stays uniform for the implicit derivatives).

fn inspect_color(mode: u32, layer: u32, is_far: bool) -> vec3f {
  if (mode == 1u) { return vec3f(0.76, 0.30, 0.88); }  // dir: always the top capture
  if (mode == 2u) { return select(vec3f(0.90, 0.22, 0.20), vec3f(0.95, 0.66, 0.15), layer == 1u); }
  return select(vec3f(0.95, 0.45, 0.10), vec3f(0.15, 0.55, 0.95), is_far);
}

fn shade_carpet(
  alb: vec3f, cov: f32, aux: vec4f, up_ws: vec3f, yaw: f32, world_texel: vec3f,
  layer: u32, is_far: bool,
) -> vec3f {
  // The baked normal is in the mesh frame; lift it into the GROUND frame so a
  // mat on a slope lights as a slope (the specimen path only yaws, which is
  // right for an upright plant and wrong for a mat).
  let basis = plant_basis_from_up(up_ws, yaw);
  var n = basis * oct_decode_yup(aux.rg * 2.0 - 1.0);
  let vdir = normalize(frame.camera_pos - world_texel);
  if (dot(n, vdir) < 0.0) { n = -n; }  // thin foliage is two-sided
  var color = light_surface(alb, n, world_texel);
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, world_texel);
    let ins = u32(uni.tune.w + 0.5);
    if (ins > 0u) { color = inspect_color(ins, layer, is_far); }
  }
  return debug_shade(color, alb, n, cov, world_texel);
}

struct FOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_carpet(in: COut) -> FOut {
  let layer = u32(in.fdat.y + 0.5);
  let li = i32(layer);
  let plane_y = in.fdat.w;

  // One-tap parallax: the card plane sits at the layer's mean capture depth, so
  // a texel dy metres above it is really seen dy/v.y further along the view ray.
  // One precomputed-depth lookup, not marching.
  let bias = uni.tune2.x;
  let aux0 = textureSampleBias(carpet_aux, carpet_samp, in.uv, li, bias);
  let dy = (uni.geom.z - uni.geom.w * carpet_depth(aux0)) - plane_y;
  let v = normalize(frame.camera_pos - in.world);
  let shift = clamp(carpet_rot(v.xz * (dy / v.y), -in.fdat.z) / uni.geom.x, vec2f(-0.35), vec2f(0.35));
  let uv = in.uv + select(vec2f(0.0), shift, uni.tune.y > 0.5 && v.y > 0.12);

  let alb = textureSampleBias(carpet_albedo, carpet_samp, uv, li, bias);
  let aux = textureSampleBias(carpet_aux, carpet_samp, uv, li, bias);
  let cov = alb.a * in.fdat.x;
  if (cov < uni.tune.z) { discard; }

  // Per-texel height: the cushion's real relief, straight into the depth
  // buffer. This is what a card cannot do — capitula occlude each other and
  // interleave with the neighbouring tile's cushion.
  let rel = uni.geom.z - uni.geom.w * carpet_depth(aux);
  let dxz = carpet_rot((uv - in.uv) * uni.geom.x, in.fdat.z);
  let world_texel = vec3f(in.world.x + dxz.x, in.world.y + rel - plane_y, in.world.z + dxz.y);
  let clip = frame.view_proj * vec4f(world_texel, 1.0);

  var out: FOut;
  out.color = vec4f(shade_carpet(alb.rgb, cov, aux, in.up_ws, in.fdat.z, world_texel, layer, false), 1.0);
  out.depth = clamp(clip.z / max(clip.w, 1e-4), 0.0, 1.0);
  return out;
}

/// Far tiles: no parallax, no frag_depth (early-z stays on), flat card plane.
@fragment
fn fs_carpet_far(in: COut) -> @location(0) vec4f {
  let alb = textureSampleBias(carpet_albedo, carpet_samp, in.uv, 0, uni.tune2.x);
  let aux = textureSampleBias(carpet_aux, carpet_samp, in.uv, 0, uni.tune2.x);
  let cov = alb.a * in.fdat.x;
  if (cov < uni.tune.z) { discard; }
  return vec4f(shade_carpet(alb.rgb, cov, aux, in.up_ws, in.fdat.z, in.world, 0u, true), 1.0);
}
