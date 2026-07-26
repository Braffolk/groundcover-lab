#include "src/wgsl/scatter.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./carpet_info.wgsl"

// SLICE PRISMS FOR A CARPET — the same idea with the cutting plane rotated.
//
// A Sphagnum tile is 0.18m across and 0.09m tall. It has no silhouette, so the
// vertical prisms of prisms.wgsl describe nothing: what it has is 3.3cm of
// capitulum relief, and that is a HEIGHT structure. So the bake cuts the tile
// into five equal-mass HORIZONTAL slices and stores each slice's centroid
// height, and here each slice is drawn as its own ground-parallel quad at that
// height:
//
//   * every texel belongs to exactly ONE slice (exclusive ownership in the
//     bake), so seen from straight above the stack is bit-identical to the
//     single merged card — nothing is drawn twice and nothing is lost;
//   * off vertical, the slices SLIDE against each other by exactly the
//     parallax their real height difference implies, and occlude each other
//     with real depth writes. That is cushion relief, not a flat texture;
//   * the mat gets thickness at a ridge silhouette instead of a razor edge;
//   * coverage only ever increases as the view tilts (the layers overlap more),
//     so unlike a vertical slice stack this cannot open see-through holes.
//
// The lattice is taken exactly as the stand gives it: constant scale, 90-degree
// rotations only, one quad per grid node spanning exactly footprint_m * scale,
// so neighbouring tiles abut and agree. Terrain fitting is rung 3 — every quad
// corner samples the ground under itself, which is the only rung that keeps
// neighbouring tiles sharing an edge (a per-tile plane fit cracks at every
// boundary).
//
// Beyond slice_dist the stack collapses (continuously) into one merged card,
// i.e. exactly what a flat billboard mat draws, at a sixth of the vertices.

@group(1) @binding(0) var<uniform> info: CarpetInfo;
@group(1) @binding(1) var<storage, read> nodes: array<u32>;
@group(1) @binding(2) var carpet_albedo: texture_2d<f32>;
@group(1) @binding(3) var carpet_normal: texture_2d<f32>;
@group(1) @binding(4) var carpet_sampler: sampler;

// Residual separation at the LOD boundary. Two exactly coplanar slices would
// z-fight along the one-texel fringe where bilinear filtering leaves both above
// the alpha reference; 4% of 3cm is a sub-millimetre offset that keeps the
// stack strictly ordered (top slice wins, which is what the merged tile shows).
const SEP_MIN: f32 = 0.04;
// The base card sits a third of a millimetre below the merged height so a
// slice that has collapsed onto it still wins the depth test cleanly.
const BASE_BIAS: f32 = 0.0003;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) world: vec3f,
  @location(2) @interpolate(flat) yaw_cs: vec2f, // cos(yaw), sin(yaw)
  @location(3) @interpolate(flat) erode: f32,    // effective alpha reference
  @location(4) @interpolate(flat) shade: f32,    // interstitial occlusion
  @location(5) up_ws: vec3f,                     // ground normal under the vertex
}

fn quad_corner2(vi: u32) -> vec2f {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  return corners[vi % 6u];
}

/**
 * Exact quarter turn. Not rot_y(v, q * PI/2): the tile lattice only stays
 * seamless while neighbours agree on their shared corners to the last bit, and
 * cos(PI/2) is not exactly 0.
 */
fn rot_quarter(v: vec2f, q: u32) -> vec2f {
  switch q {
    case 1u: { return vec2f(v.y, -v.x); }
    case 2u: { return -v; }
    case 3u: { return vec2f(-v.y, v.x); }
    default: { return v; }
  }
}

fn quarter_cs(q: u32) -> vec2f {
  switch q {
    case 1u: { return vec2f(0.0, 1.0); }
    case 2u: { return vec2f(-1.0, 0.0); }
    case 3u: { return vec2f(0.0, -1.0); }
    default: { return vec2f(1.0, 0.0); }
  }
}

/** Atlas uv of tile `tile` at tile-local [0,1]^2, inside the baked margin. */
fn tile_uv(tile: u32, local: vec2f) -> vec2f {
  let cell = vec2f(f32(tile % u32(CARPET_COLS)), f32(tile / u32(CARPET_COLS)));
  return (cell + info.uv_inset + local * (1.0 - 2.0 * info.uv_inset)) * info.uv_step;
}

/** The grid node a packed tile sits on, rebuilt with the scatter's arithmetic. */
fn carpet_node(packed: u32) -> vec2f {
  let cx = i32((packed >> 22u) & 1023u) - 512;
  let cz = i32((packed >> 12u) & 1023u) - 512;
  let slot = (packed >> 2u) & 1023u;
  let entry = stand_table[u32(info.entry_index)];
  let n = u32(entry.carpet_div);
  let step = SCATTER_CELL_SIZE / entry.carpet_div;
  let g = vec2f(f32(slot % n), f32(slot / n));
  return vec2f(f32(cx), f32(cz)) * SCATTER_CELL_SIZE + (g + 0.5) * step;
}

/**
 * One ground-parallel tile quad. `slice` selects the atlas tile — index
 * n_slice is the merged BASE CARD, anything below it a relief slice; `sep` is
 * how far the stack is opened (1 = full relief, 0 = collapsed onto the base).
 *
 * Why a base card under the slices at all, when the bake made the slices an
 * exact partition of it? Because the partition is only exact at mip 0. Under
 * minification each slice's coverage mips toward its OWN mean (a fifth), so
 * with a low alpha reference every slice passes everywhere and the mat turns
 * into "whatever the top slice averaged to" — brighter and blobbier than the
 * truth. So the base card carries coverage at the low reference (a mat must
 * never dissolve) and the slices are drawn over it at a MAJORITY reference:
 * a slice only paints where it owns more than half of the footprint it is
 * being minified into. Relief therefore survives exactly as long as it is
 * spatially coherent — a whole capitulum keeps its height, fine speckle falls
 * back to the flat card — and it dissolves continuously with distance instead
 * of popping.
 */
fn carpet_quad(packed: u32, node: vec2f, slice: u32, sep: f32, vi: u32) -> VOut {
  let quarter = packed & 3u;
  let c = quad_corner2(vi);
  let half_m = info.tile_m * info.tile_scale * 0.5;
  let xz = node + rot_quarter(c, quarter) * half_m;

  // Rung 3 of the terrain ladder: the ground under THIS vertex. terrain_sample
  // returns height and (nx, nz) from the same four texel loads, so the shading
  // basis costs nothing extra. Neighbouring tiles share corner positions, so
  // the mat stays C0 continuous — no per-tile plane can promise that.
  let ground = terrain_sample(xz);
  let up = vec3f(ground.y, sqrt(max(1.0 - ground.y * ground.y - ground.z * ground.z, 0.0)), ground.z);

  let base = slice >= u32(info.n_slice);
  let merged_h = info.slice_h[u32(info.n_slice) >> 2u][u32(info.n_slice) & 3u];
  let slice_h = info.slice_h[slice >> 2u][slice & 3u];
  let h = mix(merged_h, slice_h, sep) * info.tile_scale - select(0.0, BASE_BIAS, base);
  // A mat is a closed surface: the base card keeps the low reference so it
  // never dissolves. A relief slice must EARN its fragment instead.
  let a_ref = select(info.relief_ref, info.alpha_ref, base);

  // No camera-inside fade: a mat you are standing on must not open a hole
  // under you. Region edge only, and measured from the tile CENTRE — a fade
  // evaluated per vertex would emit some corners behind the near plane and
  // stretch the quad into a sliver.
  let d_xz = length(frame.camera_pos.xz - node);
  let fade = 1.0 - smoothstep(info.region_r * 0.82, info.region_r, d_xz);

  var out: VOut;
  let world = vec3f(xz.x, ground.x + h, xz.y);
  if (fade < a_ref) {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  } else {
    out.pos = frame.view_proj * vec4f(world, 1.0);
  }
  out.uv = tile_uv(slice, c * 0.5 + 0.5);
  out.world = world;
  out.yaw_cs = quarter_cs(quarter);
  out.erode = a_ref / max(fade, 1e-3);
  out.up_ws = up;
  // Interstitial occlusion: the exposed capitula read hotter than the shaded
  // gaps between them. Centred on the base card's 1.0 and scaled by sep, so
  // the mean luminance is unchanged and the LOD switch cannot step.
  let layer = f32(min(slice, u32(info.n_slice) - 1u)) / max(info.n_slice - 1.0, 1.0);
  out.shade = select(1.0 + info.ao * (0.5 - layer) * sep, 1.0, base);
  return out;
}

/**
 * Distance-driven collapse of the stack. Built from the camera's height above
 * ITS OWN ground (a CPU-side lookup in info.cam_height) rather than a terrain
 * fetch under the tile: it must be instance-uniform or the quad would bend,
 * and carpet_cull.wgsl decides the near/far bucket with the identical formula.
 */
fn carpet_sep(node: vec2f) -> f32 {
  let dy = info.cam_height - (info.y0 + info.y1) * 0.5 * info.tile_scale;
  let d = length(vec2f(length(frame.camera_pos.xz - node), dy));
  return max(1.0 - smoothstep(info.slice_dist * 0.6, info.slice_dist, d), SEP_MIN);
}

// --- near LOD: base card + the relief stack, 6 * (CARPET_SLICES + 1) verts --
@vertex
fn vs_carpet_near(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let packed = nodes[ii];
  let node = carpet_node(packed);
  return carpet_quad(packed, node, vi / 6u, carpet_sep(node), vi);
}

// --- far LOD: one merged card, 6 vertices ----------------------------------
@vertex
fn vs_carpet_far(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let packed = nodes[ii];
  return carpet_quad(packed, carpet_node(packed), u32(info.n_slice), 0.0, vi);
}

/**
 * The carpet normal map stores the plain TANGENT (n.x, n.z) of an upper-
 * hemisphere normal, not an octahedral pair: this map exists to be filtered
 * (one screen pixel is already ~4 texels at the carpet-close bookmark), and
 * (x, z) is linear, so bilinear + the mip chain average to the true mean
 * tangent and y = sqrt(1 - |t|^2) flattens exactly as fast as the surface
 * decorrelates. `gain` scales the tangent, i.e. it dials the whole cushion
 * relief up or down at runtime — gain 0 is a perfectly flat-lit mat.
 */
fn tile_normal(e: vec2f, gain: f32) -> vec3f {
  let t = (e * 2.0 - 1.0) * gain;
  let l2 = dot(t, t);
  let tc = select(t, t * inverseSqrt(l2), l2 > 1.0);
  return vec3f(tc.x, sqrt(max(1.0 - dot(tc, tc), 0.0)), tc.y);
}

@fragment
fn fs_carpet(in: VOut) -> @location(0) vec4f {
  // Sample before any non-uniform discard (uniform-control-flow rule).
  let alb = textureSample(carpet_albedo, carpet_sampler, in.uv);
  let enc = textureSample(carpet_normal, carpet_sampler, in.uv).xy;
  if (alb.a < in.erode) {
    discard;
  }
  let n_mesh = tile_normal(enc, info.normal_gain);
  // The tile was captured over flat ground, so its normals must be lifted into
  // the LOCAL GROUND frame, not merely yawed — otherwise a mat on a slope
  // lights as if it were level. This is plant_basis_from_up(up, yaw) from
  // terrain.wgsl, inlined because the quad already carries cos/sin(yaw).
  let up = normalize(in.up_ws);
  var t = vec3f(in.yaw_cs.x, 0.0, -in.yaw_cs.y);
  let proj = t - up * dot(up, t);
  t = select(normalize(vec3f(up.y, -up.x, 0.0)), normalize(proj), dot(proj, proj) > 1.0e-6);
  let n = t * n_mesh.x + up * n_mesh.y + cross(t, up) * n_mesh.z;

  // in.shade is fake occlusion, so it belongs to the light term: `albedo` then
  // shows the baked capture exactly and `lighting` shows sun+ambient x it.
  var color = light_surface(alb.rgb * in.shade, n, in.world);
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  return vec4f(debug_shade(color, alb.rgb, n, alb.a, in.world), 1.0);
}
