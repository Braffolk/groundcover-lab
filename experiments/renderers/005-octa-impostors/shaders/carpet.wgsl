#include "src/wgsl/scatter.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./impostor_info.wgsl"

// A carpet tile of the view set: ONE ground-parallel quad, and the view set
// used as a precomputed raycast instead of as a billboard.
//
// The trick, and the reason this beats a flat top-view card:
//
//   Every baked view is an ORTHOGRAPHIC capture along a known direction d, so
//   view_k(uv) is the colour of the first surface a ray in direction d_k
//   through the point that maps to uv hits. Projecting into view k's basis
//   DROPS the d_k component, so every point along one ray maps to the same uv.
//   Therefore: take any proxy surface the ray crosses — here the tile's own
//   ground plane, lifted to the mean capitulum height — project the proxy point
//   into the view basis, and the texel you get is exactly what the camera would
//   see along that ray. The proxy's own height is irrelevant to the colour.
//
// So a flat quad reproduces the cushion's real relief, parallax and
// inter-capitulum occlusion, view-dependently, at 4 taps per fragment. What it
// cannot reproduce is DEPTH (the fragment reports the proxy plane, not the hit
// point, so it is off by up to the 3.3 cm of relief along the ray) and the
// silhouette of the mat against the sky at the far horizon.
//
// Everything the lattice needs is preserved: the quad is exactly one grid step
// (footprint_m * scale_min, never height-derived), the yaw is one of the
// scatter's four quarter turns, the scale is the entry's constant, and no tile
// is ever billboarded, jittered or LOD-shrunk.
//
// Terrain fitting is ladder RUNG 3 (per-vertex): each corner takes its own
// terrain_sample, and neighbouring tiles share corner positions exactly, so the
// mat is C0-continuous. A per-tile plane fit (rung 1/2) would be cheaper and
// WRONG — two neighbours fit two planes and crack apart at the shared edge.
// The tile's ground BASIS (used to take the view direction into the tile frame
// and to lift the baked normal) is per-tile instead of per-vertex, because view
// selection has to be primitive-uniform; the resulting parallax discontinuity
// between neighbours is ~1 mm.

@group(1) @binding(0) var<uniform> info: EntryInfo;
// 4 B per tile: the packed grid node, not a 16 B transform.
@group(1) @binding(1) var<storage, read> recs: array<u32>;
@group(1) @binding(2) var<storage, read> view_table: array<ViewBasis>;
@group(1) @binding(3) var atlas_albedo: texture_2d_array<f32>;
@group(1) @binding(4) var atlas_normal: texture_2d_array<f32>;
@group(1) @binding(5) var atlas_samp: sampler;

/**
 * Mip level whose value is used as the LOCAL MEAN of the normal atlas: level 4
 * of a 256 px tile is 16x16, i.e. one texel per ~13 mm of tile. Everything
 * coarser than that is subtracted out (see resolve_carpet).
 */
const NORMAL_MEAN_LOD: f32 = 4.0;

struct COut {
  @builtin(position) pos: vec4f,
  @location(0) off_mesh: vec3f,                    // mesh-frame offset from the capture centre
  @location(1) world: vec3f,
  @location(2) @interpolate(flat) weights: vec4f,  // w0, w1, w2, effective alpha ref
  @location(3) @interpolate(flat) sel: vec4u,      // layer0, layer1, layer2, dominant slot
  @location(4) @interpolate(flat) ground_t: vec3f, // tile ground basis: tangent
  @location(5) @interpolate(flat) ground_up: vec3f, // tile ground basis: up
}

@vertex
fn vs_carpet(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> COut {
  let rec = recs[ii];
  let entry = stand_table[u32(info.setup.y)];
  let div = u32(entry.carpet_div);
  let slot = rec & 511u;
  let cx = f32(i32(info.region.x) + i32((rec >> 9u) & 127u));
  let cz = f32(i32(info.region.y) + i32((rec >> 16u) & 127u));
  let quarter = (rec >> 23u) & 3u;

  // Constant scale for every tile of the species (lattice invariant), taken
  // straight from the stand so it carries no quantization error: a 0.1% shrink
  // would open a hairline crack at every tile edge.
  let scale = entry.scale_min;
  let step = entry.footprint_m * scale;   // == SCATTER_CELL_SIZE / carpet_div
  let g = vec2f(f32(slot % div), f32(slot / div));
  let base_xz = vec2f(cx, cz) * SCATTER_CELL_SIZE + (g + 0.5) * step;
  // Quarter turns exactly, not cos/sin of a quantized angle.
  var cs = vec2f(1.0, 0.0);
  if (quarter == 1u) { cs = vec2f(0.0, 1.0); }
  else if (quarter == 2u) { cs = vec2f(-1.0, 0.0); }
  else if (quarter == 3u) { cs = vec2f(0.0, -1.0); }

  var quad = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  let c = quad[vi];
  let local = c * (step * 0.5);                    // tile-local, unrotated, metres
  let off_w = rot_y_cs(vec3f(local.x, 0.0, local.y), cs);
  let xz = base_xz + off_w.xz;

  // Proxy plane height, as a fraction of the GRID STEP and never of this
  // species' own mesh. All carpet entries sharing a grid must share it: with the
  // per-mesh canopy height (the three Sphagnum states are 9.1 / 6.9 / 7.2 cm
  // tall) neighbouring states sat 1.6 cm apart, and at a grazing angle a 1.6 cm
  // step exposes ~6 cm of bare peat along the lower tile's far edge — a third of
  // a tile, which reads as a scatter of tile-shaped HOLES exactly where the
  // wetness jitter interlocks two states.
  // CARPET_LIFT_FRAC * 0.18 m = 6.8 cm is also the mean capitulum apex of these
  // meshes, i.e. the surface the baked views show, and it keeps the quad clear
  // of the base pass's 1 m terrain triangles (which interpolate the heightmap
  // coarser than the bilinear terrain_sample below).
  let lift = entry.footprint_m * CARPET_LIFT_FRAC;
  let plane_mesh_y = lift / scale;                 // same value in mesh units
  let ts = terrain_sample(xz);                     // rung 3: per-vertex ground
  let world = vec3f(xz.x, ts.x + lift, xz.y);

  // --- tile ground basis (instance-uniform) ---------------------------------
  let cs_g = terrain_sample(base_xz);
  let up = vec3f(cs_g.y, sqrt(max(1.0 - cs_g.y * cs_g.y - cs_g.z * cs_g.z, 0.0)), cs_g.z);
  let t0 = vec3f(cs.x, 0.0, -cs.y);
  let proj = t0 - up * dot(up, t0);
  let g_tan = select(normalize(vec3f(up.y, -up.x, 0.0)), normalize(proj), dot(proj, proj) > 1.0e-6);
  let g_bit = cross(g_tan, up);

  let centre_w = vec3f(base_xz.x, cs_g.x + lift, base_xz.y);
  let to_cam = frame.camera_pos - centre_w;
  let dir_w = to_cam / max(length(to_cam), 1.0e-4);
  // Into the tile's own frame: this is what makes the mat's relief lean with
  // the slope instead of standing bolt upright on a hillside.
  let dir_l = vec3f(dot(dir_w, g_tan), dot(dir_w, up), dot(dir_w, g_bit));

  // --- view selection: hemi-octahedral cell + barycentric weights -----------
  let dir_up = normalize(vec3f(dir_l.x, max(dir_l.y, 0.0), dir_l.z));
  let gn = f32(GRID_N - 1);
  // Clamped one node INSIDE the octahedral square. Its boundary is the horizon,
  // and a horizon view is degenerate for a ground proxy: the whole plane
  // projects onto a single line of texels. The innermost ring is ~19.5 deg of
  // elevation, and below that the parallax simply stops deepening.
  let oct = clamp(hemioct_uv(dir_up) * gn, vec2f(1.0), vec2f(gn - 1.0));
  let cell = clamp(floor(oct), vec2f(1.0), vec2f(gn - 2.0));
  let f = oct - cell;
  var n0 = cell;
  var n1 = cell + vec2f(1.0, 0.0);
  var n2 = cell + vec2f(0.0, 1.0);
  var w = vec3f(1.0 - f.x - f.y, f.x, f.y);
  if (f.x + f.y >= 1.0) {
    n0 = cell + vec2f(1.0, 1.0);
    w = vec3f(f.x + f.y - 1.0, 1.0 - f.y, 1.0 - f.x);
  }
  let stride = f32(GRID_N);
  let layers = vec3u(
    u32(n0.y * stride + n0.x),
    u32(n1.y * stride + n1.x),
    u32(n2.y * stride + n2.x),
  );
  var dom = 0u;
  var wm = w.x;
  if (w.y > wm) { dom = 1u; wm = w.y; }
  if (w.z > wm) { dom = 2u; }

  // Mesh-frame position of this corner. The tile square is [0, footprint_m]^2
  // with its origin at the mesh origin for every source mesh, so this is the
  // crop that keeps the sampled texels inside the tile's own period — and it is
  // pulled INWARD by info.carpet.x. Reason: the source mesh overhangs its period
  // by 1.3-2.1 cm, so in a real tiling the outer ~10% of a tile is covered
  // partly by the NEIGHBOUR's overhang, which a single-tile capture cannot
  // contain. Sampling right to the crop edge therefore renders a coverage- and
  // colour-deficient band along every tile boundary: bare peat strips up close
  // and a woven diagonal lattice over the whole mid-field at grazing. Skipping
  // that band costs a ~25% magnification of a 2-5 mm noise texture, which is
  // invisible, and removes the lattice entirely (measured at 0 / 0.04 / 0.08 /
  // 0.10).
  let ins = entry.footprint_m * info.carpet.x;
  let half_mesh = entry.footprint_m * 0.5 - ins;
  let off_mesh = vec3f(ins + half_mesh * (1.0 + c.x), plane_mesh_y, ins + half_mesh * (1.0 + c.y)) - info.center.xyz;

  // No camera-inside fade: a mat you are standing on must not open a hole under
  // you. Only the region edge erodes, measured from the tile centre so a tile
  // straddling the threshold cannot be emitted half-clipped.
  let edge_fade = 1.0 - smoothstep(info.setup.z * 0.88, info.setup.z, length(to_cam.xz));
  let a_ref = info.setup.w;

  var out: COut;
  if (edge_fade <= a_ref) {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  } else {
    out.pos = frame.view_proj * vec4f(world, 1.0);
  }
  out.off_mesh = off_mesh;
  out.world = world;
  out.weights = vec4f(w, a_ref / max(edge_fade, 1.0e-3));
  out.sel = vec4u(layers, dom);
  out.ground_t = g_tan;
  out.ground_up = up;
  return out;
}

/** Reproject a mesh-frame offset into one baked view's tile uv. */
fn view_uv(layer: u32, off: vec3f) -> vec2f {
  let b = view_table[layer];
  let u = dot(off, b.right.xyz) / b.right.w;
  let v = dot(off, b.up.xyz) / b.up.w;
  return vec2f(0.5 + 0.5 * u, 0.5 - 0.5 * v);
}

fn view_tint_color(layer: u32) -> vec3f {
  let h = f32(layer) * 0.61803399;
  return vec3f(
    0.35 + 0.5 * fract(h),
    0.35 + 0.5 * fract(h * 2.7 + 0.33),
    0.35 + 0.5 * fract(h * 5.3 + 0.66),
  );
}

/**
 * Alpha test, then the baked mesh-frame normal lifted into the tile's GROUND
 * frame (not just yawed — a mat on a slope has to light as a slope), shared
 * lighting, fog only outside the debug views.
 *
 * The normal needs one correction the plant path does not. Two facts about this
 * species' normal atlas:
 *
 *  * the bake flips every stored normal toward its OWN view node (right for a
 *    one-sided blade), so a tile's MEAN normal follows whichever of the 121
 *    views it happened to pick. An upright plant picks one view and gets away
 *    with it; a mat picks twenty views in one screenful — every tile a
 *    different quarter turn at a different angle — and the field then reads as
 *    a CHEQUERBOARD of flat patches. It did exactly that (obvious in
 *    debug=normals) before this.
 *  * at 0.8 mm/texel the moss normals are pure per-texel leaf noise: dumping
 *    the atlas shows no structure at all above ~2 texels, so there is no
 *    capitulum-scale normal signal to preserve, and any local average of them
 *    is just the flip bias.
 *
 * So treat the atlas normal as a BUMP FIELD, not an orientation: subtract its
 * own local mean (one extra tap at mip NORMAL_MEAN_LOD, ~13 mm of tile), keep
 * the tangential part of the residual, and perturb the ground normal with it.
 * The mean of the mat's normal is then the ground normal — the honest aggregate
 * for a carpet, and identical between neighbours — while the roughness that
 * makes the surface read as moss instead of paint survives. Subtracting the
 * mean from the SAME view cancels that view's bias exactly, and the residual
 * decays to zero under minification, so distant mat shades as ground.
 */
fn resolve_carpet(in: COut, albedo_in: vec3f, coverage: f32, uv_dom: vec2f, layer_dom: u32) -> vec4f {
  if (coverage < in.weights.w) {
    discard;
  }
  let enc = textureSample(atlas_normal, atlas_samp, uv_dom, layer_dom).xy;
  let enc_mean = textureSampleLevel(atlas_normal, atlas_samp, uv_dom, layer_dom, NORMAL_MEAN_LOD).xy;
  let d_l = oct_decode_y(enc * 2.0 - 1.0) - oct_decode_y(enc_mean * 2.0 - 1.0);
  let g_bit = cross(in.ground_t, in.ground_up);
  let d_g = in.ground_t * d_l.x + in.ground_up * d_l.y + g_bit * d_l.z;
  // Tangential part only, so the perturbation can never cancel the ground
  // normal however large the local deviation is.
  let bump = d_g - in.ground_up * dot(d_g, in.ground_up);
  let n = normalize(in.ground_up + bump * info.shape.w);
  var albedo = albedo_in;
  if (info.shape.z > 0.5) {
    albedo = view_tint_color(layer_dom);
  }
  var color = light_surface(albedo, n, in.world);
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, in.world);
  }
  return vec4f(debug_shade(color, albedo, n, coverage, in.world), 1.0);
}

/** Barycentric blend of the three views around the tile-frame view direction. */
@fragment
fn fs_carpet_blend3(in: COut) -> @location(0) vec4f {
  let off = in.off_mesh;
  let uv0 = view_uv(in.sel.x, off);
  let uv1 = view_uv(in.sel.y, off);
  let uv2 = view_uv(in.sel.z, off);
  let s0 = textureSample(atlas_albedo, atlas_samp, uv0, in.sel.x);
  let s1 = textureSample(atlas_albedo, atlas_samp, uv1, in.sel.y);
  let s2 = textureSample(atlas_albedo, atlas_samp, uv2, in.sel.z);
  var acc = vec4f(s0.rgb * s0.a, s0.a) * in.weights.x;
  acc += vec4f(s1.rgb * s1.a, s1.a) * in.weights.y;
  acc += vec4f(s2.rgb * s2.a, s2.a) * in.weights.z;
  let uv_dom = select(select(uv0, uv1, in.sel.w == 1u), uv2, in.sel.w == 2u);
  let layer_dom = select(select(in.sel.x, in.sel.y, in.sel.w == 1u), in.sel.z, in.sel.w == 2u);
  return resolve_carpet(in, acc.rgb / max(acc.a, 1.0e-4), acc.a, uv_dom, layer_dom);
}

/** Nearest view only — crisper relief, hard switch every ~11 deg of azimuth. */
@fragment
fn fs_carpet_nearest(in: COut) -> @location(0) vec4f {
  let layer_dom = select(select(in.sel.x, in.sel.y, in.sel.w == 1u), in.sel.z, in.sel.w == 2u);
  let uv_dom = view_uv(layer_dom, in.off_mesh);
  let s = textureSample(atlas_albedo, atlas_samp, uv_dom, layer_dom);
  return resolve_carpet(in, s.rgb, s.a, uv_dom, layer_dom);
}
