#include "src/wgsl/scatter.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/hash.wgsl"
#include "src/wgsl/debug.wgsl"
#include "./rayfield-common.wgsl"

// Ray-LUT impostor: one camera-facing quad per plant; each fragment builds
// the pixel's actual eye ray, moves it into the plant's unsheared local
// frame, and answers "what does this ray see?" with ~2 lookups into the
// baked ray-answer atlas: pick the nearest baked ray direction, fetch depth
// at the ray's closest-approach slab offset, reproject once along the true
// ray (kills most of the direction-quantization parallax), then take the
// final albedo/depth/normal/height answer. The baked depth is turned back
// into a world-space hit point, so frag_depth gives true inter-plant
// occlusion — no billboard "cardboard" sorting artifacts.
//
// Both deformations are SHEARS of plant space, and the whole method rests on
// the fact that lines stay lines under a shear, so bending the RAY is exact
// rather than approximate:
//   * wind      — horizontal displacement linear in local height y;
//   * terrain   — vertical displacement linear in local xz (gradient `grad`),
//                 which is how a carpet species conforms to the ground.
// The terrain shear is preferred over a rotation for a tiled mat because it
// leaves the tile's xz footprint EXACTLY on the scatter's grid step: a rotated
// tile's footprint shrinks by cos(tilt) and opens gaps in a closed mat.

struct EntryParams {
  center: vec3f,   // baked box center (anchored mesh space)
  radius: f32,     // bounding-sphere radius (quad proxy + ray reject)
  half: vec3f,     // baked box half-extents — the per-view slab fit
  top_h: f32,      // mesh-space shear reference height
  max_dist: f32,
  fade_band: f32,
  height_ao: f32,
  entry_index: f32,
  sway_mul: f32,
  show_cells: f32, // method-specific view: tint by baked direction cell
  stride: f32,     // u32 words per instance record (8 scattered / 2 carpet)
  mat_cov: f32,    // carpet alpha reference (hard test, no dither)
}
@group(1) @binding(0) var<storage, read> plants: array<u32>;
@group(1) @binding(1) var<uniform> ep: EntryParams;
@group(1) @binding(2) var atlas_surf: texture_2d<f32>;
@group(1) @binding(3) var atlas_geom: texture_2d<f32>;

// Per-fragment terrain conforming (ladder rung 3) is worth its heightmap fetch
// only while the mismatch it removes is more than a pixel; past this the
// linearized tile plane is used instead. Measured tangent-plane residual over a
// tile footprint on this terrain: mean 1.3mm, max 34mm.
const CONFORM_DIST: f32 = 30.0;
// The residual is clamped so a pathological terrain cannot push the conformed
// surface outside the quad proxy (which is sized for it, see vs_main).
const CONFORM_SLACK: f32 = 0.035;
// Band (in incidence-corrected distance) over which a carpet's per-texel normal
// is filtered toward the mat surface normal. Held deliberately far out: the
// per-texel normals are what make the cushion read as intricate 3D up close, so
// they are kept everywhere you are actually looking at the moss, and only the
// far field — where they alias to noise and shimmer, with no mip chain to fall
// back on — converges to the smooth surface.
const CARPET_NORMAL_LOD_NEAR: f32 = 6.0;
const CARPET_NORMAL_LOD_FAR: f32 = 25.0;

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) world: vec3f,
  @location(1) @interpolate(flat) plant_pos: vec3f,
  @location(2) @interpolate(flat) data: vec4f, // cos_yaw, sin_yaw, scale, fade
  @location(3) @interpolate(flat) sway: vec3f,
  // Terrain gradient (dh/dx, dh/dz) already scaled by the species' slope_align
  // — the shear that lays this plant into the ground. (0,0) = bolt upright.
  @location(4) @interpolate(flat) grad: vec2f,
}

// rot(yaw): local -> world (about +Y). rot_inv is its transpose.
fn rot_fwd(v: vec2f, cy: f32, sy: f32) -> vec2f {
  return vec2f(v.x * cy + v.y * sy, -v.x * sy + v.y * cy);
}
fn rot_inv(v: vec2f, cy: f32, sy: f32) -> vec2f {
  return vec2f(v.x * cy - v.y * sy, v.x * sy + v.y * cy);
}

/// Ground gradient (dh/dx, dh/dz) from one terrain_sample() result, which
/// carries (h, n.x, n.z) — cheaper than terrain_normal() re-fetching.
fn terrain_grad(ts: vec3f) -> vec2f {
  let ny = sqrt(max(1.0 - ts.y * ts.y - ts.z * ts.z, 1.0e-4));
  return -ts.yz / ny;
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let st = stand_table[u32(ep.entry_index)];
  let is_carpet = st.carpet_div > 0.0;
  let base = ii * u32(ep.stride);
  var pos: vec3f;
  var yaw: f32;
  var scale: f32;
  var phase = 0.0;
  var grad = vec2f(0.0);
  if (is_carpet) {
    // 8-byte record: the grid node, rebuilt with the scatter's own arithmetic.
    let w0 = plants[base];
    let w1 = plants[base + 1u];
    let cell = vec2i(vec2u(w0 & 0xffffu, w0 >> 16u)) - vec2i(32768);
    let n = u32(st.carpet_div);
    let step = SCATTER_CELL_SIZE / st.carpet_div;
    let slot = w1 & 0xffffu;
    let g = vec2f(f32(slot % n), f32(slot / n));
    let xz = vec2f(cell) * SCATTER_CELL_SIZE + (g + 0.5) * step;
    let ts = terrain_sample(xz);
    pos = vec3f(xz.x, ts.x, xz.y);
    yaw = f32(w1 >> 16u) * QUARTER_TURN;
    scale = st.scale_min;
    grad = terrain_grad(ts) * st.slope_align;
  } else {
    pos = vec3f(bitcast<f32>(plants[base]), bitcast<f32>(plants[base + 1u]), bitcast<f32>(plants[base + 2u]));
    yaw = bitcast<f32>(plants[base + 3u]);
    scale = bitcast<f32>(plants[base + 4u]);
    phase = bitcast<f32>(plants[base + 6u]);
    if (st.slope_align > 0.0) {
      grad = terrain_grad(terrain_sample(pos.xz)) * st.slope_align;
    }
  }
  let k = wind_sway(pos, frame.time, st.sway, phase) * ep.sway_mul;
  let cy = cos(yaw);
  let sy = sin(yaw);

  // Sphere center in world, following both shears at its own offset.
  let c_rot = rot_fwd(ep.center.xz, cy, sy);
  var wc = pos + vec3f(c_rot.x, ep.center.y, c_rot.y) * scale + k * (ep.center.y / ep.top_h);
  var rw = ep.radius * scale * 1.06 + length(k) * 0.6;
  if (st.slope_align > 0.0) {
    wc.y += dot(grad, c_rot * scale);
    // The shear turns the baked sphere into an ellipsoid: its vertical
    // half-extent grows by |grad| * horizontal half-extent, plus the slack the
    // per-fragment conforming can add. The baked sphere is loose for a flat
    // tile (its extreme vertex is a corner, not the pole), so on this terrain
    // the tight sheared bound is still inside it and nothing widens.
    let hh = length(vec2f(ep.half.x, ep.half.z)) * scale;
    let hv = ep.half.y * scale + length(grad) * hh + CONFORM_SLACK;
    rw = max(rw, sqrt(hh * hh + hv * hv) * 1.06 + length(k) * 0.6);
  }

  let to_c = wc - frame.camera_pos;
  let d = length(to_c);
  let fwd = to_c / max(d, 1e-4);
  var right = vec3f(1.0, 0.0, 0.0);
  if (abs(fwd.y) < 0.999) {
    right = normalize(cross(vec3f(0.0, 1.0, 0.0), fwd));
  }
  let up = normalize(cross(fwd, right));

  // Quad sized by the SUPPORT FUNCTION of the baked box, not by the bounding
  // sphere. The box's three world axes under (yaw, scale, wind shear, terrain
  // shear) are e1/e2/e3, and the quad's half-size along an axis is the box's
  // support there. For a 7cm-thick carpet tile seen at a grazing angle that is
  // ~5x fewer fragments than the sphere disc, and ~40% fewer for an upright
  // grass clump — pure overdraw, since every fragment outside the box could
  // only ever discard. The quad PLANE is unchanged, so each pixel still
  // reconstructs exactly the same eye ray and the image does not move.
  let xz1 = rot_fwd(vec2f(1.0, 0.0), cy, sy) * scale;
  let xz3 = rot_fwd(vec2f(0.0, 1.0), cy, sy) * scale;
  let inv_top = 1.0 / ep.top_h;
  let e1 = vec3f(xz1.x, dot(grad, xz1), xz1.y);
  let e3 = vec3f(xz3.x, dot(grad, xz3), xz3.y);
  let e2 = vec3f(k.x * inv_top, scale + k.y * inv_top + dot(grad, k.xz * inv_top), k.z * inv_top);
  let h = ep.half;
  let pad = select(0.0, CONFORM_SLACK, st.slope_align > 0.0);
  let sup_r = abs(dot(right, e1)) * h.x + abs(dot(right, e2)) * h.y + abs(dot(right, e3)) * h.z + pad;
  let sup_u = abs(dot(up, e1)) * h.x + abs(dot(up, e2)) * h.y + abs(dot(up, e3)) * h.z + pad;
  let sup_f = abs(dot(fwd, e1)) * h.x + abs(dot(fwd, e2)) * h.y + abs(dot(fwd, e3)) * h.z + pad;
  // Perspective enlargement: the box's near face is sup_f closer than the quad
  // plane, so its silhouette is that much wider in angle.
  let grow = clamp(d / max(d - sup_f, sup_f * 0.2), 1.0, 5.0);
  let size = vec2f(sup_r, sup_u) * grow * 1.02;

  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  let cn = corners[vi];
  let world = wc + (right * cn.x * size.x + up * cn.y * size.y);

  // Camera-inside-plant rule: dissolve as the eye enters the sphere. NOT for a
  // carpet — a mat is the ground you stand on, and eroding it opens a hole
  // under the camera.
  var near_fade = clamp((d / rw - 0.55) / 0.65, 0.0, 1.0);
  if (is_carpet) {
    near_fade = 1.0;
  }
  let dist_fade = clamp((ep.max_dist - d) / max(ep.fade_band, 0.01), 0.0, 1.0);

  var out: VOut;
  out.clip = frame.view_proj * vec4f(world, 1.0);
  out.world = world;
  out.plant_pos = pos;
  out.data = vec4f(cy, sy, scale, near_fade * dist_fade);
  out.sway = k;
  out.grad = grad;
  return out;
}

struct FOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_main(in: VOut) -> FOut {
  let st = stand_table[u32(ep.entry_index)];
  let is_carpet = st.carpet_div > 0.0;
  let cy = in.data.x;
  let sy = in.data.y;
  let scale = in.data.z;
  let fade = in.data.w;
  let dither = hash_f32(hash2(u32(in.clip.x), u32(in.clip.y)));
  if (!is_carpet && fade <= dither * 0.999) {
    discard;
  }
  let k = in.sway;
  let g = in.grad;

  // Eye ray -> unsheared plant-local (mesh units) space. The composite shear
  // map is q = rot(p)*scale + k*(p.y/top_h) horizontally and
  // q.y = p.y*denom + dot(g, q.xz) vertically; invert both, exactly.
  let denom = scale + k.y / ep.top_h;
  let rd_w = normalize(in.world - frame.camera_pos);
  let qo = frame.camera_pos - in.plant_pos;
  let oy = (qo.y - dot(g, qo.xz)) / denom;
  let oxz = rot_inv(qo.xz - k.xz * (oy / ep.top_h), cy, sy) / scale;
  let o_l = vec3f(oxz.x, oy, oxz.y);
  let uy = (rd_w.y - dot(g, rd_w.xz)) / denom;
  let uxz = rot_inv(rd_w.xz - k.xz * (uy / ep.top_h), cy, sy) / scale;
  let d_l = normalize(vec3f(uxz.x, uy, uxz.y));

  // Closest approach to the baked sphere — the seed slab offset.
  let m = o_l - ep.center;
  let t0 = -dot(m, d_l);
  let p0 = o_l + d_l * t0;
  if (length(p0 - ep.center) > ep.radius * 1.02) {
    discard;
  }

  // Nearest baked ray direction and its slab basis.
  let e = rf_oct_encode(d_l);
  let cellf = clamp(floor(e * RF_VIEWS), vec2f(0.0), vec2f(RF_VIEWS - 1.0));
  let dv = rf_oct_decode((cellf + vec2f(0.5)) / RF_VIEWS);
  let r_ax = rf_view_right(dv);
  let u_ax = normalize(cross(dv, r_ax));
  // This view's slab half-sizes: the baked box's support along each axis, the
  // same fit the bake rasterized with.
  let ext = vec2f(rf_extent(r_ax, ep.half), rf_extent(u_ax, ep.half));
  let ext_d = rf_extent(dv, ep.half);

  // Lookup 1: depth answer at the closest-approach offset.
  var sp = vec2f(dot(p0 - ep.center, r_ax), dot(p0 - ep.center, u_ax));
  var uv = rf_atlas_uv(cellf, sp, ext);
  var surf = textureSampleLevel(atlas_surf, linear_sampler, uv, 0.0);
  var geom = textureSampleLevel(atlas_geom, linear_sampler, uv, 0.0);
  // Beyond ~40m one LUT texel is subpixel — the parallax refinement fetch
  // would be invisible, so spend it only up close.
  let cam_d = length(in.world - frame.camera_pos);
  if (surf.a > 0.01 && cam_d < 40.0) {
    // Reproject the baked hit onto the TRUE eye ray and ask again — one
    // parallax-correction step against direction quantization.
    let depth0 = geom.r / surf.a;
    let hit0 = ep.center + r_ax * sp.x + u_ax * sp.y + dv * ((depth0 * 2.0 - 1.0) * ext_d);
    let p1 = o_l + d_l * dot(hit0 - o_l, d_l);
    sp = vec2f(dot(p1 - ep.center, r_ax), dot(p1 - ep.center, u_ax));
    uv = rf_atlas_uv(cellf, sp, ext);
    surf = textureSampleLevel(atlas_surf, linear_sampler, uv, 0.0);
    geom = textureSampleLevel(atlas_geom, linear_sampler, uv, 0.0);
  }

  let cov = surf.a;
  let inv_cov = 1.0 / max(cov, 1e-3);
  let depth01 = geom.r * inv_cov;
  // The baked hit in local space — used both for the clutter hash and for the
  // true-depth reprojection below, so reconstruct it exactly once.
  let hitp = ep.center + r_ax * sp.x + u_ax * sp.y + dv * ((depth01 * 2.0 - 1.0) * ext_d);

  // Sharpened coverage: fuzzy half-covered LUT texels would screen-door the
  // whole plant; remapping keeps interiors solid and dithers only rims.
  var cov_sharp = clamp((cov - 0.26) * 1.9, 0.0, 1.0);
  var albedo = surf.rgb * inv_cov;
  if (is_carpet) {
    // A mat is a closed, depth-writing surface: HARD alpha test, no dither.
    // Dithering a carpet screen-doors the ground you look straight at, and its
    // holes in the depth buffer are exactly what stops the mat from occluding
    // everything below it. The distance fade erodes the reference instead of
    // dissolving texels, so the mat thins with hard edges.
    if (cov < mix(1.5, ep.mat_cov, fade)) {
      discard;
    }
    cov_sharp = cov;
  } else {
    // Plant-local clutter noise: LUT texels are ~2.4cm on the grasses, pure
    // magnification turns close plants into smooth blobs. A hash keyed to the
    // (stable, unsheared) local hit cell restores leafy high-frequency
    // structure. Its weight is zero past ~8m, so the whole hash is skipped out
    // there instead of being computed and multiplied by 0 for every far
    // fragment. Carpets skip it entirely: their texels are ~5mm, they need no
    // invented detail, and the coverage notches would punch holes in the mat.
    let near_w = clamp(1.0 - cam_d * 0.12, 0.0, 1.0);
    if (near_w > 0.0) {
      let cell3 = vec3i(floor(hitp * 44.0)) + vec3i(1024);
      let clutter = hash_f32(hash3(u32(cell3.x), u32(cell3.y), u32(cell3.z)));
      cov_sharp *= 1.0 - near_w * (0.75 - 1.1 * clutter);
      albedo *= 1.0 - near_w * (0.28 - 0.56 * clutter);
    }
    if (cov_sharp * fade <= dither) {
      discard;
    }
  }
  let oct = clamp(geom.gb * inv_cov, vec2f(0.0), vec2f(1.0));
  let height01 = clamp(geom.a * inv_cov, 0.0, 1.0);

  // Reproject the baked hit onto the true eye ray for real world depth.
  let p_l = o_l + d_l * dot(hitp - o_l, d_l);
  let pxz = rot_fwd(p_l.xz, cy, sy);
  var p_w = in.plant_pos + vec3f(pxz.x, p_l.y, pxz.y) * scale + k * (p_l.y / ep.top_h);
  if (st.slope_align > 0.0) {
    // Ladder rung 3, per FRAGMENT rather than per vertex: displace by the true
    // ground height under the hit, not by the tile's linearized plane. That is
    // what keeps neighbouring tiles continuous — a per-tile plane fit (rungs
    // 1-2) has each tile agreeing with its own centre and cracking against its
    // neighbour, up to 41mm apart on this terrain, half the cushion's height.
    // Beyond CONFORM_DIST the same term is taken from the linearized plane,
    // where it is subpixel.
    var dy = dot(g, p_w.xz - in.plant_pos.xz);
    if (cam_d < CONFORM_DIST) {
      // Clamp the RESIDUAL against the linearized plane (mean 1.3mm, max 34mm
      // on this terrain), never the whole offset: clamping the offset itself
      // would flatten every tile on a slope and bury its uphill half in the
      // ground, which is exactly how this first went wrong.
      // ('target' is a WGSL reserved keyword — hence the name.)
      let conform_dy = st.slope_align * (terrain_sample(p_w.xz).x - in.plant_pos.y);
      dy += clamp(conform_dy - dy, -CONFORM_SLACK, CONFORM_SLACK);
    }
    p_w.y += dy;
  }
  let clip = frame.view_proj * vec4f(p_w, 1.0);
  if (clip.w <= 0.0) {
    discard;
  }

  // Baked mesh normal, yaw-rotated, foliage-style two-sided. rf_mesh_normal()
  // and not rf_oct_decode(): GCMESH1's oct convention derives Z, not Y.
  var n_l = rf_mesh_normal(oct);
  let nxz = rot_fwd(n_l.xz, cy, sy);
  var n = vec3f(nxz.x, n_l.y, nxz.y);
  if (st.slope_align > 0.0) {
    // Normals transform by the inverse transpose of the shear jacobian, which
    // for a vertical shear is n.xz -= grad * n.y. A flat cushion top (0,1,0)
    // therefore comes out as the terrain normal, exactly as it should.
    n = normalize(vec3f(n.x - g.x * n.y, n.y, n.z - g.y * n.y));
  }
  if (dot(n, frame.camera_pos - p_w) < 0.0) {
    n = -n;
  }
  if (is_carpet) {
    // Normal-field LOD. A carpet texel is 2-5mm, so past a few metres it is
    // subpixel and the single normal the LUT returns is an ALIASED sample of a
    // much richer distribution: noise in a still, shimmer in motion, and there
    // is no mip chain to fall back on (view cells would bleed across atlas
    // tiles). Converging to the mat's own surface normal is the correct filtered
    // answer for a sub-texel footprint, and only the NORMAL converges — albedo,
    // coverage, depth and the height AO keep their per-texel detail at every
    // distance. Kept deliberately far out (6-25m) so the near field, where you
    // actually look at the moss, still shows real cushion relief.
    let up_mat = normalize(vec3f(-g.x, 1.0, -g.y));
    let far_w = smoothstep(CARPET_NORMAL_LOD_NEAR, CARPET_NORMAL_LOD_FAR, cam_d);
    n = normalize(mix(n, up_mat, far_w) + up_mat * 1.0e-4);
  }
  // Method-specific inspection view: flat tint per baked direction cell, so
  // the 24x24 direction quantization (and its popping) is directly visible.
  if (ep.show_cells > 0.5) {
    let ci = u32(cellf.x + cellf.y * RF_VIEWS);
    albedo = vec3f(
      hash_f32(hash2(ci, 7u)),
      hash_f32(hash2(ci, 11u)),
      hash_f32(hash2(ci, 13u)),
    ) * 0.6 + 0.2;
  }

  let ao = mix(1.0 - ep.height_ao, 1.0, height01);
  var color = light_surface(albedo * ao, n, p_w);
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, p_w);
  }

  var out: FOut;
  out.color = vec4f(debug_shade(color, albedo, n, cov_sharp * fade, p_w), 1.0);
  out.depth = clamp(clip.z / clip.w, 0.0, 1.0);
  return out;
}
