#include "src/wgsl/scatter.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"

// Secant relief cards. Every plant is one camera-facing card. Its appearance
// comes from a small hemi-octahedral set (5x5 = 25) of DEPTH-AUGMENTED
// orthographic captures: albedo+coverage plus a signed heightfield h(u,v)
// (distance of the foliage surface from the plant's mid-plane along the
// capture axis) and a baked normal.
//
// Per fragment, a FIXED number of taps — never a loop, never a march —
// intersects the true eye ray with the selected view's heightfield:
//
//   tap 1  h0 at the card point           (reliefMode >= flat-1tap)
//   step   analytic reprojection: slide along the ray to the plane f·p = h0
//   tap 2  h1 there                       (reliefMode >= linear-2tap)
//   solve  secant / false position on e(t) = f·(o+td) - h(uv(t)) using the
//          two exact samples e(0), e(t1) — one division, no iteration
//   tap 3  final surface fetch at t2      (reliefMode = secant-3tap)
//
// The solved hit is snapped onto the sampled height sheet and written as real
// @builtin(frag_depth), so plants inter-occlude by their reconstructed 3D
// surfaces, not by card planes. View cells are chosen stochastically per pixel
// with bilinear weights (dithered view selection): no 4-view crossfade
// ghosting, and at a node direction all pixels agree.
//
// Wind is exact-by-construction: sway is a linear-in-height shear, and lines
// stay lines under shear — so instead of bending the plant we inverse-shear
// the eye ray into the un-swayed baked frame (per plant, per fragment).
//
// Everything that is CONSTANT over a card — the eye position in the plant's
// mesh frame, the view-cell coordinates it selects, cos/sin of the plant yaw —
// is computed once in the vertex stage and passed flat. The one quantity that
// varies, the card point in mesh frame, is an AFFINE function of the world
// position, so perspective-correct interpolation reproduces it exactly: the
// fragment stage never redoes the world->mesh transform (two rotations with
// trig, per pixel, for a value the primitive already knows).
//
// CARPET TILES (stand_table[e].carpet_div > 0, e.g. Sphagnum) use the second
// entry-point pair, vs_carpet/fs_carpet, and a completely different card:
// GROUND-PARALLEL, tile-sized, terrain-conformed per vertex, relief-mapped
// against a single cropped zenith capture. A camera-facing card is wrong for a
// mat twice over — it breaks the lattice that makes neighbouring tiles agree,
// and a 7 cm cushion seen edge-on has almost no silhouette to show. What a
// carpet DOES have is 3.3 cm of capitulum relief across a lumpy, near-height-
// field surface, which is the one shape relief mapping is actually good at, so
// this is where the method can beat a flat billboard. Kept as separate entry
// points so the grass path above is untouched, byte for byte.

struct Meta {
  center: vec3f,        // mesh bounds center, local mesh units
  radius: f32,          // bounding radius, local mesh units
  half_ext: vec3f,      // mesh bounds half extents — tight card sizing
  top_h: f32,           // mesh top height (m) — shear normalization
  grid: f32,            // hemi-oct grid side (5)
  tile: f32,            // tile px
  atlas_px: f32,
  max_dist: f32,
  fade_band: f32,
  cov_thresh: f32,
  ao: f32,
  entry_index: f32,
  sway_mul: f32,
  relief_mode: f32,     // 0 flat, 1 linear, 2 secant
  view_mode: f32,       // 0 stochastic, 1 nearest
  capacity: f32,
  inspect: f32,         // method-specific inspector: 0 off, 1 height, 2 view-cell
  h_plane: f32,         // carpet: coverage-weighted mean canopy height (unit coords)
  h_sigma: f32,         // carpet: its std-dev — scales the crevice darkening
  _p1: f32,
}

// Written by cull.wgsl: 2 vec4 per scattered plant, 1 vec4 per carpet tile
// (a mat has one constant scale and no wind phase, so position + yaw is all
// there is to store — and at ~1.13M tiles on the bog stand, halving the stride
// is 4.5 MB of the species budget).
@group(1) @binding(0) var<storage, read> plants: array<vec4f>;
@group(1) @binding(1) var<uniform> mp: Meta;
@group(1) @binding(2) var atlas_albedo: texture_2d<f32>;
@group(1) @binding(3) var atlas_geom: texture_2d<f32>;
// Carpet tiles only: the cropped zenith tile is exactly periodic, so it is
// sampled with address mode REPEAT (a relief march walking off one edge is
// continuing into the neighbouring tile, which is a copy) and with mipmapping,
// which the 25-view atlas cannot have because levels would bleed across view
// cells.
@group(1) @binding(4) var carpet_sampler: sampler;

const INSPECT_HEIGHT: u32 = 1u;
const INSPECT_VIEW_CELL: u32 = 2u;

// Yaw rotation with the cosine/sine supplied — the plant yaw is a per-instance
// constant, so its trig is evaluated once per vertex and reused everywhere.
fn rot_y_cs(v: vec3f, c: f32, s: f32) -> vec3f {
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

fn hemioct_encode(v: vec3f) -> vec2f {
  let s = abs(v.x) + abs(v.y) + abs(v.z);
  let p = vec2f(v.x, v.z) / max(s, 1e-6);
  return vec2f(p.x + p.y, p.x - p.y);
}

fn hemioct_decode(e: vec2f) -> vec3f {
  let px = (e.x + e.y) * 0.5;
  let pz = (e.x - e.y) * 0.5;
  let y = 1.0 - abs(px) - abs(pz);
  return normalize(vec3f(px, y, pz));
}

fn oct_decode(e: vec2f) -> vec3f {
  let y = 1.0 - abs(e.x) - abs(e.y);
  var x = e.x;
  var z = e.y;
  if (y < 0.0) {
    x = (1.0 - abs(e.y)) * sign(e.x);
    z = (1.0 - abs(e.x)) * sign(e.y);
  }
  return normalize(vec3f(x, y, z));
}

struct Basis { f: vec3f, x: vec3f, y: vec3f }

// MUST match the bake basis construction exactly.
fn node_basis(node: vec2f) -> Basis {
  let gm1 = mp.grid - 1.0;
  let e = node / gm1 * 2.0 - 1.0;
  let f = hemioct_decode(e);
  var up_ref = vec3f(0.0, 1.0, 0.0);
  if (abs(f.y) > 0.999) { up_ref = vec3f(0.0, 0.0, 1.0); }
  var b: Basis;
  b.f = f;
  b.x = normalize(cross(up_ref, f));
  b.y = normalize(cross(f, b.x));
  return b;
}

// Plane coords [-1,1]^2 -> atlas uv inside tile `node`, 1-texel inset to
// avoid bilinear bleed across tiles.
fn atlas_uv(plane: vec2f, node: vec2f) -> vec2f {
  let inset = 1.0 / mp.tile;
  let uv = clamp(plane * 0.5 + 0.5, vec2f(inset), vec2f(1.0 - inset));
  return (node + uv) * mp.tile / mp.atlas_px;
}

fn tap_h(p: vec3f, b: Basis, node: vec2f) -> f32 {
  let plane = vec2f(dot(p, b.x), dot(p, b.y));
  return textureSampleLevel(atlas_geom, linear_sampler, atlas_uv(plane, node), 0.0).x;
}

// World -> un-swayed local mesh frame. Forward map of a baked point q is
//   world(q) = root + rot_y(q, yaw)*scale + sway * (q.y / top_h)
// which is affine in q, so it inverts exactly (including for ray directions
// taken as point differences) — and, being affine, its image interpolates
// exactly across the card.
fn to_mesh(p: vec3f, root: vec3f, cs: f32, sn: f32, inv_scale: f32, sway: vec3f, inv_shear: f32) -> vec3f {
  let p1 = p - root;
  let qy = p1.y * inv_shear; // 1 / (scale + sway.y/top_h)
  return rot_y_cs((p1 - sway * (qy / mp.top_h)) * inv_scale, cs, -sn);
}

struct VOut {
  @builtin(position) pos: vec4f,
  // Card point in the plant's un-swayed, un-yawed, unit-radius mesh frame.
  // Affine in world position -> perspective-correct interpolation is exact.
  @location(0) o_u: vec3f,
  // --- constant over the card, computed once per vertex ---------------------
  @location(1) @interpolate(flat) cam_u: vec3f,  // eye in the same frame
  @location(2) @interpolate(flat) root: vec3f,
  @location(3) @interpolate(flat) sway: vec3f,   // world-space tip sway
  @location(4) @interpolate(flat) cell: vec4f,   // view grid: g0.xy, frac.xy
  @location(5) @interpolate(flat) misc: vec4f,   // cos(yaw), sin(yaw), scale, fade
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var out: VOut;
  if (ii >= u32(mp.capacity)) {
    out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
    return out;
  }
  let d0 = plants[ii * 2u];
  let d1 = plants[ii * 2u + 1u];
  let root = d0.xyz;
  let yaw = d0.w;
  let scale = d1.x;
  let phase = d1.z;
  let entry = u32(mp.entry_index);
  let cs = cos(yaw);
  let sn = sin(yaw);

  let sway = wind_sway(root, frame.time, stand_table[entry].sway, phase) * mp.sway_mul;
  let R = mp.radius * scale;
  let center = root + rot_y_cs(mp.center, cs, sn) * scale + sway * (mp.center.y / mp.top_h);

  let dcam = distance(frame.camera_pos, center);
  // Camera-inside-plant rule: dissolve when the camera enters the bounding
  // sphere; also dissolve toward max_dist.
  let near_fade = smoothstep(R * 0.5, R * 1.2, dcam);
  let far_fade = 1.0 - smoothstep(mp.max_dist - mp.fade_band, mp.max_dist, dcam);
  let fade = near_fade * far_fade;
  if (fade <= 0.001) {
    out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
    return out;
  }

  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  let c = corners[vi];

  // Camera-facing billboard sized to the plant's PROJECTED tight box, not the
  // bounding-sphere square — big overdraw saver on tall-narrow species.
  let dir = normalize(frame.camera_pos - center);
  var up_ref = vec3f(0.0, 1.0, 0.0);
  if (abs(dir.y) > 0.99) { up_ref = vec3f(1.0, 0.0, 0.0); }
  let right = normalize(cross(up_ref, dir));
  let up = normalize(cross(dir, right));
  let bx = vec3f(cs, 0.0, -sn);   // rot_y(+x, yaw)
  let bz = vec3f(sn, 0.0, cs);    // rot_y(+z, yaw)
  let he = mp.half_ext * scale;
  let half_w = abs(dot(right, bx)) * he.x + abs(right.y) * he.y + abs(dot(right, bz)) * he.z;
  let half_h = abs(dot(up, bx)) * he.x + abs(up.y) * he.y + abs(dot(up, bz)) * he.z;
  // Keep covering the silhouette under perspective, plus sway slack.
  let enlarge = clamp(dcam / sqrt(max(dcam * dcam - R * R, 1e-4)), 1.0, 2.0);
  let slack = 0.5 * length(sway);

  let world = center
    + right * (c.x * (half_w * enlarge + slack))
    + up * (c.y * (half_h * enlarge + slack));

  // --- per-card constants: the eye ray frame and the baked view cell --------
  let inv_scale = 1.0 / scale;
  let inv_shear = 1.0 / (scale + sway.y / mp.top_h);
  let inv_r = 1.0 / (mp.radius + 1e-6);
  let cam_u = (to_mesh(frame.camera_pos, root, cs, sn, inv_scale, sway, inv_shear) - mp.center) * inv_r;

  // View direction from plant center -> baked view cell (upper hemisphere).
  var v = normalize(cam_u);
  v = normalize(vec3f(v.x, max(v.y, 0.02), v.z));
  let gm1 = mp.grid - 1.0;
  let g = clamp(hemioct_encode(v) * 0.5 + 0.5, vec2f(0.0), vec2f(1.0)) * gm1;
  let g0 = clamp(floor(g), vec2f(0.0), vec2f(gm1 - 1.0));

  out.pos = frame.view_proj * vec4f(world, 1.0);
  out.o_u = (to_mesh(world, root, cs, sn, inv_scale, sway, inv_shear) - mp.center) * inv_r;
  out.cam_u = cam_u;
  out.root = root;
  out.sway = sway;
  out.cell = vec4f(g0, clamp(g - g0, vec2f(0.0), vec2f(1.0)));
  out.misc = vec4f(cs, sn, scale, fade);
  return out;
}

struct FOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_main(in: VOut) -> FOut {
  let cs = in.misc.x;
  let sn = in.misc.y;
  let scale = in.misc.z;
  let fade = in.misc.w;

  // Eye ray in the plant's un-swayed, un-yawed, unit-radius frame — both ends
  // arrive from the vertex stage.
  let o_u = in.o_u;
  let d_u = normalize(o_u - in.cam_u);
  let g0 = in.cell.xy;
  let f = in.cell.zw;

  // Deterministic per-pixel dither (shared PCG hash of pixel coords).
  let px = vec2u(in.pos.xy);
  let hh = hash2(px.x, px.y);
  let r1 = hash_f32(hh);
  let r2 = hash_f32(hash2(hh, 17u));
  let r3 = hash_f32(hash2(hh, 41u));

  var node: vec2f;
  if (mp.view_mode < 0.5) {
    // Stochastic view selection: expectation equals the bilinear blend.
    node = g0 + vec2f(select(0.0, 1.0, r1 < f.x), select(0.0, 1.0, r2 < f.y));
  } else {
    node = g0 + step(vec2f(0.5), f);
  }
  let b = node_basis(node);

  // ---- fixed-tap relief ----------------------------------------------------
  let o_f = dot(o_u, b.f);
  let d_f = min(dot(d_u, b.f), -0.05); // f points at the (quantized) camera
  var t = 0.0;
  let h0 = tap_h(o_u, b, node); // tap 1
  if (mp.relief_mode > 0.5) {
    // Analytic reprojection: slide along the eye ray to the plane f·p = h0.
    let t1 = clamp((h0 - o_f) / d_f, -2.0, 2.0);
    t = t1;
    if (mp.relief_mode > 1.5) {
      let h1 = tap_h(o_u + d_u * t1, b, node); // tap 2
      // Secant / false-position on e(t) = f·(o + t d) - h(uv(t)), using the
      // two exact samples e(0) = o_f - h0 and e(t1) = o_f + d_f t1 - h1.
      let e0 = o_f - h0;
      let e1 = o_f + d_f * t1 - h1;
      let denom = e1 - e0;
      if (abs(denom) > 1e-5) {
        t = clamp(t1 - e1 * t1 / denom, -2.0, 2.0);
      }
    }
  }

  let p2 = o_u + d_u * t;
  let plane2 = vec2f(dot(p2, b.x), dot(p2, b.y));
  if (max(abs(plane2.x), abs(plane2.y)) > 1.0) { discard; }
  let uv2 = atlas_uv(plane2, node);
  let surf = textureSampleLevel(atlas_albedo, linear_sampler, uv2, 0.0); // tap 3 (a)
  let geom = textureSampleLevel(atlas_geom, linear_sampler, uv2, 0.0);  // tap 3 (b)

  // Alpha test + dithered near/far dissolve.
  let coverage = surf.a;
  if (coverage < mp.cov_thresh || fade < r3) { discard; }

  // Snap the hit onto the sampled height sheet and reconstruct the world
  // point -> true frag_depth (plants inter-occlude in 3D).
  let p_hit = p2 + b.f * (geom.x - dot(p2, b.f));
  let q_hit = p_hit * mp.radius + mp.center;
  let world_hit = in.root + rot_y_cs(q_hit, cs, sn) * scale + in.sway * (q_hit.y / mp.top_h);
  let clip = frame.view_proj * vec4f(world_hit, 1.0);

  // Real per-fragment normal: the baked local-frame normal of the surface the
  // ray actually hit, rotated into world by the plant yaw.
  let n_w = rot_y_cs(oct_decode(geom.yz), cs, sn);
  let ynorm = clamp(q_hit.y / mp.top_h, 0.0, 1.0);
  let albedo = surf.rgb * mix(1.0 - mp.ao, 1.0, ynorm);

  var color = light_surface(albedo, n_w, world_hit);
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    let ins = u32(mp.inspect + 0.5);
    if (ins == INSPECT_HEIGHT) {
      // Relief height of the solved hit along the capture axis, [-1,1] -> grey.
      color = vec3f(geom.x * 0.5 + 0.5);
    } else if (ins == INSPECT_VIEW_CELL) {
      // Which of the 25 baked views this pixel sampled (stochastic dither
      // shows up as grain between two cell colours).
      let gm1 = mp.grid - 1.0;
      color = vec3f(node.x / gm1, node.y / gm1, 0.35);
    } else {
      color = apply_fog(color, world_hit);
    }
  }

  var out: FOut;
  out.color = vec4f(debug_shade(color, albedo, n_w, coverage, world_hit), 1.0);
  out.depth = clamp(clip.z / max(clip.w, 1e-4), 0.0, 1.0);
  return out;
}

// ===========================================================================
// CARPET TILES — ground-parallel relief quads over one cropped zenith capture
// ===========================================================================
//
// The bake stores ONE view for a mat species, cropped to the species' periodic
// tile square ([0, tileM]^2 in the mesh frame) with centre (tileM/2, ., tileM/2)
// and radius tileM/2. Two things follow, and both are used below:
//
//   * the card point in the tile's unit mesh frame is just the quad corner:
//     p = (c2.x, h_top, c2.y) with h_top = (top_h - centre.y) / radius,
//     because the crop makes plane coords and corner coords the same thing;
//   * plane coords outside [-1,1] are the neighbouring tile, which is a copy —
//     so the relief march never has to be clipped, only wrapped, which the
//     REPEAT sampler does for free.
//
// The card plane sits at the cushion's MEAN canopy height, not its top. A plane
// at the top is the textbook convention (all relief is then below it, so no
// geometry is unreachable), but this heightfield is not textbook: measured on
// the baked tile, h varies by ~10 mm within any 3x3 texel neighbourhood at EVERY
// mip level, i.e. ~13:1 slopes at 0.35 mm texels, with a p05..p95 spread of
// 23 mm. From a top plane every pixel would have to march ~20 mm inward through
// that, and three taps on a surface that steep resolve to noise. From the mean
// the march is a few millimetres and lands close; the price is that peaks above
// the plane are shaded and depth-written correctly but cannot extend the
// silhouette (they have no quad to rasterize into).
//
// For the same reason the ray is intersected against a SMOOTHED height sheet
// (mip level >= CARPET_RELIEF_LOD, ~5.6 mm texels, where the slopes are about
// 1:1) while colour and normal are fetched at the true screen footprint. The
// cushion-scale lumpiness is what parallax can carry; the per-capitulum spikes
// are carried by the normal map, where they belong.
//
// Terrain fitting is rung 3 (per-vertex conforming): every vertex takes the
// ground height under its own xz, and neighbouring tiles share corner xz, so
// the mat is C0 continuous across the whole field. A per-tile plane fit
// (rung 1/2) is not merely cheaper here, it is WRONG — adjacent tiles would fit
// different planes and crack apart along their shared edge.

// Zenith capture basis — MUST match nodeBasis(0, 0, 1) in bake.ts:
// f = +Y, x = -X, y = +Z. Axis-aligned, so every dot product folds away.
const CARPET_H_CLAMP: f32 = -0.05;
// Longest ray step the solve may take, in tile-radius units (1 = 9 cm here).
// Parallax offset limiting: a grazing ray's intersection is unresolvable in
// three taps, and letting it slide a whole tile only smears the texture.
const CARPET_T_MAX: f32 = 1.0;
// Mip level floor for the HEIGHT taps: ~5.6 mm texels on a 0.18 m tile, the
// scale at which this cushion's heightfield stops being a spike field.
const CARPET_RELIEF_LOD: f32 = 4.0;
// How much of the shading normal is the baked per-leaf normal versus the
// cushion-scale height gradient. 0 = macro shape only (plasticky), 1 = baked
// only (flat grain, no cushions).
const CARPET_DETAIL: f32 = 0.45;
// Alpha reference for carpet tiles INSTEAD of covThresh. A mat is a closed
// surface, so its tiles must not dissolve with distance: tile coverage is ~80%
// up close but the mip chain pulls it toward the tile mean, and at the grass
// threshold (0.38) whole distant tiles fail the test and punch holes in the
// carpet. A low reference keeps the mat a solid depth-writing occluder while
// the genuinely empty texels — the gaps down to the peat — still open.
const CARPET_COV: f32 = 0.10;

/// Tile-square plane coords [-1,1]^2 -> uv. No inset and no atlas offset: the
/// texture IS the tile, and out-of-range wraps (REPEAT) into its own copy.
fn carpet_uv(p: vec3f) -> vec2f {
  return vec2f(-p.x, p.z) * 0.5 + 0.5;
}

fn carpet_h(p: vec3f, lod: f32) -> f32 {
  return textureSampleLevel(atlas_geom, carpet_sampler, carpet_uv(p), lod).x;
}

struct COut {
  @builtin(position) pos: vec4f,
  // Card point in the tile's unit mesh frame (affine in the corner).
  @location(0) o_u: vec3f,
  // World position of that card point on the CONFORMED quad — the relief hit
  // is reconstructed relative to this, not to the tile's tangent plane, so
  // neighbouring tiles agree on depth along their shared edge.
  @location(1) pw: vec3f,
  @location(2) @interpolate(flat) cam_u: vec3f,
  @location(3) @interpolate(flat) t_ws: vec3f,   // ground tangent, yawed
  @location(4) @interpolate(flat) up_ws: vec3f,  // ground up at the tile centre
  @location(5) @interpolate(flat) misc: vec4f,   // scale, _, erode, _
}

@vertex
fn vs_carpet(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> COut {
  var out: COut;
  if (ii >= u32(mp.capacity)) {
    out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
    return out;
  }
  let inst = plants[ii];
  let base = inst.xyz;               // tile centre, y = terrain height
  let entry = stand_table[u32(mp.entry_index)];
  // Taken AS GIVEN from the stand — one constant scale for every tile of the
  // species and yaw only in 90-degree steps are exactly what keeps neighbouring
  // tiles agreeing, i.e. a mat instead of confetti.
  let scale = entry.scale_min;
  let cs = cos(inst.w);
  let sn = sin(inst.w);
  let radius = mp.radius;            // = footprint_m * 0.5, by the bake's crop

  // Region fade measured from the tile CENTRE, never per vertex: it decides
  // whether the whole quad is emitted, and two vertices disagreeing would
  // stretch the quad into a screen-long sliver. NO camera-inside fade — a mat
  // you are standing on must not open a hole under you.
  let d_xz = length(frame.camera_pos.xz - base.xz);
  let fade = 1.0 - smoothstep(mp.max_dist - mp.fade_band, mp.max_dist, d_xz);

  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  let c2 = corners[vi];

  // Footprint from the species' PERIODIC TILE, never from height_scale (which
  // would make a 0.07 m tall, 0.24 m wide cushion ~3.5x too small and open gaps
  // in ground that must be closed). radius is that footprint's half width.
  let mesh_off = c2 * radius;
  let off = vec2f(cs * mesh_off.x + sn * mesh_off.y, -sn * mesh_off.x + cs * mesh_off.y) * scale;
  let xz = base.xz + off;
  // Rung 3: the ground under THIS vertex, height and normal from one bilinear
  // fetch. The quad plane rides at the cushion top.
  let g = terrain_sample(xz);
  // Card plane at the MEAN canopy height (mesh y = h_plane * radius + centre.y,
  // measured from the baked tile), with mesh y = 0 — the tile's own base — on
  // the ground.
  let plane_y = mp.h_plane * radius + mp.center.y;
  let world = vec3f(xz.x, g.x + plane_y * scale, xz.y);

  // Ray frame: ground up at the tile CENTRE. It has to be constant over the
  // primitive (the relief solve is a straight line in this frame), and the
  // per-vertex conforming above is what keeps the surface continuous.
  let gc = terrain_sample(base.xz);
  let up = vec3f(gc.y, sqrt(max(1.0 - gc.y * gc.y - gc.z * gc.z, 0.0)), gc.z);
  var tan0 = vec3f(cs, 0.0, -sn);
  let proj = tan0 - up * dot(up, tan0);
  let tng = select(normalize(vec3f(up.y, -up.x, 0.0)), normalize(proj), dot(proj, proj) > 1.0e-6);
  let bit = cross(tng, up);

  // Camera in the tile's unit mesh frame. Forward map:
  //   world(q) = base + [tng, up, bit] * ((q - q0) * scale),  q0 = (r, 0, r)
  // so the inverse is a transpose, and q0 - centre = (0, -centre.y, 0).
  let rel = frame.camera_pos - base;
  let loc = vec3f(dot(rel, tng), dot(rel, up), dot(rel, bit)) / scale;
  let cam_u = vec3f(loc.x, loc.y - mp.center.y, loc.z) / radius;

  // A tile whose fade dropped below the alpha reference has every fragment
  // discarding, so skip rasterizing it at all.
  if (fade < CARPET_COV) {
    out.pos = vec4f(0.0, 0.0, -1.0, 1.0);
  } else {
    out.pos = frame.view_proj * vec4f(world, 1.0);
  }
  out.o_u = vec3f(c2.x, mp.h_plane, c2.y);
  out.pw = world;
  out.cam_u = cam_u;
  out.t_ws = tng;
  out.up_ws = up;
  out.misc = vec4f(scale, 0.0, CARPET_COV / max(fade, 1.0e-3), 0.0);
  return out;
}

@fragment
fn fs_carpet(in: COut) -> FOut {
  let radius = mp.radius;
  let scale = in.misc.x;
  let erode = in.misc.z;
  let o_u = in.o_u;
  let d_u = normalize(o_u - in.cam_u);

  // Texture footprint from the SMOOTH card-plane uv — never from the solved uv,
  // whose screen derivatives blow up wherever the relief crosses a depth gap.
  // Passed as an explicit GRADIENT PAIR rather than collapsed to one mip level:
  // a mat is looked at almost entirely at grazing angles, where the footprint is
  // wildly anisotropic (metres along the view, millimetres across it), and any
  // single level has to be chosen for the long axis — which is what turned the
  // whole field into one flat khaki wash of per-tile mean colours. With the
  // gradients intact the sampler's anisotropy keeps the across-view axis sharp.
  let base_uv = carpet_uv(o_u);
  let ddx = dpdx(base_uv);
  let ddy = dpdy(base_uv);
  let tex_px = f32(textureDimensions(atlas_albedo, 0).x);
  let lod_iso = max(0.0, log2(max(max(length(ddx), length(ddy)) * tex_px, 1.0e-6)));
  let lod_h = max(lod_iso, CARPET_RELIEF_LOD);

  // ---- fixed-tap relief, capture axis = +Y so the dots fold away -----------
  let o_f = o_u.y;
  let d_f = min(d_u.y, CARPET_H_CLAMP); // downward into the cushion
  var t = 0.0;
  let h0 = carpet_h(o_u, lod_h); // tap 1
  if (mp.relief_mode > 0.5) {
    let t1 = clamp((h0 - o_f) / d_f, -CARPET_T_MAX, CARPET_T_MAX);
    t = t1;
    if (mp.relief_mode > 1.5) {
      let h1 = carpet_h(o_u + d_u * t1, lod_h); // tap 2
      let e0 = o_f - h0;
      let e1 = o_f + d_f * t1 - h1;
      let denom = e1 - e0;
      if (abs(denom) > 1.0e-5) {
        t = clamp(t1 - e1 * t1 / denom, -CARPET_T_MAX, CARPET_T_MAX);
      }
    }
  }

  let p2 = o_u + d_u * t;
  let uv2 = carpet_uv(p2);
  let surf = textureSampleGrad(atlas_albedo, carpet_sampler, uv2, ddx, ddy); // tap 3 (a)
  let geom = textureSampleGrad(atlas_geom, carpet_sampler, uv2, ddx, ddy);   // tap 3 (b)

  // Hard alpha-test edge; the region fade erodes coverage through the
  // reference instead of dithering (no screen door, and the mat keeps writing
  // solid depth so it stays an occluder).
  let coverage = surf.a;
  if (coverage < erode) { discard; }

  // Snap the hit onto the sampled height sheet and rebuild the world point
  // relative to the conformed quad -> true frag_depth, so cushions occlude
  // each other and the emergent grass stems intersect the mat correctly.
  let p_hit = vec3f(p2.x, geom.x, p2.z);
  let bit = cross(in.t_ws, in.up_ws);
  let dq = (p_hit - o_u) * (radius * scale);
  let world_hit = in.pw + in.t_ws * dq.x + in.up_ws * dq.y + bit * dq.z;
  let clip = frame.view_proj * vec4f(world_hit, 1.0);

  // Cushion-scale surface normal from the SMOOTHED height sheet, blended with
  // the baked per-leaf normal. This matters more than it looks: filtered to the
  // pixel footprint the baked normals average to straight up plus high-frequency
  // noise (visible in debug=normals), which is honest — a horizontal cushion's
  // foliage really does point every which way — but it leaves the mat with no
  // macro shading at all, so the intricacy reads as flat grain. The height
  // gradient at the relief LOD is where the cushion's SHAPE lives: ~11 mm of
  // height over ~11 mm of ground, i.e. real 45-degree lumps.
  // The centre height is free: the secant's root is where the linear model
  // meets the sheet, so o_f + d_f*t already is h(p2).
  let step_u = 2.0 / max(tex_px / exp2(min(lod_h, 7.0)), 2.0);
  let hc = o_f + d_f * t;
  let hu = carpet_h(p2 + vec3f(step_u, 0.0, 0.0), lod_h);
  let hv = carpet_h(p2 + vec3f(0.0, 0.0, step_u), lod_h);
  let n_macro = normalize(vec3f(-(hu - hc), step_u, -(hv - hc)));
  // The baked normal is a MESH-frame normal captured over flat ground, so it
  // has to be lifted into the local ground frame, not just yawed — otherwise a
  // mat on a slope lights as if it were level.
  let n_mesh = normalize(mix(n_macro, oct_decode(geom.yz), CARPET_DETAIL));
  let n_w = in.t_ws * n_mesh.x + in.up_ws * n_mesh.y + bit * n_mesh.z;

  // Crevice darkening, scaled by the tile's own height spread rather than by
  // its absolute height: the whole cushion sits in the top quarter of the mesh
  // box, so a ramp over [0, top_h] would only span 0.65..0.90 and do nothing.
  let ao_t = clamp(0.5 + (geom.x - mp.h_plane) / (3.0 * max(mp.h_sigma, 1.0e-3)), 0.0, 1.0);
  let albedo = surf.rgb * mix(1.0 - mp.ao, 1.0, ao_t);

  var color = light_surface(albedo, n_w, world_hit);
  if (debug_mode() == DEBUG_OFF) {
    let ins = u32(mp.inspect + 0.5);
    if (ins == INSPECT_HEIGHT) {
      color = vec3f(geom.x * 0.5 + 0.5);
    } else if (ins == INSPECT_VIEW_CELL) {
      // A carpet has exactly one view cell; show the texture footprint instead
      // (the isotropic mip level it would have taken), which is what varies.
      color = vec3f(lod_iso / 9.0, 0.15, 0.35);
    } else {
      color = apply_fog(color, world_hit);
    }
  }

  var out: FOut;
  out.color = vec4f(debug_shade(color, albedo, n_w, coverage, world_hit), 1.0);
  out.depth = clamp(clip.z / max(clip.w, 1e-4), 0.0, 1.0);
  return out;
}
