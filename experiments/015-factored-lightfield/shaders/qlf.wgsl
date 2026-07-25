#include "src/wgsl/scatter.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/debug.wgsl"

// Factored quantized light field. Each plant is one camera-facing PROXY card —
// a window, not the geometry: every fragment reconstructs a 3D hit point and
// writes its true depth, so what lands on screen is a depth-correct surface,
// never a flat quad. The 4D light field L(view dir, image offset) is factored
// as geometry x radiance: grid^2 hemi-oct ortho DEPTH maps (8-bit, one rg8
// atlas) answer "which surface does this ray see"; the reconstructed 3D hit
// point then indexes view-INDEPENDENT 3D albedo/normal volumes. The 4 views
// nearest the eye ray are blended by coverage-weighted hit position — since
// colour is a function of the hit point only, view interpolation can never
// double-image colours, only soften geometry. Placement is the scatter twin
// over a bounded camera-centred region: per-frame cost is independent of total
// plant count.
//
// Carpet species (stand_table.carpet_div > 0, i.e. the bog's Sphagnum) keep
// the stand's grid, its 90-degree yaw and its constant scale exactly as given
// — the proxy card is camera-facing but the SAMPLED tile never is, so the
// lattice invariant holds. What differs for a mat: the tile is anchored on its
// periodic square (not on the mesh origin), the plant frame is tilted into the
// ground plane, the proxy hugs the tile's flat bounding slab instead of its
// bounding sphere, the alpha reference is low (a mat is a closed surface), the
// camera-inside fade is off (never open a hole under the viewer's feet), and
// per-fragment normals come from the light field's own depth gradient.

struct QlfInfo {
  center: vec3f, radius: f32,      // mesh-local bbox center + bounding radius (m)
  bmin: vec3f, grid: f32,
  bmax: vec3f, tile: f32,
  atlas_px: f32, entry_index: f32, origin_cell: vec2f,
  side: f32, seed: f32, region_r: f32, cov_threshold: f32,
  refine: f32, height_m: f32, show_view_grid: f32, relief: f32,
  // Padded half-extents of the mesh bbox in unit-sphere space. Each ortho view
  // is framed on the BOX support along its own axes, not on the bounding
  // sphere: a 0.21 x 0.09 x 0.23 m Sphagnum slab inside a 0.33 m sphere would
  // otherwise waste 70% of every tile's texels on empty space.
  half_u: vec3f, cell_rad: f32,
  // Carpet distance collapse, in unit-sphere local space: the periodic tile
  // square (centre.xy, half size.z) and the mean capitulum height (.w).
  slab: vec4f,
  base_y: f32, collapse_near: f32, collapse_far: f32, _q0: f32,
  // Inward normalized frustum planes — the cell-level reject. Rejecting a cell
  // here skips the scatter's 8 heightmap taps per candidate, which is what
  // makes 484 slots per cell affordable.
  planes: array<vec4f, 6>,
}
@group(1) @binding(0) var<uniform> qlf: QlfInfo;
@group(1) @binding(1) var depth_lf: texture_2d<f32>;
@group(1) @binding(2) var vol_albedo: texture_3d<f32>;
@group(1) @binding(3) var vol_normal: texture_3d<f32>;
@group(1) @binding(4) var qlf_samp: sampler;

/// Extra size of a carpet's proxy card over the tile's exact box support, so
/// the eye-ray refine tap (which slides the query along the ray) still has its
/// answer inside the card.
const CARPET_CARD_PAD: f32 = 1.25;

fn hemioct_encode(v: vec3f) -> vec2f {
  let s = abs(v.x) + abs(v.y) + abs(v.z);
  let p = vec2f(v.x, v.z) / max(s, 1e-6);
  return vec2f(p.x + p.y, p.x - p.y); // [-1,1]
}

fn hemioct_decode(e: vec2f) -> vec3f {
  let px = (e.x + e.y) * 0.5;
  let pz = (e.x - e.y) * 0.5;
  let y = 1.0 - abs(px) - abs(pz);
  return normalize(vec3f(px, y, pz));
}

// Everything that is constant across one card is interpolated FLAT (or lives
// in the vertex stage): only the sample position varies per fragment.
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) off_unit: vec3f,                       // sample point, rest-pose unit-sphere local
  @location(1) @interpolate(flat) center: vec3f,      // plant bbox center, world
  @location(2) @interpolate(flat) cam_unit: vec3f,    // camera, rest-pose unit-sphere local
  @location(3) @interpolate(flat) sway: vec3f,        // wind displacement at full lean
  @location(4) @interpolate(flat) bx: vec3f,          // plant basis, columns (local -> world)
  @location(5) @interpolate(flat) by: vec3f,
  @location(6) @interpolate(flat) bz: vec3f,
  @location(7) @interpolate(flat) rw_fade: vec2f,     // bounding radius (world), coverage fade
  @location(8) @interpolate(flat) base_h_d: vec4f,    // base y (world), height (world), camera distance, collapse metric
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var out: VOut;
  let entry = u32(qlf.entry_index);
  let st = stand_table[entry];
  let is_carpet = st.carpet_div > 0.0;
  // Slots per cell come from the ENTRY, never from the global scatter budget:
  // a carpet has carpet_div^2 (484 at Sphagnum life size), and enumerating 128
  // of them would render a striped quarter of the mat.
  let slots = select(SCATTER_MAX_PER_CELL, u32(st.carpet_div) * u32(st.carpet_div), is_carpet);
  let side = u32(qlf.side);
  let slot = ii % slots;
  let cell_lin = ii / slots;
  let cxi = i32(qlf.origin_cell.x) + i32(cell_lin % side);
  let czi = i32(qlf.origin_cell.y) + i32(cell_lin / side);

  // --- cell-level rejects: cheap, cell-uniform, and they run BEFORE the
  // scatter, whose wetness field costs 8 heightmap taps per candidate.
  let cell_mid = (vec2f(f32(cxi), f32(czi)) + vec2f(0.5)) * SCATTER_CELL_SIZE;
  let to_cell = cell_mid - frame.camera_pos.xz;
  if (length(to_cell) - qlf.cell_rad > qlf.region_r) {
    out.pos = vec4f(0.0, 0.0, 2.0, 1.0); // degenerate, no fragments
    return out;
  }
  // Conservative box over every card this cell can hold: its footprint plus the
  // widest card in xz, and the whole terrain height range in y (the shared FBM
  // never leaves +-1.05 * height_scale, and reading the real height here would
  // cost the taps this test exists to avoid).
  let box_c = vec3f(cell_mid.x, 0.0, cell_mid.y);
  let box_e = vec3f(qlf.cell_rad, frame.terrain_height_scale * 1.05 + qlf.cell_rad, qlf.cell_rad);
  for (var i = 0u; i < 6u; i++) {
    let pl = qlf.planes[i];
    if (dot(pl.xyz, box_c) + pl.w + dot(abs(pl.xyz), box_e) < 0.0) {
      out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
      return out;
    }
  }

  let sp = scatter_candidate(u32(qlf.seed), entry, vec2i(cxi, czi), slot);
  if (!sp.exists) {
    out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
    return out;
  }

  let scale = sp.scale;
  let cs = vec2f(cos(sp.yaw), sin(sp.yaw));
  // Plant frame. slope_align = 0 keeps the exact yaw-only rotation (grasses
  // grow upright); a carpet (align 1) is tilted into the ground plane, which
  // is rung 1 of the fitting ladder and here effectively exact: a 0.18 m tile
  // is smaller than ONE 0.5 m heightmap texel, so the ground under it is a
  // bilinear patch that its tangent plane tracks to ~3 mm, and neighbouring
  // tiles therefore cannot crack apart the way a plane fit over a large
  // footprint would (see NOTES.md).
  var basis = mat3x3f(vec3f(cs.x, 0.0, -cs.y), vec3f(0.0, 1.0, 0.0), vec3f(cs.y, 0.0, cs.x));
  if (st.slope_align > 0.001) {
    let g = terrain_sample(sp.pos.xz); // (height, nx, nz) in ONE bilinear fetch
    let n_t = vec3f(g.y, sqrt(max(1.0 - g.y * g.y - g.z * g.z, 0.0)), g.z);
    basis = plant_basis_from_up(mix(vec3f(0.0, 1.0, 0.0), n_t, st.slope_align), sp.yaw);
  }

  // Where the scatter's position sits in the mesh frame. For a carpet that is
  // the centre of the periodic TILE (footprint_m/2), not the mesh origin: the
  // 90-degree yaws rotate about the tile centre, so every rotation still lands
  // its square on the same grid. Anchoring on the mesh origin (a tile corner)
  // shifts each yaw into a different quadrant and shreds the mat.
  let anchor = select(vec3f(0.0), vec3f(st.footprint_m * 0.5, 0.0, st.footprint_m * 0.5), is_carpet);
  let center = sp.pos + basis * ((qlf.center - anchor) * scale);
  let rw = qlf.radius * scale;

  // Far fade at the region edge + fade-out when the camera is inside a plant.
  // A carpet gets NO near fade: a mat you are standing on must not open a hole.
  let dcam = distance(frame.camera_pos, center);
  let far_fade = 1.0 - smoothstep(qlf.region_r * 0.72, qlf.region_r * 0.97, dcam);
  let near_fade = select(smoothstep(rw * 0.45, rw * 1.1, dcam), 1.0, is_carpet);
  let fade = far_fade * near_fade;
  if (fade < 0.004) {
    out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
    return out;
  }

  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  let c = corners[vi];

  // Spherical (camera-facing) billboard basis for the proxy card.
  let dir = normalize(frame.camera_pos - center);
  var up_ref = vec3f(0.0, 1.0, 0.0);
  if (abs(dir.y) > 0.99) { up_ref = vec3f(1.0, 0.0, 0.0); }
  let right = normalize(cross(up_ref, dir));
  let up = normalize(cross(dir, right));

  // Card axes in the plant's local frame — the sample-space basis.
  let right_l = transpose(basis) * right;
  let up_l = transpose(basis) * up;

  // Card extent. An upright plant spans the bounding sphere (a tighter card
  // was tried and rejected in the audit — the refine tap reads outside the
  // projected footprint, see NOTES.md). A CARPET tile is a flat slab: 3.3 cm
  // of relief in a 0.33 m sphere, so the sphere card wastes ~3x the fill and
  // hangs metres of empty quad below ground. Size it from the exact box
  // support along the card axes instead.
  var hw = rw;
  var hh = rw;
  if (is_carpet) {
    let half_w = qlf.half_u * rw; // world half-extents of the mesh bbox
    hw = dot(abs(right_l), half_w) * CARPET_CARD_PAD;
    hh = dot(abs(up_l), half_w) * CARPET_CARD_PAD;
  }

  // Wind: the card leans with height, and the fragment shader samples in the
  // plant's REST pose (the un-sheared card point below), which is exactly what
  // makes the sprite itself lean (unlike leaning only the card window).
  let sway_vec = wind_sway(sp.pos, frame.time, st.sway, sp.phase);
  let base_y = sp.pos.y;
  let height_w = qlf.height_m * scale;
  let corner_y = center.y + c.y * hh;
  let t_lean = clamp((corner_y - base_y) / max(height_w, 1e-4), 0.0, 1.0);
  let world = center + right * (c.x * hw) + up * (c.y * hh) + sway_vec * t_lean;

  out.pos = frame.view_proj * vec4f(world, 1.0);
  // The rest-pose sample point is affine in the card corner, so interpolating
  // it directly is identical to un-shearing world space per fragment — and the
  // frame rotation happens once per PLANT instead of once per fragment.
  out.off_unit = right_l * (c.x * hw / rw) + up_l * (c.y * hh / rw);
  out.center = center;
  out.cam_unit = (transpose(basis) * (frame.camera_pos - center)) / max(rw, 1e-4);
  out.sway = sway_vec;
  out.bx = basis[0];
  out.by = basis[1];
  out.bz = basis[2];
  out.rw_fade = vec2f(rw, fade);
  // Collapse metric: distance, stretched as the view flattens. A mat seen at a
  // few degrees is foreshortened along the ground, so its texels per pixel
  // collapse long before a top-down view of the same tile does — but its
  // vertical relief profile is NOT foreshortened, so the stretch is mild (1.8x
  // at full grazing) rather than the full 1/sin(elevation).
  let flatten = 1.0 + 0.8 * (1.0 - clamp(abs(dir.y), 0.0, 1.0));
  out.base_h_d = vec4f(base_y, height_w, dcam, dcam * flatten);
  return out;
}

// --- light-field lookup ------------------------------------------------------

struct ViewTap {
  cov: f32,     // bilinear coverage of this view's 2x2 depth footprint
  hit: vec3f,   // reconstructed hit point, unit-sphere local space
  nrm: vec3f,   // relief normal from the depth gradient (unit space)
  nw: f32,      // 1 when the whole 2x2 footprint is surface, so nrm is valid
}

// One view of the depth light field: project the query point into the view's
// ortho basis (framed on the box support along that basis, so no texels are
// spent on empty space), read fractional coverage (bilinear — it is
// continuous) and gather the 2x2 depth footprint (miss sentinel = 1.0),
// reconstruct the 3D hit point on that view's ortho ray, and take the depth
// gradient of the same footprint as a mesoscale relief normal.
fn tap_view(node: vec2f, p_unit: vec3f) -> ViewTap {
  var t: ViewTap;
  let gm1 = qlf.grid - 1.0;
  let e = node / gm1 * 2.0 - 1.0;
  let fwd = hemioct_decode(e);
  var up_ref = vec3f(0.0, 1.0, 0.0);
  if (abs(fwd.y) > 0.999) { up_ref = vec3f(0.0, 0.0, 1.0); }
  let bx = normalize(cross(up_ref, fwd));
  let by = normalize(cross(fwd, bx));
  // Ortho half-extents of this view = support function of the bbox along its
  // axes. MUST match the bake (bake.ts writes 1/support per view).
  let su = dot(abs(bx), qlf.half_u);
  let sv = dot(abs(by), qlf.half_u);
  let sw = dot(abs(fwd), qlf.half_u);

  let perp = p_unit - fwd * dot(p_unit, fwd);
  // NDC +y is framebuffer row 0, so v flips.
  var uv = vec2f(dot(p_unit, bx) / su * 0.5 + 0.5, 0.5 - dot(p_unit, by) / sv * 0.5);
  let inset = 1.5 / qlf.tile;
  uv = clamp(uv, vec2f(inset), vec2f(1.0 - inset));
  let atlas_uv = (node + uv) * qlf.tile / qlf.atlas_px;

  let cov_frac = textureSampleLevel(depth_lf, qlf_samp, atlas_uv, 0.0).g;
  let g = textureGather(0, depth_lf, qlf_samp, atlas_uv);
  // Gather components: x=(0,1) y=(1,1) z=(1,0) w=(0,0) of the footprint.
  let f = fract(atlas_uv * qlf.atlas_px - 0.5);
  let hit_mask = step(g, vec4f(0.998)); // 1 where a surface was captured
  let w4 = vec4f((1.0 - f.x) * f.y, f.x * f.y, f.x * (1.0 - f.y), (1.0 - f.x) * (1.0 - f.y));
  let cw = w4 * hit_mask;
  let wsum = dot(cw, vec4f(1.0));
  let d8 = dot(cw, g) / max(wsum, 1e-5);
  let z01 = d8 * (255.0 / 254.0); // undo the sentinel rescale
  t.cov = select(0.0, cov_frac, wsum > 1e-4);
  t.hit = perp + fwd * ((1.0 - 2.0 * z01) * sw);

  // Relief normal: the gather already fetched the 2x2 neighbourhood, so the
  // depth gradient across it is free apart from ALU. Only meaningful where all
  // four texels are surface (otherwise the difference spans a silhouette).
  t.nw = min(min(hit_mask.x, hit_mask.y), min(hit_mask.z, hit_mask.w));
  // Depth codes -> metres along fwd (the same (1-2z)*sw mapping, differenced).
  let k = -2.0 * sw * (255.0 / 254.0);
  let dh_du = ((g.y + g.z) - (g.x + g.w)) * 0.5 * k; // per texel along +bx
  let dh_dv = ((g.x + g.y) - (g.z + g.w)) * 0.5 * k; // per texel along -by
  let da = 2.0 * su / qlf.tile;
  let db = -2.0 * sv / qlf.tile;
  t.nrm = fwd - bx * (dh_du / da) - by * (dh_dv / db);
  return t;
}

struct QuadTap {
  cov: f32,
  hit: vec3f,  // coverage-weighted sum; divide by cov
  nrm: vec3f,  // coverage-weighted relief normal sum
  nw: f32,
}

fn tap_quad(g0: vec2f, wd: vec4f, p_unit: vec3f) -> QuadTap {
  let t00 = tap_view(g0, p_unit);
  let t10 = tap_view(g0 + vec2f(1.0, 0.0), p_unit);
  let t01 = tap_view(g0 + vec2f(0.0, 1.0), p_unit);
  let t11 = tap_view(g0 + vec2f(1.0, 1.0), p_unit);
  var q: QuadTap;
  q.cov = wd.x * t00.cov + wd.y * t10.cov + wd.z * t01.cov + wd.w * t11.cov;
  q.hit = wd.x * t00.cov * t00.hit + wd.y * t10.cov * t10.hit
        + wd.z * t01.cov * t01.hit + wd.w * t11.cov * t11.hit;
  let n4 = vec4f(wd.x * t00.nw, wd.y * t10.nw, wd.z * t01.nw, wd.w * t11.nw);
  q.nrm = n4.x * normalize(t00.nrm) + n4.y * normalize(t10.nrm)
        + n4.z * normalize(t01.nrm) + n4.w * normalize(t11.nrm);
  q.nw = n4.x + n4.y + n4.z + n4.w;
  return q;
}

/// Debug param `showViewGrid`: stable colour per hemi-oct view node, so the
/// angular grid and its 4-view blend region are visible directly.
fn node_tint(node: vec2f) -> vec3f {
  let h = hash2(u32(node.x), u32(node.y));
  return vec3f(hash_f32(hash2(h, 1u)), hash_f32(hash2(h, 2u)), hash_f32(hash2(h, 3u))) * 0.7 + 0.15;
}

struct FOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_main(in: VOut) -> FOut {
  let basis = mat3x3f(in.bx, in.by, in.bz);
  let rw = in.rw_fade.x;
  let fade = in.rw_fade.y;
  let off_unit = in.off_unit;
  let cam_unit = in.cam_unit;

  // Per-fragment eye direction (toward camera), clamped to the baked
  // hemisphere.
  var vdir = normalize(cam_unit - off_unit);
  if (vdir.y < 0.02) { vdir = normalize(vec3f(vdir.x, 0.02, vdir.z)); }

  // --- distance collapse (carpets only; collapse_near is 1e9 for plants) -----
  // Past a few metres a 0.9mm texel is far below one pixel, so the light field
  // can only alias — one point sample per tile, flickering between leaf and
  // gap. What the mat honestly IS at that range is a closed slab of cushion
  // tops, so it becomes exactly that: ray vs the periodic tile square between
  // the ground and the mean capitulum height. Neighbouring slabs abut, so the
  // far field is continuous, quiet and free of taps. No dithering involved.
  let eray = normalize(off_unit - cam_unit);
  var far_w = smoothstep(qlf.collapse_near, qlf.collapse_far, in.base_h_d.w);
  var slab_hit = off_unit;
  var slab_cov = 0.0;
  if (far_w > 0.001) {
    let lo = vec3f(qlf.slab.x - qlf.slab.z, qlf.base_y, qlf.slab.y - qlf.slab.z);
    let hi = vec3f(qlf.slab.x + qlf.slab.z, qlf.slab.w, qlf.slab.y + qlf.slab.z);
    let ad = max(abs(eray), vec3f(1e-6));
    let dsafe = select(ad, -ad, eray < vec3f(0.0)); // no zero components -> no NaN
    let t0 = (lo - cam_unit) / dsafe;
    let t1 = (hi - cam_unit) / dsafe;
    let ta = min(t0, t1);
    let tb = max(t0, t1);
    let t_in = max(max(ta.x, ta.y), max(ta.z, 0.0));
    let t_out = min(min(tb.x, tb.y), tb.z);
    slab_cov = select(0.0, 1.0, t_out > t_in);
    slab_hit = cam_unit + eray * t_in;
  }

  let e = hemioct_encode(vdir);
  let gpos = clamp((e * 0.5 + 0.5) * (qlf.grid - 1.0), vec2f(0.0), vec2f(qlf.grid - 1.001));
  let g0 = min(floor(gpos), vec2f(qlf.grid - 2.0));
  let gf = gpos - g0;
  let wd = vec4f((1.0 - gf.x) * (1.0 - gf.y), gf.x * (1.0 - gf.y), (1.0 - gf.x) * gf.y, gf.x * gf.y);

  var hit = slab_hit;
  var cov = slab_cov * far_w;
  var relief_n = vec3f(0.0);
  var relief_w = 0.0;
  if (far_w < 0.999) {
    var q = tap_quad(g0, wd, off_unit);

    // One eye-ray reprojection: slide the query to where the first answer says
    // the surface is, along the true eye ray, and ask again. A precomputed
    // lookup refinement, not a march. Beyond ~45m the residual parallax of the
    // view grid is subpixel, so skip the second round of taps.
    // ...and not once the collapse is taking over anyway (far_w), which is what
    // spares the mid-field band a second round of 4 taps.
    if (q.cov > 1e-4 && qlf.refine > 0.5 && in.base_h_d.z < 45.0 && far_w < 0.35) {
      let hp = cam_unit + eray * dot(q.hit / q.cov - cam_unit, eray);
      let q2 = tap_quad(g0, wd, hp);
      if (q2.cov > 0.05) { q = q2; }
    }
    if (q.cov > 1e-4) {
      hit = mix(q.hit / q.cov, slab_hit, far_w);
      cov = mix(q.cov, slab_cov, far_w);
      relief_n = q.nrm;
      relief_w = q.nw;
    }
  }

  // Hard alpha-test edge (honest coverage falloff via fade, no dithering).
  let coverage = cov * fade;
  if (coverage < qlf.cov_threshold) { discard; }

  // Radiance factor: the hit point indexes the view-independent volumes. A
  // collapsed tile reads its colour from the cushion TOPS (the surface it is
  // standing in for), not from the slab face the ray happened to enter.
  let hit_alb = vec3f(hit.x, mix(hit.y, qlf.slab.w, far_w), hit.z);
  let hit_m = qlf.center + hit_alb * qlf.radius; // mesh-local metres
  let uvw = clamp((hit_m - qlf.bmin) / max(qlf.bmax - qlf.bmin, vec3f(1e-4)), vec3f(0.0), vec3f(1.0));
  let alb_tap = textureSampleLevel(vol_albedo, qlf_samp, uvw, 0.0);
  var n_mesh = textureSampleLevel(vol_normal, qlf_samp, uvw, 0.0).xyz * 2.0 - 1.0;
  if (length(n_mesh) < 1e-3) { n_mesh = vec3f(0.0, 1.0, 0.0); }
  n_mesh = normalize(n_mesh);

  // A dense cushion averages every leaf orientation into one voxel, so its
  // volume normal is close to noise. The light field's own depth gradient is
  // the honest mesoscale normal for that geometry — cushion relief instead of
  // salt-and-pepper — so a carpet leans on it (relief > 0) and only keeps the
  // volume normal where the footprint straddled a silhouette.
  if (qlf.relief > 0.001 && relief_w > 0.05) {
    let n_relief = normalize(relief_n / relief_w);
    n_mesh = normalize(mix(n_mesh, n_relief, qlf.relief * clamp(relief_w, 0.0, 1.0)));
  }
  // A collapsed tile stands in for the whole cushion mass, so its normal is the
  // AREA-WEIGHTED MEAN of the slab faces the viewer can see: mostly up from
  // above, tilting toward horizontal at grazing in proportion to the mat's
  // thickness/width ratio. That is what keeps the collapsed band from lighting
  // up brighter than the light-field band it replaces — a flat up normal would
  // give every distant fragment full sun, which is exactly the "far field gets
  // brighter" failure mode.
  if (far_w > 0.001) {
    let vv = -eray; // toward the camera, local space
    let t_rel = (qlf.slab.w - qlf.base_y) / max(2.0 * qlf.slab.z, 1e-4);
    let w = vec3f(t_rel * abs(vv.x), abs(vv.y), t_rel * abs(vv.z));
    let n_slab = normalize(vec3f(sign(vv.x) * w.x, w.y, sign(vv.z) * w.z));
    n_mesh = normalize(mix(n_mesh, n_slab, far_w));
  }

  // Re-inject the baked per-voxel luminance sigma as deterministic speckle
  // (hash of the quantized mesh-local hit — stable under wind and camera),
  // so voxel-averaged interiors keep sub-voxel texture.
  let spk = vec3i(floor(hit_m * 66.0)); // ~1.5cm cells
  let hn = hash_f32(hash3(bitcast<u32>(spk.x), bitcast<u32>(spk.y), bitcast<u32>(spk.z)));
  let sigma = alb_tap.a * 0.5;
  var albedo = alb_tap.rgb * clamp(1.0 + (hn - 0.5) * sigma * 2.6, 0.55, 1.55);
  if (qlf.show_view_grid > 0.5) {
    albedo = node_tint(g0) * wd.x + node_tint(g0 + vec2f(1.0, 0.0)) * wd.y
           + node_tint(g0 + vec2f(0.0, 1.0)) * wd.z + node_tint(g0 + vec2f(1.0, 1.0)) * wd.w;
  }

  // World-space hit: re-frame (yaw + ground tilt), re-scale, re-shear, then
  // true depth.
  var hit_w = in.center + basis * (hit * rw);
  let t_hit = clamp((hit_w.y - in.base_h_d.x) / max(in.base_h_d.y, 1e-4), 0.0, 1.0);
  hit_w += in.sway * t_hit;
  let clip = frame.view_proj * vec4f(hit_w, 1.0);

  var n_ws = basis * n_mesh;
  let view_ws = normalize(frame.camera_pos - hit_w);
  if (dot(n_ws, view_ws) < 0.0) { n_ws = -n_ws; } // two-sided foliage

  // Height-based baked AO stand-in: canopy tops lit, roots dimmed.
  let ao = mix(0.6, 1.0, clamp(hit_m.y / max(qlf.height_m, 1e-3), 0.0, 1.0));

  var color = light_surface(albedo * ao, n_ws, hit_w);
  // Fog only in the normal view — debug views stay unfogged and honest.
  if (debug_mode() == DEBUG_OFF) {
    color = apply_fog(color, hit_w);
  }

  var o: FOut;
  o.color = vec4f(debug_shade(color, albedo, n_ws, coverage, hit_w), 1.0);
  o.depth = clamp(clip.z / max(clip.w, 1e-4), 0.0, 1.0);
  return o;
}
