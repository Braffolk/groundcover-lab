#include "src/wgsl/wind.wgsl"
#include "src/wgsl/hash.wgsl"
#include "src/wgsl/terrain.wgsl"
#include "./common.wgsl"

// Pass 2: half-res splat scatter. One indexed indirect draw per (stand entry,
// LOD); instances are the culled plant records, vertices enumerate that LOD's
// splats (4 unique corners per splat, 6 shared indices — a quad only has four
// corners, so shading six would waste a third of this pass's vertex work
// across the ~1M splats a frame draws). Each splat is an anisotropic quad spanned by
// its baked elongation axis and the view-perpendicular, dither-discarded to
// its elliptical footprint so depth stays exact. Outputs an albedo+height
// target and a normal+viewdepth target for the reconstruction pass.

struct DrawInfo {
  bounds_min: vec4f, // xyz = splat bounds min (plant-local), w = topH
  bounds_ext: vec4f, // xyz = splat bounds extent, w = max ra (m)
  misc: vec4f,       // x = max rb (m), y = far bucket flag, z = mean local y
  idx: vec4u,        // splat offset, splat count, bucket base, entry index
  far: vec4f,        // whole-tile mean albedo + colour variance (far bucket)
}

@group(1) @binding(0) var<uniform> di: DrawInfo;
@group(1) @binding(1) var<storage, read> records: array<PlantRec>;
@group(1) @binding(2) var<storage, read> splats: array<vec4u>;
@group(1) @binding(3) var<uniform> params: SrParams;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) uv_hf: vec3f,      // ellipse uv, height fraction
  @location(2) normal_vz: vec4f,  // world normal, linear view depth
  // colorVar, per-splat dither offset (NEGATIVE = hard-edged, see fs_splat).
  // Deliberately two components and not three: adding an interpolant perturbs
  // the interpolation of the others enough to flip dithered pixels on the
  // upright-plant path, which must stay identical.
  @location(3) misc: vec2f,
}

fn rot_y(v: vec3f, c: f32, s: f32) -> vec3f {
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

@vertex
fn vs_splat(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let rec = records[di.idx.z + ii];
  // Indexed quads: 4 corners per splat, the index buffer spans the two
  // triangles (0,1,2, 1,3,2) — same triangles as an unindexed 6-vertex quad.
  let s = vi / 4u;
  var corners = array<vec2f, 4>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(1.0, 1.0),
  );
  let c = corners[vi % 4u];

  let w = splats[di.idx.x + s];
  let local = di.bounds_min.xyz + vec3f(
    f32(w.x & 0xffffu), f32(w.x >> 16u), f32(w.y & 0xffffu),
  ) / 65535.0 * di.bounds_ext.xyz;
  let albedo = vec3f(
    f32((w.y >> 16u) & 0xffu), f32(w.y >> 24u), f32(w.z & 0xffu),
  ) / 255.0;
  let cvar = f32((w.z >> 8u) & 0xffu) / 255.0;
  let n_local = sr_oct_decode(f32((w.z >> 16u) & 0xffu) / 255.0, f32(w.z >> 24u) / 255.0);
  let a_local = sr_oct_decode(f32(w.w & 0xffu) / 255.0, f32((w.w >> 8u) & 0xffu) / 255.0);
  let ra_q = f32((w.w >> 16u) & 0xffu) / 255.0;
  let rb_q = f32(w.w >> 24u) / 255.0;

  let ys = unpack2x16float(rec.ys);
  let yaw = ys.x;
  let scale = ys.y;
  let base_y = unpack2x16float(rec.pfy & 0xffffu).x;
  let phase = f32((rec.pfy >> 16u) & 0xffu) / 255.0 * 6.2831853;
  let fade = f32(rec.pfy >> 24u) / 255.0;

  let cy = cos(yaw);
  let sy = sin(yaw);
  let base = vec3f(rec.xz.x, base_y, rec.xz.y);
  let entry = stand_table[di.idx.w];
  // A carpet (stand_table.carpet_div > 0) is a periodic MAT, not an upright
  // plant: it conforms to the ground per splat and never fades under the
  // camera. Everything else keeps the upright-plant path bit-for-bit.
  let carpet = entry.carpet_div > 0.0;

  // Plant fade dissolves whole splats (not pixels): a partially-faded plant
  // must never leave sparse dithered depth samples that occlude the field
  // behind it — that punches sky-colored holes into the reconstruction.
  let pid = hash2(bitcast<u32>(rec.xz.x), bitcast<u32>(rec.xz.y));
  let rnd = hash_f32(hash2(pid, s));
  if (rnd >= fade) {
    var dead: VOut;
    dead.pos = vec4f(2.0, 2.0, 2.0, 1.0); // zero-area, off-screen
    return dead;
  }

  // --- Far carpet bucket: ONE ground-parallel disc per tile ------------------
  // Beyond ~25m a life-size mat tile is ~2px, where the six splats of the
  // coarsest baked level are all clamped to the same one-pixel minimum: 20x the
  // fragments needed, and (every tile painting the identical six-blob pattern)
  // a corduroy moire across the mid field. One disc per tile at the tile centre
  // carries exactly the information a 2px tile can show: its mean colour.
  // GROUND-PARALLEL, not view-facing: a view-facing disc wide enough to cover a
  // 0.18m tile would stand 0.2m proud of a 0.09m mat and, at a grazing camera,
  // paint a fuzzy band floating above the ground.
  if (di.misc.y > 0.5) {
    let g = terrain_sample(rec.xz);
    let t_ny = sqrt(max(1.0 - g.y * g.y - g.z * g.z, 1.0e-4));
    let up = vec3f(g.y, t_ny, g.z);
    let center_f = vec3f(rec.xz.x, g.x + di.misc.z * scale, rec.xz.y);
    // Hard-edged, so the disc only has to contain the tile square:
    // r >= sqrt(2)/2 * step, plus a hair for the grid's terrain shear.
    let vz_f = -(frame.view * vec4f(center_f, 1.0)).z;
    let r_f = max(
      0.72 * entry.footprint_m * scale,
      2.5 * vz_f / (frame.proj[1][1] * frame.viewport.y * 0.5),
    );
    let t_a = normalize(cross(up, vec3f(0.0, 0.0, 1.0)));
    let t_b = cross(t_a, up);
    let world_f = center_f + t_a * (r_f * c.x) + t_b * (r_f * c.y);
    var far_out: VOut;
    far_out.pos = frame.view_proj * vec4f(world_f, 1.0);
    far_out.color = di.far.rgb * (0.82 + 0.36 * hash_f32(hash2(pid, 0x5bd1e995u)));
    // The far disc IS the top surface of the cushion, so it takes the height
    // fraction of the cushion top, not the mean over the whole 9cm of moss —
    // using the mean would make the distant mat ~20% darker than the near field
    // through the height AO in the reconstruction pass.
    far_out.uv_hf = vec3f(c, CARPET_TOP_HF);
    far_out.normal_vz = vec4f(up, -(frame.view * vec4f(world_f, 1.0)).z);
    far_out.misc = vec2f(di.far.a, -1.0);
    return far_out;
  }

  // Height fraction: drives wind (tips move, roots do not) and the height AO in
  // the reconstruction pass. Tried and rejected for mats: a steeper curve
  // (hf^2.3, on the theory that a dense cushion shadows its gaps harder than a
  // grass tuft). It measured 10% darker with LOWER local contrast — the visible
  // surface of a mat is its top, so hf is nearly constant there and the exponent
  // only shifts the level.
  let hf = clamp(local.y / max(di.bounds_min.w, 1e-3), 0.0, 1.0);
  let sway = wind_sway(base, frame.time, entry.sway, phase) * (hf * (0.55 + 0.45 * hf));
  var center = base + rot_y(local, cy, sy) * scale + sway;
  var normal_ws = rot_y(n_local, cy, sy);
  if (carpet) {
    // Terrain fitting, ladder rung 3-4: every SPLAT is conformed at its own xz
    // (one bilinear terrain_sample gives height AND the (nx, nz) the shading
    // needs). A vertical shear rather than a rigid per-tile tilt, because two
    // neighbouring tiles that each fit their own plane crack apart along their
    // shared edge, whereas any two splats at the same xz get the same ground
    // height by construction — the mat stays exactly continuous. Over 9cm of
    // moss the difference between "vertical" and "normal to the slope" is
    // sub-millimetre, so shearing costs nothing visually.
    let g = terrain_sample(center.xz);
    center.y = g.x + local.y * scale;
    // Normal under that shear: n' = J^-T n, with dH/dx = -nx/ny.
    let t_ny = sqrt(max(1.0 - g.y * g.y - g.z * g.z, 1.0e-4));
    normal_ws = normalize(vec3f(
      normal_ws.x + g.y / t_ny * normal_ws.y,
      normal_ws.y,
      normal_ws.z + g.z / t_ny * normal_ws.y,
    ));
  }

  // Radii are sqrt-encoded against the per-species maxima.
  var ra = ra_q * ra_q * di.bounds_ext.w * scale * params.splat_scale;
  var rb = rb_q * rb_q * di.misc.x * scale * params.splat_scale;
  if (carpet) {
    // Closure floor. A mat has to stay a closed surface at every LOD, and a
    // coarse LOD's splats are baked from the mesh's own point spread, not from
    // "cover this tile". n discs of effective radius 0.65r (the dither falloff
    // saturates there) close a footprint^2 square when r >= 0.87*footprint/sqrt(n).
    // Binds only where the bake would otherwise leave the tile holed.
    let r_floor = 0.87 * entry.footprint_m * scale / sqrt(f32(max(di.idx.y, 1u)));
    // ...and a ceiling of half a tile. A Morton chunk that straddles a high-level
    // boundary of the curve can span two distant parts of the mesh, and its
    // covariance then describes the gap rather than any geometry: the moss bake
    // has a few 13cm splats at levels where the typical radius is 3cm. On a mat
    // that is a quarter of the whole tile smeared into one blob. (The same bake
    // artifact at 0.6m in the grass levels is where the 48px screen clamp came
    // from; it is left alone there because it would change `default`.)
    let r_lim = max(0.5 * entry.footprint_m * scale, r_floor);
    ra = clamp(ra, r_floor, r_lim);
    rb = clamp(rb, r_floor, r_lim);
  }

  // Quad basis: baked elongation axis projected off the view direction
  // (min width rb when a blade is seen end-on), view-perpendicular second axis.
  let vdir = normalize(center - frame.camera_pos);
  let a_ws = rot_y(a_local, cy, sy);
  var a_p = a_ws - vdir * dot(a_ws, vdir);
  let la = length(a_p);
  var a_e: vec3f;
  if (la > 0.1) {
    a_e = a_p / la;
  } else {
    a_e = normalize(cross(vdir, vec3f(0.017, 1.0, 0.013)));
  }
  let b_e = cross(vdir, a_e);
  var ra_eff = max(ra * max(la, 0.35), rb);
  var rb_eff = rb;
  // Screen-space radius clamp: very close splats would otherwise cover
  // hundreds of pixels as one featureless blur-bomb. ~48px cap keeps the
  // camera-adjacent stems made of many small splats instead.
  let vz_c = -(frame.view * vec4f(center, 1.0)).z;
  let px_limit = 48.0 * vz_c / (frame.proj[1][1] * frame.viewport.y * 0.5);
  ra_eff = min(ra_eff, px_limit);
  rb_eff = min(rb_eff, px_limit);
  if (carpet) {
    // Screen-space closure floor: a 3cm splat on a life-size tile is under one
    // HALF-RES pixel by ~15m, and a sub-pixel splat can miss the sample grid
    // entirely, so the mat dissolves into speckle over bare peat. 2.5 full-res px
    // (~1.25 half-res) keeps every splat rasterizing at least one sample. Applies
    // to buckets 0-4; the far bucket has its own, on a hard-edged disc.
    let floor_px = 2.5 * vz_c / (frame.proj[1][1] * frame.viewport.y * 0.5);
    ra_eff = max(ra_eff, floor_px);
    rb_eff = max(rb_eff, floor_px);
  }
  let world = center + a_e * (ra_eff * c.x) + b_e * (rb_eff * c.y);

  var out: VOut;
  out.pos = frame.view_proj * vec4f(world, 1.0);
  // Per-splat brightness jitter: survives the reconstruction averaging that
  // would wash out pure per-pixel noise, and breaks up flat color masses.
  let jitter = 0.82 + 0.36 * hash_f32(hash2(pid, s ^ 0xa511e9b3u));
  out.color = albedo * jitter;
  out.uv_hf = vec3f(c, hf);
  let vz = -(frame.view * vec4f(world, 1.0)).z;
  out.normal_vz = vec4f(normal_ws, vz);
  out.misc = vec2f(cvar, rnd);
  return out;
}

struct FragOut {
  @location(0) color: vec4f, // albedo rgb, height fraction
  @location(1) aux: vec4f,   // world normal, linear view depth
}

@fragment
fn fs_splat(in: VOut) -> FragOut {
  let r2 = dot(in.uv_hf.xy, in.uv_hf.xy);
  if (r2 > 1.0) { discard; }
  let p = vec2u(in.pos.xy);
  // The far carpet disc is HARD: it is a closed ground surface, not a fuzzy
  // volume element, so it writes solid depth and needs no stochastic coverage
  // (which at that distance would only add screen-door and shimmer).
  if (in.misc.y >= 0.0) {
    // Dithered coverage: soft elliptical falloff x plant fade, resolved by the
    // reconstruction pass into smooth alpha. Screen-hash keeps it static.
    let noise = hash_f32(hash2(p.x, p.y ^ 0x517cc1b7u));
    let cover = clamp((1.0 - r2) * 2.4, 0.0, 1.0);
    if (cover <= fract(noise + in.misc.y)) { discard; }
  }

  var albedo = in.color;
  let vn = hash_f32(hash2(p.x * 3u + 1u, p.y ^ 0x9e3779b9u)) - 0.5;
  albedo = albedo * (1.0 + vn * in.misc.x * params.var_amt * 2.0);

  var out: FragOut;
  out.color = vec4f(albedo, in.uv_hf.z);
  out.aux = vec4f(in.normal_vz.xyz, in.normal_vz.w);
  return out;
}
