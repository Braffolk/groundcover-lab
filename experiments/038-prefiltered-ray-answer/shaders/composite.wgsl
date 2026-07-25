// BAKE STAGE 2 — composite one PATCH ray answer per baked direction.
//
// The patch is a 6x6 m periodic window of the ACTIVE STAND's own scatter (real
// positions, yaws, scales, species mix). For a direction d the answer is a 2D
// table indexed by E = where the ray crosses the canopy-top plane y = H:
//
//     E(X) = X.xz - d.xz * (X.y - H) / d.y            (invariant along d)
//
// E is affine in a plant's ortho image coordinates, so each plant's stage-1
// tile is drawn as ONE sheared quad into E space, with the tile's baked height
// as the depth. Along a downward ray y decreases monotonically, so "first hit"
// is exactly "greatest y" — the depth test composites the union of all plants
// exactly, at any density, with no marching.
//
// SS target 0 (surf): rgb = albedo, a = baked AO
// SS target 1 (geom): r = hit height / H, gb = oct normal * 0.5 + 0.5 (patch
//                     frame), a = 1 (the coverage marker the downsample weighs by)

struct CSlice {
  ax: vec4f,    // xyz: ortho right in patch space, w: 1 / atlas grid
  bx: vec4f,    // xyz: ortho up in patch space,    w: SS target dimension
  dir: vec4f,   // xyz: slice direction (unit, y < 0), w: reference plane y (= H)
  info: vec4f,  // x: azimuth index, y: elevation index, z: wrap side W, w: is_top
  cfg: vec4f,   // x: tile L (m), y: canopy H (m), z: AO range (m), w: atlas grid
  misc: vec4f,  // x: atlas side (px)
}
@group(0) @binding(0) var<uniform> sc: CSlice;
@group(0) @binding(1) var<storage, read> plants: array<vec4f>;
@group(0) @binding(2) var atlas_surf: texture_2d_array<f32>;
@group(0) @binding(3) var atlas_geom: texture_2d_array<f32>;
@group(0) @binding(4) var atlas_samp: sampler;
@group(0) @binding(5) var top_geom: texture_2d_array<f32>;

const N_AZ: u32 = 24u;
const N_EL: u32 = 6u;
const TWO_PI: f32 = 6.2831853;

fn oct_decode_mesh(e: vec2f) -> vec3f {
  var x = e.x;
  var z = e.y;
  let y = 1.0 - abs(e.x) - abs(e.y);
  if (y < 0.0) {
    x = (1.0 - abs(e.y)) * select(-1.0, 1.0, e.x >= 0.0);
    z = (1.0 - abs(e.x)) * select(-1.0, 1.0, e.y >= 0.0);
  }
  return normalize(vec3f(x, y, z));
}

fn oct_encode_mesh(n: vec3f) -> vec2f {
  let s = abs(n.x) + abs(n.y) + abs(n.z);
  if (s < 1e-6) {
    return vec2f(0.0);
  }
  var u = n.x / s;
  var v = n.z / s;
  if (n.y < 0.0) {
    let fu = (1.0 - abs(v)) * select(-1.0, 1.0, n.x >= 0.0);
    let fv = (1.0 - abs(u)) * select(-1.0, 1.0, n.z >= 0.0);
    u = fu;
    v = fv;
  }
  return vec2f(u, v);
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) auv: vec2f,
  @location(1) @interpolate(flat) slot: u32,
  @location(2) @interpolate(flat) rot: vec2f,     // cos/sin of the snapped yaw
  @location(3) @interpolate(flat) plant_h: f32,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var quad = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  let q = quad[vi];

  let w_side = u32(sc.info.z);
  let per_plant = w_side * w_side;
  let pi = ii / per_plant;
  let oi = ii % per_plant;
  let half_w = f32(w_side / 2u);
  let wrap = (vec2f(f32(oi % w_side), f32(oi / w_side)) - vec2f(half_w)) * sc.cfg.x;

  let d0 = plants[pi * 2u];
  let d1 = plants[pi * 2u + 1u];
  let plant_h = d0.z;
  let r = d0.w;
  let cs = d1.x;
  let sn = d1.y;
  let is_top = sc.info.w > 0.5;

  // Ortho basis in patch space. For a slanted slice the basis built from the
  // direction is yaw-invariant (rotating the mesh rotates the direction with
  // it), so it is shared by every plant; the straight-down tile is degenerate
  // and needs the plant's own snapped yaw.
  var ax = sc.ax.xyz;
  var bx = sc.bx.xyz;
  if (is_top) {
    ax = vec3f(cs, 0.0, sn);
    bx = vec3f(-sn, 0.0, cs);
  }

  let centre = vec3f(d0.x, plant_h * 0.5, d0.y);
  // The plant's own E-space centre, wrapped into the tile, so the periodic copy
  // window only has to cover the quad's extent and not the (large, grazing)
  // shear offset of the plant itself.
  var ec = centre.xz;
  var doff = r * (q.x * ax.xz + q.y * bx.xz);
  if (!is_top) {
    ec -= sc.dir.xz * ((centre.y - sc.dir.w) / sc.dir.y);
    doff -= sc.dir.xz * (r * (q.x * ax.y + q.y * bx.y) / sc.dir.y);
  }
  ec -= floor(ec / sc.cfg.x) * sc.cfg.x;
  let e = ec + doff + wrap;

  // Atlas tile: relative azimuth folds the plant's snapped yaw into the slice.
  let az = u32(sc.info.x);
  let el = u32(sc.info.y);
  let az_yaw = u32(d1.w);
  var tile = N_AZ * N_EL;
  if (!is_top) {
    tile = el * N_AZ + (az + N_AZ - az_yaw) % N_AZ;
  }
  let grid = sc.cfg.w;
  let scale = sc.ax.w;
  let origin = vec2f(f32(tile % u32(grid)), f32(tile / u32(grid))) * scale;
  let inset = 0.5 / sc.misc.x;
  var auv = origin + (vec2f(q.x, -q.y) * 0.5 + 0.5) * scale;
  auv = clamp(auv, origin + vec2f(inset), origin + vec2f(scale - inset));

  let uv = e / sc.cfg.x;
  var out: VOut;
  out.pos = vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.5, 1.0);
  out.auv = auv;
  out.slot = u32(d1.z);
  out.rot = vec2f(cs, sn);
  out.plant_h = plant_h;
  return out;
}

struct FOut {
  @builtin(frag_depth) depth: f32,
  @location(0) surf: vec4f,
  @location(1) geom: vec4f,
}

@fragment
fn fs(in: VOut) -> FOut {
  let s = textureSampleLevel(atlas_surf, atlas_samp, in.auv, in.slot, 0.0);
  let cov = s.a;
  if (cov < 0.5) {
    discard;
  }
  let g = textureSampleLevel(atlas_geom, atlas_samp, in.auv, in.slot, 0.0);
  let inv = 1.0 / cov;
  let y = (g.r * inv) * in.plant_h;
  let h_top = sc.cfg.y;
  if (y > h_top) {
    discard; // canopy truncated at H (documented in NOTES)
  }

  // Where on the ground this hit actually sits, for the AO lookup.
  let e = in.pos.xy / sc.bx.w * sc.cfg.x;
  var hit_xz = e;
  if (sc.info.w < 0.5) {
    hit_xz = e + sc.dir.xz * ((y - sc.dir.w) / sc.dir.y);
  }

  var ao = 1.0;
  if (sc.info.w < 0.5) {
    let t = textureSampleLevel(top_geom, atlas_samp, fract(hit_xz / sc.cfg.x), 0, 0.0);
    let top_h = (t.r / max(t.a, 1e-4)) * h_top;
    ao = 1.0 - clamp((top_h - y) / sc.cfg.z, 0.0, 1.0);
  }

  var n = oct_decode_mesh((g.gb * inv) * 2.0 - 1.0);
  n = vec3f(n.x * in.rot.x - n.z * in.rot.y, n.y, n.x * in.rot.y + n.z * in.rot.x);

  var out: FOut;
  out.depth = clamp(1.0 - y / h_top, 0.0, 1.0);
  out.surf = vec4f(s.rgb * inv, ao);
  out.geom = vec4f(clamp(y / h_top, 0.0, 1.0), oct_encode_mesh(n) * 0.5 + 0.5, 1.0);
  return out;
}
