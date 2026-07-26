// Bake resolve. Runs once per atlas row of tiles, over the supersampled
// capture of that row.
//
//   cs_albedo — coverage-weighted 2x downsample into the albedo/coverage atlas.
//   cs_geom   — 6x downsample into the geometry atlas AND the actual
//               self-occlusion solve: each output texel reconstructs its 3D
//               plant-local position from the captured view depth, then tests
//               32 precomputed visibility depth maps to get
//                 ao   = fraction of the upper hemisphere that is open
//                 bent = mean unoccluded direction
//               Stored as [octN.x, octN.y, ao, q] where N is the shading normal
//               (geometric normal bent toward the openness direction).

const GRID: u32 = 5u;
const TILE_A: u32 = 384u;
const TILE_B: u32 = 128u;
const SS_TILE: u32 = 768u;      // TILE_A * 2 supersample
const B_TO_SS: u32 = 6u;        // SS_TILE / TILE_B
const N_VIS: u32 = 32u;
const VIS_TILE: u32 = 256u;
const VIS_COLS: u32 = 8u;
const BENT_MIX: f32 = 0.32;     // how far the shading normal leans to `bent`

struct Tables {
  // 25 tiles x 4: (r.xyz, boxC.x) (u.xyz, boxC.y) (b.xyz, dNear) (hx, hy, D, _)
  tiles: array<vec4f, 100>,
  // 32 probes x 3: (r.xyz,_) (u.xyz,_) (d.xyz,_)
  probes: array<vec4f, 96>,
  sphere: vec4f, // xyz = bounding-sphere centre, w = radius
  misc: vec4f,   // x = plant height (plant-local, m)
}

struct RowUni {
  v: vec4f, // x = atlas row index
}

@group(0) @binding(0) var<uniform> tables: Tables;
@group(0) @binding(1) var<uniform> row_uni: RowUni;
@group(0) @binding(2) var src_albedo: texture_2d<f32>;
@group(0) @binding(3) var src_nrm: texture_2d<f32>;
@group(0) @binding(4) var vis_atlas: texture_2d<f32>;
@group(0) @binding(5) var dst_albedo: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(6) var dst_geom: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(7) var dst_mask: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn cs_albedo(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= GRID * TILE_A || gid.y >= TILE_A) {
    return;
  }
  let base = vec2i(vec2u(gid.x * 2u, gid.y * 2u));
  var rgb = vec3f(0.0);
  var a_sum = 0.0;
  for (var j = 0; j < 2; j++) {
    for (var i = 0; i < 2; i++) {
      let t = textureLoad(src_albedo, base + vec2i(i, j), 0);
      rgb += t.rgb * t.a;
      a_sum += t.a;
    }
  }
  let out_rgb = select(vec3f(0.0), rgb / max(a_sum, 1e-4), a_sum > 0.0);
  let row = u32(row_uni.v.x);
  textureStore(dst_albedo, vec2u(gid.x, row * TILE_A + gid.y), vec4f(out_rgb, a_sum * 0.25));
}

// Octahedral encode, y-primary — inverse of oct_decode in cards.wgsl.
fn oct_encode(n: vec3f) -> vec2f {
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

@compute @workgroup_size(8, 8)
fn cs_geom(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= GRID * TILE_B || gid.y >= TILE_B) {
    return;
  }
  let row = u32(row_uni.v.x);
  let col = gid.x / TILE_B;
  let tile_idx = row * GRID + col;
  let dst = vec2u(gid.x, row * TILE_B + gid.y);

  // --- resolve of the 6x6 supersample block ---------------------------------
  // Depth is coverage-weighted averaged: the warp wants a SMOOTH field.
  // The normal is NOT averaged — it is point-sampled from the most opaque,
  // most central covered sample in the block. Averaging normals over a block
  // this size mushes per-blade facets into one near-vertical mean, and because
  // the lighting term is non-linear in the normal that shows up as a flat,
  // uniformly-bright canopy. A picked normal keeps the real facet spread at
  // this atlas's resolution for exactly zero extra bytes.
  let base = vec2i(vec2u(gid.x * B_TO_SS, gid.y * B_TO_SS));
  let mid = (f32(B_TO_SS) - 1.0) * 0.5;
  var q_acc = 0.0;
  var w_sum = 0.0;
  var best_score = -1e9;
  var best_n = vec3f(0.0, 1.0, 0.0);
  for (var j = 0u; j < B_TO_SS; j++) {
    for (var i = 0u; i < B_TO_SS; i++) {
      let c = base + vec2i(vec2u(i, j));
      let a = textureLoad(src_albedo, c, 0).a;
      if (a <= 0.0) {
        continue;
      }
      let nq = textureLoad(src_nrm, c, 0);
      q_acc += nq.w * a;
      w_sum += a;
      let d = vec2f(f32(i) - mid, f32(j) - mid);
      let score = a * 20.0 - dot(d, d);
      if (score > best_score) {
        best_score = score;
        best_n = nq.xyz * 2.0 - 1.0;
      }
    }
  }
  if (w_sum <= 0.0) {
    textureStore(dst_geom, dst, vec4f(0.5, 0.5, 0.0, 0.0));
    textureStore(dst_mask, dst, vec4f(0.0));
    return;
  }
  let n_geom = normalize(best_n);
  let q = q_acc / w_sum;

  // --- reconstruct the plant-local 3D position of this texel ----------------
  let t0 = tables.tiles[tile_idx * 4u];
  let t1 = tables.tiles[tile_idx * 4u + 1u];
  let t2 = tables.tiles[tile_idx * 4u + 2u];
  let t3 = tables.tiles[tile_idx * 4u + 3u];
  let uu = (f32(gid.x % TILE_B) + 0.5) / f32(TILE_B);
  let vv = (f32(gid.y) + 0.5) / f32(TILE_B);
  let xr = t0.w + (uu * 2.0 - 1.0) * t3.x;
  let yu = t1.w - (vv * 2.0 - 1.0) * t3.y;
  let dd = t2.w - q * t3.z;
  let pos = t0.xyz * xr + t1.xyz * yu + t2.xyz * dd;

  // --- 32 precomputed visibility maps, one texel fetch each -----------------
  let o = pos - tables.sphere.xyz;
  let inv_r = 1.0 / tables.sphere.w;
  let bias = tables.sphere.w * 0.035;
  var vis_sum = 0.0;
  var bent = vec3f(0.0);
  for (var k = 0u; k < N_VIS; k++) {
    let pr = tables.probes[k * 3u].xyz;
    let pu = tables.probes[k * 3u + 1u].xyz;
    let pd = tables.probes[k * 3u + 2u].xyz;
    let su = dot(o, pr) * inv_r;
    let sv = dot(o, pu) * inv_r;
    var v = 1.0;
    if (abs(su) < 1.0 && abs(sv) < 1.0) {
      let tx = clamp(i32((su * 0.5 + 0.5) * f32(VIS_TILE)), 0, i32(VIS_TILE) - 1);
      let ty = clamp(i32((0.5 - sv * 0.5) * f32(VIS_TILE)), 0, i32(VIS_TILE) - 1);
      let cell = vec2i(i32((k % VIS_COLS) * VIS_TILE) + tx, i32((k / VIS_COLS) * VIS_TILE) + ty);
      let front = textureLoad(vis_atlas, cell, 0).r;
      v = select(0.0, 1.0, dot(o, pd) > front - bias);
    }
    vis_sum += v;
    bent += pd * v;
  }
  // Canopy-layer ramp, folded in here rather than per fragment: a texel near
  // the soil has more canopy overhead than a tip, so its openness is raised to
  // a larger power. Baking it means the runtime spends one pow, not three ops
  // plus a varying, and the height used is the texel's OWN height rather than
  // the card quad's.
  let hf = clamp(pos.y / max(tables.misc.x, 1e-3), 0.0, 1.0);
  let ao = pow(vis_sum / f32(N_VIS), 1.7 - 0.7 * hf);
  let bent_len = length(bent);
  let bent_n = select(n_geom, bent / max(bent_len, 1e-5), bent_len > 1e-4);
  let n_out = normalize(mix(n_geom, bent_n, BENT_MIX));

  textureStore(dst_geom, dst, vec4f(oct_encode(n_out) * 0.5 + 0.5, ao, q));
  textureStore(dst_mask, dst, vec4f(1.0));
}
