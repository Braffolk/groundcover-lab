// BAKE STAGE 2 — turn the atomic accumulation buffers into two filterable 3D
// textures and compute sky openness in the same sweep.
//
// One thread per (x, z) column: it walks the column from the top down, which is
// exactly the order needed to accumulate vertical transmittance, so ambient
// occlusion ("how buried is this point") comes out for free.
//
//   vol_geo : xyz = PROJECTED area density per axis (1/length, normalized units)
//             w   = total area density (unused by the trace, kept for debugging)
//   vol_nrm : rg = oct-encoded area-weighted mean normal, b = its coherence
//   vol_col : rgb = area-weighted mean authored colour, a = sky openness

struct ResolveParams {
  dims: vec3u,
  area_fixed: f32,
  voxel: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
}

@group(0) @binding(0) var<storage, read> acc_u: array<u32>;
@group(0) @binding(1) var<storage, read> acc_i: array<i32>;
@group(0) @binding(5) var<storage, read> acc_p: array<u32>;
@group(0) @binding(6) var vol_nrm: texture_storage_3d<rgba8unorm, write>;
@group(0) @binding(2) var vol_geo: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var vol_col: texture_storage_3d<rgba8unorm, write>;
@group(0) @binding(4) var<uniform> rp: ResolveParams;

fn oct_encode(n: vec3f) -> vec2f {
  let s = abs(n.x) + abs(n.y) + abs(n.z);
  if (s < 1e-6) {
    return vec2f(0.5, 0.5);
  }
  var u = n.x / s;
  var v = n.z / s;
  if (n.y < 0.0) {
    let fu = (1.0 - abs(v)) * select(-1.0, 1.0, n.x >= 0.0);
    let fv = (1.0 - abs(u)) * select(-1.0, 1.0, n.z >= 0.0);
    u = fu;
    v = fv;
  }
  return vec2f(u, v) * 0.5 + 0.5;
}

@compute @workgroup_size(8, 8)
fn cs_resolve(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= rp.dims.x || gid.y >= rp.dims.z) {
    return;
  }
  let ix = gid.x;
  let iz = gid.y;
  var transmittance = 1.0;
  var y = i32(rp.dims.y) - 1;
  loop {
    if (y < 0) {
      break;
    }
    let base = (u32(y) * rp.dims.z + iz) * rp.dims.x + ix;
    let area = f32(acc_u[base * 4u]);
    let inv_unit = 1.0 / (rp.area_fixed * rp.voxel);
    var proj_a = vec3f(0.0);
    var nbar = vec3f(0.0);
    var albedo = vec3f(0.0);
    if (area > 0.0) {
      proj_a = vec3f(f32(acc_p[base * 4u]), f32(acc_p[base * 4u + 1u]), f32(acc_p[base * 4u + 2u])) * inv_unit;
      albedo = vec3f(f32(acc_u[base * 4u + 1u]), f32(acc_u[base * 4u + 2u]), f32(acc_u[base * 4u + 3u])) / area;
      nbar = vec3f(f32(acc_i[base * 4u]), f32(acc_i[base * 4u + 1u]), f32(acc_i[base * 4u + 2u])) / area;
    }
    let coh = min(length(nbar), 1.0);
    let oct = oct_encode(select(vec3f(0.0, 1.0, 0.0), nbar / max(coh, 1e-4), coh > 1e-4));
    // Openness is sampled BEFORE this voxel attenuates, so a point on a blade
    // is not occluded by its own voxel.
    let openness = transmittance;
    textureStore(vol_geo, vec3i(i32(ix), y, i32(iz)), vec4f(proj_a, area * inv_unit));
    textureStore(vol_nrm, vec3i(i32(ix), y, i32(iz)), vec4f(oct, coh, 1.0));
    textureStore(vol_col, vec3i(i32(ix), y, i32(iz)), vec4f(clamp(albedo, vec3f(0.0), vec3f(1.0)), openness));
    // Vertical extinction is exactly the y component of the projected area.
    transmittance *= exp(-proj_a.y * rp.voxel);
    y = y - 1;
  }
}
