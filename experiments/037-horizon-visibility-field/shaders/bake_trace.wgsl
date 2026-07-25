// BAKE STAGE 3 — the precomputed raycast. THIS is the only place a ray is ever
// marched: offline, once per species, never at runtime.
//
// For every (entry point u,v on the canopy top plane) x (azimuth, elevation) the
// canopy volume is integrated along the ray and the ANSWER is stored:
//   out_a: r = 1/(1+distance) to the first blocker, gb = oct normal there,
//          a = coverage (1 - transmittance) = the visibility answer
//   out_b: rgb = coverage-weighted albedo, a = sky openness (AO) at the hit
//
// The integral is a proper volume integral over surface-area density, so the
// answer is PREFILTERED: coverage is honest partial occlusion rather than a
// binary blade test, and the recorded distance is where accumulated coverage
// crosses one half (a real silhouette depth), falling back to the mean.

struct TraceParams {
  res: u32,
  n_phi: u32,
  n_theta: u32,
  max_steps: u32,
  tile_norm: f32,
  height_norm: f32,
  sin_min: f32,
  step_base: f32,
  step_grow: f32,
  step_max: f32,
  _pad0: f32,
  _pad1: f32,
}

@group(0) @binding(0) var vol_geo: texture_3d<f32>;
@group(0) @binding(1) var vol_col: texture_3d<f32>;
@group(0) @binding(6) var vol_nrm: texture_3d<f32>;
@group(0) @binding(2) var vol_samp: sampler;
@group(0) @binding(3) var<storage, read_write> out_a: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_b: array<u32>;
@group(0) @binding(5) var<uniform> tp: TraceParams;

fn oct_decode(e: vec2f) -> vec3f {
  let f = e * 2.0 - 1.0;
  var x = f.x;
  var z = f.y;
  let y = 1.0 - abs(f.x) - abs(f.y);
  if (y < 0.0) {
    x = (1.0 - abs(f.y)) * select(-1.0, 1.0, f.x >= 0.0);
    z = (1.0 - abs(f.x)) * select(-1.0, 1.0, f.y >= 0.0);
  }
  return normalize(vec3f(x, y, z));
}

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

fn pack_rgba8(v: vec4f) -> u32 {
  let q = vec4u(round(clamp(v, vec4f(0.0), vec4f(1.0)) * 255.0));
  return q.x | (q.y << 8u) | (q.z << 16u) | (q.w << 24u);
}

@compute @workgroup_size(8, 8, 1)
fn cs_trace(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= tp.res || gid.y >= tp.res) {
    return;
  }
  let layer = gid.z;
  let phi_bin = layer / tp.n_theta;
  let theta_bin = layer % tp.n_theta;
  let phi = f32(phi_bin) * 6.28318531 / f32(tp.n_phi);
  let sin_t = mix(tp.sin_min, 1.0, f32(theta_bin) / f32(tp.n_theta - 1u));
  let cos_t = sqrt(max(1.0 - sin_t * sin_t, 0.0));
  let dir = vec3f(cos(phi) * cos_t, -sin_t, sin(phi) * cos_t);

  let start = vec3f(
    (f32(gid.x) + 0.5) / f32(tp.res) * tp.tile_norm,
    tp.height_norm,
    (f32(gid.y) + 0.5) / f32(tp.res) * tp.tile_norm,
  );

  var transmittance = 1.0;
  var dist_sum = 0.0;
  var col_sum = vec3f(0.0);
  var nrm_sum = vec3f(0.0);
  var ao_sum = 0.0;
  var t_cross = -1.0;
  var t = 0.0;
  var steps = 0u;
  let inv_tile = 1.0 / tp.tile_norm;
  let inv_h = 1.0 / tp.height_norm;
  loop {
    if (steps >= tp.max_steps) {
      break;
    }
    steps = steps + 1u;
    let h = min(tp.step_base * (1.0 + tp.step_grow * t), tp.step_max);
    let p = start + dir * (t + h * 0.5);
    if (p.y <= 0.0) {
      break;
    }
    let uvw = vec3f(p.x * inv_tile, p.y * inv_h, p.z * inv_tile);
    let geo = textureSampleLevel(vol_geo, vol_samp, uvw, 0.0);
    // Extinction for THIS direction straight out of the per-axis projected area:
    // near-vertical blades stop a horizontal ray and barely touch a vertical one.
    let ext = dot(abs(dir), geo.xyz);
    if (ext > 1e-4) {
      let alpha = 1.0 - exp(-ext * h);
      let w = alpha * transmittance;
      if (w > 1e-5) {
        let col = textureSampleLevel(vol_col, vol_samp, uvw, 0.0);
        let nrm_tex = textureSampleLevel(vol_nrm, vol_samp, uvw, 0.0);
        let accum = 1.0 - transmittance;
        if (t_cross < 0.0 && accum + w >= 0.5) {
          t_cross = t + h * 0.5;
        }
        dist_sum += w * (t + h * 0.5);
        col_sum += w * col.rgb;
        ao_sum += w * col.a;
        // Shading normal: the mean normal where the voxel's normals agree, the
        // anisotropy axis where they cancel (the two faces of one blade), always
        // flipped toward the incoming ray because foliage is two-sided.
        let coh = nrm_tex.b;
        var n = oct_decode(nrm_tex.rg);
        let aniso = normalize(geo.xyz * -sign(dir) + vec3f(1e-5));
        n = normalize(mix(aniso, n, coh));
        if (dot(n, dir) > 0.0) {
          n = -n;
        }
        nrm_sum += w * n;
        transmittance *= 1.0 - alpha;
        if (transmittance < 0.01) {
          break;
        }
      }
    }
    t += h;
  }

  let cov = 1.0 - transmittance;
  // Distance from the entry point to where the canopy blocks this ray. With no
  // canopy on the ray, the answer is the closed-form ground crossing.
  let t_ground = tp.height_norm / sin_t;
  var dist = t_ground;
  var nrm = vec3f(0.0, 1.0, 0.0);
  var albedo = vec3f(0.0);
  var ao = 1.0;
  if (cov > 1e-4) {
    dist = select(dist_sum / cov, t_cross, t_cross >= 0.0);
    albedo = col_sum / cov;
    ao = ao_sum / cov;
    let nl = length(nrm_sum);
    if (nl > 1e-5) {
      nrm = nrm_sum / nl;
    }
  }
  let oct = oct_encode(nrm);
  let idx = (layer * tp.res + gid.y) * tp.res + gid.x;
  out_a[idx] = pack_rgba8(vec4f(1.0 / (1.0 + dist), oct.x, oct.y, cov));
  out_b[idx] = pack_rgba8(vec4f(albedo, ao));
}
