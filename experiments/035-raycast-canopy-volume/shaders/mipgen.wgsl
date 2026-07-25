// Load-time mip chain for the ray-answer table — this is the antialiasing of
// the technique: mip N is an honestly averaged canopy answer over 2^N texels,
// so distant grass is a prefiltered surface instead of a shimmering one.
//
// q and the normal are COVERAGE-WEIGHTED (a missing sub-ray must not drag the
// hit depth toward the ground), the normal is averaged as a direction (decode,
// average, re-encode) rather than in oct space, and taps wrap instead of
// clamping because the table is periodic.

@group(0) @binding(0) var src_level: texture_2d_array<f32>;
@group(0) @binding(1) var dst_level: texture_storage_2d_array<rgba8unorm, write>;

fn oct_decode(e: vec2f) -> vec3f {
  let f = e * 2.0 - vec2f(1.0);
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
  if (s < 1.0e-6) { return vec2f(0.5); }
  var u = n.x / s;
  var v = n.z / s;
  if (n.y < 0.0) {
    let fu = (1.0 - abs(v)) * select(-1.0, 1.0, n.x >= 0.0);
    let fv = (1.0 - abs(u)) * select(-1.0, 1.0, n.z >= 0.0);
    u = fu;
    v = fv;
  }
  return vec2f(u, v) * 0.5 + vec2f(0.5);
}

@compute @workgroup_size(8, 8, 1)
fn reduce(@builtin(global_invocation_id) gid: vec3u) {
  let dst_dims = textureDimensions(dst_level);
  if (gid.x >= dst_dims.x || gid.y >= dst_dims.y) { return; }
  let src_dims = vec2i(textureDimensions(src_level));
  let base = vec2i(gid.xy) * 2;

  var cov = 0.0;
  var q = 0.0;
  var nrm = vec3f(0.0);
  for (var j = 0; j < 2; j++) {
    for (var i = 0; i < 2; i++) {
      let c = (base + vec2i(i, j)) % src_dims;
      let t = textureLoad(src_level, c, gid.z, 0);
      cov += t.g;
      q += t.r * t.g;
      nrm += oct_decode(t.ba) * t.g;
    }
  }
  var out_q = 1.0;
  var out_n = vec3f(0.0, 1.0, 0.0);
  if (cov > 1.0e-4) {
    out_q = q / cov;
    let l = length(nrm);
    if (l > 1.0e-5) { out_n = nrm / l; }
  }
  let oct = oct_encode(out_n);
  textureStore(dst_level, vec2i(gid.xy), i32(gid.z), vec4f(out_q, cov * 0.25, oct.x, oct.y));
}
