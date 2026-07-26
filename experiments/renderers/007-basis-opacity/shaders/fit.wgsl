// Fourier fit pass — runs once per elevation ring at bake time, entirely on
// the GPU (no readback ever happens; coefficients are written straight into
// the persistent coefficient textures).
//
// For every card texel (u,v) it reads the V azimuth captures of this ring and
// projects the view-dependent functions onto a truncated Fourier basis in the
// view azimuth theta:
//   coverage  w(theta)  -> order 3 (7 coefficients)  [the silhouette]
//   luminance g(theta)  -> order 1 relative modulation (2 coefficients)
//   rel depth d(theta)  -> DC + cos term (2 coefficients)
//   color rgb           -> coverage-weighted DC
//   normal              -> coverage-weighted DC (mesh frame), oct encoded
//
// Packing (all rgba8unorm array layers, one layer per elevation ring; biased
// channels store x*0.5+0.5):
//   out_a : [a0, a1c*, a1s*, a2c*]
//   out_b : [a2s*, a3c*, a3s*, ddc*]
//   out_c : [r, g, b, m1c*]
//   out_d : [oct_u, oct_v, m1s*, d1c*]
// (* = biased)

const V: u32 = 24u;
const TILE: u32 = 256u;
const TWO_PI: f32 = 6.2831853;

@group(0) @binding(0) var cap_color: texture_2d<f32>;
@group(0) @binding(1) var cap_normal: texture_2d<f32>;
@group(0) @binding(2) var out_a: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var out_b: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var out_c: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var out_d: texture_storage_2d<rgba8unorm, write>;

// Octahedral encode over the full sphere; decoded by oct_decode_card() in
// card.wgsl (matching pair, kept local to this experiment).
fn oct_encode(v: vec3f) -> vec2f {
  let s = abs(v.x) + abs(v.y) + abs(v.z);
  var p = vec2f(v.x, v.z) / max(s, 1e-6);
  if (v.y < 0.0) {
    p = (vec2f(1.0) - abs(vec2f(p.y, p.x))) * vec2f(sign(p.x), sign(p.y));
  }
  return p * 0.5 + 0.5;
}

fn bias(x: f32) -> f32 {
  return clamp(x, -1.0, 1.0) * 0.5 + 0.5;
}

@compute @workgroup_size(8, 8)
fn cs(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= TILE || gid.y >= TILE) { return; }

  var sw = 0.0;                    // sum of coverage
  var wc = vec3f(0.0);             // sum w * cos(k theta), k = 1..3
  var ws = vec3f(0.0);             // sum w * sin(k theta)
  var col = vec3f(0.0);            // sum w * rgb
  var nrm = vec3f(0.0);            // sum w * normal
  var gsum = 0.0;                  // sum w * luma
  var gc1 = 0.0;                   // sum w * luma * cos
  var gs1 = 0.0;                   // sum w * luma * sin
  var dsum = 0.0;                  // sum w * depth
  var dc1 = 0.0;                   // sum w * depth * cos

  for (var j = 0u; j < V; j++) {
    let px = vec2u(gid.x + j * TILE, gid.y);
    let a = textureLoad(cap_color, px, 0);
    let b = textureLoad(cap_normal, px, 0);
    let w = a.a;
    if (w <= 0.0) { continue; }
    let th = TWO_PI * f32(j) / f32(V);
    let c1 = cos(th);
    let s1 = sin(th);
    let c2 = 2.0 * c1 * c1 - 1.0;
    let s2 = 2.0 * s1 * c1;
    let c3 = c1 * c2 - s1 * s2;
    let s3 = s1 * c2 + c1 * s2;
    sw += w;
    wc += w * vec3f(c1, c2, c3);
    ws += w * vec3f(s1, s2, s3);
    col += w * a.rgb;
    nrm += w * (b.xyz * 2.0 - 1.0);
    let g = dot(a.rgb, vec3f(0.299, 0.587, 0.114));
    let d = b.a * 2.0 - 1.0;
    gsum += w * g;
    gc1 += w * g * c1;
    gs1 += w * g * s1;
    dsum += w * d;
    dc1 += w * d * c1;
  }

  let inv_v = 1.0 / f32(V);
  let a0 = sw * inv_v;
  let ak_c = 2.0 * inv_v * wc;
  let ak_s = 2.0 * inv_v * ws;

  var rgb = vec3f(0.0);
  var m1c = 0.0;
  var m1s = 0.0;
  var ddc = 0.0;
  var d1c = 0.0;
  var oct = vec2f(0.5, 1.0); // straight-up normal fallback
  if (sw > 1e-4) {
    rgb = col / sw;
    let gbar = max(gsum / sw, 1e-3);
    // Relative first-order luminance modulation: g(theta)/gbar - 1.
    m1c = clamp((2.0 * gc1 / sw - gbar * ak_c.x / max(a0, 1e-4)) / gbar, -1.0, 1.0);
    m1s = clamp((2.0 * gs1 / sw - gbar * ak_s.x / max(a0, 1e-4)) / gbar, -1.0, 1.0);
    ddc = dsum / sw;
    d1c = clamp(2.0 * dc1 / sw - ddc * ak_c.x / max(a0, 1e-4), -1.0, 1.0);
    let n = nrm / sw;
    if (dot(n, n) > 1e-6) { oct = oct_encode(normalize(n)); }
  }

  let p = vec2u(gid.xy);
  textureStore(out_a, p, vec4f(clamp(a0, 0.0, 1.0), bias(ak_c.x), bias(ak_s.x), bias(ak_c.y)));
  textureStore(out_b, p, vec4f(bias(ak_s.y), bias(ak_c.z), bias(ak_s.z), bias(ddc)));
  textureStore(out_c, p, vec4f(rgb, bias(m1c)));
  textureStore(out_d, p, vec4f(oct, bias(m1s), bias(d1c)));
}
