// Mip generation for one layer of the view-set texture array, both maps in a
// single pass. Colour and normal are averaged COVERAGE-WEIGHTED so a minified
// silhouette never drifts toward the cleared background or toward a normal
// nobody stored; alpha is the plain average, i.e. real coverage.
//
// Layers are independent textures as far as filtering is concerned, so unlike
// an atlas there is no bleed between views at any mip level and no gutters are
// needed.

@group(0) @binding(0) var src_albedo: texture_2d<f32>;
@group(0) @binding(1) var src_normal: texture_2d<f32>;

fn oct_decode_y(e: vec2f) -> vec3f {
  var x = e.x;
  var z = e.y;
  let y = 1.0 - abs(e.x) - abs(e.y);
  if (y < 0.0) {
    x = (1.0 - abs(e.y)) * select(-1.0, 1.0, e.x >= 0.0);
    z = (1.0 - abs(e.x)) * select(-1.0, 1.0, e.y >= 0.0);
  }
  return normalize(vec3f(x, y, z));
}

fn oct_encode_y(n: vec3f) -> vec2f {
  let s = abs(n.x) + abs(n.y) + abs(n.z);
  if (s < 1.0e-6) {
    return vec2f(0.5);
  }
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

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[vi], 0.0, 1.0);
}

struct FOut {
  @location(0) albedo: vec4f,
  @location(1) nrm: vec2f,
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> FOut {
  let base = vec2i(pos.xy) * 2;
  var acc = vec4f(0.0);
  var n = vec3f(0.0);
  for (var j = 0; j < 2; j++) {
    for (var i = 0; i < 2; i++) {
      let c = base + vec2i(i, j);
      let a = textureLoad(src_albedo, c, 0);
      let e = textureLoad(src_normal, c, 0).xy;
      acc += vec4f(a.rgb * a.a, a.a);
      n += oct_decode_y(e * 2.0 - 1.0) * a.a;
    }
  }
  var out: FOut;
  // Dilated texels carry colour with alpha 0; keep their contribution as the
  // fallback so a fully-empty quad stays plausible instead of black.
  if (acc.a > 1.0e-5) {
    out.albedo = vec4f(acc.rgb / acc.a, acc.a * 0.25);
    out.nrm = oct_encode_y(n); // oct_encode_y normalizes by the L1 norm itself
  } else {
    var rgb = vec3f(0.0);
    var nn = vec3f(0.0);
    for (var j = 0; j < 2; j++) {
      for (var i = 0; i < 2; i++) {
        let c = base + vec2i(i, j);
        rgb += textureLoad(src_albedo, c, 0).rgb * 0.25;
        nn += oct_decode_y(textureLoad(src_normal, c, 0).xy * 2.0 - 1.0);
      }
    }
    out.albedo = vec4f(rgb, 0.0);
    out.nrm = oct_encode_y(nn);
  }
  return out;
}
