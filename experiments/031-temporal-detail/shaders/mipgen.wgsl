// Atlas mip generation at load time. Albedo mips are COVERAGE-WEIGHTED (a plain
// box filter would drag silhouette colours toward whatever the dilation left in
// the empty texels). Normal mips decode the octahedral pair, average the real
// directions and re-encode, so a distant tile keeps a meaningful mean normal
// instead of the mean of two encoded numbers.

@group(0) @binding(0) var src: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[vi], 0.0, 1.0);
}

@fragment
fn fs_albedo(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let base = vec2i(pos.xy) * 2;
  var acc = vec3f(0.0);
  var aSum = 0.0;
  for (var j = 0; j < 2; j++) {
    for (var i = 0; i < 2; i++) {
      let s = textureLoad(src, base + vec2i(i, j), 0);
      acc += s.rgb * s.a;
      aSum += s.a;
    }
  }
  if (aSum <= 0.0) {
    // Keep the dilated colour so filtering never reaches for black.
    var rgb = vec3f(0.0);
    for (var j = 0; j < 2; j++) {
      for (var i = 0; i < 2; i++) {
        rgb += textureLoad(src, base + vec2i(i, j), 0).rgb;
      }
    }
    return vec4f(rgb * 0.25, 0.0);
  }
  return vec4f(acc / aSum, aSum * 0.25);
}

fn oct_decode(e: vec2f) -> vec3f {
  var x = e.x;
  var z = e.y;
  let y = 1.0 - abs(e.x) - abs(e.y);
  if (y < 0.0) {
    x = (1.0 - abs(e.y)) * select(-1.0, 1.0, e.x >= 0.0);
    z = (1.0 - abs(e.x)) * select(-1.0, 1.0, e.y >= 0.0);
  }
  return normalize(vec3f(x, y, z));
}

fn oct_encode(n: vec3f) -> vec2f {
  let s = abs(n.x) + abs(n.y) + abs(n.z);
  if (s < 1e-6) {
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
  return vec2f(u, v) * 0.5 + 0.5;
}

@fragment
fn fs_normal(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let base = vec2i(pos.xy) * 2;
  var acc = vec3f(0.0);
  for (var j = 0; j < 2; j++) {
    for (var i = 0; i < 2; i++) {
      acc += oct_decode(textureLoad(src, base + vec2i(i, j), 0).xy * 2.0 - 1.0);
    }
  }
  if (dot(acc, acc) < 1e-8) {
    return vec4f(0.5, 0.5, 0.0, 1.0);
  }
  return vec4f(oct_encode(normalize(acc)), 0.0, 1.0);
}
