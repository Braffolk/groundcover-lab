// Load-time mip generation for the clump atlases (one pass per mip level).
// Every atlas tile origin and size is a multiple of 32, so at each of the
// levels we generate a 2x2 block never straddles a tile boundary — the chain
// is bleed-free by construction and needs no per-tile clamping here.
// Albedo is coverage-weighted so transparent texels never darken the average;
// normals are decoded, averaged as vectors and re-encoded, so a distant tile
// keeps a real mean direction instead of an average of oct coordinates.

@group(0) @binding(0) var src_level: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var tri = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(3.0, 1.0), vec2f(-1.0, 1.0));
  return vec4f(tri[vi], 0.0, 1.0);
}

fn src_texel(base: vec2i, dx: i32, dy: i32) -> vec4f {
  let dims = vec2i(textureDimensions(src_level));
  let c = min(base + vec2i(dx, dy), dims - vec2i(1));
  return textureLoad(src_level, c, 0);
}

@fragment
fn fs_albedo(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let base = vec2i(pos.xy) * 2;
  var rgb_sum = vec3f(0.0);
  var a_sum = 0.0;
  for (var j = 0; j < 2; j++) {
    for (var i = 0; i < 2; i++) {
      let t = src_texel(base, i, j);
      rgb_sum += t.rgb * t.a;
      a_sum += t.a;
    }
  }
  let rgb = rgb_sum / max(a_sum, 1e-4);
  return vec4f(rgb, a_sum * 0.25);
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

fn oct_encode(n_in: vec3f) -> vec2f {
  let n = normalize(n_in);
  let s = abs(n.x) + abs(n.y) + abs(n.z);
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
      acc += oct_decode(src_texel(base, i, j).xy * 2.0 - 1.0);
    }
  }
  if (dot(acc, acc) < 1e-8) {
    return vec4f(0.5, 0.5, 0.0, 1.0);
  }
  return vec4f(oct_encode(acc), 0.0, 1.0);
}
