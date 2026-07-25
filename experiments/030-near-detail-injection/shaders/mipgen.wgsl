// Load-time mip generation for the two card atlases (one pass per mip level).
//
//  fs_albedo — coverage-weighted, so transparent texels never darken the mean.
//  fs_geo    — normals are DECODED, averaged as vectors and re-encoded (a box
//              filter over oct codes drifts toward the tile mean, which shows up
//              as the far field losing its normal character); depth and sky
//              visibility are plain box averages.

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

fn oct_decode_mip(e: vec2f) -> vec3f {
  var x = e.x;
  var z = e.y;
  let y = 1.0 - abs(e.x) - abs(e.y);
  if (y < 0.0) {
    x = (1.0 - abs(e.y)) * select(-1.0, 1.0, e.x >= 0.0);
    z = (1.0 - abs(e.x)) * select(-1.0, 1.0, e.y >= 0.0);
  }
  return normalize(vec3f(x, y, z));
}

fn oct_encode_mip(n: vec3f) -> vec2f {
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

@fragment
fn fs_geo(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let base = vec2i(pos.xy) * 2;
  var nsum = vec3f(0.0);
  var ba = vec2f(0.0);
  for (var j = 0; j < 2; j++) {
    for (var i = 0; i < 2; i++) {
      let t = src_texel(base, i, j);
      nsum += oct_decode_mip(t.rg * 2.0 - 1.0);
      ba += t.ba;
    }
  }
  return vec4f(oct_encode_mip(normalize(nsum)), ba * 0.25);
}
