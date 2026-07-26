// Load-time mip generation, one pass per level per texture. Tile sizes stay
// even at every generated level, so a 2x2 box never crosses an atlas tile.
//   fs_cov    r8    plain average of the coverage mask
//   fs_albedo rgba8 coverage-weighted colour (a = mean coverage)
//   fs_oct    rg8   oct normals: decoded, averaged, re-encoded (a plain box
//                   filter over oct pairs drifts toward nonsense at distance)
//   fs_aux    rgba8 canopy aux: oct normal decoded/averaged/re-encoded in rg,
//                   plain average of the height byte in b
//   fs_aux_ao rgba8 same, plus the plain average of the baked cavity occlusion
//                   in a — averaging the OCCLUSION (rather than deriving it
//                   from the averaged height) is what keeps a mat's shading
//                   constant with distance instead of brightening as the
//                   height field flattens

@group(0) @binding(0) var src_level: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var tri = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(3.0, 1.0), vec2f(-1.0, 1.0));
  return vec4f(tri[vi], 0.0, 1.0);
}

fn src_texel(base: vec2i, dx: i32, dy: i32) -> vec4f {
  let dims = vec2i(textureDimensions(src_level));
  return textureLoad(src_level, min(base + vec2i(dx, dy), dims - vec2i(1)), 0);
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

@fragment
fn fs_cov(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let base = vec2i(pos.xy) * 2;
  var acc = 0.0;
  for (var j = 0; j < 2; j++) {
    for (var i = 0; i < 2; i++) {
      acc += src_texel(base, i, j).r;
    }
  }
  return vec4f(acc * 0.25, 0.0, 0.0, 1.0);
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
  return vec4f(rgb_sum / max(a_sum, 1.0e-4), a_sum * 0.25);
}

@fragment
fn fs_oct(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let base = vec2i(pos.xy) * 2;
  var n = vec3f(0.0);
  for (var j = 0; j < 2; j++) {
    for (var i = 0; i < 2; i++) {
      n += oct_decode_mip(src_texel(base, i, j).xy * 2.0 - 1.0);
    }
  }
  return vec4f(oct_encode_mip(normalize(n)), 0.0, 1.0);
}

@fragment
fn fs_aux(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let base = vec2i(pos.xy) * 2;
  var n = vec3f(0.0);
  var h = 0.0;
  for (var j = 0; j < 2; j++) {
    for (var i = 0; i < 2; i++) {
      let t = src_texel(base, i, j);
      n += oct_decode_mip(t.xy * 2.0 - 1.0);
      h += t.z;
    }
  }
  return vec4f(oct_encode_mip(normalize(n)), h * 0.25, 1.0);
}

@fragment
fn fs_aux_ao(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let base = vec2i(pos.xy) * 2;
  var n = vec3f(0.0);
  var h = 0.0;
  var ao = 0.0;
  for (var j = 0; j < 2; j++) {
    for (var i = 0; i < 2; i++) {
      let t = src_texel(base, i, j);
      n += oct_decode_mip(t.xy * 2.0 - 1.0);
      h += t.z;
      ao += t.w;
    }
  }
  return vec4f(oct_encode_mip(normalize(n)), h * 0.25, ao * 0.25);
}
