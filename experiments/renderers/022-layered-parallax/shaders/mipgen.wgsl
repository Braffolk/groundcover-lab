// Load-time mip generation for the slab atlases (one pass per mip level).
// Albedo is coverage-weighted so transparent texels never darken the average;
// the oct-encoded normal atlas gets a plain box filter (only sampled at
// distance, where the approximation is invisible).

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

@fragment
fn fs_normal(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let base = vec2i(pos.xy) * 2;
  var acc = vec4f(0.0);
  for (var j = 0; j < 2; j++) {
    for (var i = 0; i < 2; i++) {
      acc += src_texel(base, i, j);
    }
  }
  return acc * 0.25;
}

// Carpet normal maps store PLAIN unit vectors in the +Y hemisphere (rgb) plus
// coverage (a), not an octahedral pair — octahedral encoding is not
// mip-averageable. Averaging the vectors is linear and safe; weighting by
// coverage keeps empty texels from dragging the average toward the dilation
// fringe. The result is renormalized, so every level stays a unit vector.
@fragment
fn fs_normal_cov(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let base = vec2i(pos.xy) * 2;
  var v = vec3f(0.0);
  var a_sum = 0.0;
  for (var j = 0; j < 2; j++) {
    for (var i = 0; i < 2; i++) {
      let t = src_texel(base, i, j);
      let w = t.a + 0.002;
      v += (t.xyz * 2.0 - 1.0) * w;
      a_sum += t.a;
    }
  }
  let len = length(v);
  let n = select(vec3f(0.0, 1.0, 0.0), v / len, len > 1e-5);
  return vec4f(n * 0.5 + 0.5, a_sum * 0.25);
}
