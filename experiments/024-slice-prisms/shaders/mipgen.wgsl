// Load-time mip generation for the prism atlas (one pass per mip level).
// The chain stops at the level where one atlas tile is still 1x1 texel, so a
// clean 2x2 box filter NEVER mixes two tiles: tile boundaries stay texel
// aligned at every level it produces.
//
// Albedo is coverage-weighted so transparent texels never darken the average;
// normals use a plain box filter over the oct-encoded values (only sampled at
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
