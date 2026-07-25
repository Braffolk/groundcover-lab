// Bake reductions: supersampled composite -> field slab slice, plus 3D mip
// generation. Everything is stored PREMULTIPLIED by coverage, which makes a
// plain box filter the correct (coverage-weighted) prefilter — that is what
// makes distant grass an honestly averaged canopy instead of a sharp,
// shimmering one, with no dithering anywhere.

struct Cfg {
  // slice index, supersample window, unused, unused
  info: vec4u,
}
@group(0) @binding(0) var<uniform> cfg: Cfg;
@group(0) @binding(1) var src_surf: texture_2d<f32>;
@group(0) @binding(2) var src_geom: texture_2d<f32>;
@group(0) @binding(3) var dst_surf: texture_storage_3d<rgba8unorm, write>;
@group(0) @binding(4) var dst_geom: texture_storage_3d<rgba8snorm, write>;

@compute @workgroup_size(8, 8, 1)
fn reduce_surf(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(dst_surf).xy;
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let win = i32(cfg.info.y);
  let base = vec2i(gid.xy) * win;
  var rgb = vec3f(0.0);
  var cov = 0.0;
  for (var j = 0; j < win; j++) {
    for (var i = 0; i < win; i++) {
      let t = textureLoad(src_surf, base + vec2i(i, j), 0);
      rgb += t.rgb * t.a;
      cov += t.a;
    }
  }
  let n = f32(win * win);
  let c = cov / n;
  let albedo = rgb / max(cov, 1e-4);
  textureStore(dst_surf, vec3i(vec2i(gid.xy), i32(cfg.info.x)), vec4f(albedo * c, c));
}

@compute @workgroup_size(8, 8, 1)
fn reduce_geom(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(dst_geom).xy;
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let win = i32(cfg.info.y);
  let base = vec2i(gid.xy) * win;
  var hsum = 0.0;
  var cov = 0.0;
  // The NORMAL is point-sampled from the subsample that actually won the ray
  // (highest hit), not averaged: averaging a texel full of blades pointing
  // every which way collapses to straight up and the canopy lights up like
  // linoleum. Height and coverage stay averaged, which is the honest prefilter.
  var best_h = -1.0;
  var best_n = vec2f(0.0);
  for (var j = 0; j < win; j++) {
    for (var i = 0; i < win; i++) {
      let s = textureLoad(src_surf, base + vec2i(i, j), 0);
      if (s.a < 0.5) { continue; }
      let g = textureLoad(src_geom, base + vec2i(i, j), 0);
      hsum += g.b;
      cov += 1.0;
      if (g.b > best_h) {
        best_h = g.b;
        best_n = vec2f((g.r - 0.5) * 2.0, (g.g - 0.5) * 2.0);
      }
    }
  }
  let c = cov / f32(win * win);
  let h = hsum / max(cov, 1e-4);
  textureStore(dst_geom, vec3i(vec2i(gid.xy), i32(cfg.info.x)), vec4f(best_n * c, h * c, c));
}
