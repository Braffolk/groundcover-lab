// Single-pass dilation of both baked atlases: empty texels take the average of
// the nearest ring of covered neighbours, so bilinear/mip filtering can never
// pull background black (albedo) or a bogus depth (geometry) across a
// silhouette. Search is clamped to the owning tile so views never bleed.
// Coverage itself is left at zero — the alpha test still cuts the true edge.

const GRID: u32 = 5u;
const TILE_A: u32 = 384u;
const TILE_B: u32 = 128u;

@group(0) @binding(0) var src_albedo: texture_2d<f32>;
@group(0) @binding(1) var src_geom: texture_2d<f32>;
@group(0) @binding(2) var src_mask: texture_2d<f32>;
@group(0) @binding(3) var dst_albedo: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var dst_geom: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn cs_albedo(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = GRID * TILE_A;
  if (gid.x >= n || gid.y >= n) {
    return;
  }
  let c = vec2i(gid.xy);
  let here = textureLoad(src_albedo, c, 0);
  if (here.a > 0.0) {
    textureStore(dst_albedo, gid.xy, here);
    return;
  }
  let lo = vec2i((gid.xy / TILE_A) * TILE_A);
  let hi = lo + vec2i(i32(TILE_A) - 1);
  var rgb = vec3f(0.0);
  var count = 0.0;
  for (var r = 1; r <= 3; r++) {
    for (var j = -r; j <= r; j++) {
      for (var i = -r; i <= r; i++) {
        if (max(abs(i), abs(j)) != r) {
          continue;
        }
        let s = c + vec2i(i, j);
        if (any(s < lo) || any(s > hi)) {
          continue;
        }
        let t = textureLoad(src_albedo, s, 0);
        if (t.a <= 0.0) {
          continue;
        }
        rgb += t.rgb;
        count += 1.0;
      }
    }
    if (count > 0.0) {
      break;
    }
  }
  let out_rgb = select(vec3f(0.0), rgb / max(count, 1.0), count > 0.0);
  textureStore(dst_albedo, gid.xy, vec4f(out_rgb, 0.0));
}

@compute @workgroup_size(8, 8)
fn cs_geom(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = GRID * TILE_B;
  if (gid.x >= n || gid.y >= n) {
    return;
  }
  let c = vec2i(gid.xy);
  if (textureLoad(src_mask, c, 0).r > 0.0) {
    textureStore(dst_geom, gid.xy, textureLoad(src_geom, c, 0));
    return;
  }
  let lo = vec2i((gid.xy / TILE_B) * TILE_B);
  let hi = lo + vec2i(i32(TILE_B) - 1);
  var acc = vec4f(0.0);
  var count = 0.0;
  for (var r = 1; r <= 3; r++) {
    for (var j = -r; j <= r; j++) {
      for (var i = -r; i <= r; i++) {
        if (max(abs(i), abs(j)) != r) {
          continue;
        }
        let s = c + vec2i(i, j);
        if (any(s < lo) || any(s > hi)) {
          continue;
        }
        if (textureLoad(src_mask, s, 0).r <= 0.0) {
          continue;
        }
        acc += textureLoad(src_geom, s, 0);
        count += 1.0;
      }
    }
    if (count > 0.0) {
      break;
    }
  }
  let out_v = select(vec4f(0.5, 0.5, 0.0, 1.0), acc / max(count, 1.0), count > 0.0);
  textureStore(dst_geom, gid.xy, out_v);
}
