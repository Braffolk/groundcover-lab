// Whole-frame |A-B| reduction for the diff HUD readout.

@group(0) @binding(0) var tex_a: texture_2d<f32>;
@group(0) @binding(1) var tex_b: texture_2d<f32>;

struct Stats {
  // Fixed point: sum of per-pixel max-channel |A-B| * 255.
  sum: atomic<u32>,
  // Pixels with max-channel |A-B| above ~1/255.
  over: atomic<u32>,
}
@group(0) @binding(2) var<storage, read_write> out_stats: Stats;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(tex_a);
  if (gid.x >= dims.x || gid.y >= dims.y) {
    return;
  }
  let a = textureLoad(tex_a, vec2i(gid.xy), 0).rgb;
  let b = textureLoad(tex_b, vec2i(gid.xy), 0).rgb;
  let d = abs(a - b);
  let m = max(d.r, max(d.g, d.b));
  atomicAdd(&out_stats.sum, u32(m * 255.0 + 0.5));
  if (m > 0.004) {
    atomicAdd(&out_stats.over, 1u);
  }
}
