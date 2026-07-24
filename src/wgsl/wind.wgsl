#include "src/wgsl/frame.wgsl"

// Shared wind model — pure function of `time`, so fixed-timestep runs are
// deterministic. CPU mirror: Wind.sway() in src/scene/wind.ts.
// `sway` comes from the species table (0 = moss-still), `phase` is the
// per-plant phase offset from the scatter hash.

fn wind_gust(time: f32) -> f32 {
  let t = time * frame.wind_gust_freq;
  // Slow asymmetric envelope in [0.25, 1]: mostly calm, periodic surges.
  let g = 0.5 + 0.5 * sin(t) * sin(t * 0.731 + 1.3);
  return 0.25 + 0.75 * g * g;
}

// Horizontal sway displacement (m) for a point at `world_pos`, to be scaled by
// normalized height up the plant by the caller (tips move, roots do not).
fn wind_sway(world_pos: vec3f, time: f32, sway: f32, phase: f32) -> vec3f {
  let along = dot(world_pos.xz, frame.wind_dir);
  let w1 = sin(time * 1.9 - along * 0.35 + phase);
  let w2 = sin(time * 3.7 - along * 1.10 + phase * 1.7) * 0.35;
  let amp = frame.wind_strength * wind_gust(time) * sway;
  let d = (w1 + w2) * amp;
  // Slight vertical dip as the tip leans over (small-angle arc approximation).
  return vec3f(frame.wind_dir.x * d, -abs(d) * 0.3, frame.wind_dir.y * d);
}
