#include "src/wgsl/scatter.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/lighting.wgsl"

// Stochastic thinning cascade. Placement is evaluated procedurally with the
// shared scatter twin over concentric cell annuli around the camera. A plant
// at distance d survives iff its slot rank r < keep(d) = (R0/d)^2; survivors
// widen by 1/sqrt(keep) so the ENSEMBLE frontal area is statistically
// conserved. Ring k doubles the covered radius and quarters the slot cap, so
// per-ring instance count is constant and total per-frame work is
// O(log(view radius)) — independent of stand plant count.
//
// Coverage is realized per pixel: the atlas alpha is mip-averaged fractional
// coverage (premultiplied bake), tested against a world/texel-anchored hash
// threshold. Near the camera the threshold is a hard 0.5 (crisp silhouettes,
// solid early-z); with distance it morphs into the hashed test that IS the
// distance-collapse mechanism (see NOTES.md for the taste-rule justification).

struct CardUni {
  rings: array<vec4<u32>, 6>, // start_instance, n (outer half-cells), h (hole half-cells), cap
  ring_count: u32,
  seed: u32,
  entry_index: u32,
  debug_rings: u32,
  cam_cell: vec2i,
  stand_radius: f32,
  detail_r: f32,
  card_center: vec3f,
  r_card: f32,
  width_cap: f32,
  densify: f32,
  stoch_start: f32,
  stoch_full: f32,
}

@group(1) @binding(0) var<uniform> card_uni: CardUni;
@group(1) @binding(1) var atlas_albedo: texture_2d<f32>;
@group(1) @binding(2) var atlas_normal: texture_2d<f32>;
@group(1) @binding(3) var atlas_samp: sampler;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,                          // card uv, v up
  @location(1) world: vec3f,
  @location(2) @interpolate(flat) pid: u32,        // stable per-plant hash
  @location(3) @interpolate(flat) yad: vec4f,      // yaw, amp, dist, vf (azimuth view coord)
  @location(4) @interpolate(flat) axes: vec4f,     // right.xz, up2.xz (top-view mapping)
  @location(5) @interpolate(flat) ktop: f32,
  @location(6) @interpolate(flat) ring_id: u32,
}

fn kill_vertex() -> VOut {
  var out: VOut;
  out.pos = vec4f(0.0, 0.0, 2.0, 1.0); // clipped
  return out;
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  // --- ring lookup (<= 6 iterations) ---
  var r = 0u;
  loop {
    if (r + 1u >= card_uni.ring_count || ii < card_uni.rings[r + 1u].x) { break; }
    r = r + 1u;
  }
  let ring = card_uni.rings[r];
  let n = i32(ring.y);
  let h = i32(ring.z);
  let cap = ring.w;
  let local = ii - ring.x;
  let cell_idx = local / cap;
  let slot = local % cap;

  // --- annulus cell decomposition: 2 full-width strips + 2 side blocks ---
  let w_cells = u32(2 * n);
  var cx: i32;
  var cz: i32;
  if (h == 0) {
    cx = i32(cell_idx % w_cells) - n;
    cz = i32(cell_idx / w_cells) - n;
  } else {
    let strip = w_cells * u32(n - h);
    let side_w = u32(n - h);
    let side_c = side_w * u32(2 * h);
    if (cell_idx < strip) {
      cx = i32(cell_idx % w_cells) - n;
      cz = h + i32(cell_idx / w_cells);
    } else if (cell_idx < 2u * strip) {
      let j = cell_idx - strip;
      cx = i32(j % w_cells) - n;
      cz = -n + i32(j / w_cells);
    } else {
      let j = cell_idx - 2u * strip;
      let side = j / side_c;
      let l = j % side_c;
      cx = select(-n, h, side == 1u) + i32(l % side_w);
      cz = i32(l / side_w) - h;
    }
  }
  let cell = card_uni.cam_cell + vec2i(cx, cz);

  // --- cheap existence + rank rejection before touching the terrain ---
  let eh = hash4(card_uni.seed, bitcast<u32>(cell.x), bitcast<u32>(cell.y), (card_uni.entry_index << 16u) ^ slot);
  let entry = stand_table[card_uni.entry_index];
  if (hash_f32(hash2(eh, 0u)) >= min(entry.density, SCATTER_MAX_DENSITY) / SCATTER_MAX_DENSITY) {
    return kill_vertex();
  }
  let ox = hash_f32(hash2(eh, 1u));
  let oz = hash_f32(hash2(eh, 2u));
  let xz = (vec2f(cell) + vec2f(ox, oz)) * SCATTER_CELL_SIZE;
  if (abs(xz.x) > card_uni.stand_radius || abs(xz.y) > card_uni.stand_radius) {
    return kill_vertex();
  }
  let d = distance(frame.camera_pos.xz, xz);

  // --- stochastic thinning: survival by slot rank vs keep(d) ---
  let rd = card_uni.detail_r / max(d, card_uni.detail_r);
  let keep = rd * rd; // min(1, (R0/d)^2)
  let rank = (f32(slot) + 0.5) / 128.0;
  if (rank >= keep) {
    return kill_vertex();
  }
  let fade_w = clamp((keep - rank) / (0.35 * keep), 0.0, 1.0);
  let amp = min(card_uni.width_cap, inverseSqrt(keep));

  // Full candidate (shared twin) — exact stand position/yaw/scale/phase.
  let sp = scatter_candidate(card_uni.seed, card_uni.entry_index, cell, slot);

  let scale = sp.scale;
  let cyaw = cos(sp.yaw);
  let syaw = sin(sp.yaw);
  let off_l = card_uni.card_center;
  let off_w = vec2f(cyaw * off_l.x - syaw * off_l.z, syaw * off_l.x + cyaw * off_l.z) * scale;
  let center_w = vec3f(sp.pos.x + off_w.x, sp.pos.y + off_l.y * scale, sp.pos.z + off_w.y);

  var fwd = frame.camera_pos - center_w;
  let fl = length(fwd);
  let near_w = clamp((fl - 0.35) / 0.4, 0.0, 1.0); // camera-inside-plant fade
  fwd = fwd / max(fl, 1e-4);
  var right = cross(vec3f(0.0, 1.0, 0.0), fwd);
  let rl = length(right);
  right = select(right / max(rl, 1e-5), vec3f(1.0, 0.0, 0.0), rl < 1e-3);
  let up2 = cross(fwd, right);

  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  let c = corners[vi];
  let half_size = card_uni.r_card * scale;
  let sz = fade_w * near_w;
  var world = center_w
    + right * (c.x * half_size * amp * sz)
    + up2 * (c.y * half_size * sz);

  // Wind: shear by normalized height so roots stay put and tips travel.
  let hf = clamp((world.y - sp.pos.y) / max(2.0 * half_size, 1e-3), 0.0, 1.0);
  world += wind_sway(sp.pos, frame.time, entry.sway, sp.phase) * hf;

  var out: VOut;
  out.pos = frame.view_proj * vec4f(world, 1.0);
  out.uv = c * 0.5 + 0.5;
  out.world = world;
  out.pid = eh;
  var vf = (atan2(fwd.z, fwd.x) - sp.yaw) * 1.2732395; // / (pi/4)
  vf = vf - 8.0 * floor(vf / 8.0);
  out.yad = vec4f(sp.yaw, amp, d, vf);
  out.axes = vec4f(right.x, right.z, up2.x, up2.z);
  out.ktop = smoothstep(0.30, 0.95, fwd.y);
  out.ring_id = r;
  return out;
}

fn tile_uv(idx: u32, uvl: vec2f) -> vec2f {
  let cuv = clamp(uvl, vec2f(0.004), vec2f(0.996));
  return (cuv + vec2f(f32(idx % 3u), f32(idx / 3u))) / 3.0;
}

fn tap_albedo(idx: u32, uvl: vec2f, gx: vec2f, gy: vec2f) -> vec4f {
  return textureSampleGrad(atlas_albedo, atlas_samp, tile_uv(idx, uvl), gx, gy);
}

fn tap_normal(idx: u32, uvl: vec2f, gx: vec2f, gy: vec2f) -> vec4f {
  return textureSampleGrad(atlas_normal, atlas_samp, tile_uv(idx, uvl), gx, gy);
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let yaw = in.yad.x;
  let amp = in.yad.y;
  let dist = in.yad.z;

  let uv_t = vec2f(in.uv.x, 1.0 - in.uv.y);
  let gx = dpdx(uv_t) / 3.0;
  let gy = dpdy(uv_t) / 3.0;

  // Two nearest azimuth views, then the top view by elevation.
  let i0 = u32(floor(in.yad.w)) % 8u;
  let i1 = (i0 + 1u) % 8u;
  let t = fract(in.yad.w);
  var alb = mix(tap_albedo(i0, uv_t, gx, gy), tap_albedo(i1, uv_t, gx, gy), t);
  var nrm = mix(tap_normal(i0, uv_t, gx, gy), tap_normal(i1, uv_t, gx, gy), t);
  if (in.ktop > 0.004) {
    // Map the card plane into the plant-local top frame (foreshortened by
    // the projected card axes, de-rotated by yaw).
    let q = uv_t - 0.5;
    let wxz = in.axes.xy * q.x - in.axes.zw * q.y;
    let cy = cos(yaw);
    let sy = sin(yaw);
    let l = vec2f(cy * wxz.x + sy * wxz.y, -sy * wxz.x + cy * wxz.y);
    let uv_top = l + 0.5;
    alb = mix(alb, tap_albedo(8u, uv_top, gx, gy), in.ktop);
    nrm = mix(nrm, tap_normal(8u, uv_top, gx, gy), in.ktop);
  }

  let a = alb.a;
  if (a < 0.004) { discard; }

  // Opacity as statistics: densify amplified clumps (a card standing in for
  // 1/keep plants raises its interior coverage like independent overlap
  // would), then realize the fractional coverage with a stable hashed test.
  let gamma = 1.0 + (amp - 1.0) * card_uni.densify;
  let a_eff = 1.0 - pow(1.0 - min(a, 0.995), gamma);
  let px = vec2u(clamp(uv_t, vec2f(0.0), vec2f(0.999)) * 128.0);
  let xi = hash_f32(hash3(in.pid, px.x, px.y));
  let sigma = smoothstep(card_uni.stoch_start, card_uni.stoch_full, dist);
  let tau = mix(0.5, mix(0.03, 0.97, xi), sigma);
  if (a_eff < tau) { discard; }

  let mean_rgb = alb.rgb / a; // un-premultiply: mean radiance of covered area
  var n_l = nrm.rgb / max(nrm.a, 0.01) * 2.0 - 1.0;
  n_l = select(n_l, vec3f(0.0, 1.0, 0.0), dot(n_l, n_l) < 1e-4);
  n_l = normalize(n_l);
  let cyw = cos(yaw);
  let syw = sin(yaw);
  let n_w0 = vec3f(cyw * n_l.x - syw * n_l.z, n_l.y, syw * n_l.x + cyw * n_l.z);
  // Far cards drift toward the statistical canopy normal (up-dominated).
  let n_w = normalize(mix(n_w0, vec3f(0.0, 1.0, 0.0), 0.25 * sigma));

  let ao = 0.62 + 0.38 * in.uv.y;
  var color = light_surface(mean_rgb * ao, n_w, in.world);
  color = apply_fog(color, in.world);

  if (card_uni.debug_rings == 1u) {
    var ring_tints = array<vec3f, 6>(
      vec3f(1.0, 0.2, 0.2), vec3f(1.0, 1.0, 0.2), vec3f(0.2, 1.0, 0.2),
      vec3f(0.2, 0.6, 1.0), vec3f(0.8, 0.2, 1.0), vec3f(1.0, 1.0, 1.0),
    );
    color = mix(color, ring_tints[in.ring_id], 0.55);
  }
  return vec4f(color, 1.0);
}
