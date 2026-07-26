#include "src/wgsl/lighting.wgsl"
#include "src/wgsl/wind.wgsl"
#include "src/wgsl/debug.wgsl"

// Cache composite — the cheap per-frame half of the amortization. Each active
// cluster slot is one instanced quad anchored in the world at the slot's
// frozen capture plane. The fragment shader re-projects every cached texel to
// the CURRENT camera: it unpacks the slot's 16-bit capture depth, unprojects
// through the slot's stored inverse view-proj to true world position, and
// writes frag_depth from the live camera — so cached clusters depth-compose
// correctly against the terrain, the near ring, and each other, from any
// angle, without ever re-rendering their contents. Wind is a coherent
// per-cluster shear of the card (tips move, roots pinned), applied to the
// stale image so cached grass never freezes.

struct SlotMeta {
  origin: vec4f,    // xyz = card plane centre (world), w = cluster sway
  right: vec4f,     // xyz = half-extent right vector,  w = cluster wind phase
  up_v: vec4f,      // xyz = half-extent up vector,     w = unused
  slot_info: vec4f, // x = cache array layer, y = quadtree level, w = tint mode
  inv_vp: mat4x4f,  // slot frustum inverse view-proj
}

@group(1) @binding(0) var<storage, read> slot_meta: array<SlotMeta>;
@group(1) @binding(1) var cache_color: texture_2d_array<f32>;
@group(1) @binding(2) var cache_aux: texture_2d_array<f32>;
@group(1) @binding(3) var cache_sampler: sampler;

const QUAD_U = array<f32, 6>(0.0, 1.0, 0.0, 1.0, 1.0, 0.0);
const QUAD_V = array<f32, 6>(1.0, 1.0, 0.0, 1.0, 0.0, 0.0);

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) @interpolate(flat) slot: u32,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let m = slot_meta[ii];
  let u = QUAD_U[vi];
  let v = QUAD_V[vi];
  var world = m.origin.xyz + m.right.xyz * (2.0 * u - 1.0) + m.up_v.xyz * (1.0 - 2.0 * v);
  // Coherent cluster sway: shear the card, pinned at its base.
  let sway = wind_sway(m.origin.xyz, frame.time, m.origin.w, m.right.w);
  world += sway * (1.0 - v);

  var o: VOut;
  o.pos = frame.view_proj * vec4f(world, 1.0);
  o.uv = vec2f(u, v);
  o.slot = ii;
  return o;
}

struct FOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
}

/// Distinct hue per index — cluster LOD / slot inspection only.
fn cluster_tint(t: f32) -> vec3f {
  return 0.5 + 0.5 * cos(6.2831853 * (vec3f(0.0, 0.33, 0.67) + t));
}

@fragment
fn fs_main(i: VOut) -> FOut {
  let m = slot_meta[i.slot];
  let layer = i32(m.slot_info.x + 0.5);
  // The quad maps to slot texel CENTRES (half-texel inset), so a bilinear tap
  // never reaches past the slot's own texels.
  let dims = vec2f(textureDimensions(cache_color));
  let cache_uv = (0.5 + i.uv * (dims - 1.0)) / dims;
  // Color is stored premultiplied-by-coverage (empty texels are 0): the
  // bilinear tap averages covered neighbours only, so silhouettes stay clean.
  let c = textureSampleLevel(cache_color, cache_sampler, cache_uv, layer, 0.0);
  if (c.a < 0.38) { discard; }

  // Depth must NOT be filtered across the hi/lo byte split: load the nearest
  // texel exactly.
  let texel = vec2i(clamp(cache_uv * dims, vec2f(0.0), dims - 1.0));
  let aux = textureLoad(cache_aux, texel, layer, 0);
  let z_slot = (floor(aux.r * 255.0 + 0.5) * 256.0 + floor(aux.g * 255.0 + 0.5)) / 65535.0;
  if (z_slot >= 0.9998) { discard; } // background

  // Re-project the cached texel: slot ndc -> world -> current clip.
  let ndc = vec4f(i.uv.x * 2.0 - 1.0, 1.0 - 2.0 * i.uv.y, z_slot, 1.0);
  var world = m.inv_vp * ndc;
  let wpos = world.xyz / world.w;
  let clip = frame.view_proj * vec4f(wpos, 1.0);
  if (clip.w <= 0.0) { discard; }

  // The slot already holds whatever the refresh baked — the shaded colour, or
  // the albedo/normal/lighting debug view (cards.wgsl runs debug_shade there,
  // and main.ts re-reconstructs every slot when the selector changes).
  // Coverage and depth are answered live instead, from this frame's camera.
  let stored = c.rgb / max(c.a, 1e-3);
  let mode = debug_mode();
  var col = stored;
  if (mode == DEBUG_OFF) {
    col = apply_fog(stored, wpos);
  } else if (mode == DEBUG_COVERAGE) {
    col = vec3f(clamp(c.a, 0.0, 1.0));
  } else if (mode == DEBUG_DEPTH) {
    col = debug_shade(stored, stored, vec3f(0.0, 1.0, 0.0), c.a, wpos);
  }

  let tint_mode = u32(m.slot_info.w + 0.5);
  if (tint_mode != 0u) {
    let t = select(fract(m.slot_info.x * 0.618034), m.slot_info.y * 0.19, tint_mode == 1u);
    col = mix(col, cluster_tint(t), 0.55);
  }

  var o: FOut;
  o.depth = clamp(clip.z / clip.w, 0.0, 1.0);
  o.color = vec4f(col, 1.0);
  return o;
}
