// Declarations shared by the fin-cards cull pass and the render pass. Both
// passes bind the SAME uniform buffer and must agree bit-for-bit on the
// per-plant fade (the cull rejects plants whose fade is zero; the vertex
// shader re-derives it for the surviving ones), so the fade lives here.

struct FinCfg {
  cx: f32, yc: f32, cz: f32, half_w: f32,
  half_h: f32, y_low: f32, y_high: f32, radius: f32,
  origin_cell_x: f32, origin_cell_z: f32, side: f32, seed: f32,
  max_dist: f32, entry_index: f32, edge_fade: f32, top_blend: f32,
  alpha_sharp: f32, cell_min: f32, cell_max: f32, card_tint: f32,
  // Carpet entries (stand_table[i].carpet_div > 0) — see shaders/moss.wgsl.
  // `slots` is the entry's candidate slots per scatter cell: carpet_div^2 for a
  // mat (484 for the bog moss), SCATTER_MAX_PER_CELL otherwise. NEVER assume
  // the latter — a carpet has more slots per cell than the scatter budget.
  slots: f32, carpet: f32, layer_count: f32, moss_ref: f32,
  lvl_a: vec4f,  // cross-section heights 0..3 (mesh frame)
  lvl_b: vec4f,  // heights 4, 5, mean capitulum apex, pad
}

/// One surviving plant, written by the cull pass and read per vertex (24 B).
/// Scalars, not vec3f — a vec3 member would pad the array stride to 32 B.
struct FinPlant {
  px: f32, py: f32, pz: f32,
  yaw: f32, scale: f32, phase: f32,
}

fn rot_y(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

/// Per-plant opacity envelope: fade out toward the region boundary, and
/// collapse the cards when the camera moves inside the plant (the rules'
/// camera-inside-plant escape hatch).
fn fin_fade(dcam: f32, r_world: f32, max_dist: f32) -> f32 {
  let far_fade = 1.0 - smoothstep(max_dist * 0.72, max_dist * 0.97, dcam);
  let near_fade = smoothstep(r_world * 0.35, r_world * 0.95, dcam);
  return far_fade * near_fade;
}

/// Distance (m) beyond which a carpet tile draws only its closure card. Both
/// the cull pass (which tile goes in the near list) and moss.wgsl (how many
/// cards a tile has) must agree on this, or the extra cards are either never
/// dispatched or dispatched for tiles that discard them.
const MOSS_NEAR_M: f32 = 9.0;

/// Per-tile opacity envelope for a CARPET entry: the region edge only, measured
/// from the tile centre in XZ. No camera-inside term — eroding a mat you are
/// standing on opens a hole under your feet.
fn carpet_fade(d_xz: f32, max_dist: f32) -> f32 {
  return 1.0 - smoothstep(max_dist * 0.82, max_dist, d_xz);
}

/// Plant-space card center in world space (bake framing center, scaled+yawed).
fn fin_center(pos: vec3f, scale: f32, yaw: f32, cfg: FinCfg) -> vec3f {
  return pos + rot_y(vec3f(cfg.cx, cfg.yc, cfg.cz) * scale, yaw);
}
