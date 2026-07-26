// Shared math of the baked ray-answer field ("ray LUT"). The atlas is a
// RF_VIEWS x RF_VIEWS grid of orthographic ray-bundle answers: cell (i,j)
// holds every precomputed ray with direction oct_decode((ij+0.5)/RF_VIEWS),
// resolved over a RF_SLAB x RF_SLAB grid of ray offsets across the plant's
// baked bounding BOX as seen from that direction. TS mirror of the direction /
// basis / extent math lives in bake.ts — keep them in sync.

const RF_VIEWS: f32 = 24.0;
const RF_SLAB: f32 = 64.0;
// Slack on the fitted slab so the silhouette (and the first/last depth slice)
// is never clipped by the ortho frustum at the bake.
const RF_EXT_MARGIN: f32 = 1.02;

fn rf_sgn(x: f32) -> f32 {
  return select(-1.0, 1.0, x >= 0.0);
}

// Full-sphere octahedral direction <-> [0,1]^2. Same fold as the GCMESH1
// normal encoding, so one decoder serves both directions and baked normals.
fn rf_oct_encode(d: vec3f) -> vec2f {
  let n = d / (abs(d.x) + abs(d.y) + abs(d.z));
  var e = n.xz;
  if (n.y < 0.0) {
    e = (vec2f(1.0) - abs(vec2f(n.z, n.x))) * vec2f(rf_sgn(n.x), rf_sgn(n.z));
  }
  return e * 0.5 + 0.5;
}

fn rf_oct_decode(e: vec2f) -> vec3f {
  let f = e * 2.0 - 1.0;
  var n = vec3f(f.x, 1.0 - abs(f.x) - abs(f.y), f.y);
  if (n.y < 0.0) {
    n = vec3f((1.0 - abs(f.y)) * rf_sgn(f.x), n.y, (1.0 - abs(f.x)) * rf_sgn(f.y));
  }
  return normalize(n);
}

/**
 * Baked MESH normal from the atlas' oct pair. GCMESH1 stores the standard
 * octahedral encoding — (u, v) = (n.x, n.y) with n.z derived as 1-|u|-|v| —
 * while rf_oct_decode() (which also serves the view directions, where the
 * encode/decode pair is self-consistent and the convention is free) derives the
 * Y component instead. The two differ by exactly a y<->z swap, verified against
 * face normals computed from raw vertex positions: mean |cos| to the geometric
 * normal is 0.75 for the (x,y)+z convention vs 0.57 for (x,z)+y (sphagnum,
 * 120k vertex samples).
 *
 * Consequence of getting this wrong, for the record: a straight-down view of the
 * moss returned a mean normal of (0, 0.05, 0.97) — horizontal — instead of up,
 * so the mat lit as if every capitulum faced sideways, and the scatter's 90
 * degree tile rotations turned that into a hard chequerboard.
 *
 * The swap is an isometry, so it commutes with the bake's normal averaging and
 * needs no rebake: the atlas holds swap(true mean normal), and swapping back
 * here is exact.
 */
fn rf_mesh_normal(e: vec2f) -> vec3f {
  let n = rf_oct_decode(e);
  return vec3f(n.x, n.z, n.y);
}

// Deterministic per-view slab basis. Must match viewBasis() in bake.ts.
fn rf_view_right(dv: vec3f) -> vec3f {
  if (abs(dv.y) > 0.999) {
    return vec3f(1.0, 0.0, 0.0);
  }
  return normalize(cross(vec3f(0.0, 1.0, 0.0), dv));
}

/**
 * Half-size of the slab along `ax`: the support function of the baked box
 * (half-extents `half`) in that direction. Every view therefore spends its
 * whole 64x64 cell on the box's own silhouette instead of on the loose
 * bounding sphere — for a 7cm-tall carpet tile seen edge-on that is a 3.3x
 * finer vertical texel, which is exactly where a cushion's relief lives.
 * Must match rfExtent() in bake.ts.
 */
fn rf_extent(ax: vec3f, half: vec3f) -> f32 {
  return (abs(ax.x) * half.x + abs(ax.y) * half.y + abs(ax.z) * half.z) * RF_EXT_MARGIN;
}

// Atlas uv for slab coordinates sp (metres along r_ax/u_ax from the box
// center) inside view cell `cellf`, given that view's fitted half-extents.
// Half-texel inset stops bilinear bleeding across neighbouring view cells.
fn rf_atlas_uv(cellf: vec2f, sp: vec2f, ext: vec2f) -> vec2f {
  let hp = 0.5 / RF_SLAB;
  let suv = clamp(sp / (2.0 * ext) + vec2f(0.5), vec2f(hp), vec2f(1.0 - hp));
  return (cellf + suv) / RF_VIEWS;
}
