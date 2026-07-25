// Per-stand-entry parameters shared by the cull compute pass and the card
// draw. Written every frame by main.ts — keep the field order in sync.

struct EntryInfo {
  planes: array<vec4f, 6>, // world-space frustum planes (nx,ny,nz,d), normalized
  region: vec4f,           // origin_cell.x, origin_cell.y, side_x, side_z
  ids: vec4f,              // seed, entry_index, region_r, near_split
  caps: vec4f,             // capacity_near, capacity_far, alpha_ref, sphere_r
  aabb_c: vec4f,           // plant-local AABB centre xyz, w = plant height (local)
  aabb_h: vec4f,           // plant-local AABB half extents xyz
  shade: vec4f,            // occlusion, transmission, canopy depth, parallax
}

// 25 baked views x 4 vec4:
//   (r.xyz, boxC.x) (u.xyz, boxC.y) (b.xyz, dNear) (halfW, halfH, depthRange, 0)
struct TileTable {
  v: array<vec4f, 100>,
}
