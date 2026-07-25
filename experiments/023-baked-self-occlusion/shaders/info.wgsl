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
  // Carpet (mat) layout. enum_slots is carpet_div^2 for a mat and the ordinary
  // scatter budget otherwise — the number of candidate slots per cell the cull
  // must EVALUATE, which is NOT the number expected to survive. plane_h is the
  // height (plant-local m, scale 1) of the ground-parallel quad, i.e. the mean
  // capitulum apex the straight-down view shows.
  carpet: vec4f,           // enum_slots, plane_h, is_carpet, 0
  // Tile-local [0,1]^2 -> straight-down view UV: uv * scale.xy + bias.zw,
  // measured from the tile centre. This is the species' periodic square, which
  // is smaller than the captured box (the mesh overflows its own period).
  carpet_uv: vec4f,        // scale.x, scale.y, bias.x, bias.y
}

// 25 baked views x 4 vec4:
//   (r.xyz, boxC.x) (u.xyz, boxC.y) (b.xyz, dNear) (halfW, halfH, depthRange, 0)
struct TileTable {
  v: array<vec4f, 100>,
}
