#include "src/wgsl/frame.wgsl"
#include "src/wgsl/terrain.wgsl"

// Pass 1 of the bounded enumeration: one thread per scatter cell of the
// camera-centred region rect. Cells that survive the region circle and the
// frustum are compacted into cell_list, and cell_args[0] doubles as the
// compaction counter AND the x dimension of pass 2's indirect dispatch — so
// the plant expansion never touches a cell that cannot contribute.

const CELL_SIZE: f32 = 4.0;

struct CellInfo {
  planes: array<vec4f, 6>,
  // x,y: first cell of the rect; z,w: rect size in cells
  origin: vec4f,
  // x: region radius (m), y: tallest plant (m), z: cell list capacity
  cfg: vec4f,
}

@group(1) @binding(0) var<uniform> cell_info: CellInfo;
@group(1) @binding(1) var<storage, read_write> cell_list: array<vec2i>;
@group(1) @binding(2) var<storage, read_write> cell_args: array<atomic<u32>>;

@compute @workgroup_size(64)
fn cs_cells(@builtin(global_invocation_id) gid: vec3u) {
  let side_x = u32(cell_info.origin.z);
  let side_z = u32(cell_info.origin.w);
  if (gid.x >= side_x * side_z) {
    return;
  }
  let cx = i32(cell_info.origin.x) + i32(gid.x % side_x);
  let cz = i32(cell_info.origin.y) + i32(gid.x / side_x);
  let x0 = f32(cx) * CELL_SIZE;
  let z0 = f32(cz) * CELL_SIZE;
  let x1 = x0 + CELL_SIZE;
  let z1 = z0 + CELL_SIZE;

  // Region circle (the rect that feeds this pass is its bounding box).
  let nearest = clamp(frame.camera_pos.xz, vec2f(x0, z0), vec2f(x1, z1));
  if (distance(nearest, frame.camera_pos.xz) > cell_info.cfg.x) {
    return;
  }

  // Vertical extent: terrain corners + slack for the interior, plus canopy.
  let h00 = terrain_height(vec2f(x0, z0));
  let h10 = terrain_height(vec2f(x1, z0));
  let h01 = terrain_height(vec2f(x0, z1));
  let h11 = terrain_height(vec2f(x1, z1));
  let y_lo = min(min(h00, h10), min(h01, h11)) - 0.75;
  let y_hi = max(max(h00, h10), max(h01, h11)) + 0.75 + cell_info.cfg.y;

  for (var p = 0u; p < 6u; p = p + 1u) {
    let pl = cell_info.planes[p];
    // Positive vertex of the box against this plane.
    let pv = vec3f(
      select(x0, x1, pl.x > 0.0),
      select(y_lo, y_hi, pl.y > 0.0),
      select(z0, z1, pl.z > 0.0),
    );
    if (dot(pl.xyz, pv) + pl.w < 0.0) {
      return;
    }
  }

  let idx = atomicAdd(&cell_args[0], 1u);
  if (idx < u32(cell_info.cfg.z)) {
    cell_list[idx] = vec2i(cx, cz);
  }
}
