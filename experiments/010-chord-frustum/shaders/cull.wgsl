#include "src/wgsl/scatter.wgsl"

// Per-frame instance culling. Threads enumerate the camera-bounded region's
// scatter slots (cells x 128 candidates) for ONE stand entry, evaluate the
// shared placement hash, and compact surviving plants (exists + inside view
// frustum + inside the region ring) into an instance list + indirect draw
// args. Work is bounded by the region, never by the stand's plant count.

struct CullCfg {
  origin_cell: vec2f,  // integer cell coords of region corner
  side: f32,           // cells per region side
  seed: f32,
  entry_index: f32,
  max_dist: f32,
  cull_rad: f32,       // proxy bounding radius at scale 1 (+ margins)
  cell_min: f32,       // stand-region cell clamp (matches Scatter.region)
  cell_max: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
}

struct PlantInst {
  pos_yaw: vec4f,
  scale_phase: vec4f,
}

struct DrawArgs {
  vertex_count: u32,
  instance_count: atomic<u32>,
  first_vertex: u32,
  first_instance: u32,
}

@group(1) @binding(0) var<uniform> cull: CullCfg;
@group(1) @binding(1) var<storage, read_write> instances: array<PlantInst>;
@group(1) @binding(2) var<storage, read_write> draw_args: DrawArgs;

const WIND_MARGIN: f32 = 0.5;

@compute @workgroup_size(64)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
  let side = u32(cull.side);
  let cell_lin = gid.x / SCATTER_MAX_PER_CELL;
  let slot = gid.x % SCATTER_MAX_PER_CELL;
  if (cell_lin >= side * side) { return; }
  let cx = i32(cull.origin_cell.x) + i32(cell_lin % side);
  let cz = i32(cull.origin_cell.y) + i32(cell_lin / side);
  if (cx < i32(cull.cell_min) || cx > i32(cull.cell_max) ||
      cz < i32(cull.cell_min) || cz > i32(cull.cell_max)) { return; }
  let entry = u32(cull.entry_index);

  let sp = scatter_candidate(u32(cull.seed), entry, vec2i(cx, cz), slot);
  if (!sp.exists) { return; }

  let rad = cull.cull_rad * sp.scale + WIND_MARGIN;
  let center = sp.pos + vec3f(0.0, rad * 0.5, 0.0);

  // Region ring: hard cut where the fragment fade reaches zero.
  let dcam = distance(frame.camera_pos, center);
  if (dcam > cull.max_dist + rad) { return; }

  // Frustum cull against planes extracted from view_proj (clip z in [0,1]).
  let m = frame.view_proj;
  let row0 = vec4f(m[0][0], m[1][0], m[2][0], m[3][0]);
  let row1 = vec4f(m[0][1], m[1][1], m[2][1], m[3][1]);
  let row2 = vec4f(m[0][2], m[1][2], m[2][2], m[3][2]);
  let row3 = vec4f(m[0][3], m[1][3], m[2][3], m[3][3]);
  let planes = array<vec4f, 5>(
    row3 + row0,  // left
    row3 - row0,  // right
    row3 + row1,  // bottom
    row3 - row1,  // top
    row2,         // near
  );
  let c4 = vec4f(center, 1.0);
  for (var i = 0u; i < 5u; i++) {
    let pl = planes[i];
    if (dot(pl, c4) < -rad * length(pl.xyz)) { return; }
  }

  let idx = atomicAdd(&draw_args.instance_count, 1u);
  if (idx >= arrayLength(&instances)) {
    atomicSub(&draw_args.instance_count, 1u);
    return;
  }
  instances[idx].pos_yaw = vec4f(sp.pos, sp.yaw);
  instances[idx].scale_phase = vec4f(sp.scale, sp.phase, 0.0, 0.0);
}
