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
  // Slots per cell for THIS entry — carpetDiv^2 (484 for the bog moss), which
  // is deliberately larger than SCATTER_MAX_PER_CELL. Hardcoding 128 here
  // enumerated 6 of the carpet's 22 grid rows and rendered the mat as 4m-spaced
  // stripes.
  slots_per_cell: f32,
  wind_margin: f32,    // 0 for a rigid species (moss): no sway to bound
  proxy_h: f32,        // proxy height at scale 1 (cull sphere center)
  // Carpet lift, in proxy heights, along the conformed up axis. The base pass
  // draws the terrain as 1m quads while placement uses the 0.5m bilinear
  // heightmap, and the drawn surface sits up to 6.5cm ABOVE the placed height
  // (>1cm over 21% of the area) — enough to bury a 9cm mat. See NOTES.md.
  base_lift: f32,
}

struct PlantInst {
  pos_yaw: vec4f,
  // x scale, y wind phase, zw the terrain-conformed up vector's x/z (up.y is
  // recovered as sqrt(1-x^2-z^2); terrain normals never point down).
  scale_phase: vec4f,
}

// draw-indexed-indirect args (the proxy is an indexed 18-vertex mesh).
struct DrawArgs {
  index_count: u32,
  instance_count: atomic<u32>,
  first_index: u32,
  base_vertex: i32,
  first_instance: u32,
}

@group(1) @binding(0) var<uniform> cull: CullCfg;
@group(1) @binding(1) var<storage, read_write> instances: array<PlantInst>;
@group(1) @binding(2) var<storage, read_write> draw_args: DrawArgs;

@compute @workgroup_size(64)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
  let side = u32(cull.side);
  let slots = max(u32(cull.slots_per_cell), 1u);
  let cell_lin = gid.x / slots;
  let slot = gid.x % slots;
  if (cell_lin >= side * side) { return; }
  let cx = i32(cull.origin_cell.x) + i32(cell_lin % side);
  let cz = i32(cull.origin_cell.y) + i32(cell_lin / side);
  if (cx < i32(cull.cell_min) || cx > i32(cull.cell_max) ||
      cz < i32(cull.cell_min) || cz > i32(cull.cell_max)) { return; }
  let entry = u32(cull.entry_index);

  let sp = scatter_candidate(u32(cull.seed), entry, vec2i(cx, cz), slot);
  if (!sp.exists) { return; }

  let rad = cull.cull_rad * sp.scale + cull.wind_margin;
  let center = sp.pos + vec3f(0.0, cull.proxy_h * sp.scale * 0.5, 0.0);

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
  // Fit the plant to the ground here, once per instance (not per vertex): the
  // proxy is a rigid convex volume, so its frame must stay affine for the
  // closed-form chord to hold — see NOTES.md on the terrain-fitting rung. The
  // heightmap's stored normal is already a 1m-wide central difference, so this
  // single bilinear tap is the plane fit over a footprint this small (a 0.18m
  // moss tile is a third of a 0.5m heightmap texel).
  let align = clamp(stand_table[entry].slope_align, 0.0, 1.0);
  var up = vec3f(0.0, 1.0, 0.0);
  if (align > 0.0) {
    let g = terrain_sample(sp.pos.xz);
    let ny = sqrt(max(1.0 - g.y * g.y - g.z * g.z, 0.0));
    up = normalize(mix(up, vec3f(g.y, ny, g.z), align));
  }
  let pos = sp.pos + up * (cull.base_lift * cull.proxy_h * sp.scale);
  instances[idx].pos_yaw = vec4f(pos, sp.yaw);
  instances[idx].scale_phase = vec4f(sp.scale, sp.phase, up.x, up.z);
}
