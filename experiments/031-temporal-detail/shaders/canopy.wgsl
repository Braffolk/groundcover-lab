#include "src/wgsl/scatter.wgsl"
#include "./tables.wgsl"

// The amortised canopy cache — the "temporal detail" half of the method.
//
// A world-anchored 128x8x128 grid follows the camera by TOROIDAL addressing
// (cell index = world cell & 127), so scrolling moves no data: only the column
// that wraps in at the window rim is momentarily stale, and the renderer fades
// the cache's influence out exactly there.
//
//   cs_clear  zeroes the density accumulator (cheap, whole grid)
//   cs_splat  re-derives the stand's plants inside the window from the shared
//             scatter and accumulates foliage density with a bilinear xz
//             footprint and a vertical mass profile. Bounded by window area,
//             never by the stand's plant count.
//   cs_light  the expensive part: per cell, marches the density field toward
//             the sun for transmittance and straight up for sky visibility.
//             Only 1/`groups` of the cells run per frame (round robin). Because
//             the grid is world-anchored and plants do not move, a cell's cached
//             value stays CORRECT while it waits its turn — the amortization
//             costs freshness only where the window has just scrolled.
//
// Renderers then buy volumetric self-occlusion, contact shading and inter-plant
// sun shadowing for ONE texture tap per fragment.

@group(1) @binding(0) var<uniform> canopy: CanopyInfo;
@group(1) @binding(1) var<storage, read_write> dens: array<atomic<u32>>;
@group(1) @binding(2) var canopy_out: texture_storage_3d<rgba8unorm, write>;

const NX: u32 = 128u;
const NY: u32 = 8u;
const NZ: u32 = 128u;
const CELLS: u32 = NX * NY * NZ;
const FIXED: f32 = 4096.0;
const SUN_STEPS: i32 = 12;

fn cell_index(tx: u32, iy: u32, tz: u32) -> u32 {
  return (iy * NZ + tz) * NX + tx;
}

fn dens_at(tx: u32, iy: u32, tz: u32) -> f32 {
  return f32(atomicLoad(&dens[cell_index(tx, iy, tz)])) * (1.0 / FIXED);
}

@compute @workgroup_size(64)
fn cs_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= CELLS) {
    return;
  }
  atomicStore(&dens[gid.x], 0u);
}

@compute @workgroup_size(64)
fn cs_splat(@builtin(global_invocation_id) gid: vec3<u32>) {
  let side_x = u32(canopy.splat_side.x);
  let per_entry = side_x * u32(canopy.splat_side.y) * SCATTER_MAX_PER_CELL;
  if (per_entry == 0u || gid.x >= per_entry * u32(canopy.n_entries)) {
    return;
  }
  let entry_index = min(gid.x / per_entry, CANOPY_MAX_ENTRIES - 1u);
  let rest = gid.x % per_entry;
  let slot = rest % SCATTER_MAX_PER_CELL;
  let cell_lin = rest / SCATTER_MAX_PER_CELL;
  let cx = i32(canopy.splat_origin.x) + i32(cell_lin % side_x);
  let cz = i32(canopy.splat_origin.y) + i32(cell_lin / side_x);
  let sp = scatter_candidate(u32(canopy.seed), entry_index, vec2i(cx, cz), slot);
  if (!sp.exists) {
    return;
  }

  let geo = canopy.entry_geo[entry_index];
  let top = geo.y * sp.scale;
  let cell = canopy.cell;
  let cell_y = canopy.y_top / f32(NY);
  // Foliage per cell volume: a clump of radius r spread over one grid cell.
  let amount = canopy.density_scale * min(1.0, (geo.x * sp.scale) * (geo.x * sp.scale) / (cell * cell));

  // Bilinear xz footprint around the plant (cell centers sit at +0.5).
  let gx = sp.pos.x / cell - 0.5;
  let gz = sp.pos.z / cell - 0.5;
  let ix = i32(floor(gx));
  let iz = i32(floor(gz));
  let fx = gx - f32(ix);
  let fz = gz - f32(iz);
  let tx0 = u32(ix & i32(NX - 1u));
  let tx1 = u32((ix + 1) & i32(NX - 1u));
  let tz0 = u32(iz & i32(NZ - 1u));
  let tz1 = u32((iz + 1) & i32(NZ - 1u));
  let w00 = (1.0 - fx) * (1.0 - fz);
  let w10 = fx * (1.0 - fz);
  let w01 = (1.0 - fx) * fz;
  let w11 = fx * fz;

  for (var iy = 0u; iy < NY; iy++) {
    let cy = (f32(iy) + 0.5) * cell_y;
    if (cy > top) {
      break;
    }
    // Blade mass is dense low and thins toward the tips.
    let t = cy / max(top, 1e-3);
    let profile = mix(1.0, 0.3, smoothstep(0.3, 1.0, t));
    let a = amount * profile;
    atomicAdd(&dens[cell_index(tx0, iy, tz0)], u32(a * w00 * FIXED));
    atomicAdd(&dens[cell_index(tx1, iy, tz0)], u32(a * w10 * FIXED));
    atomicAdd(&dens[cell_index(tx0, iy, tz1)], u32(a * w01 * FIXED));
    atomicAdd(&dens[cell_index(tx1, iy, tz1)], u32(a * w11 * FIXED));
  }
}

@compute @workgroup_size(64)
fn cs_light(@builtin(global_invocation_id) gid: vec3<u32>) {
  let groups = u32(canopy.groups);
  let nx_g = NX / groups;
  if (gid.x >= nx_g * NY * NZ) {
    return;
  }
  let ix_g = gid.x % nx_g;
  let iy = (gid.x / nx_g) % NY;
  let tz = gid.x / (nx_g * NY);
  let tx = ix_g * groups + u32(canopy.group);

  // Toroidal cell -> the world cell it currently represents.
  let cx0 = i32(canopy.win_cell.x);
  let cz0 = i32(canopy.win_cell.y);
  let wcx = cx0 + i32((tx - u32(cx0 & i32(NX - 1u))) & (NX - 1u));
  let wcz = cz0 + i32((tz - u32(cz0 & i32(NZ - 1u))) & (NZ - 1u));
  let cell = canopy.cell;
  let cell_y = canopy.y_top / f32(NY);
  let wx = (f32(wcx) + 0.5) * cell;
  let wz = (f32(wcz) + 0.5) * cell;
  let ly = (f32(iy) + 0.5) * cell_y;

  // The grid's y axis is height above the terrain, so a ray toward the sun
  // climbs by sun.y and the ground climbs by the terrain gradient; the
  // difference is the ray's rise in grid space (locally planar terrain).
  let ts = terrain_sample(vec2f(wx, wz));
  let ny = sqrt(max(1.0 - ts.y * ts.y - ts.z * ts.z, 1e-4));
  let grad = -vec2f(ts.y, ts.z) / ny;
  let sun = frame.sun_dir;
  let rise = sun.y - dot(grad, sun.xz);

  let step = canopy.sun_step;
  var od = dens_at(tx, iy, tz) * 0.5;
  for (var s = 1; s <= SUN_STEPS; s++) {
    let t = f32(s) * step;
    let ly_s = ly + rise * t;
    if (ly_s >= canopy.y_top) {
      break;
    }
    let iy_s = u32(clamp(ly_s / cell_y, 0.0, f32(NY) - 1.0));
    let sx = u32(i32(floor((wx + sun.x * t) / cell)) & i32(NX - 1u));
    let sz = u32(i32(floor((wz + sun.z * t) / cell)) & i32(NZ - 1u));
    od += dens_at(sx, iy_s, sz);
  }
  let sun_t = exp(-od * step);

  // Sky visibility: how much canopy sits directly above this cell.
  var od_sky = dens_at(tx, iy, tz) * 0.5;
  for (var k = iy + 1u; k < NY; k++) {
    od_sky += dens_at(tx, k, tz);
  }
  let sky = exp(-od_sky * cell_y * canopy.sky_k);

  let d_norm = clamp(dens_at(tx, iy, tz) * 0.25, 0.0, 1.0);
  textureStore(canopy_out, vec3i(i32(tx), i32(iy), i32(tz)), vec4f(d_norm, sun_t, sky, 1.0));
}
