#include "src/wgsl/scatter.wgsl"

// THE CANOPY-HULL FIELD — this is where the stand's real placement enters.
//
// Every plant the active stand specifies (exact position, scale, species entry,
// wind phase, from the shared scatter twin) stamps its own footprint into a
// world-space field: the tallest plant covering a column wins, via atomicMax on
// a packed (height | phase | entry) word. Nothing is rasterized and no plant
// primitive exists — this is a data-structure build, run once per field origin,
// not per frame.
//
// The field is what makes the baked ray table stand-faithful: it says how tall
// the canopy is at every 25cm of the world, which species owns it, how it sways,
// and where there is bare ground.

struct EntryInfo {
  radius: f32,  // horizontal support radius of the source mesh (m, at scale 1)
  hmax: f32,    // height_scale * scale_max
  taper: f32,   // hull dome falloff across the footprint
  _pad: f32,
}

struct FieldParams {
  origin: vec2f,
  texel: f32,
  res: f32,
  cell_min: vec2i,
  cells_x: i32,
  cells_z: i32,
  n_entries: u32,
  seed: u32,
  h_quant: f32,     // 4095 / H_RANGE
  inv_h_quant: f32, // H_RANGE / 4095
  ent: array<EntryInfo, 8>,
}

@group(1) @binding(0) var<uniform> fp: FieldParams;
@group(1) @binding(1) var<storage, read_write> pack: array<atomic<u32>>;

@compute @workgroup_size(64)
fn cs_splat(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let flat = gid.y * (nwg.x * 64u) + gid.x;
  let cells = u32(fp.cells_x * fp.cells_z);
  let total = cells * fp.n_entries * SCATTER_MAX_PER_CELL;
  if (flat >= total) {
    return;
  }
  let slot = flat % SCATTER_MAX_PER_CELL;
  let rest = flat / SCATTER_MAX_PER_CELL;
  let entry = rest % fp.n_entries;
  let cell_idx = i32(rest / fp.n_entries);
  let cell = fp.cell_min + vec2i(cell_idx % fp.cells_x, cell_idx / fp.cells_x);

  let pt = scatter_candidate(fp.seed, entry, cell, slot);
  if (!pt.exists) {
    return;
  }
  let info = fp.ent[entry];
  let height = stand_table[entry].height_scale * pt.scale;
  let radius = max(info.radius * pt.scale, fp.texel * 0.6);
  let inv_r2 = 1.0 / (radius * radius);

  // Footprint stamp. Its texel extent is a property of the species mesh, not of
  // the plant count: a fixed few texels per plant.
  let c_uv = (pt.pos.xz - fp.origin) / fp.texel;
  let rad_t = radius / fp.texel;
  let i0 = vec2i(floor(c_uv - rad_t));
  let i1 = vec2i(ceil(c_uv + rad_t));
  let n = i32(fp.res);
  let phase_q = u32(clamp(pt.phase / 6.28318531, 0.0, 0.999) * 64.0) & 63u;
  let word_lo = (phase_q << 14u) | ((entry & 15u) << 10u);
  for (var z = max(i0.y, 0); z <= min(i1.y, n - 1); z = z + 1) {
    for (var x = max(i0.x, 0); x <= min(i1.x, n - 1); x = x + 1) {
      let p = (vec2f(f32(x), f32(z)) + 0.5) * fp.texel + fp.origin;
      let dsq = dot(p - pt.pos.xz, p - pt.pos.xz) * inv_r2;
      if (dsq > 1.0) {
        continue;
      }
      let h = height * (1.0 - info.taper * dsq);
      let hq = min(u32(h * fp.h_quant), 4095u);
      let word = (hq << 20u) | word_lo;
      atomicMax(&pack[u32(z) * u32(fp.res) + u32(x)], word);
    }
  }
}

@group(2) @binding(0) var field_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn cs_resolve(@builtin(global_invocation_id) gid: vec3u) {
  let n = u32(fp.res);
  if (gid.x >= n || gid.y >= n) {
    return;
  }
  let word = atomicLoad(&pack[gid.y * n + gid.x]);
  let h = f32(word >> 20u) * fp.inv_h_quant;
  let phase = f32((word >> 14u) & 63u) / 64.0 * 6.28318531;
  let entry = f32((word >> 10u) & 15u);
  // Phase is stored as its sine/cosine so the field can be filtered bilinearly
  // without wrapping discontinuities tearing the wind.
  textureStore(field_out, vec2i(gid.xy), vec4f(h, sin(phase), cos(phase), entry));
}
