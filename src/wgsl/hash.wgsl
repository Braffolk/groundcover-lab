// Integer PCG hash — bit-identical twin of src/scene/hash.ts.
// All placement/randomness in the project flows through these functions so that
// CPU-generated instance buffers and fully in-shader procedural techniques
// agree on every plant.

fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn hash2(a: u32, b: u32) -> u32 {
  return pcg(a ^ pcg(b));
}

fn hash3(a: u32, b: u32, c: u32) -> u32 {
  return pcg(a ^ pcg(b ^ pcg(c)));
}

fn hash4(a: u32, b: u32, c: u32, d: u32) -> u32 {
  return pcg(a ^ pcg(b ^ pcg(c ^ pcg(d))));
}

// Uniform [0,1) with a 24-bit mantissa — exactly representable in f32, so the
// f64 TypeScript twin produces the identical value.
fn hash_f32(h: u32) -> f32 {
  return f32(h >> 8u) / 16777216.0;
}
