// GCMESH1 source-vertex decode — the WGSL twin of src/mesh/gcmesh.ts.
//
// Include this instead of hand-rolling the decode. The octahedral convention
// here is NOT the one most references print, and getting it wrong produces
// lighting that looks plausible rather than broken, which is how the wrong
// version survived for months:
//
//   the stored pair is (x, y) and the RECONSTRUCTED component is Z, positive.
//
// Measured, not assumed. AXIS: decoding three ways and comparing against face
// normals computed from raw dequantized POSITIONS over 217,113 triangles gives
// mean |cos| 0.4225 (x-derived) / 0.5354 (y-derived) / 0.7871 (z-derived), and z
// wins by more still on the other two species (0.85, 0.94). SIGN: not from the
// winding — the meshes are open shells and are not even wound alike (mean signed
// cos -0.73 / +0.71 / +0.89) — but from a 4mm ray fired along +n and -n from
// every sampled vertex, skipping its own triangles. A surface has material on one
// side only, and on all three meshes that side is -n, so +n is outward and there
// is no negation. See tools/probe-oct-normal.ts and the "Harness bugs" section
// of docs/ORCHESTRATION.md.
//
// This applies to SOURCE mesh normals only. A bake that encodes its own
// octahedral normals and decodes them with its own matching pair is
// self-consistent and none of this touches it.

/// Decode a GCMESH1 octahedral normal from the two stored UNORM components.
fn gcmesh_normal_decode(u01: f32, v01: f32) -> vec3f {
  let u = u01 * 2.0 - 1.0;
  let v = v01 * 2.0 - 1.0;
  var x = u;
  var y = v;
  let z = 1.0 - abs(u) - abs(v);
  if (z < 0.0) {
    x = (1.0 - abs(v)) * select(-1.0, 1.0, u >= 0.0);
    y = (1.0 - abs(u)) * select(-1.0, 1.0, v >= 0.0);
  }
  return normalize(vec3f(x, y, z));
}

/// Same, from the raw u16 pair as stored in the vertex record.
fn gcmesh_normal_decode_u16(nu: u32, nv: u32) -> vec3f {
  return gcmesh_normal_decode(f32(nu) / 65535.0, f32(nv) / 65535.0);
}
