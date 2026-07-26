// Stage 5 — the BRDF. Authored WGSL, deliberately: this is where 039-nd-moss'
// wrap / fuzz / AO-Fresnel / micro-shadow / bounce stack would go, and none of
// it is expressible as node soup. 001-flat-maps' BRDF is the honest floor, so
// this file is the honest floor too.
//
// Do NOT #include anything: everything below is already in scope.

fn material_shade(g: Geo, s: Surface, vu: ViewUv) -> Shaded {
  var sh = mat_shaded_default(s);

  // Occlusion multiplies the WHOLE light term, which keeps debug_shade's
  // LIGHTING view exact: shaded / albedo is (sun + ambient*hemi) * occlusion
  // and nothing else. (039 splits the two halves apart by subtraction so each
  // gets a different occluder — that is the interesting version, and it is
  // deliberately not what this floor does.)
  //
  // `vu.shadow` is 1.0 unless the view-uv stage marched a second ray along the
  // sun, so at defaults this is exactly 001's `light_surface(...) * occlusion`.
  sh.color = light_surface(s.albedo, s.normal, g.world) * s.ao * vu.shadow;
  sh.albedo = s.albedo;
  sh.normal = s.normal;
  sh.coverage = s.opacity;
  return sh;
}
