fn material_surface(g: Geo, ch: Channels, base: Surface) -> Surface {
  var s = base;
  s.albedo = ch.albedo;
  let mapped = normalize(
    g.tangent * ch.normal.x * u.p.normal_strength
      + g.geo_n * ch.normal.y
      + g.bitangent * ch.normal.z * u.p.normal_strength
  );
  s.normal = mapped;
  s.geo_normal = g.geo_n;
  s.ao = ch.ao;
  s.height = ch.height;
  s.opacity = 1.0;
  s.roughness = mix(0.99, 0.88, smoothstep(0.42, 0.86, ch.height));
  s.sheen = mix(vec3f(0.055, 0.10, 0.018), vec3f(0.29, 0.41, 0.085), smoothstep(0.36, 0.84, ch.height));
  s.subsurface = mix(vec3f(0.018, 0.043, 0.010), vec3f(0.070, 0.115, 0.026), smoothstep(0.28, 0.82, ch.height));
  return s;
}
