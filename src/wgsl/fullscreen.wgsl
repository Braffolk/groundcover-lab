// Fullscreen triangle vertex stage shared by sky/composite/diff passes.
// Emits z = 1 so it can double as a far-plane sky with depthCompare
// 'less-equal' (depth ignored entirely when no depth attachment is bound).

struct FullscreenOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> FullscreenOut {
  let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  var out: FullscreenOut;
  out.pos = vec4f(xy * 2.0 - 1.0, 1.0, 1.0);
  out.uv = vec2f(xy.x, 1.0 - xy.y);
  return out;
}
