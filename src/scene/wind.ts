/**
 * Global wind parameters + CPU mirror of the WGSL wind model
 * (src/wgsl/wind.wgsl). Everything derives from `time`, so fixed-timestep
 * runs are deterministic.
 */

export interface WindParams {
  /** Unit direction in the xz plane. */
  dir: [number, number]
  /** Base sway amplitude in meters at the tip of a sway=1 plant. */
  strength: number
  gustFreq: number
}

export const WIND_DEFAULTS: WindParams = (() => {
  const len = Math.hypot(0.8, 0.35)
  return { dir: [0.8 / len, 0.35 / len], strength: 0.12, gustFreq: 0.5 }
})()

export function windGust(params: WindParams, time: number): number {
  const t = time * params.gustFreq
  const g = 0.5 + 0.5 * Math.sin(t) * Math.sin(t * 0.731 + 1.3)
  return 0.25 + 0.75 * g * g
}

/** CPU mirror of wind_sway() — same formula, not bit-critical (visual only). */
export function windSway(
  params: WindParams,
  worldPos: [number, number, number],
  time: number,
  sway: number,
  phase: number,
): [number, number, number] {
  const along = worldPos[0] * params.dir[0] + worldPos[2] * params.dir[1]
  const w1 = Math.sin(time * 1.9 - along * 0.35 + phase)
  const w2 = Math.sin(time * 3.7 - along * 1.1 + phase * 1.7) * 0.35
  const amp = params.strength * windGust(params, time) * sway
  const d = (w1 + w2) * amp
  return [params.dir[0] * d, -Math.abs(d) * 0.3, params.dir[1] * d]
}
