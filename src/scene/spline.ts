import { lerp, lerpAngle } from '../util/math.ts'
import type { CameraPose } from './camera.ts'
import type { Terrain } from './terrain.ts'

/**
 * Catmull-Rom camera path, arc-length reparametrized so bench flythroughs
 * move at constant speed. Yaw/pitch use shortest-arc interpolation.
 */
export class CameraSpline {
  private cumulative: number[] = []
  private total = 0

  constructor(
    private poses: CameraPose[],
    private loop = false,
  ) {
    if (poses.length < 2) throw new Error('CameraSpline needs at least 2 poses')
    // Arc-length table from dense position sampling.
    const STEPS = 512
    let prev = this.raw(0)
    this.cumulative.push(0)
    for (let i = 1; i <= STEPS; i++) {
      const cur = this.raw(i / STEPS)
      this.total += Math.hypot(cur.x - prev.x, cur.y - prev.y, cur.z - prev.z)
      this.cumulative.push(this.total)
      prev = cur
    }
  }

  /** Pose at normalized arc length t in [0, 1]. */
  sample(t01: number): CameraPose {
    const t = Math.min(1, Math.max(0, t01))
    const target = t * this.total
    // Binary search the cumulative table, then linear within the step.
    let lo = 0
    let hi = this.cumulative.length - 1
    while (lo + 1 < hi) {
      const mid = (lo + hi) >> 1
      if (this.cumulative[mid]! <= target) lo = mid
      else hi = mid
    }
    const seg = this.cumulative[hi]! - this.cumulative[lo]!
    const f = seg > 0 ? (target - this.cumulative[lo]!) / seg : 0
    return this.raw((lo + f) / (this.cumulative.length - 1))
  }

  /** Pose at raw parameter u in [0, 1] (not arc-length corrected). */
  private raw(u: number): CameraPose {
    const n = this.poses.length
    const segments = this.loop ? n : n - 1
    const s = Math.min(u * segments, segments - 1e-6)
    const i = Math.floor(s)
    const f = s - i
    const at = (k: number): CameraPose =>
      this.loop ? this.poses[((k % n) + n) % n]! : this.poses[Math.min(n - 1, Math.max(0, k))]!
    const [p0, p1, p2, p3] = [at(i - 1), at(i), at(i + 1), at(i + 2)]
    const cr = (a: number, b: number, c: number, d: number): number => {
      const t = f
      return 0.5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * t * t + (3 * b - a - 3 * c + d) * t * t * t)
    }
    return {
      x: cr(p0.x, p1.x, p2.x, p3.x),
      y: cr(p0.y, p1.y, p2.y, p3.y),
      z: cr(p0.z, p1.z, p2.z, p3.z),
      yaw: lerpAngle(p1.yaw, p2.yaw, f),
      pitch: lerp(p1.pitch, p2.pitch, f),
      fov: lerp(p1.fov, p2.fov, f),
    }
  }
}

/** Built-in bench splines — every experiment is measured on the same paths. */
export function builtinSplines(terrain: Terrain): Record<string, CameraSpline> {
  const h = (x: number, z: number): number => terrain.height(x, z)
  const orbitLow: CameraPose[] = []
  const R = 18
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * Math.PI * 2
    const x = Math.sin(a) * R
    const z = Math.cos(a) * R
    // Face the center: yaw such that forward points to origin.
    orbitLow.push({ x, y: h(x, z) + 1.3, z, yaw: Math.atan2(-x, z), pitch: -0.12, fov: 60 })
  }
  const flythrough: CameraPose[] = []
  for (let z = 40; z >= -40; z -= 10) {
    // Moving toward -z, facing -z (yaw 0).
    flythrough.push({ x: 5, y: h(5, z) + 1.1, z, yaw: 0, pitch: -0.08, fov: 60 })
  }
  const QUARTER = -Math.PI / 4 // at (+x, +z), faces the origin
  const pullBack: CameraPose[] = [
    { x: 1.6, y: h(1.6, 1.6) + 0.5, z: 1.6, yaw: QUARTER, pitch: -0.05, fov: 60 },
    { x: 6, y: h(6, 6) + 1.6, z: 6, yaw: QUARTER, pitch: -0.18, fov: 60 },
    { x: 16, y: h(16, 16) + 6, z: 16, yaw: QUARTER, pitch: -0.35, fov: 60 },
    { x: 34, y: h(34, 34) + 16, z: 34, yaw: QUARTER, pitch: -0.5, fov: 60 },
  ]
  return {
    'orbit-low': new CameraSpline(orbitLow, true),
    flythrough: new CameraSpline(flythrough, false),
    'pull-back': new CameraSpline(pullBack, false),
  }
}
