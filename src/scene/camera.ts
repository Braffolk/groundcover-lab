import { mat4, vec3, type Mat4 } from 'wgpu-matrix'
import { clamp } from '../util/math.ts'

export interface CameraPose {
  x: number
  y: number
  z: number
  /** Radians; 0 faces -Z, positive turns toward +X. */
  yaw: number
  /** Radians; positive looks up. */
  pitch: number
  /** Vertical field of view, degrees. */
  fov: number
}

export interface CameraMatrices {
  view: Mat4
  proj: Mat4
  viewProj: Mat4
  invViewProj: Mat4
}

export function poseForward(pose: CameraPose): [number, number, number] {
  const cp = Math.cos(pose.pitch)
  return [Math.sin(pose.yaw) * cp, Math.sin(pose.pitch), -Math.cos(pose.yaw) * cp]
}

export function poseMatrices(pose: CameraPose, aspect: number): CameraMatrices {
  const eye = [pose.x, pose.y, pose.z]
  const fwd = poseForward(pose)
  const view = mat4.lookAt(eye, vec3.add(eye, fwd), [0, 1, 0])
  const proj = mat4.perspective((pose.fov * Math.PI) / 180, aspect, 0.05, 1500)
  const viewProj = mat4.multiply(proj, view)
  return { view, proj, viewProj, invViewProj: mat4.inverse(viewProj) }
}

export function serializePose(p: CameraPose): string {
  return [p.x.toFixed(2), p.y.toFixed(2), p.z.toFixed(2), p.yaw.toFixed(4), p.pitch.toFixed(4), p.fov.toFixed(1)].join(',')
}

export function parsePose(s: string): CameraPose | null {
  const parts = s.split(',').map(Number)
  if (parts.length !== 6 || parts.some((v) => !Number.isFinite(v))) return null
  const [x, y, z, yaw, pitch, fov] = parts as [number, number, number, number, number, number]
  return { x, y, z, yaw, pitch, fov }
}

const PITCH_LIMIT = Math.PI / 2 - 0.01

/**
 * The single camera controller. One instance drives every view on a page —
 * that is what makes the A/B camera EXACTLY shared: there is one pose and one
 * frame UBO, by construction.
 */
export class CameraController {
  mode: 'orbit' | 'fly' = 'fly'
  pose: CameraPose
  /** Set false while a bench spline drives the camera. */
  inputEnabled = true

  private orbitDistance = 12
  private keys = new Set<string>()
  private dragButton = -1
  private flySpeed = 4
  private changed = true
  private detach: (() => void) | null = null

  constructor(initial: CameraPose) {
    this.pose = { ...initial }
  }

  attach(canvas: HTMLCanvasElement): void {
    const onPointerDown = (ev: PointerEvent): void => {
      if (!this.inputEnabled) return
      this.dragButton = ev.button
      canvas.setPointerCapture(ev.pointerId)
      ev.preventDefault()
    }
    const onPointerUp = (ev: PointerEvent): void => {
      this.dragButton = -1
      canvas.releasePointerCapture(ev.pointerId)
    }
    const onPointerMove = (ev: PointerEvent): void => {
      if (!this.inputEnabled || this.dragButton === -1) return
      if (this.dragButton === 0) {
        const sens = this.mode === 'orbit' ? 0.005 : 0.003
        this.pose.yaw += ev.movementX * sens
        this.pose.pitch = clamp(this.pose.pitch - ev.movementY * sens, -PITCH_LIMIT, PITCH_LIMIT)
        if (this.mode === 'orbit') this.snapToOrbit()
        this.changed = true
      } else {
        // Right/middle drag: pan target (orbit) or slide (fly).
        const right = this.right()
        const scale = (this.mode === 'orbit' ? this.orbitDistance : 10) * 0.0015
        this.pose.x -= (right[0] * ev.movementX - 0) * scale
        this.pose.z -= right[2] * ev.movementX * scale
        this.pose.y += ev.movementY * scale
        this.changed = true
      }
    }
    const onWheel = (ev: WheelEvent): void => {
      if (!this.inputEnabled) return
      ev.preventDefault()
      if (this.mode === 'orbit') {
        const target = this.orbitTarget()
        this.orbitDistance = clamp(this.orbitDistance * Math.exp(ev.deltaY * 0.0012), 0.2, 400)
        this.snapToOrbit(target)
      } else {
        this.flySpeed = clamp(this.flySpeed * Math.exp(-ev.deltaY * 0.001), 0.1, 100)
      }
      this.changed = true
    }
    const onKeyDown = (ev: KeyboardEvent): void => {
      if (!this.inputEnabled) return
      if (ev.target instanceof HTMLInputElement || ev.target instanceof HTMLTextAreaElement) return
      if (ev.code === 'Tab') {
        ev.preventDefault()
        this.mode = this.mode === 'orbit' ? 'fly' : 'orbit'
        if (this.mode === 'orbit') this.orbitDistance = Math.min(this.orbitDistance, 40)
        return
      }
      this.keys.add(ev.code)
    }
    const onKeyUp = (ev: KeyboardEvent): void => {
      this.keys.delete(ev.code)
    }
    const onContextMenu = (ev: Event): void => ev.preventDefault()

    canvas.addEventListener('pointerdown', onPointerDown)
    canvas.addEventListener('pointerup', onPointerUp)
    canvas.addEventListener('pointermove', onPointerMove)
    canvas.addEventListener('wheel', onWheel, { passive: false })
    canvas.addEventListener('contextmenu', onContextMenu)
    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('keyup', onKeyUp)
    this.detach = () => {
      canvas.removeEventListener('pointerdown', onPointerDown)
      canvas.removeEventListener('pointerup', onPointerUp)
      canvas.removeEventListener('pointermove', onPointerMove)
      canvas.removeEventListener('wheel', onWheel)
      canvas.removeEventListener('contextmenu', onContextMenu)
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('keyup', onKeyUp)
    }
  }

  dispose(): void {
    this.detach?.()
    this.detach = null
  }

  /** Advance fly movement; returns true if the pose changed since last call. */
  update(dt: number): boolean {
    if (this.inputEnabled && this.mode === 'fly' && this.keys.size > 0) {
      const fwd = poseForward(this.pose)
      const right = this.right()
      let mx = 0
      let my = 0
      let mz = 0
      const speed = this.flySpeed * (this.keys.has('ShiftLeft') || this.keys.has('ShiftRight') ? 4 : 1) * dt
      const add = (v: readonly number[], s: number): void => {
        mx += v[0]! * s
        my += v[1]! * s
        mz += v[2]! * s
      }
      if (this.keys.has('KeyW')) add(fwd, speed)
      if (this.keys.has('KeyS')) add(fwd, -speed)
      if (this.keys.has('KeyD')) add(right, speed)
      if (this.keys.has('KeyA')) add(right, -speed)
      if (this.keys.has('KeyE')) my += speed
      if (this.keys.has('KeyQ')) my -= speed
      if (mx !== 0 || my !== 0 || mz !== 0) {
        this.pose.x += mx
        this.pose.y += my
        this.pose.z += mz
        this.changed = true
      }
    }
    const was = this.changed
    this.changed = false
    return was
  }

  setPose(pose: CameraPose): void {
    this.pose = { ...pose }
    this.changed = true
  }

  private right(): [number, number, number] {
    return [Math.cos(this.pose.yaw), 0, Math.sin(this.pose.yaw)]
  }

  private orbitTarget(): [number, number, number] {
    const f = poseForward(this.pose)
    return [
      this.pose.x + f[0] * this.orbitDistance,
      this.pose.y + f[1] * this.orbitDistance,
      this.pose.z + f[2] * this.orbitDistance,
    ]
  }

  private snapToOrbit(target = this.orbitTarget()): void {
    const f = poseForward(this.pose)
    this.pose.x = target[0] - f[0] * this.orbitDistance
    this.pose.y = target[1] - f[1] * this.orbitDistance
    this.pose.z = target[2] - f[2] * this.orbitDistance
  }
}
