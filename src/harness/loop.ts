import blitSrc from '../gpu/blit.wgsl'
import { configureCanvas, createGpu, type Gpu } from '../gpu/context.ts'
import { createFullscreenPipeline } from '../gpu/screenPass.ts'
import { ShaderRegistry, type ShaderMessage } from '../gpu/shaders.ts'
import { PassTimer, RollingStats, ScopedTimer, type FrameTiming } from '../gpu/timing.ts'
import { VramTracker, type VramReport, type VramScope } from '../gpu/resources.ts'
import { meshCatalog } from '../mesh/catalog.ts'
import { standardBookmarks, loadUserBookmarks } from '../scene/bookmarks.ts'
import { CameraController, poseMatrices, type CameraPose } from '../scene/camera.ts'
import { BasePass } from '../scene/basePass.ts'
import { FrameGroup } from '../scene/frameUbo.ts'
import { Scatter } from '../scene/scatter.ts'
import { SPECIES } from '../scene/species.ts'
import { createStandBuffer, type Stand } from '../scene/stands.ts'
import { builtinSplines } from '../scene/spline.ts'
import { Terrain } from '../scene/terrain.ts'
import { WIND_DEFAULTS } from '../scene/wind.ts'
import type {
  Experiment,
  ExperimentContext,
  FrameInfo,
  RegistryEntry,
  SceneServices,
  ViewTargets,
} from './registry.ts'
import { paramDefaults, paramsFromQuery, type ParamSchema, type ParamValues } from './params.ts'

export const DEPTH_FORMAT: GPUTextureFormat = 'depth32float'

export interface ViewSlot {
  /** URL param namespace: 'p' solo, 'a'/'b' in compare views. */
  ns: 'p' | 'a' | 'b'
  name: 'solo' | 'A' | 'B'
  entry: RegistryEntry
  experiment: Experiment
  ctx: ExperimentContext
  params: ParamValues<ParamSchema>
  scope: VramScope
  color: GPUTexture
  colorView: GPUTextureView
  depth: GPUTexture
  depthView: GPUTextureView
}

/** Final on-screen composition — solo blit, or the A/B compare modes. */
export interface Compositor {
  composite(enc: GPUCommandEncoder, timer: PassTimer, views: ViewSlot[], canvasView: GPUTextureView): void
  /** Called when view targets are (re)created. */
  targetsChanged(views: ViewSlot[]): void
  rebuild?(): void
  dispose?(): void
}

export interface LabAppOptions {
  canvas: HTMLCanvasElement
  seed: number
  /** The placement setup every view on this page renders. */
  stand: Stand
  experiments: { entry: RegistryEntry; ns: 'p' | 'a' | 'b'; query: URLSearchParams }[]
  onError: (message: string) => void
  onShaderMessage: (msg: ShaderMessage) => void
  onDeviceLost: (reason: string) => void
  /** Fixed timestep (s) — bench/diff/capture determinism. */
  fixedDt?: number
  /** Fixed canvas size (device px) — bench reproducibility. */
  fixedSize?: { width: number; height: number }
  timerRingSize?: number
  initialPose?: CameraPose
}

export class LabApp {
  time = 0
  frameIndex = 0
  paused = false
  /** When set (bench), drives the camera instead of user input. */
  cameraDriver: ((time: number) => CameraPose) | null = null
  /** Bench/HUD hook — GPU timings arrive a few frames late, keyed by index. */
  onAfterFrame: ((gpu: FrameTiming[], cpuFrameMs: number, cpuJsMs: number) => void) | null = null

  readonly stats = new RollingStats()

  private raf = 0
  private lastNow = 0
  private width = 0
  private height = 0
  private resizeObserver: ResizeObserver | null = null
  private disposed = false

  private constructor(
    readonly gpu: Gpu,
    readonly canvas: HTMLCanvasElement,
    private canvasCtx: GPUCanvasContext,
    readonly tracker: VramTracker,
    readonly shaders: ShaderRegistry,
    readonly timer: PassTimer,
    readonly scene: SceneServices,
    private frameGroup: FrameGroup,
    private basePass: BasePass,
    private sceneScope: VramScope,
    readonly camera: CameraController,
    readonly views: ViewSlot[],
    private compositor: Compositor,
    private opts: LabAppOptions,
  ) {}

  static async create(opts: LabAppOptions): Promise<LabApp> {
    const gpu = await createGpu({ onError: opts.onError, onDeviceLost: opts.onDeviceLost })
    const { device } = gpu
    const tracker = new VramTracker(device)
    const shaders = new ShaderRegistry(device, opts.onShaderMessage)
    const timer = tracker.exempt(() => new PassTimer(device, gpu.hasTimestamps, opts.timerRingSize ?? 4))

    const sceneScope = tracker.scope('scene')
    const terrain = Terrain.generate(sceneScope, device.queue)
    const standBuffer = createStandBuffer(sceneScope, device.queue, opts.stand)
    const frameGroup = new FrameGroup(device, sceneScope, terrain, standBuffer)
    const basePass = new BasePass(device, shaders, frameGroup.layout, gpu.format, DEPTH_FORMAT)
    const scatter = new Scatter(terrain, opts.seed, opts.stand)
    const scene: SceneServices = {
      terrain,
      scatter,
      wind: WIND_DEFAULTS,
      species: SPECIES,
      bookmarks: { ...standardBookmarks(terrain), ...loadUserBookmarks() },
      splines: builtinSplines(terrain),
    }

    const camera = new CameraController(opts.initialPose ?? scene.bookmarks['grazing']!)
    camera.attach(opts.canvas)
    const canvasCtx = configureCanvas(gpu, opts.canvas)

    const app = new LabApp(
      gpu, opts.canvas, canvasCtx, tracker, shaders, timer, scene, frameGroup, basePass,
      sceneScope, camera, [], null as unknown as Compositor, opts,
    )

    app.updateSize()
    for (const req of opts.experiments) {
      app.views.push(await app.createView(req.entry, req.ns, req.query))
    }
    app.setCompositor(new BlitCompositor(device, shaders, gpu.format))
    shaders.onReload(() => {
      basePass.rebuild()
      app.compositor.rebuild?.()
    })
    return app
  }

  private async createView(entry: RegistryEntry, ns: 'p' | 'a' | 'b', query: URLSearchParams): Promise<ViewSlot> {
    const manifest = entry.manifest
    if (!manifest) throw new Error(`experiment "${entry.id}": ${entry.error ?? 'no manifest'}`)
    const name = ns === 'p' ? 'solo' : ns === 'a' ? 'A' : 'B'
    const scope = this.tracker.scope(entry.id + (name === 'solo' ? '' : `:${name}`))
    const params = paramsFromQuery(manifest.params, query, ns)
    const { color, colorView, depth, depthView } = this.createTargets(name)
    const app = this
    const ctx: ExperimentContext = {
      id: entry.id,
      device: this.gpu.device,
      colorFormat: this.gpu.format,
      depthFormat: DEPTH_FORMAT,
      scene: this.scene,
      frame: { layout: this.frameGroup.layout, bindGroup: this.frameGroup.bindGroup },
      res: scope,
      shaders: this.shaders,
      timing: new ScopedTimer(this.timer, name === 'solo' ? '' : `${name}/`),
      params,
      seed: this.opts.seed,
      stand: this.opts.stand,
      meshes: meshCatalog,
      size: () => ({ width: app.width, height: app.height }),
    }
    const module = await manifest.load()
    const experiment = await module.create(ctx)
    return { ns, name, entry, experiment, ctx, params, scope, color, colorView, depth, depthView }
  }

  private createTargets(name: string): Pick<ViewSlot, 'color' | 'colorView' | 'depth' | 'depthView'> {
    const usage =
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.COPY_DST
    const color = this.sceneScope.createTexture(
      { label: `target/${name}/color`, size: [this.width, this.height], format: this.gpu.format, usage },
      { tag: 'render-target' },
    )
    const depth = this.sceneScope.createTexture(
      { label: `target/${name}/depth`, size: [this.width, this.height], format: DEPTH_FORMAT, usage },
      { tag: 'render-target' },
    )
    return { color, colorView: color.createView(), depth, depthView: depth.createView() }
  }

  setCompositor(compositor: Compositor): void {
    this.compositor?.dispose?.()
    this.compositor = compositor
    compositor.targetsChanged(this.views)
  }

  getCompositor(): Compositor {
    return this.compositor
  }

  private updateSize(): boolean {
    const fixed = this.opts.fixedSize
    const dpr = Math.min(devicePixelRatio || 1, 2)
    const width = fixed?.width ?? Math.max(1, Math.round(this.canvas.clientWidth * dpr))
    const height = fixed?.height ?? Math.max(1, Math.round(this.canvas.clientHeight * dpr))
    if (width === this.width && height === this.height) return false
    this.width = width
    this.height = height
    this.canvas.width = width
    this.canvas.height = height
    for (const view of this.views) {
      view.color.destroy()
      view.depth.destroy()
      Object.assign(view, this.createTargets(view.name))
      view.experiment.resize?.(width, height)
    }
    this.compositor?.targetsChanged(this.views)
    return true
  }

  start(): void {
    if (!this.opts.fixedSize) {
      this.resizeObserver = new ResizeObserver(() => this.updateSize())
      this.resizeObserver.observe(this.canvas)
    }
    this.lastNow = performance.now()
    const tick = (): void => {
      if (this.disposed) return
      this.raf = requestAnimationFrame(tick)
      this.frame()
    }
    this.raf = requestAnimationFrame(tick)
  }

  stop(): void {
    cancelAnimationFrame(this.raf)
    this.resizeObserver?.disconnect()
    this.resizeObserver = null
  }

  /** One frame. Public so bench/capture can single-step deterministically. */
  frame(): void {
    const now = performance.now()
    const realDtMs = now - this.lastNow
    this.lastNow = now
    const jsStart = performance.now()

    const dt = this.paused ? 0 : (this.opts.fixedDt ?? Math.min(realDtMs / 1000, 0.1))
    if (this.cameraDriver) this.camera.setPose(this.cameraDriver(this.time))
    this.camera.update(dt)

    const { device } = this.gpu
    const pose = { ...this.camera.pose }
    const matrices = poseMatrices(pose, this.width / this.height)
    this.frameGroup.write(device.queue, {
      ...matrices,
      cameraPos: [pose.x, pose.y, pose.z],
      time: this.time,
      dt,
      frameIndex: this.frameIndex,
      wind: this.scene.wind,
      viewport: [this.width, this.height],
    })
    const frame: FrameInfo = {
      frameIndex: this.frameIndex,
      time: this.time,
      dt,
      deterministic: this.opts.fixedDt !== undefined,
      camera: { pose, ...matrices },
      wind: this.scene.wind,
    }

    for (const view of this.views) view.experiment.update?.(frame)

    const enc = device.createCommandEncoder({ label: 'frame' })
    this.timer.beginFrame(this.frameIndex)

    const first = this.views[0]
    if (first) {
      this.basePass.encode(enc, this.timer, this.frameGroup.bindGroup, first.colorView, first.depthView)
      const second = this.views[1]
      if (second) {
        // Bit-identical base inputs for B — the fairness invariant.
        enc.copyTextureToTexture({ texture: first.color }, { texture: second.color }, [this.width, this.height])
        enc.copyTextureToTexture({ texture: first.depth }, { texture: second.depth }, [this.width, this.height])
      }
      for (const view of this.views) {
        const targets: ViewTargets = {
          colorView: view.colorView,
          depthView: view.depthView,
          depthTexture: view.depth,
          width: this.width,
          height: this.height,
          view: view.name,
        }
        view.experiment.encode(enc, frame, targets)
      }
    }

    this.compositor.composite(enc, this.timer, this.views, this.canvasCtx.getCurrentTexture().createView())
    this.timer.endFrame(enc)
    device.queue.submit([enc.finish()])
    this.timer.afterSubmit()

    const cpuJsMs = performance.now() - jsStart
    const timings = this.timer.poll()
    for (const t of timings) {
      for (const pass of t.passes) this.stats.push(`gpu/${pass.label}`, pass.ms)
    }
    if (this.frameIndex > 0) this.stats.push('cpu/frame', realDtMs)
    this.stats.push('cpu/js', cpuJsMs)
    this.onAfterFrame?.(timings, realDtMs, cpuJsMs)

    this.time += dt
    this.frameIndex++
  }

  vramReport(): VramReport {
    return this.tracker.report()
  }

  /** Update one param value, notifying the experiment. */
  setParam(view: ViewSlot, key: string, value: unknown): void {
    ;(view.params as Record<string, unknown>)[key] = value
    view.experiment.onParamsChanged?.(new Set([key]))
  }

  resetParams(view: ViewSlot): void {
    const defaults = paramDefaults(view.entry.manifest!.params)
    Object.assign(view.params, defaults)
    view.experiment.onParamsChanged?.(new Set(Object.keys(defaults)))
  }

  dispose(): void {
    this.disposed = true
    this.stop()
    this.camera.dispose()
    for (const view of this.views) {
      try {
        view.experiment.dispose()
      } catch (err) {
        console.warn(`[dispose] ${view.entry.id}:`, err)
      }
      view.scope.dispose()
    }
    this.compositor?.dispose?.()
    this.sceneScope.dispose()
    this.timer.dispose()
    this.gpu.device.destroy()
  }
}

/** Solo compositor — blit view 0 to the canvas. */
export class BlitCompositor implements Compositor {
  private pipeline!: GPURenderPipeline
  private layout: GPUBindGroupLayout
  private sampler: GPUSampler
  private bindGroup: GPUBindGroup | null = null

  constructor(
    private device: GPUDevice,
    private shaders: ShaderRegistry,
    private format: GPUTextureFormat,
  ) {
    this.layout = device.createBindGroupLayout({
      label: 'blit',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    })
    this.sampler = device.createSampler({ label: 'blit', magFilter: 'linear', minFilter: 'linear' })
    this.rebuild()
  }

  rebuild(): void {
    this.pipeline = createFullscreenPipeline(this.device, this.shaders, blitSrc, {
      label: 'composite/blit',
      format: this.format,
      bindGroupLayouts: [this.layout],
    })
  }

  targetsChanged(views: ViewSlot[]): void {
    const first = views[0]
    this.bindGroup = first
      ? this.device.createBindGroup({
          label: 'blit',
          layout: this.layout,
          entries: [
            { binding: 0, resource: first.colorView },
            { binding: 1, resource: this.sampler },
          ],
        })
      : null
  }

  composite(enc: GPUCommandEncoder, timer: PassTimer, _views: ViewSlot[], canvasView: GPUTextureView): void {
    const pass = timer.renderPass(enc, 'composite', {
      colorAttachments: [{ view: canvasView, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store' }],
    })
    if (this.bindGroup) {
      pass.setPipeline(this.pipeline)
      pass.setBindGroup(0, this.bindGroup)
      pass.draw(3)
    }
    pass.end()
  }
}
