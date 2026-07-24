import meshShader from './mesh-view.wgsl'
import { createGpu, configureCanvas } from '../../gpu/context.ts'
import { ShaderRegistry } from '../../gpu/shaders.ts'
import { meshCatalog } from '../../mesh/catalog.ts'
import { CameraController, poseMatrices } from '../../scene/camera.ts'
import type { HashState } from '../../url/state.ts'
import { Overlay } from '../../ui/overlay.ts'
import { button, el, topbar, type View } from './shared.ts'

/**
 * #/mesh/<id> — brute-force render of a raw GCMESH1 source mesh. Tooling
 * only (experiments must NOT render real geometry): validates the parser,
 * and is the visual ground truth to eyeball bakes against.
 */
export async function meshView(root: HTMLElement, state: HashState): Promise<View> {
  const id = state.path[1] ?? ''
  const page = el('div', 'page')
  root.appendChild(page)
  const overlay = new Overlay(page)
  let disposed = false
  let raf = 0
  let device: GPUDevice | null = null

  const dispose = (): void => {
    disposed = true
    cancelAnimationFrame(raf)
    device?.destroy()
    overlay.dispose()
    page.remove()
  }

  try {
    const info = meshCatalog.info(id)
    page.appendChild(
      topbar(`mesh: ${id}`, [
        el(
          'span',
          'hint',
          `${(info.vertexCount / 1e6).toFixed(2)}M verts · ${(info.triangleCount / 1e6).toFixed(2)}M tris · ${(info.bytes / 1e6).toFixed(1)}MB`,
        ),
        el('span', 'hint', 'drag=orbit · wheel=zoom · Tab=fly'),
      ]),
    )
    const content = el('div', 'content')
    const viewer = el('div', 'viewer')
    const canvas = el('canvas')
    viewer.appendChild(canvas)
    content.appendChild(viewer)
    page.appendChild(content)

    const loading = el('div', 'hud', `fetching ${(info.bytes / 1e6).toFixed(0)}MB…`)
    viewer.appendChild(loading)

    const [gpu, mesh] = await Promise.all([
      createGpu({
        onError: (m) => overlay.toast(m, 'error'),
        onDeviceLost: (r) => overlay.toast(`device lost: ${r}`, 'error'),
      }),
      meshCatalog.load(id),
    ])
    if (disposed) {
      gpu.device.destroy()
      return { dispose }
    }
    device = gpu.device
    const shaders = new ShaderRegistry(device, (m) => overlay.toast(m.text, m.kind === 'error' ? 'error' : 'info'))
    const canvasCtx = configureCanvas(gpu, canvas)
    loading.textContent = 'uploading to GPU…'

    const vertexBuffer = device.createBuffer({
      label: `mesh/${id}/vertices`,
      size: mesh.vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(vertexBuffer, 0, mesh.vertices)
    const indices = mesh.indices()
    const indexBuffer = device.createBuffer({
      label: `mesh/${id}/indices`,
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(indexBuffer, 0, indices)

    const uniform = device.createBuffer({
      label: `mesh/${id}/uniforms`,
      size: 32 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const bgl = device.createBindGroupLayout({
      label: 'mesh-view',
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
    })
    const bindGroup = device.createBindGroup({
      label: 'mesh-view',
      layout: bgl,
      entries: [{ binding: 0, resource: { buffer: uniform } }],
    })

    let pipeline!: GPURenderPipeline
    const build = (): void => {
      const module = shaders.module(meshShader)
      pipeline = device!.createRenderPipeline({
        label: 'mesh-view',
        layout: device!.createPipelineLayout({ label: 'mesh-view', bindGroupLayouts: [bgl] }),
        vertex: {
          module,
          entryPoint: 'vs_main',
          buffers: [
            {
              arrayStride: 16,
              attributes: [
                { shaderLocation: 0, offset: 0, format: 'uint16x4' },
                { shaderLocation: 1, offset: 8, format: 'uint16x4' },
              ],
            },
          ],
        },
        fragment: { module, entryPoint: 'fs_main', targets: [{ format: gpu.format }] },
        primitive: { topology: 'triangle-list' },
        depthStencil: { format: 'depth32float', depthCompare: 'less', depthWriteEnabled: true },
      })
    }
    build()
    shaders.onReload(build)

    const center = mesh.header.boundsMin.map((v, i) => (v + mesh.header.boundsMax[i]!) / 2) as [number, number, number]
    const camera = new CameraController({ x: center[0] + 0.9, y: center[1] + 0.5, z: center[2] + 0.9, yaw: -2.36, pitch: -0.25, fov: 55 })
    camera.mode = 'orbit'
    camera.attach(canvas)

    let mode = 0
    let repeat = false
    const modeBtn = button('view: color', () => {
      mode = (mode + 1) % 3
      modeBtn.textContent = `view: ${['color', 'normals', 'lit'][mode]}`
    })
    const repeatBtn = button('tile ×1', () => {
      repeat = !repeat
      repeatBtn.textContent = repeat ? 'tile ×9' : 'tile ×1'
    })
    const toolbar = el('div', 'toolbar')
    toolbar.append(modeBtn, repeatBtn)
    viewer.appendChild(toolbar)

    loading.remove()
    const fps = el('div', 'hud')
    viewer.appendChild(fps)

    let depth: GPUTexture | null = null
    let last = performance.now()
    let frameMsAvg = 16
    const uniforms = new Float32Array(32)
    const tick = (): void => {
      if (disposed) return
      raf = requestAnimationFrame(tick)
      const now = performance.now()
      frameMsAvg = frameMsAvg * 0.95 + (now - last) * 0.05
      last = now

      const dpr = Math.min(devicePixelRatio || 1, 2)
      const w = Math.max(1, Math.round(canvas.clientWidth * dpr))
      const h = Math.max(1, Math.round(canvas.clientHeight * dpr))
      if (canvas.width !== w || canvas.height !== h || !depth) {
        canvas.width = w
        canvas.height = h
        depth?.destroy()
        depth = device!.createTexture({
          label: 'mesh-view/depth',
          size: [w, h],
          format: 'depth32float',
          usage: GPUTextureUsage.RENDER_ATTACHMENT,
        })
      }
      camera.update((now - last) / 1000 + 1e-6)
      const { viewProj } = poseMatrices(camera.pose, w / h)
      uniforms.set(viewProj, 0)
      uniforms.set(mesh.header.boundsMin, 16)
      uniforms[19] = mode
      uniforms.set(mesh.header.boundsMax, 20)
      uniforms[23] = repeat ? 1 : 0
      uniforms.set(mesh.header.tileSize, 24)
      device!.queue.writeBuffer(uniform, 0, uniforms)

      const enc = device!.createCommandEncoder()
      const pass = enc.beginRenderPass({
        label: 'mesh-view',
        colorAttachments: [
          { view: canvasCtx.getCurrentTexture().createView(), loadOp: 'clear', clearValue: { r: 0.09, g: 0.1, b: 0.12, a: 1 }, storeOp: 'store' },
        ],
        depthStencilAttachment: {
          view: depth.createView(),
          depthLoadOp: 'clear',
          depthClearValue: 1,
          depthStoreOp: 'store',
        },
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bindGroup)
      pass.setVertexBuffer(0, vertexBuffer)
      pass.setIndexBuffer(indexBuffer, 'uint32')
      pass.drawIndexed(indices.length, repeat ? 9 : 1)
      pass.end()
      device!.queue.submit([enc.finish()])

      fps.textContent = `${frameMsAvg.toFixed(1)} ms cpu-frame · ${((indices.length / 3) * (repeat ? 9 : 1) / 1e6).toFixed(1)}M tris`
    }
    raf = requestAnimationFrame(tick)

    return {
      dispose: () => {
        camera.dispose()
        depth?.destroy()
        dispose()
      },
    }
  } catch (err) {
    overlay.fatal(`Mesh inspector failed for "${id}"`, err instanceof Error ? (err.stack ?? err.message) : String(err))
    return { dispose }
  }
}
