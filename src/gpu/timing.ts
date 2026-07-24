/**
 * Per-pass GPU timing via timestamp-query, plus rolling statistics.
 *
 * Every render/compute pass an experiment encodes goes through
 * PassTimer.renderPass/computePass, which attaches `timestampWrites`. At the
 * end of the frame the query set is resolved and copied into a ring of
 * readback buffers that are mapped asynchronously — timing NEVER stalls the
 * frame loop; if the ring is full that frame's numbers are simply dropped
 * (sampled), and results arrive ~2-3 frames late keyed by frame index.
 */

const MAX_PASSES_PER_FRAME = 32
const QUERY_COUNT = MAX_PASSES_PER_FRAME * 2

export interface FrameTiming {
  frameIndex: number
  passes: { label: string; ms: number }[]
}

interface Slot {
  buffer: GPUBuffer
  state: 'free' | 'copied' | 'mapping'
  labels: string[]
  frameIndex: number
}

export class PassTimer {
  readonly enabled: boolean
  private querySet: GPUQuerySet | undefined
  private resolveBuffer: GPUBuffer | undefined
  private slots: Slot[] = []
  private used = 0
  private labels: string[] = []
  private frameIndex = 0
  private completed: FrameTiming[] = []

  constructor(device: GPUDevice, enabled: boolean, ringSize = 4) {
    this.enabled = enabled
    if (!enabled) return
    this.querySet = device.createQuerySet({ label: 'pass-timer', type: 'timestamp', count: QUERY_COUNT })
    this.resolveBuffer = device.createBuffer({
      label: 'pass-timer/resolve',
      size: QUERY_COUNT * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    })
    for (let i = 0; i < ringSize; i++) {
      this.slots.push({
        buffer: device.createBuffer({
          label: `pass-timer/readback-${i}`,
          size: QUERY_COUNT * 8,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        }),
        state: 'free',
        labels: [],
        frameIndex: -1,
      })
    }
  }

  beginFrame(frameIndex: number): void {
    this.frameIndex = frameIndex
    this.used = 0
    this.labels = []
  }

  renderPass(enc: GPUCommandEncoder, label: string, desc: GPURenderPassDescriptor): GPURenderPassEncoder {
    return enc.beginRenderPass({ ...desc, label, ...this.claim(label) })
  }

  computePass(enc: GPUCommandEncoder, label: string): GPUComputePassEncoder {
    return enc.beginComputePass({ label, ...this.claim(label) })
  }

  private claim(label: string): { timestampWrites?: GPURenderPassTimestampWrites } {
    if (!this.enabled || this.used >= MAX_PASSES_PER_FRAME) {
      if (this.enabled) console.warn(`[timing] more than ${MAX_PASSES_PER_FRAME} passes in a frame; "${label}" untimed`)
      return {}
    }
    const pair = this.used++
    this.labels.push(label)
    return {
      timestampWrites: {
        querySet: this.querySet!,
        beginningOfPassWriteIndex: pair * 2,
        endOfPassWriteIndex: pair * 2 + 1,
      },
    }
  }

  /** Encode the resolve+copy. Call after all passes, before submit. */
  endFrame(enc: GPUCommandEncoder): void {
    if (!this.enabled || this.used === 0) return
    const slot = this.slots.find((s) => s.state === 'free')
    if (!slot) return // ring full — drop this frame's timing rather than stall
    enc.resolveQuerySet(this.querySet!, 0, this.used * 2, this.resolveBuffer!, 0)
    enc.copyBufferToBuffer(this.resolveBuffer!, 0, slot.buffer, 0, this.used * 2 * 8)
    slot.state = 'copied'
    slot.labels = [...this.labels]
    slot.frameIndex = this.frameIndex
  }

  /** Kick off async readback of any copied slots. Call right after queue.submit(). */
  afterSubmit(): void {
    for (const slot of this.slots) {
      if (slot.state !== 'copied') continue
      slot.state = 'mapping'
      slot.buffer
        .mapAsync(GPUMapMode.READ)
        .then(() => {
          const times = new BigInt64Array(slot.buffer.getMappedRange(0, slot.labels.length * 16))
          const passes = slot.labels.map((label, i) => ({
            label,
            // Timestamps are ns; clamp — some drivers report reordered pairs.
            ms: Math.max(0, Number(times[i * 2 + 1]! - times[i * 2]!) / 1e6),
          }))
          slot.buffer.unmap()
          this.completed.push({ frameIndex: slot.frameIndex, passes })
          slot.state = 'free'
        })
        .catch(() => {
          slot.state = 'free' // device lost / destroyed mid-map
        })
    }
  }

  /** Drain timings completed since the last poll. */
  poll(): FrameTiming[] {
    const out = this.completed
    this.completed = []
    return out
  }

  dispose(): void {
    this.querySet?.destroy()
    this.resolveBuffer?.destroy()
    for (const s of this.slots) s.buffer.destroy()
    this.slots = []
  }
}

/** Pass factory handed to experiments — labels auto-prefixed per view (A/, B/). */
export class ScopedTimer {
  constructor(
    private inner: PassTimer,
    private prefix: string,
  ) {}

  get enabled(): boolean {
    return this.inner.enabled
  }

  renderPass(enc: GPUCommandEncoder, label: string, desc: GPURenderPassDescriptor): GPURenderPassEncoder {
    return this.inner.renderPass(enc, this.prefix + label, desc)
  }

  computePass(enc: GPUCommandEncoder, label: string): GPUComputePassEncoder {
    return this.inner.computePass(enc, this.prefix + label)
  }
}

export interface StatSummary {
  n: number
  mean: number
  p50: number
  p95: number
  p99: number
}

export function summarize(samples: readonly number[]): StatSummary {
  const n = samples.length
  if (n === 0) return { n: 0, mean: 0, p50: 0, p95: 0, p99: 0 }
  const sorted = [...samples].sort((a, b) => a - b)
  const q = (p: number): number => sorted[Math.min(n - 1, Math.floor(p * n))]!
  return {
    n,
    mean: samples.reduce((a, b) => a + b, 0) / n,
    p50: q(0.5),
    p95: q(0.95),
    p99: q(0.99),
  }
}

/** Rolling per-label sample windows feeding the live HUD. */
export class RollingStats {
  private windows = new Map<string, number[]>()

  constructor(private capacity = 300) {}

  push(label: string, ms: number): void {
    let w = this.windows.get(label)
    if (!w) this.windows.set(label, (w = []))
    w.push(ms)
    if (w.length > this.capacity) w.shift()
  }

  labels(): string[] {
    return [...this.windows.keys()]
  }

  stats(label: string): StatSummary | undefined {
    const w = this.windows.get(label)
    return w && w.length > 0 ? summarize(w) : undefined
  }

  clear(): void {
    this.windows.clear()
  }
}
