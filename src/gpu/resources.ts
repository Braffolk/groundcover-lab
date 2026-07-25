/**
 * VRAM accounting. All buffer/texture allocation goes through a VramScope so
 * the README's hard 25MB-per-species budget is a live, per-owner meter instead
 * of a post-hoc audit. In dev, raw `device.create*` calls made behind the
 * tracker's back are called out on the console.
 */

export interface VramEntry {
  kind: 'buffer' | 'texture'
  label: string
  owner: string
  species?: string
  tag?: string
  bytes: number
  alive: boolean
}

export interface VramReport {
  totalBytes: number
  byOwner: Record<string, number>
  bySpecies: Record<string, number>
  entries: VramEntry[]
}

export interface VramAttr {
  /** Species id this allocation counts against (25MB budget each). */
  species?: string
  tag?: string
}

const BYTES_PER_TEXEL: Record<string, number> = {
  r8unorm: 1, r8snorm: 1, r8uint: 1, r8sint: 1,
  rg8unorm: 2, rg8snorm: 2, rg8uint: 2, rg8sint: 2, r16float: 2, r16uint: 2, r16sint: 2, r16unorm: 2,
  rgba8unorm: 4, 'rgba8unorm-srgb': 4, rgba8snorm: 4, rgba8uint: 4, rgba8sint: 4,
  bgra8unorm: 4, 'bgra8unorm-srgb': 4,
  rg16float: 4, rg16uint: 4, rg16sint: 4,
  r32float: 4, r32uint: 4, r32sint: 4,
  rgb10a2unorm: 4, rg11b10ufloat: 4, rgb9e5ufloat: 4,
  rgba16float: 8, rgba16uint: 8, rgba16sint: 8, rg32float: 8, rg32uint: 8, rg32sint: 8,
  rgba32float: 16, rgba32uint: 16, rgba32sint: 16,
  depth16unorm: 2, depth24plus: 4, 'depth24plus-stencil8': 5, depth32float: 4, 'depth32float-stencil8': 5,
  stencil8: 1,
}

/** Bytes per texel for an uncompressed format, or undefined if unknown. */
export function textureFormatBytes(format: GPUTextureFormat): number | undefined {
  return BYTES_PER_TEXEL[format]
}

export function textureBytes(desc: GPUTextureDescriptor): number {
  const perTexel = BYTES_PER_TEXEL[desc.format]
  if (perTexel === undefined) {
    console.warn(`[vram] unknown texel size for format ${desc.format}; assuming 4 bytes`)
  }
  const bpt = perTexel ?? 4
  const size = desc.size as GPUExtent3DDict & readonly number[]
  const w = Array.isArray(size) ? (size[0] ?? 1) : (size.width ?? 1)
  const h = Array.isArray(size) ? (size[1] ?? 1) : (size.height ?? 1)
  const d = Array.isArray(size) ? (size[2] ?? 1) : (size.depthOrArrayLayers ?? 1)
  const samples = desc.sampleCount ?? 1
  let bytes = 0
  let [mw, mh] = [w, h]
  for (let level = 0; level < (desc.mipLevelCount ?? 1); level++) {
    bytes += mw * mh * d * bpt * samples
    mw = Math.max(1, mw >> 1)
    mh = Math.max(1, mh >> 1)
  }
  return bytes
}

export class VramTracker {
  private entries: VramEntry[] = []
  private inTracker = false

  constructor(readonly device: GPUDevice) {
    if (import.meta.env.DEV) this.installUntrackedWarning()
  }

  scope(owner: string): VramScope {
    return new VramScope(this, owner)
  }

  /**
   * Run `fn` with the untracked-allocation warning suppressed — for tiny
   * harness-internal buffers (timers, compositors, readbacks) that would
   * only add noise to the budget meter.
   */
  exempt<T>(fn: () => T): T {
    this.inTracker = true
    try {
      return fn()
    } finally {
      this.inTracker = false
    }
  }

  /** @internal */
  allocBuffer(owner: string, desc: GPUBufferDescriptor, attr?: VramAttr): { resource: GPUBuffer; entry: VramEntry } {
    this.inTracker = true
    try {
      const buffer = this.device.createBuffer(desc)
      const entry = this.register(buffer, 'buffer', desc.label, owner, desc.size, attr)
      return { resource: buffer, entry }
    } finally {
      this.inTracker = false
    }
  }

  /** @internal */
  allocTexture(owner: string, desc: GPUTextureDescriptor, attr?: VramAttr): { resource: GPUTexture; entry: VramEntry } {
    this.inTracker = true
    try {
      const texture = this.device.createTexture(desc)
      const entry = this.register(texture, 'texture', desc.label, owner, textureBytes(desc), attr)
      return { resource: texture, entry }
    } finally {
      this.inTracker = false
    }
  }

  private register(
    resource: GPUBuffer | GPUTexture,
    kind: VramEntry['kind'],
    label: string | undefined,
    owner: string,
    bytes: number,
    attr?: VramAttr,
  ): VramEntry {
    const entry: VramEntry = {
      kind,
      label: label ?? '(unlabeled)',
      owner,
      bytes,
      alive: true,
      ...(attr?.species !== undefined && { species: attr.species }),
      ...(attr?.tag !== undefined && { tag: attr.tag }),
    }
    this.entries.push(entry)
    const originalDestroy = resource.destroy.bind(resource)
    resource.destroy = () => {
      entry.alive = false
      originalDestroy()
    }
    return entry
  }

  report(): VramReport {
    const live = this.entries.filter((e) => e.alive)
    const sum = (group: (e: VramEntry) => string | undefined): Record<string, number> => {
      const out: Record<string, number> = {}
      for (const e of live) {
        const key = group(e)
        if (key !== undefined) out[key] = (out[key] ?? 0) + e.bytes
      }
      return out
    }
    return {
      totalBytes: live.reduce((a, e) => a + e.bytes, 0),
      byOwner: sum((e) => e.owner),
      bySpecies: sum((e) => e.species),
      entries: live,
    }
  }

  private installUntrackedWarning(): void {
    const device = this.device
    const rawBuffer = device.createBuffer.bind(device)
    const rawTexture = device.createTexture.bind(device)
    device.createBuffer = (desc) => {
      if (!this.inTracker) {
        console.warn(`[vram] untracked createBuffer(${desc.label ?? 'unlabeled'}, ${desc.size}B) — allocate via ctx.res so the budget meter stays honest`)
      }
      return rawBuffer(desc)
    }
    device.createTexture = (desc) => {
      if (!this.inTracker) {
        console.warn(`[vram] untracked createTexture(${desc.label ?? 'unlabeled'}) — allocate via ctx.res so the budget meter stays honest`)
      }
      return rawTexture(desc)
    }
  }
}

export class VramScope {
  private resources: (GPUBuffer | GPUTexture)[] = []
  private mine: VramEntry[] = []

  constructor(
    private tracker: VramTracker,
    readonly owner: string,
  ) {}

  /** The device this scope allocates on — so helpers can take just a scope. */
  get device(): GPUDevice {
    return this.tracker.device
  }

  /**
   * Run `fn` with the untracked-allocation warning suppressed. For transient
   * harness-side staging/readback buffers that would only add noise to the
   * budget meter — never for anything an experiment keeps alive.
   */
  exempt<T>(fn: () => T): T {
    return this.tracker.exempt(fn)
  }

  createBuffer(desc: GPUBufferDescriptor, attr?: VramAttr): GPUBuffer {
    const { resource, entry } = this.tracker.allocBuffer(this.owner, desc, attr)
    this.resources.push(resource)
    this.mine.push(entry)
    return resource
  }

  createTexture(desc: GPUTextureDescriptor, attr?: VramAttr): GPUTexture {
    const { resource, entry } = this.tracker.allocTexture(this.owner, desc, attr)
    this.resources.push(resource)
    this.mine.push(entry)
    return resource
  }

  /**
   * Destroys every resource this scope still holds. This IS the sanctioned
   * cleanup path — experiments are not required to destroy their own
   * allocations on dispose.
   */
  dispose(): void {
    for (const r of this.resources) r.destroy()
    this.resources = []
    this.mine = []
  }
}

export const SPECIES_BUDGET_BYTES = 25 * 1024 * 1024
