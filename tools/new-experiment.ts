/**
 * Scaffold a new experiment from a `_template` directory.
 *
 *   npm run new -- <slug>                               experiments/renderers/<nnn>-<slug>
 *   npm run new -- <slug> --kind material --group mosses/sphagnum
 *                                                       experiments/materials/mosses/sphagnum/<nnn>-<slug>
 *   npm run new -- <slug> --into experiments/renderers   explicit parent directory
 *
 * This is the ENTIRE registration process — discovery globs
 * `experiments/**\/manifest.ts` and takes the id from the manifest, so no
 * shared file is touched and parallel agents never conflict.
 *
 * The number is allocated from a scan of the WHOLE tree, not one directory
 * level: ids key ratings, bench results and goldens, and a collision silently
 * mixes two experiments' history (registry.ts turns it into a broken card).
 */
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs'
import path from 'node:path'

const root = path.resolve(import.meta.dirname, '..')
const experimentsDir = path.join(root, 'experiments')

const KINDS = ['renderer', 'material'] as const
type Kind = (typeof KINDS)[number]
/** Where each kind lives, relative to experiments/. */
const KIND_ROOT: Record<Kind, string> = { renderer: 'renderers', material: 'materials' }

const usage = [
  'usage: npm run new -- <slug> [--kind renderer|material] [--group <a/b>] [--into <dir>]',
  '  <slug>    lowercase letters, digits, dashes',
  '  --kind    renderer (default) -> experiments/renderers/',
  '            material           -> experiments/materials/<group>/  (--group required)',
  '  --group   taxonomy path under the kind root, e.g. mosses/sphagnum',
  '  --into    repo-relative parent directory, overriding --kind/--group entirely',
].join('\n')

function fail(message: string): never {
  console.error(message)
  console.error(usage)
  process.exit(1)
}

// --- arguments -------------------------------------------------------------

const argv = process.argv.slice(2)
let slug: string | undefined
let kind: Kind = 'renderer'
let group = ''
let into: string | undefined

for (let i = 0; i < argv.length; i++) {
  const arg = argv[i]!
  if (arg === '--kind' || arg === '--group' || arg === '--into') {
    const value = argv[++i]
    if (value === undefined) fail(`${arg} needs a value`)
    if (arg === '--kind') {
      if (!(KINDS as readonly string[]).includes(value)) fail(`unknown kind "${value}" (${KINDS.join(', ')})`)
      kind = value as Kind
    } else if (arg === '--group') group = value.replace(/^\/+|\/+$/g, '')
    else into = value.replace(/^\/+|\/+$/g, '')
  } else if (arg.startsWith('-')) {
    fail(`unknown option "${arg}"`)
  } else if (slug === undefined) {
    slug = arg
  } else {
    fail(`unexpected argument "${arg}"`)
  }
}

if (!slug || !/^[a-z0-9][a-z0-9-]*$/.test(slug)) fail('a slug is required (lowercase letters, digits, dashes)')

const segments = (value: string): string[] => value.split('/').filter((s) => s.length > 0)
for (const part of [...segments(group), ...segments(into ?? '')]) {
  if (!/^[a-zA-Z0-9._-]+$/.test(part) || part === '..') fail(`invalid path segment "${part}"`)
}

if (into === undefined) {
  if (kind === 'material' && group === '') {
    fail('a material needs --group <class>/<subject>, e.g. --group mosses/sphagnum')
  }
  into = ['experiments', KIND_ROOT[kind], ...segments(group)].join('/')
} else if (segments(into)[0] !== 'experiments') {
  fail(`--into must be a path under experiments/, got "${into}"`)
}

// --- numbering: per BRANCH, with a whole-tree uniqueness check --------------

/** Directory names anywhere under `dir`, recursively. */
function namesInTree(dir: string): string[] {
  const out: string[] = []
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (!entry.isDirectory() || entry.name.startsWith('.')) continue
    out.push(entry.name)
    out.push(...namesInTree(path.join(dir, entry.name)))
  }
  return out
}

const numberOf = (name: string): number | undefined => {
  const n = /^(\d{3})-/.exec(name)?.[1]
  return n === undefined ? undefined : Number(n)
}

// The sequence restarts in each branch, so a subject reads 001/002/003 rather
// than inheriting the whole repo's count — the point of having a tree at all.
// `_template` is skipped by discovery but NOT by numbering.
const branchNumbers = existsSync(path.join(root, into))
  ? namesInTree(path.join(root, into))
      .map(numberOf)
      .filter((n): n is number => n !== undefined)
  : []
let seq = branchNumbers.length > 0 ? Math.max(...branchNumbers) + 1 : 1

// Ids stay unique across the WHOLE tree — they key ratings, bench results and
// goldens, and the registry surfaces a collision as a broken card. Per-branch
// numbering only collides when the number AND the slug match, so bump past it.
const taken = new Set(namesInTree(experimentsDir))
while (taken.has(`${String(seq).padStart(3, '0')}-${slug}`)) seq++

const id = `${String(seq).padStart(3, '0')}-${slug}`
const dir = path.join(root, into, id)

if (existsSync(dir)) fail(`${path.relative(root, dir)} already exists`)

// --- template: the kind's own if it has one, else the renderer template -----

const templates = [path.join(root, into, '_template'), path.join(experimentsDir, KIND_ROOT[kind], '_template'), path.join(experimentsDir, 'renderers', '_template')]
const template = templates.find((t) => existsSync(t))
if (template === undefined) fail(`no _template found (looked in ${templates.map((t) => path.relative(root, t)).join(', ')})`)

mkdirSync(path.dirname(dir), { recursive: true })
cpSync(template, dir, { recursive: true })

const title = slug.replace(/-/g, ' ')
for (const file of ['manifest.ts', 'NOTES.md']) {
  const p = path.join(dir, file)
  if (!existsSync(p)) continue
  writeFileSync(p, readFileSync(p, 'utf8').replaceAll('__SLUG__', id).replaceAll('__NAME__', title))
}

const rel = path.relative(root, dir)
console.log(`created ${rel}/  (from ${path.relative(root, template)}/)`)
console.log(`run it:  http://localhost:5175/#/run/${id}`)
console.log(`rules:   see CLAUDE.md — edit ONLY files inside ${rel}/`)
if (kind === 'material' && !template.includes(path.join('materials', '_template'))) {
  console.log('note:    that is the RENDERER template — switch manifest.ts to defineMaterial() (see CLAUDE.md "Material rules")')
}
