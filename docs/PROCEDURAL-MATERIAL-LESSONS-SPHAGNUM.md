# Building complex procedural materials that remain convincing at every scale

## Why this document exists

The Sphagnum material did not become convincing through incremental polishing of
the first plausible generator. More than twenty approaches failed before
`025-statistical-assemblage` found a representation that the project owner judged
worth preserving and refining. The useful result is therefore not only the
material. It is the change in reasoning that produced it.

The last explicitly approved pre-march checkpoint is Git commit `4f5b460`
(`refine separated packed sphagnum cushions`). Earlier recovery points are:

- `8974915`: first statistical assemblage;
- `54fb579`: layered Sphagnum canopy, before cushion work;
- `4a4b5a4`: coherent cushions;
- `4f5b460`: separated, tightly packed cushions.

This report describes the successful representation at `4f5b460`. Subsequent
height-continuity and POM work should be evaluated as changes on top of that
known-good state, not confused with the breakthrough that made the material read
as Sphagnum in the first place.

## The actual target

The target was not “green moss-like noise,” a collection of recognizable plant
icons, or a good-looking height map in isolation. The acceptance criterion was a
material whose final render could plausibly be mistaken for photographed
Sphagnum, at both carpet and close-inspection scales.

That requires several truths to hold at once:

1. Individual shoots and capitula must remain detectable.
2. Individuals must vary, but still belong to one botanical family.
3. A shoot is not flat: its crown, radiating branches, lower fascicles, and
   pendent structure occupy different heights.
4. A carpet is not merely a random collection of isolated heads above a lower
   layer. It is interpenetrating biomass with several continuously populated
   depth cohorts.
5. At larger scales, plants organize into dense cushions with coherent changes
   in height, packing, posture, vitality, and colour.
6. Height, albedo, normal, AO, and microshadow must describe the same structure.
7. The result must tile, must survive POM, and must work at both `tileScale=1`
   and `tileScale=8`; the latter must be inspected at 2048 px and
   `reliefGain=2` rather than hidden by low resolution or weak relief.

Any construction that satisfies only one scale is the wrong construction. A
perfectly drawn capitulum stamp fails when repeated as wallpaper. A convincing
macro colour field fails when its close view is Perlin noise or confetti.

## What repeatedly failed

The rejected materials were visually different, but their underlying mistakes
were remarkably consistent.

### Noise was mistaken for structure

Several attempts placed arbitrary dots, capsules, cuts, spikes, or fragments on
top of low-frequency Perlin/Worley-like fields. The result could have spectral
variation and still had no biological grammar. It read as confetti, mould,
snake skin, extruded circles, or a generic green blob because neighboring marks
had no reason to be related.

Noise is useful for modulating a model. It is not a substitute for the model.
The successful material still uses stochastic fields, but every field has a
semantic job: population intensity, colony state, shoot posture, tissue age,
pigmentation, or elevation. None is added merely because the image “needs more
detail.”

### A repeated symbol was mistaken for a population

Other attempts designed one recognizable node or radial head and stamped it
across the tile. Even with random rotation, size, and height, the eye found the
template immediately. Improving the single symbol made the wallpaper more
obvious rather than more natural.

The opposite failure also occurred: abandoning individuals entirely and using a
continuous field. That sometimes produced pleasant macro variation but nothing
that could be identified as a Sphagnum plant.

The successful representation occupies the middle: each event is a varied,
coherent plant-scale assemblage, while the population process and cushion field
control how thousands of those events interact.

### “Heads plus a floor” was mistaken for a canopy

A two-level construction—random plant heads over a generic below layer—cannot
represent Sphagnum. Openings between top crowns exposed either flat substrate or
zero-height holes. Under POM those holes became implausibly deep black shafts.
Reducing relief strength would have hidden the symptom by making the whole
material less expressive.

The correct fix was structural. Successive spreading and pendent fascicles were
placed at continuously descending heights. A gap in a top crown therefore
reveals older, darker, lower plant tissue rather than an unrelated floor. Four
depth-ranked surfaces are retained per texel, and the accepted checkpoint has
near-complete occupancy through all four cohorts.

### Channels were authored independently

Combining one noise for height, another for colour, and a third for AO can make
each channel look busy while making the material incoherent. A red patch that
does not correspond to tissue age or colony state, or AO unrelated to actual
apertures and overlaps, is noticed even when the viewer cannot name the error.

In 025, structural events carry tissue and pigment attributes into the raster.
The top depth-ranked surface supplies height and material state. AO is inferred
from local valleys, depth-layer congestion, and the distance between occupied
cohorts. Normals are derived from the exact height sampled by POM. One causal
structure therefore explains every channel.

### Reference images were used as appearance, not evidence

Reference-fitted synthesis briefly produced attractive colour and variation,
but risked copying the state, blur, lighting, and resolution of one photograph.
It did not expose controls for the biological state and could not be trusted to
generalize.

References should answer questions—scale, morphology, packing, height profile,
colour covariance—not become an unexplained source texture. The images under
`reference/mosses/sphagnum/` remained the visual acceptance set, while the final
runtime descriptor was generated from deterministic structure rather than
sampled photography.

### Refinement attacked symptoms instead of causes

Common symptom-level responses included smoothing the whole height map, reducing
relief, filling holes with a constant base, adding high-frequency detail, or
perfecting an individual stamp. Each could make one screenshot calmer while
destroying depth, scale behavior, or generalization.

The successful iterations asked what physical structure was missing. Black POM
holes called for lower real biomass. Repetition called for a marked population
with correlated variation. Flat crowns called for a nonlinear crown profile.
Absent macro grouping called for a cushion process. This rule—repair the causal
model—was the most important change in the work.

## The representation that finally worked

### 1. Start from a botanical hierarchy

The conceptual hierarchy is:

```text
carpet
  -> overlapping cushions / colony states
    -> population of neighboring shoots and exposed fascicles
      -> capitulum and successive lower fascicle tiers
        -> spreading and pendent branch packets
          -> overlapping hooded leaves and central tissue
```

This is not literal mesh modeling. It is the hierarchy used to choose compact
2D mathematical primitives and their correlations. The “imagine the carpet
from an ant’s scale” heuristic was useful only to recover the missing levels of
organization; it was not permission to spend the whole budget perfectly
modeling one leaf.

Botanical descriptions supported the key architectural decision: Sphagnum has
successive fascicles with spreading and pendent branches and a populated green
canopy beneath the capitula. That evidence ruled out the heads-plus-floor model
before more shader tuning was wasted on it.

### 2. Use a marked, spatially modulated point process

The 0.42 m periodic tile is divided into candidate cells. A deterministic
acceptance test creates a Cox-like population whose local intensity is modulated
by macro form, elevation, and cushion state. This is materially different from
both a regular grid and independent uniform scatter:

```text
lambda(x) = base density
          + macro-form contribution
          + macro-elevation contribution
          + cushion-state contribution
```

Each accepted event receives a correlated mark vector:

```text
m = (family, size, phase, openness, density, verticality,
     asymmetry, curvature, vitality, pigment, baseline height)
```

The marks are not independent sliders. Dense shoots tend to be less open and
more vertical. Cushion state affects density, size, posture, and baseline
height. Age and elevation influence vitality and pigment. Correlation keeps the
variation biologically coherent; bounded per-event randomness prevents cloned
plants.

The accepted checkpoint authors 7,664 events: 6,890 capitula and 774 exposed
fascicles, approximately 43,447 events/m², with a mean nominal capitulum width
of about 12.08 mm. Those values are useful scale anchors, not universal magic
constants.

### 3. Construct a capitulum from related branch packets

A capitulum is a phyllotactic crown, not a star-shaped signed-distance stamp.
Seven to eleven primary packets receive unequal angular weights, a shared phase,
directional asymmetry, bounded length variation, curvature, width, and vertical
offset. Inner leaves follow a golden-angle sequence with controlled jitter.

The important property is shared identity. Parts of one crown inherit the same
event marks, so they vary together. They are not independent capsules that
happen to meet near a point.

The height profile was refined through direct visual comparison. Early shapes
looked like sea stars with ribs, then like spiders, then like flat lobes. The
accepted direction removed artificial periodic ribs and made the crown
nonlinear: a fuller high centre transitions into branches whose outer portions
descend more quickly. Individual leaf stems remain legible without slicing the
whole plant into repeated ridges.

This was a case where the owner’s language—“fluffier,” “edge bits lower,”
“centre higher,” “ease in, not ease out”—identified the correct geometric
relationship more effectively than generic similarity scores.

### 4. Render the vertical ecology, not a second layer

Every capitulum generates several lower fascicle tiers. Each tier contains two
or three spreading packets and one or two shorter, narrower pendent packets,
with continuously descending heights and age/vitality changes. The deepest
cohort is broader and more collapsed, so it closes the canopy without becoming
a recognizable flat mat.

At every texel, the raster retains the four highest distinct surfaces rather
than only the final maximum. This small order-statistics buffer is the key
bridge between biological layering and an efficient runtime texture:

```text
for each structural splat sample:
    insert (height, surface id, tissue, pigment)
    into the descending top-K list for that texel
```

The top layer supplies the visible descriptor. Lower layers supply overlap and
aperture information and prove that the apparent floor is real interleaved
biomass. At the approved checkpoint, occupied fractions were 100.0%, 99.34%,
97.91%, and 95.17% for layers one through four.

A final contact closure is allowed only across residual gaps narrower than a
branch leaf. It extends the deepest neighboring real packet’s identity,
material state, and depth for a bounded number of texels. It cannot grow into a
hidden substrate. Only 4,640 texels (0.111%) required this closure at the
approved checkpoint, converging within roughly 0.62 mm.

### 5. Add cushions after plant-scale structure works

Macro noise was not allowed to rescue bad individual plants. Cushions were
introduced only after the capitula and layered canopy read correctly.

The cushion model is a periodic collection of irregular, compact-support
elliptical sites. A quartic-like onset keeps neighboring cushions distinct,
while a subdued overlap term avoids hard island boundaries. The normalized
cushion state modulates:

- local event acceptance and packing;
- shoot density and openness;
- size and posture;
- a shared baseline-height offset;
- a much weaker pigment contribution.

Two tuning decisions mattered:

1. Reduce independent shoot-height jitter inside coherent cushions.
2. Increase the shared between-cushion height variation.

This created readable colony-scale cushions without turning each cushion into a
uniform blob. The height correlation with cushion state is approximately 0.54;
the pigment correlation is only about 0.26. The existing light/dark material
mosaic therefore complements the geometry instead of becoming a one-to-one
height visualization.

Cushion edges were made more distinct but remained soft and overlapping, and
each cushion remained tightly packed. Geometry-independent cushion variation is
real, but it is only one contributor; eventual scene geometry must still vary
the material naturally.

### 6. Pack one causal descriptor and derive the rest

The authored 2048² RGBA8 descriptor stores:

```text
R: top-surface tissue/vitality state
G: height
B: structurally inferred aperture/AO
A: pigmentation/age state
```

The live graph derives albedo, height, normals, and AO from that descriptor.
The normal map is derived from the same height used by POM, in the declared
mesh-frame convention. The BRDF adds geometry-aware wrapping, fuzz, AO, and
microshadow; microshadow defaults on because correct dense moss gains much of
its depth cue from self-shadowing.

This costs 22,369,620 bytes (21.33 MiB) including the full mip chain. Runtime is
constant in population size: the thousands of biological events exist only in
the authoring process. The runtime remains one material descriptor and one
harness-owned material pass.

### 7. Vary repeats without breaking periodicity

Small repeat decorrelation is applied through a deterministic per-tile interior
warp. Its envelope is zero, with continuous slope, at every tile boundary. The
tile can therefore vary internally while neighboring periods still meet.

This distinction is reusable: periodic content may be transformed only by a
mapping whose boundary values and derivatives preserve the tile contract.
Random translation or unconstrained domain warp merely exchanges obvious
repetition for visible seams.

## Why this representation succeeded

The decisive improvement was not a clever noise function. It was matching one
mathematical mechanism to each biological scale and making those mechanisms
causally dependent:

| Visual scale | Biological observation | Mathematical representation |
| --- | --- | --- |
| leaf/packet | overlapping hooded leaves along a branch | oriented compact-support elliptical splats along a curved packet |
| shoot | coherent capitulum with lower fascicles | shared marked event, phyllotactic angles, correlated dimensions and heights |
| local canopy | openings contain older/lower biomass | top-four depth order statistics over interleaved tiers |
| cushion | dense neighboring shoots share colony state | irregular compact-support cushion sites modulating the point process |
| carpet | broad but nonidentical ecological regions | low-frequency periodic form, elevation, age, and pigment fields |
| runtime | thousands of structures must be cheap | offline descriptor synthesis plus one live material graph |

The earlier attempts usually skipped one or more rows, then tried to synthesize
the missing scale with noise or post-processing. Because 025 includes every row,
no single level has to impersonate the others.

## The iteration method that avoided another endless failure cycle

### Begin with a written representation hypothesis

Before coding, the author should inspect the complete reference set and write:

1. the recognizable structures at micro, plant, neighborhood, and carpet scale;
2. the approximate physical size and density of each;
3. which variables are correlated across levels;
4. which mathematical primitive will represent every level;
5. what observation would falsify the proposed representation.

If the plan says only “layer several noises,” “scatter plant heads,” or “match
the reference,” it is not yet a representation.

### Use short falsification rounds until the basis is right

The useful cadence was a short one-shot attempt with at most one or two repairs,
followed by rejection if the representation family was visibly wrong. Endless
improvement cycles on mould, confetti, snake skin, or a repeated stamp did not
make those bases valid.

Once a genuinely promising basis appeared, the process changed: preserve it in
Git, critique one scale at a time, and refine without abandoning the successful
general structure. This avoided both premature polishing and the opposite
failure of discarding a good direction at the first defect.

### Judge the representation in channels and final renders

Every meaningful iteration must inspect:

- raw height;
- albedo;
- normal;
- AO;
- final shaded plane at macro and grazing angles;
- final sphere and cube;
- `tileScale=1` and `tileScale=8`;
- 2048 px at `tileScale=8` and `reliefGain=2`;
- microshadow enabled.

Low-resolution 512 px previews can hide the precise repetition, discontinuity,
and ray-march failures that appear immediately at 2048 px. Conversely, a close
view alone can encourage over-modeling one plant and miss the carpet.

The raw map is diagnostic, not the verdict. A plausible height map can still
produce spikes under POM; a visually odd packed descriptor can still decode
correctly. Final renders and channel semantics must agree.

### Make one causal change at a time and checkpoint good states

The productive sequence was:

1. establish varied recognizable capitula;
2. remove artificial ribs while retaining branch identity;
3. correct the nonlinear centre-to-edge height profile;
4. add real descending fascicle cohorts to eliminate black holes;
5. establish dense, coherent canopy overlap;
6. introduce cushion-scale state;
7. shift variance from within-cushion shoot jitter to between-cushion height;
8. strengthen and soften cushion boundaries without loosening packing;
9. only then investigate extreme POM sampling artifacts.

Commits between these phases made experimentation reversible. A speculative
cushion or continuity change could not erase the accepted plant-scale result.

### Preserve expressive height and repair its failure modes directly

When strong relief exposed black shafts, the answer was not to cap
`reliefGain`. When zoomed grazing views exposed discontinuities, the first
question was whether leaf geometry or the ray marcher caused them—not how much
global blur would hide them.

This principle generalizes: never “fix” a material by disabling the condition
under which it is supposed to excel. Preserve the expressive range, then repair
the structure or traversal that breaks inside that range.

## A reusable protocol for future complex materials

### Gate A: reference decomposition

Deliver a one-page observation sheet before implementation:

- reference images at multiple distances and lighting conditions;
- physical scale estimates;
- named structures at four or more scales;
- covariance notes, such as “height and vitality partially correlate, but
  pigment does not exactly trace height”;
- negative constraints: what the material is definitely not.

Fail the gate if the description could apply equally well to mould, coral,
lichen, or carpet pile.

### Gate B: mathematical construction

Specify the generative dependency graph. A useful generic form is:

```text
ecological fields
    -> population intensity and regional state
        -> marked structural events
            -> related substructures and depth cohorts
                -> top-K surface raster / compact descriptor
                    -> all material channels
```

Every random variable needs a semantic name, scale, distribution, and parent.
Independent randomness is the exception, not the default. Every blur or noise
octave needs a stated biological or optical role.

Reject the design before coding if individual identity and macro organization
cannot both emerge from it.

### Gate C: cheap proof of representation

Build the smallest artifact that can falsify the basis:

- a shape atlas showing several individual variants;
- a top-down population tile;
- a height-only grazing render;
- one macro render.

Do not add polish, expensive channel synthesis, or UI until those views read as
the intended subject. Allow one or two structural fixes. If the basis still
looks like a repeated icon or undirected noise, drop it.

### Gate D: multichannel coherence

Derive all channels from shared structure, then explicitly inspect causal
relationships. Use probes and statistics where helpful: layer occupancy,
coverage, event density, physical feature size, gradient quantiles, and
cross-field correlation. Metrics cannot certify realism, but they expose hidden
cheats such as a zero-height floor or pigment copied directly from cushion
height.

### Gate E: adversarial rendering

Use the hardest supported conditions before declaring success:

- close and macro views;
- grazing and top-down cameras;
- simple and sharply joined geometry;
- minimum and maximum intended tile scale;
- strong intended relief;
- full-resolution textures;
- all debug channels;
- microshadow on and off.

The owner’s eyes make the acceptance decision. Automated checks establish
determinism, completeness, channel validity, and absence of GPU errors; they do
not replace visual judgment.

### Gate F: performance architecture

After visual acceptance, perform a structural waste review:

- move population synthesis and stable work to author/init time;
- eliminate eager duplicate texture samples;
- avoid per-frame work scaling with authored plant count;
- reuse staging storage and avoid allocation-heavy per-texel helpers;
- keep runtime descriptors compact;
- benchmark the hardest view, not only a friendly camera;
- verify that an apparent speedup still renders the same material.

Do not sacrifice the accepted relief range merely to make a primitive marcher
cheap. Improve the traversal or provide a bounded quality mode.

## Compact failure checklist

Stop and reconsider the representation if any of these appears:

- random confetti over smooth noise;
- one cloned symbol at many rotations;
- spikes, worms, mould, scales, or carpet pile instead of the target subject;
- a continuous macro field with no identifiable individuals;
- perfect individual icons that collapse into wallpaper;
- a two-level “heads and floor” construction;
- height, colour, and AO that do not describe the same structures;
- empty zero-height holes hidden by weaker relief;
- photographic blur or ecological state baked in from one reference image;
- arbitrary stretched artifacts from domain warps;
- validation only at 512 px or one camera angle;
- endless local tuning after the underlying basis has already failed.

## The central lesson

Convincing procedural material generation is not primarily the accumulation of
noise. It is the compression of a real multiscale system into a small hierarchy
of mathematical relationships.

The successful Sphagnum material works because it knows what exists below a
capitulum, why neighboring shoots form cushions, which properties should vary
together, which should only weakly correlate, and how those structures project
into height and shading. It models enough of each scale to make the next scale
possible, but never spends the runtime budget literally modeling the organism.

That is the reusable standard: **think in biological and visual layers, encode
their dependencies explicitly, and falsify the representation at both micro and
macro scale before polishing it.**
