# Instance Segmentation: Legacy vs. Graph Pipeline

This document describes the two instance-segmentation pipelines in `magneton/instance_segmentation/`. Both turn a voxel-affinity volume (output of the PyTC UNet) into a volume where every neuron has its own integer ID.

- **Legacy pipeline** (overlap-vote stitching): `legacy_main.py`, menu option 6.
- **Graph pipeline** (non-overlapping cores + a global region graph): `main.py`, menu option 3. It replaced the legacy pipeline on 2026-09-17.

Both pipelines have to work around the same constraint. The volume is too large to segment in one piece, so it is cut into blocks that run as a SLURM array, and the per-block results then have to be combined into one consistent labeling. They differ in **how cross-block connectivity is decided**:

| | Legacy | Graph |
|---|---|---|
| Question asked at a block border | "Did two independent labelings of the same voxels overlap enough?" | "How strong is the membrane boundary between these two fragments?" |
| Evidence used | Voxel-overlap counts between two finished segmentations | Affinity values on the contact faces (waterz merge score) |
| Where merges are decided | Inside each block (waterz), then again pairwise between neighboring blocks (voting) | Once, globally, over a single region graph |

Parameter values below come from the fib_c neuron configs (`config_fib_c_neuron_midaff_lsd_i300k.yaml` for legacy, `config_fib_c_neuron_graph_v1.yaml` for graph).

---

## Pipeline 1 — Legacy (overlap-vote stitching)

### One-line summary
Cut the volume into **overlapping** blocks, run a **full segmentation independently in each block**, then **stitch** the blocks by voting on how their labels overlap in the shared margin.

### Stages

**0. Block grid** (`utils/block_utils.py::generate_blocks_zyx`)
- The volume is tiled into **overlapping** blocks of 768³ voxels with a step of 512, so neighboring blocks share a 256-voxel slab.
- Voxels in an overlap slab belong to 2 blocks, voxels on an edge line to 4, and voxels on a corner to 8.

**1. Segmentation, per block** (`stages/segmentation_stage.py`, SLURM array; `waterz_block.py::run_waterz_block`)
- Read affinity for the entire 768³ block.
- Boundary map `B = 1 − mean(affinity)`.
- Seeds: threshold the interior (`1 − B > interior_threshold = 0.1`), apply a distance transform, and take local maxima (`min_distance = 3`).
- 3D watershed on `B` from those seeds produces **supervoxels** (fragments).
- **waterz agglomeration** merges supervoxels greedily until the merge score (`aff50_his256` = 1 − median boundary affinity) reaches `threshold = 0.45`.
- The result is a **dense, finished segmentation of the whole block**, labeled 1…max_id locally (uint32).
- Each block, **including its overlap margin**, is written as **its own precomputed volume** (`output_local_base_<i>`), plus a JSON metadata file (`coords`, `max_id`).
- In mito or bouton mode, `run_mito_block` / `run_bouton_block` (binary watershed) replaces waterz, and the rest of the pipeline is the same.

**2. Merge-pools: stitch by overlap voting** (`stages/merge_pools.py`)
- **Global ID offsets**: a prefix sum of each block's `max_id` makes the IDs unique (`gid = local_id + offset[block]`).
- For every pair of neighboring blocks (26-neighborhood) the stage reads **both blocks' labels over their shared overlap region** and counts every co-occurring pair `(label_in_A, label_in_B)` voxel by voxel.
- **`select_pairs`** (`utils/relabel_utils.py`) accepts a pair as "the same object" if:
  - overlap count ≥ `min_overlap_vox` (5), **and**
  - `frac_A ≥ 0.5` **OR** `frac_B ≥ 0.5`, where the fractions are the overlap count divided by each label's total voxels in the slab,
  - `min_iou` (0.5) is computed but **never enforced**, the reciprocal-best-match and dominance checks are commented out, and `allow_union_amb: true` keeps **every** candidate, so one label can be unioned with several.
- Accepted pairs are appended to `unions.txt`, and the offsets are written to `global_offsets.json`.

**3. Merge-apply: union-find and write** (`stages/merge_apply.py`)
- All union pairs are folded into **ID pools** (union-find), and each pool maps to its smallest ID.
- For each block: add the block offset, remap through the pool map, **crop to the block's core** (trim overlap/2 from each interior face, snapped to the 128-voxel chunk grid), and write the core into the single global output volume.
- The overlap margin that every block segmented is **thrown away** at this step. It was used only for voting.

### Data flow (for the figure)

```
Affinity volume
   │
   ▼
[Overlapping block grid: 768³ blocks, 256 overlap]
   │
   ├──► Block 0: watershed → waterz agglomerate → labels 1..N0 ──► block_0 volume (full 768³)
   ├──► Block 1: watershed → waterz agglomerate → labels 1..N1 ──► block_1 volume (full 768³)
   ├──► ...                                       (SLURM array, independent)
   └──► Block k: ...                                               ──► block_k volume
                                       │
                                       ▼
          [Merge-pools] for each neighbouring block pair:
             read both labelings in the shared overlap slab
             count (label_A, label_B) voxel co-occurrences
             accept if frac_A ≥ 0.5 OR frac_B ≥ 0.5   → unions.txt
                                       │
                                       ▼
          [Merge-apply] union-find pools → min ID per pool
             per block: offset + remap + crop to core → global volume
                                       │
                                       ▼
                         Final instance segmentation
```

### Key visual concepts for the figure
- **Overlapping tiles**: blocks drawn as squares that overlap, with the shared slab shaded.
- **Duplicate labels in the overlap**: the same voxels colored two different ways (block A's labels and block B's labels) side by side.
- **Voting**: a table or matrix of overlap counts between A's labels and B's labels, with arrows for accepted unions.
- **Transitive union-find**: a chain of unions (A1–B3–C2–…) collapsing into a single ID.
- **Core crop**: each block's inner core kept and its outer margin discarded.

### Weaknesses (why it was replaced)
1. **Merges are decided by agreement between guesses, not by image evidence.** Two blocks segment the same voxels independently, and the stitcher checks only whether their labels overlap. It never looks at the affinity (membrane strength) between the objects.
2. **Permissive acceptance rule.** The "either fraction ≥ 0.5" rule lets a small fragment that pokes into a neighbor's margin union two large objects. `min_iou` is not enforced, and ambiguous multi-matches are all kept.
3. **Errors propagate transitively.** If one block's waterz merges two cells near its edge while the neighbor keeps them separate, union-find carries the merge across the whole volume.
4. **Membrane voxels are claimed by everyone, and the vote counts them as evidence.** Neuron mode labels every voxel, including membrane, so neighbouring blocks can hand the same stretch of thick membrane to two different cells. The vote then treats that shared membrane as proof that the two cells are one object. See the mechanism below.
5. **Invisible evidence.** The voxels that caused a merge are in the overlap margin, which is discarded, so the bridge cannot be seen in the final volume.
6. **Redundant thin blocks.** `generate_blocks_zyx` emits extra trailing blocks when the previous block already reaches the volume edge (fib_c z=640 → blocks 24–47 are just z[512,640]). Their cores are empty, but they still vote and create bridges.
7. **Changing the merge threshold means re-running everything**, because the threshold is baked into every block's agglomeration.

### The membrane-overlap merge: how two neurons get joined through a membrane
This is the error that motivated the replacement. It happens through the following steps, and each one suggests a panel for the figure.

1. **Dense labelling splits each membrane between the neurons on either side.** In neuron mode the watershed labels every voxel, membrane included. Membrane voxels therefore get absorbed into whichever neuron touches them, and two touching neurons **meet roughly in the middle of the membrane** between them.
2. **Membranes vary in thickness.** In the EM, membrane can be a thin line in one place and a thick band in another. For a thick membrane, the "middle" where the two neurons meet is not a sharp line: small changes in context move the split point by several voxels.
3. **At a block border, the two blocks split the same membrane differently.** Each block segments the overlap margin on its own, with different context: for one block the region is interior, for the other it is at its edge. Their split points in a thick membrane can disagree. Neuron A's segment in one block and neuron B's segment in the other block can then **both extend into the same stretch of membrane**.
4. **Those shared membrane voxels become an overlap between A and B.** Merge-pools reads both labellings over the margin and counts co-occurring labels. The membrane voxels that both blocks claimed show up as A-in-block-1 overlapping B-in-block-2.
5. **The acceptance rule is too weak to reject it.** A pair is accepted if the overlap is at least 5 voxels and **either** side's overlap fraction is at least 0.5. A small segment that pokes into the neighbour's margin can satisfy this easily. The affinity is never checked, and IoU is not enforced.
6. **Union-find spreads the merge to both whole neurons.** Once the two pieces are unioned, every block-level piece of each neuron joins one pool, and the final volume gets one ID for both cells.
7. **The evidence disappears.** The shared voxels sit in the overlap margin, which merge-apply crops away. In Neuroglancer the membrane between the two neurons looks watertight, yet they share an ID.

**Concrete case (fib_c ID 49363, dendrite and axon merged):**
- Block 17's segment 1688 extended **58 voxels** into block 10's margin.
- **38** of those voxels overlapped block 10's segment 1205. The overlap fraction for 1688 was 38 / 58 = **0.655 ≥ 0.5**, so the pair was unioned, with **IoU = 0.00**.
- All 38 shared voxels were membrane with **affinity exactly 0 in every channel**. Blocks 10/11 had given that membrane sheet to one neuron and blocks 16/17 had given it to the other.
- In short, 38 voxels of pure membrane, with no affinity evidence of a connection at all, were enough to merge two neurons.

**Figure suggestion:** a cross-section of two neurons with a thick membrane between them, shown twice (block 1's labelling, then block 2's labelling), each with a different split point. A third panel overlays them, showing the membrane strip both blocks claimed, labelled "38 voxels, affinity = 0 → overlap vote → union".

---

## Pipeline 2 — Graph (non-overlapping cores + region graph)

### One-line summary
Cut the volume into **non-overlapping cores** (the halo is read-only context), produce **only watershed fragments** per core, score **every fragment–fragment boundary** (including those that cross a core face) with the same waterz merge function, then make **one global merge decision** by thresholding the graph and taking connected components.

### Core design ideas
- **Read ROI ≠ write ROI.** Each block has two regions. The **write ROI** is its core, the only voxels it may write. The **read ROI** is the core plus a 128-voxel halo on every interior face, which it reads for context. Write ROIs tile the volume with no overlap, so every voxel has exactly one owner and two blocks can never disagree about a voxel. Read ROIs overlap freely.
- **Fragment-making and edge-making are separate passes.** Every write ROI's fragments are finalized (pass 1) before any agglomeration runs (pass 2). So when pass 2 reads its read ROI, most of it holds **fragments made by the neighbouring blocks**, and waterz agglomerates across the core face with the neighbours' real fragments as context (see below).
- **Defer all merge decisions.** Per-block work produces fragments and edge scores, not final objects.
- **Cross-block edges are ordinary edges.** A pair of fragments that meets across a core face is scored exactly like a pair inside a core, using the membrane affinity between them.
- **One global decision.** Threshold + connected components over the whole graph. It reads no image data, finishes in seconds, and can be swept.

### Stages (five passes)

**0. Core grid** (`utils/block_utils.py::build_core_grid`)
- Each block is a dict with two boxes: `core` (the **write ROI**) and `read` (the **read ROI** = core grown by the halo, clipped to the volume).
- Non-overlapping cores of 384×512×512 (z×y×x), each with a 128-voxel **halo** on every interior face. The read ROI is up to 640×768×768, similar to the legacy block, so per-task memory is unchanged.
- Core sizes must be multiples of the 128-voxel storage chunk, so parallel writers never share a chunk file.
- The step equals the core size, which makes the redundant trailing block impossible, and `assert_cores_tile` checks that the cores cover the volume exactly once.

**Pass 1 — Fragments** (`stages/fragments_stage.py`, SLURM array; `waterz_block.py::run_watershed_fragments`)
- Read affinity over the **read ROI** (core + halo), so the watershed has context at the core edges.
- Run the same 3D watershed (seeds from `interior_threshold = 0.9`, raised from legacy 0.1 because 0.1 planted seeds inside membranes), with **no agglomeration**.
- **Crop to the write ROI** (the halo's fragments are discarded, because the neighbouring block produces the authoritative ones there) and renumber to 1…n within the core.
- **Globally unique IDs without coordination**: add `bump = core_index × voxels_per_core` (uint64).
- Write into **one shared global fragments volume**. Each core writes only its own region.

**Pass 2 — Edges / region graph** (`stages/edges_stage.py`, SLURM array)
- Hard barrier: pass 2 refuses to start until **every** core has finished pass 1 (`require_all_fragments`), because it needs the neighbours' fragments on disk.
- Read **fragments + affinity over the read ROI** (core + halo). The halo now holds the **neighbours' final fragments** from pass 1, the same IDs that will appear in the final volume, so fragments on both sides of a core face appear in one array.
- Measure the **contact area** (number of shared voxel faces) for every touching fragment pair. This is done before waterz runs, because waterz mutates the array in place.
- Run **waterz** on those fragments up to `score_threshold = 0.45` (`aff50_his256`) and record the **merge history** plus the region-adjacency graph.
- A **MergeTree** (lowest common ancestor over the merge history) gives, for every adjacent pair, the score at which waterz actually joined them. That score is **waterz's verdict**. Pairs waterz refused to merge get score = **∞** (never the raw pre-agglomeration score, which would resurrect rejected merges).
- Keep edges with at least one endpoint owned by this core, and write `edges_<i>.npz` = `(u, v, score, contact)`. Cross-core edges come out of this as ordinary edges.

#### Why fragment-making and edge-making are split: agglomeration gets context from neighbouring blocks
If one block made fragments and agglomerated them in the same step (as legacy does), its halo would hold only **its own provisional guess** of the neighbour's territory. That guess is computed with less context and thrown away afterward. Splitting the work into two passes with a global barrier between them changes what the halo contains:

1. **Pass 1** reads the read ROI, but writes **only the write ROI**. After pass 1, the fragments volume is complete, and every voxel carries the fragment ID produced by the block that owns it.
2. **Pass 2** reads the same read ROI from that completed volume. Everything outside its own core is now **fragments generated by neighbouring blocks**: real, final fragment IDs, not a local re-guess.
3. **A large share of the read ROI belongs to other blocks.** For a fully interior core (384×512×512 core, 640×768×768 read ROI), only about 27% of the voxels waterz agglomerates are the block's own; about **73% are fragments made by neighbours**. On fib_c, where the 640-deep volume allows a z halo on one side only, it is about 67%.
4. **Waterz agglomerates across the core face as if the face weren't there.** A neuron cut by a core boundary shows up as fragments from core A next to fragments from core B, and waterz grows regions through both. The merge score for a cross-face pair therefore reflects the regions on both sides as they form, not just a thin sliver at the block edge.
5. **Both sides score the same cross-face pair, each with different context** (each core sees 128 voxels into the other). Pass 3's `dedupe: max` keeps the more conservative of the two verdicts.
6. **Because the IDs match across blocks, no matching step is needed.** An edge `(u, v)` from core A and the same edge from core B refer to the same two fragments, so the edges combine directly into one global graph. Legacy had to infer correspondence by voting on voxel overlap.

The halo size (128) caps how far into a neighbour this context reaches. The context is also used only to **score** edges: the region labels waterz produces inside pass 2 are discarded, and only the per-edge verdicts are kept.

#### Why the membrane-overlap merge cannot happen here
The graph pipeline removes the legacy failure at its root. It never uses voxel overlap as evidence, so it doesn't matter who owns a membrane voxel.

- **No voxel is labelled twice.** Write ROIs don't overlap, so there is never a second labelling of a membrane strip to disagree with, and no count of shared voxels exists to vote on.
- **Merges are decided by waterz from the affinity on the shared boundary.** Two fragments that meet inside a thick membrane touch along faces whose affinity is near 0. Their merge score (1 − median boundary affinity) is therefore near 1, far above the 0.45 threshold, so waterz refuses the merge and the edge is recorded as ∞.
- **Membrane ownership becomes harmless.** Wherever the watershed splits a thick membrane between two neurons, the split only moves the boundary. It does not create a connection. The 49363 bridge (38 voxels, all affinity 0) would be scored as a strong boundary, not a match.
- **Remaining risk, to be clear about it:** waterz can still merge two cells if the affinity itself has a real gap, for example a pinhole in the predicted membrane. That is an affinity or scoring problem, not a stitching problem. `min_contact` and the ∞-for-refused rule reduce it.

**Pass 3 — Global merge** (`stages/global_merge.py`, single process, seconds)
- Concatenate all edge files and **dedupe** pairs scored by both neighboring cores (`dedupe: max` keeps the worse score, which prefers a split).
- Keep edges with `score ≤ threshold` (0.45) **and** `contact ≥ min_contact` (20). The contact rule guards against pinhole bridges.
- **Connected components** (scipy sparse) produce a **LUT**: fragment ID → segment ID, where the segment ID is the smallest fragment ID in the component. The mapping is idempotent, so it is safe to retry or apply in place.
- Reads **no image data**. A threshold sweep only re-runs passes 3 + 4.

**Pass 4 — Relabel** (`stages/relabel_stage.py`, SLURM array)
- For each core: read the fragments, apply the LUT (fragments not in the LUT keep their own ID), and write to the final uint64 output volume.
- Also write per-core segment voxel counts (`counts_<i>.npz`).

**Pass 5 — Segment properties** (`stages/segment_props_stage.py`, single process)
- Sum the per-core counts into a Neuroglancer `segment_properties` list with size tags: `#tiny` < 1k voxels, `#small` 1k–10k, `#medium` 10k–100k, `#large` ≥ 100k. An optional `#debris` tag marks segments with no confident-interior voxel.
- Without this list, Neuroglancer opens the segmentation layer with nothing visible.

### Data flow (for the figure)

```
Affinity volume
   │
   ▼
[Non-overlapping core grid: 384×512×512 cores + 128 halo (read-only)]
   │
   ▼
PASS 1  Fragments (SLURM array)
   per core: read READ ROI (core+halo) → watershed only (no merging)
             → crop to WRITE ROI (core) → ID += core_index × voxels_per_core
             → write into ONE global fragments volume
   │
   ═══ barrier: every core must finish pass 1 ═══
   ▼
PASS 2  Edges (SLURM array)
   per core: read fragments + affinity over READ ROI
             (~73% of it = fragments made by neighbouring blocks)
             → waterz agglomerates across the core face
             → contact area per touching pair
             → waterz scores each boundary (merge score or ∞ if refused)
             → edges_i.npz: (u, v, score, contact), incl. cross-core edges
   │
   ▼
PASS 3  Global merge (one process, seconds, no image I/O)
   all edges → dedupe (keep worst) → keep score ≤ 0.45 & contact ≥ 20
             → connected components → LUT (fragment → segment)
   │        ▲
   │        └── threshold sweep loops back here cheaply
   ▼
PASS 4  Relabel (SLURM array)
   per core: fragments → LUT → final volume  (+ per-core size counts)
   │
   ▼
PASS 5  Segment properties
   sum counts → Neuroglancer segment list with #tiny/#small/#medium/#large tags
   │
   ▼
Final instance segmentation
```

### Key visual concepts for the figure
- **Tiles that touch but don't overlap**: cores (write ROIs) as abutting squares, with the read ROI as a dashed outline around one core to show it is read but not written.
- **Pass 2 read ROI coloured by owner**: one read ROI whose fragments are coloured by the block that produced them. The centre core is in one colour and the neighbours' fragments fill the surrounding halo (about 73% of the box), with waterz regions growing across the core boundary line.
- **Barrier between passes**: a "wait for all cores" bar between pass 1 and pass 2.
- **Fragments, not objects**: a core filled with many small watershed fragments (over-segmented) rather than finished neurons.
- **Region graph**: nodes = fragments, edges = shared boundaries. Edge weight/color = merge score, with some edges crossing the core boundary line.
- **Global cut**: edges above the threshold drawn dashed or removed, and the remaining connected components colored as final neurons.
- **Cheap sweep loop**: a feedback arrow from the final result back to pass 3 labeled "change threshold → re-run passes 3–4 only".

---

## Side-by-side differences

| Aspect | Legacy (overlap vote) | Graph (region graph) |
|---|---|---|
| Block layout | Overlapping 768³ blocks, 256 overlap | Non-overlapping cores (384×512×512) + 128 read-only halo |
| Read vs. write region | Whole 768³ block is segmented and written, cropped to the core only at merge-apply | Read ROI (core + halo) read, write ROI (core) written, in every pass |
| Voxel ownership | Overlap voxels labeled by 2–8 blocks | Each voxel labeled by exactly one core |
| What the halo contains during agglomeration | The block's own provisional labels | Final fragments made by neighbouring blocks (about 67–73% of the read ROI) |
| Per-block output | Finished segmentation (watershed + waterz) | Watershed fragments only |
| Per-block storage | One precomputed volume per block, margin included | One shared global fragments volume |
| ID uniqueness | Prefix sum of `max_id` after all blocks finish | Arithmetic bump `core_index × voxels_per_core`, with no coordination |
| Cross-block decision | Voxel-overlap counts between two labelings (`frac ≥ 0.5` on either side) | Waterz boundary score between adjacent fragments (same function as within a core) |
| Uses affinity at block borders? | No | Yes |
| Where merges happen | In every block, then pairwise between blocks, then transitively via union-find | Once, globally: threshold + connected components |
| Membrane voxels claimed by two neurons at a block border | Counted as overlap; enough to union two cells (49363: 38 voxels at affinity 0) | Impossible: one owner per voxel, and a membrane contact scores near 1, so the merge is refused |
| Guard against thin bridges | None enforced (`min_iou` computed but ignored) | `min_contact` (contact-face area) + refused merges scored ∞ + `dedupe: max` |
| Changing the merge threshold | Re-run the whole pipeline | Re-run pass 3 (seconds) + pass 4 |
| Redundant trailing blocks | Yes (fib_c: 24 thin z-blocks) | Impossible by construction |
| Final-volume dtype | uint32 | uint64 |
| Neuroglancer segment list | Not produced | Pass 5 (with size tags) |
| Modes supported | Neuron (waterz), mito, bouton | Neuron (waterz fragments) |
| Stage count | 3 (segmentation → merge-pools → merge-apply) | 5 (fragments → edges → global-merge → relabel → segment-props) |

### The core conceptual shift (a caption for the pair of figures)
The legacy pipeline asks each block to **fully segment** its region and then reconciles the independent answers by **voting on overlap**. A merge can therefore come from two guesses agreeing, even across a membrane with zero affinity. The graph pipeline never lets a block decide that two fragments are one cell. Blocks only **measure boundaries**, and **every merge is made once, globally, from the affinity evidence** on the shared boundary. Because fragments are finalized for the whole volume before any agglomeration, each block measures those boundaries while seeing its neighbours' real fragments in its read ROI.
