# Face Aware Remesh

Remesh a surface model to a target edge length while keeping its `ModelFaceID` face labels, and
choose how the seams between labelled faces are held.

This is the module to reach for after [Paint Model](PaintModel.md) has partitioned a model into
face groups: `ModelFaceID` is the array both modules work in terms of, and it is the labelling
SimVascular's meshing and boundary-condition setup depend on. A remesher that does not
understand it forces you to relabel afterwards by proximity lookup, which smears labels across
face boundaries exactly where precision matters.

## How it works

Remeshing is a Python port of [geometry3Sharp](https://github.com/gradientspace/geometry3Sharp)'s
constrained remesher (gradientspace, Boost licence — see [Licence](#licence)) onto a dynamic mesh
of the package's own. The algorithm is the incremental one: a pass over every edge that splits it
if it is longer than 4/3 of the target, collapses it if it is shorter than 4/5, and otherwise
flips it when that brings the vertex valences closer to six, followed by a Laplacian smoothing
pass with every vertex reprojected onto the surface it started from.

Labels survive because `ModelFaceID` *is* the mesh's triangle grouping rather than something
looked up afterwards: a split inherits its parent triangle's group, and a collapse cannot merge
two groups without crossing a constrained edge. Nothing is relabelled after the fact.

There is no helper binary and no .NET runtime. The library imports numpy and VTK and nothing
else, both of which Slicer ships — see [Using it outside Slicer](#using-it-outside-slicer).

## Reading a clipped surface

The remesher runs on a manifold triangle mesh. What comes out of **Clip Vessel**, VMTK's
cappers or `vtkClipPolyData` is not reliably one, so the input is conditioned first and
everything conditioning did is reported — in the log, and in the panel's result line.

| What arrives | What happens to it |
|---|---|
| A cap emitted as a single `VTK_POLYGON` (`vtkvmtkSimpleCapPolyData` does this) | Fanned into triangles, every one of them carrying that cap's own `ModelFaceID` |
| A triangle strip | Same |
| A cell repeating a point id, `[a, b, a]` | Removed |
| The same three vertices written twice, wound either way | The second copy removed |
| A stray point or line cell, as `vtkCleanPolyData` leaves behind | Removed |
| An edge still carrying three triangles afterwards | **Refused** — see below |
| Two points sharing a position exactly | Counted and reported, never merged |

**Removing a folded cell is not a repair.** `[a, b, a]` spans no area, and its three edges are
`(a, b)`, `(b, a)` and `(a, a)` — so what it contributes to the surface is a *second copy* of
edge `(a, b)`. Together with the two genuine triangles on that edge, that makes four: the cell
is not a broken triangle, it is the thing making the edge non-manifold. Dropping it restores
the neighbourhood that was there before whatever wrote it, and moves no vertex.

**Do not run Clean first to work around a surface like this** — not on a labelled one.
`vtkCleanPolyData` converts a degenerate triangle to a line cell by default
(`ConvertPolysToLines` is on), and a `vtkPolyData` orders its cells verts, lines, polys,
strips, so that line lands *before* every triangle. `ModelFaceID` is still one tuple per cell
and still passes every check, and now names the wrong triangle. On three triangles labelled 11,
22, 33 with the middle one folded, Clean returns `[22, 11, 33]`. Conditioning subsets the arrays
instead, where a triangle and its label are carried by the same index, so it cannot do that.

A **non-manifold junction** — three sheets of surface along one edge — is refused rather than
conditioned away, because which two of the three the edge belongs to is a modelling decision.
Slicer's Surface Toolbox will show you where they are; a clip that cut through a branch it was
not meant to reach is the usual cause.

Coincident points are the one defect reported but not acted on, because merging them joins two
sheets of surface. Two coincident copies of a cap rim are a slit, and a slit shows up later as
a change in the open boundary loop count — so that refusal names the count, to say where to
look.

## Seam modes

A seam is the boundary between two labelled faces.

| Mode | What it does |
|---|---|
| **Slide** | Resamples the seam at the target edge length, but keeps every seam vertex on the *original seam curve*. This constrains the seam's geometry rather than its discretization. |
| **Pin** | Holds the seam vertex for vertex: no split, collapse or flip. Use it when the old discretization has to come back unchanged, for example when the surface will be welded to another mesh whose vertices must coincide. |

Sliding is the default, and the reason is that a pinned seam is where sliver triangles come
from: its vertex placement is inherited from whatever cut produced it and cannot be improved. On
a clinical heart case's merged flow domain at a 0.85 mm target, sliding takes the seam band's
worst aspect ratio from 935 to 8.6 and its 99th percentile from 116 to 2.0, with every face
still present.

Pinning a fine seam next to a coarsened interior is the opposite trap, and it is worse: on a
lone patch it took the seam band's worst aspect ratio from 50.6 to 21728. Pin only when a
caller downstream genuinely needs the vertex list back unchanged.

### Seam corner angle

Projecting seam vertices onto the original curve guarantees they sit *on* it, but nothing stops
a collapse from chording across a bend. The seam corner angle pins the vertices where the seam
turns by more than that angle, so the stretches that slide are the ones where sliding costs
little. Lower it to hold the seam's shape more tightly; raise it to let more of the seam
re-space.

## Settings that matter

**Smoothing speed** defaults to 0.1, which is geometry3Sharp's own default. Raising it welds
pairs of vertices into zero-area triangles wherever the surface has thin features, and it does
not buy any reduction in slivers. Unlike the original, every position change here is gated on
the `1e-12` cross-norm a volume mesher refuses a surface at, so a smoothing pass cannot take a
triangle to exactly zero area — but the gate is a floor, not a licence to raise the speed.

**Passes** repeats the complete remesh rather than adding sweeps to one. A second pass rebuilds
the seam constraints and the projection target on the first result, which is what finishes a
face that a single pass leaves uneven.

**Queued** uses the ported `RemesherPro` sweep, which enqueues the neighbourhood of whatever an
operation touched instead of walking every edge each iteration. It holds up when the input sits
within roughly 1.0–2.0 times its own median edge of the target and degrades outside that band.

## What it does not do

- **It is not a repair.** A surface that crosses itself goes in and comes out crossing itself.
  Conditioning the input, above, removes cells that are not triangles and merges nothing, moves
  nothing and closes nothing.
- **It is not a decimator you can point anywhere.** It drives toward a uniform edge length, so a
  surface whose features are finer than the target loses them.
- **It refuses rather than degrading.** A pass that would leave a degenerate triangle, tear an
  open boundary, or remesh a face away is reported in the log and the input model is left
  untouched. The panel copies the input before it starts, which is what makes that guarantee
  real.
- **Sliver triangles at the seam are reduced, not eliminated.** A triangle whose seam edges
  cannot flip without moving the seam, and whose interior edge cannot collapse without pinching
  it, survives. A few do every run.

## Using it outside Slicer

The remesher is `svremesh`, a package beside the module that imports nothing from Slicer:

```
pip install -e FaceAwareRemesh/
```

```python
import svremesh

surface, record = svremesh.remesh_preserving_faces(polydata, 0.85)
print(record["faces"])            # cells per ModelFaceID after the pass
print(record["conditioning"])     # what the input had to have removed to be read at all
```

There is no flag for turning conditioning off, because `record["conditioning"]` is more use than
one: every count in it being zero is exactly the condition "the input was already a manifold
triangle mesh", so a caller that wants to refuse such an input can test for that and refuse in
its own words.

```python
import svremesh

surface, conditioning = svremesh.condition_surface(polydata)
if any(conditioning[key] for key in ("polygons_fanned", "stray_cells",
                                     "folded_triangles", "duplicated_triangles")):
    raise RuntimeError(svremesh.describe_conditioning(conditioning))
```

`remesh_preserving_faces` is the entry point every frontend should call — going under it to
`remesh_labelled_surface` skips the reporting and the active-scalar assignment, which is how two
hosts start producing different surfaces from one input.

Inside Slicer no install is needed: the module's own directory is already on the Python path.

Its tests run headlessly and need only numpy and VTK:

```
cd FaceAwareRemesh && PYTHONPATH=. python -m unittest discover -s tests
```

## Licence

`svremesh/remesh.py` and `svremesh/dynamic_mesh.py` are ports of
[geometry3Sharp](https://github.com/gradientspace/geometry3Sharp) (Ryan Schmidt / gradientspace)
and are used under the Boost Software License 1.0, whose terms and full text are in
[`FaceAwareRemesh/NOTICE.txt`](../FaceAwareRemesh/NOTICE.txt). The rest of the module is under the
project's own licence in [`LICENSE.txt`](../LICENSE.txt).
