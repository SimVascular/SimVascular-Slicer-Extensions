"""Reading a surface a clip produced, as a surface the remesher can represent.

`DynamicMesh` is a manifold triangle mesh, and the remesher's split, collapse and flip are only
defined on one. What comes out of a clip is not always that. VMTK's `vtkvmtkSimpleCapPolyData`
-- the capper Slicer's **Clip Vessel** offers -- emits each cap as a single `VTK_POLYGON`;
`vtkCleanPolyData` turns a fold into a `VTK_LINE` and leaves it sitting in the surface; a fan
triangulation over a boundary loop that revisits a point emits a triangle with two corners on
the same point id. Every one of those used to stop `remesh_labelled_surface` with a traceback
out of `DynamicMesh.from_arrays`, which told an operator holding a freshly clipped model
nothing they could act on.

So the surface is read into shape here instead, before the mesh is built, and every cell
converted or removed is counted while it happens. Polygons and strips are fanned into
triangles and keep their labels; the folds, duplicates and stray line cells are dropped.

**Removing them is not the repair this package refuses to do.** A cell that repeats a point id
is not a triangle in need of fixing. `[a, b, a]` spans no area, and its three edges are
`(a, b)`, `(b, a)` and `(a, a)` -- so what it contributes to the surface is a *second copy* of
edge `(a, b)` plus a self-loop. On an otherwise manifold surface, the two genuine triangles on
`(a, b)` plus these two make four: the cell does not merely fail to be a triangle, it is what
makes that edge non-manifold. Dropping it restores the neighbourhood that was there before
whatever wrote it, and moves no vertex. The same argument holds for the second copy of a
triangle already present, wound either way. Nothing in this module has a tolerance, nothing in
it moves a point, and nothing in it joins two points that arrived apart.

**And it has to happen over the arrays rather than in VTK.** The advice an operator gets for a
surface like this is to run Clean first, and on a *labelled* surface that advice silently
scrambles the labels. `vtkCleanPolyData` converts a degenerate triangle to a line cell by
default -- `ConvertPolysToLines` is on -- and a `vtkPolyData` orders its cells verts, lines,
polys, strips, so that line lands *before* every triangle. `ModelFaceID` is still one tuple per
cell, still passes every check this package makes, and now names the wrong triangle. Measured
on three triangles labelled 11, 22, 33 with the middle one folded, Clean returns `[22, 11,
33]`: two of the three labels moved. Subsetting the arrays here cannot do that, because a
triangle and its label are carried by the same index.

What is left over after this is a genuine non-manifold junction -- an edge where three sheets of
surface meet -- and that is refused, because no *reading* of the surface resolves it. Which
means a `NonManifoldMeshError` out of `DynamicMesh` now says what it should always have said:
the remesher has a bug, not the input was dirty.

`remesh_patch_interior` deliberately does not come through here. Its patch is a face of a
surface this package merged itself rather than anything a host clipped, and it carries
`GlobalNodeID` point data that is indexed by point id -- so a pass that rebuilt its polydata
would drop the array the rebase is keyed on. Conditioning belongs on the entry point that takes
an arbitrary surface from a host.
"""

from __future__ import annotations

import numpy as np
import vtk
from vtk.util import numpy_support

from . import surfaces


def condition_surface(surface):
    """`(surface, record)`: the same surface with the cells that are not triangles removed.

    `ModelFaceID` is named here rather than passed in, because carrying a label array across a
    cell subset is only correct if the same array is the one read, resliced and stamped back --
    and `surfaces.stamp_face_ids` writes that name. A parameter would be a name this honoured on
    the way in and quietly renamed on the way out.

    The surface comes back *as the same object* when there was nothing to remove, which is the
    case for everything this package produces itself -- so a clean input is remeshed down
    exactly the path it was before, vertex for vertex, and pays one vectorised audit for it.

    `record` is filled in either way. That is deliberately what this offers instead of a
    strictness flag: every count in it being zero is precisely the condition "the input was
    already a manifold triangle mesh", so a caller that wants this package's older behaviour of
    refusing such an input can test for it and say so in its own words, and one that just wants
    the surface remeshed gets it.
    """
    surface, cells = _readable(surface)
    if surface.GetNumberOfPolys() == 0:
        raise RuntimeError(
            f"None of this surface's {cells['cells_in']} cells is a triangle or a polygon, so "
            "there is no surface here to remesh -- only the points, lines or strips some "
            "earlier filter left behind. Check that the model selected is the clipped surface "
            "rather than a centreline or a contour."
        )

    points = surfaces.surface_points(surface)
    triangles = surfaces.triangle_indices(surface)
    labels = surfaces.face_ids(surface)

    folded = _folded(triangles)
    duplicated = _duplicated(triangles, ~folded)
    kept = ~(folded | duplicated)
    record = {
        **cells,
        "folded_triangles": int(folded.sum()),
        "duplicated_triangles": int(duplicated.sum()),
        "duplicate_points": _duplicate_point_count(points),
        "unused_points": int(len(points) - len(np.unique(triangles[kept]))),
        "triangles_out": int(kept.sum()),
    }
    if not record["triangles_out"]:
        raise RuntimeError(
            f"Every one of this surface's {len(triangles)} triangles either repeats a point id "
            "or repeats another triangle, so none of them spans any area. There is no surface "
            "here to remesh."
        )
    _refuse_junctions(triangles[kept], record)
    if folded.any() or duplicated.any():
        surface = _rebuilt(surface, points, triangles[kept],
                           None if labels is None else labels[kept])
    return surface, record


def describe(record):
    """The one clause a log line needs, or `None` when the input needed no conditioning.

    Here rather than in each frontend because the counts are only meaningful together: a
    surface whose caps arrived as polygons and whose folds arrived from a clip is one surface
    with one history, and an operator reading two separate lines about it has to reassemble that
    themselves. The two halves are kept apart because they are not the same act -- a cap that
    arrived as one polygon is *converted* and keeps its area and its label, and the rest is
    *removed* and had neither. Reporting both as "removed" is how an operator comes to believe
    a cap was thrown away.
    """
    clauses = []
    if record["polygons_fanned"]:
        fanned = _plural(record["polygons_fanned"], "polygon or strip", "polygons or strips")
        clauses.append(f"fanned {fanned} into triangles, each carrying its parent cell's "
                       "own ModelFaceID")
    removed = [_plural(count, singular, plural) for count, singular, plural in (
        (record["stray_cells"], "stray point or line cell", "stray point and line cells"),
        (record["folded_triangles"], "folded triangle", "folded triangles"),
        (record["duplicated_triangles"], "duplicate triangle", "duplicate triangles")) if count]
    if removed:
        clauses.append(f"removed {_and(removed)}, which span no area and are what made "
                       "the edges they sat on non-manifold")
    return "; ".join(clauses) or None


def _plural(count, singular, plural):
    return f"{count} {singular if count == 1 else plural}"


def _and(parts):
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _readable(surface):
    """The surface's own cells only, all of them triangles, and what that took.

    Two steps, and the first has to be explicit rather than delegated to `vtkTriangleFilter`.
    Stray vert and line cells are removed by rebuilding the surface without them, because the
    filter's `PassLines` cannot be relied on to do it: on an input whose polygons need fanning
    it drops them, and on one whose polygons are already triangles it takes a pass-through path
    and leaves them exactly where they were. Measured on VTK 9.6 -- a line cell beside two
    triangles survives `PassLinesOff()`. That is precisely the input `vtkCleanPolyData` hands
    over, so the removal is done here, where it is conditional on nothing.

    Fanning polygons and strips *is* delegated to the filter, because it copies the parent
    cell's data onto every triangle of the fan and so keeps a cap's `ModelFaceID` across the
    whole cap rather than on one triangle of it.

    Both steps line the label array up by slicing it, which they can do because a `vtkPolyData`
    orders its cells verts, lines, polys, strips -- so the surface's own cells are the tail of
    the array and the strays are the head. That ordering is also the whole reason a line cell
    left in a labelled surface is dangerous rather than merely untidy: it shifts every label.
    """
    strays = surface.GetNumberOfVerts() + surface.GetNumberOfLines()
    record = {"cells_in": int(surface.GetNumberOfCells()), "stray_cells": int(strays)}
    if strays:
        surface = _without_stray_cells(surface, strays)
    record["polygons_fanned"] = int((_polygon_sizes(surface) != 3).sum()) \
        + surface.GetNumberOfStrips()
    if record["polygons_fanned"]:
        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputData(surface)
        # Moot after the step above, and set anyway: the two flags are the only thing standing
        # between a stray cell this module failed to see and a `triangle_indices` traceback.
        triangulate.PassVertsOff()
        triangulate.PassLinesOff()
        triangulate.Update()
        surface = triangulate.GetOutput()
    return surface, record


def _without_stray_cells(surface, strays):
    """The surface's polygons and strips, sharing its points, with the labels resliced."""
    kept = vtk.vtkPolyData()
    kept.SetPoints(surface.GetPoints())
    kept.SetPolys(surface.GetPolys())
    kept.SetStrips(surface.GetStrips())
    labels = surface.GetCellData().GetArray("ModelFaceID")
    if labels is not None and labels.GetNumberOfTuples() == surface.GetNumberOfCells():
        surfaces.stamp_face_ids(kept, [int(labels.GetTuple1(index)) for index
                                       in range(strays, surface.GetNumberOfCells())], labels)
    return kept


def _polygon_sizes(surface):
    """Corners per polygon, off the offsets array rather than a call per cell.

    Because this runs over every cell of a ninety-thousand-triangle clip result before
    anything else happens, and the answer for almost all of them is three.
    """
    polygons = surface.GetPolys()
    if not polygons.GetNumberOfCells():
        return np.zeros(0, dtype=np.int64)
    return np.diff(numpy_support.vtk_to_numpy(polygons.GetOffsetsArray()))


def _folded(triangles):
    """Cells that repeat a point id, which is what stops them being triangles."""
    return ((triangles[:, 0] == triangles[:, 1])
            | (triangles[:, 1] == triangles[:, 2])
            | (triangles[:, 2] == triangles[:, 0]))


def _duplicated(triangles, considered):
    """Cells repeating a triangle already in the surface, keeping the first of each set.

    Matched on sorted corners rather than on the cells' own winding, because the form a clip
    leaves is a fin -- the same three vertices wound both ways -- and to the edge either side of
    it that is the same triangle twice.
    """
    duplicated = np.zeros(len(triangles), dtype=bool)
    indices = np.flatnonzero(considered)
    if len(indices) < 2:
        return duplicated
    keys = np.sort(triangles[indices], axis=1)
    # `lexsort` is stable, so the first row of each run of equal keys is the lowest cell id
    # among them, and the one kept is the one the surface listed first.
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    repeats = np.all(keys[order][1:] == keys[order][:-1], axis=1)
    duplicated[indices[order[1:][repeats]]] = True
    return duplicated


def _duplicate_point_count(points):
    """Points sharing a position with another, exactly -- reported, never merged.

    Reported because it is the number that explains the refusal an operator is most likely to
    hit next: two coincident copies of a cap rim are a slit in the surface, which shows up as
    boundary loops the remesh is then blamed for. Never merged because merging them joins two
    sheets of surface, and this module's whole claim is that it moves nothing and joins nothing.
    """
    if not len(points):
        return 0
    return int(len(points) - len(np.unique(points, axis=0)))


def _refuse_junctions(triangles, record):
    """Refuse an edge that still carries more than two triangles once the non-cells are gone."""
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]],
                                    triangles[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    crowded = int((counts > 2).sum())
    if not crowded:
        return
    removed = record["folded_triangles"] + record["duplicated_triangles"]
    raise RuntimeError(
        "This surface has "
        f"{_plural(crowded, 'non-manifold junction', 'non-manifold junctions')} left after "
        f"{_plural(removed, 'folded or duplicate cell was', 'folded and duplicate cells were')}"
        " removed: that many of its edges carry more than two triangles, which is three sheets "
        "of surface meeting along one edge. Collapse and flip are not defined there, and no "
        "reading of the surface settles it -- which two of the three sheets the edge belongs "
        "to is a modelling decision, not a repair. Slicer's Surface Toolbox will show where "
        "they are, and a clip that cut through a branch it was not meant to reach is the usual "
        "cause."
    )


def _rebuilt(surface, points, triangles, labels):
    """The surface again with only the kept triangles, labels carried by the same index."""
    rebuilt = surfaces.polydata_from_arrays(points, triangles)
    if labels is not None:
        surfaces.stamp_face_ids(rebuilt, labels,
                                surface.GetCellData().GetArray("ModelFaceID"))
    return rebuilt
