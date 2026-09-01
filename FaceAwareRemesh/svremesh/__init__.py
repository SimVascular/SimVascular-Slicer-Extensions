"""Remesh a `ModelFaceID`-labelled surface to a uniform edge length without losing its labels.

Every clustering remesher, and Slicer's own, treats a surface as one sheet of triangles. A
surface that carries `ModelFaceID` is not one sheet: it is several faces meeting along seams,
and those seams are what a mesher's boundary conditions and everything else downstream select
by. Remesh it without knowing that and the labels come back scrambled, the seams drift off
their curves, or a thin face disappears entirely.

This remeshes the *labelled* surface instead. The labels are the mesh's own triangle groups
rather than something looked up afterwards, so every face comes back; the seams between faces
are constrained to their original curves and resampled along them at the target edge length;
and the corners where three faces meet are pinned.

It is not a repair -- a surface that crosses itself comes out crossing itself -- and it is not
a decimator, since it drives toward a uniform edge length. It refuses rather than degrading: a
pass that would leave a degenerate triangle, tear an open boundary, or remesh a face away is
reported and the input is left alone.

It does *read* a surface that is not yet a manifold triangle mesh, which is what a clip hands
you: `conditioning` removes the cells that are not triangles at all -- a cap emitted as one
polygon, a fold whose two corners share a point id, a triangle written twice -- before the mesh
is built, and reports each of them. That module states why doing so is not the repair the
paragraph above refuses, and why it cannot be delegated to `vtkCleanPolyData` on a surface
carrying labels.

`remesh_preserving_faces` is the entry point a host should call. `remesh_labelled_surface`
underneath it is the arithmetic without the reporting, and `remesh_patch_interior` is the
variant that pins an open boundary instead of sliding along inter-face seams.

The remesher is a port of geometry3Sharp's (gradientspace, Boost licence -- `NOTICE.txt`
beside the module has the terms, and names which files are the port); `remesh.py`'s docstring
records where it departs from the original and why.
"""

# `describe` is the right name inside the module and too general outside it, so the public
# name says what it describes.
from .conditioning import condition_surface, describe as describe_conditioning
from .labelled import (
    assign_active_face_scalars,
    face_cell_counts,
    remesh_preserving_faces,
)
from .quality import BAD_ASPECT_RATIO, triangle_quality
from .remesh import (
    DEFAULT_CORNER_ANGLE_DEGREES,
    DEFAULT_ITERATIONS,
    DEFAULT_SMOOTHING_SPEED,
    SEAM_PINNED,
    SEAM_SLIDES,
    MeshConstraints,
    PolylineTarget,
    QueuedRemesher,
    Remesher,
    SurfaceTarget,
    face_band_quality,
    remesh_labelled_surface,
    remesh_patch_interior,
)
from .surfaces import (
    boundary_point_ids,
    count_feature_edges,
    face_ids,
    feature_edges,
    open_boundary_curves,
    polydata_from_arrays,
    stamp_face_ids,
    surface_points,
    triangle_indices,
)

__version__ = "0.1.0"

__all__ = [
    "BAD_ASPECT_RATIO",
    "DEFAULT_CORNER_ANGLE_DEGREES",
    "DEFAULT_ITERATIONS",
    "DEFAULT_SMOOTHING_SPEED",
    "MeshConstraints",
    "PolylineTarget",
    "QueuedRemesher",
    "Remesher",
    "SEAM_PINNED",
    "SEAM_SLIDES",
    "SurfaceTarget",
    "assign_active_face_scalars",
    "boundary_point_ids",
    "condition_surface",
    "count_feature_edges",
    "describe_conditioning",
    "face_band_quality",
    "face_cell_counts",
    "face_ids",
    "feature_edges",
    "open_boundary_curves",
    "polydata_from_arrays",
    "remesh_labelled_surface",
    "remesh_patch_interior",
    "remesh_preserving_faces",
    "stamp_face_ids",
    "surface_points",
    "triangle_indices",
    "triangle_quality",
]
