"""One remesher, in two modes, and it keeps `ModelFaceID` by construction.

This is a port of geometry3Sharp's constrained remesher -- `Remesher`,
`MeshConstraints`, `MeshConstraintUtil` and `DCurveProjectionTarget` -- onto the
dynamic mesh in `dynamic_mesh.py`. It replaces two remeshers used before it:
ACVD clustering for an initial patch, with the
save-trim-stitch apparatus it needed because clustering cannot hold a boundary at
all, and PyMeshLab's incremental remesher for a sculpted patch's interior, which
could hold a boundary only by freezing a collar of triangles behind it. Both are
gone, and with them pyacvd, PyVista and PyMeshLab -- so this package imports nothing
Slicer's own Python does not ship, and needs no external interpreter.

Measurements below cite three clinical heart cases, called A, B and C. They are the
surfaces this was developed against: a segmented cardiac anatomy capped into a flow
domain, a patch stitched into it along a contour, and the merged result.

The algorithm is the incremental one: a pass over every edge that splits it if it
is longer than 4/3 of the target, collapses it if it is shorter than 4/5, and
otherwise flips it when that brings the vertex valences closer to six, followed
by a uniform Laplacian smoothing pass with every vertex reprojected onto the
surface it started from. What makes it *ours* is the constraint container between
the two: an `EdgeConstraint` per edge and a `VertexConstraint` per vertex, so a
face seam is held while the mesh around it changes, continuously, rather than
repaired afterwards.

**Two modes, and the difference is what a seam is allowed to do.**

`SEAM_SLIDES` is the one most callers want: seam vertices
may be split, collapsed and moved, but they are reprojected onto the *original
seam curve* every time they move, so the seam's geometry is held to a deviation
tolerance while its discretisation is free. That is the difference between
constraining the shape of a seam and constraining its vertex list, and only the
first is load-bearing anywhere. It is what lets seam triangle quality be set by
the target edge length rather than by whatever the contour cut produced: on
case B's merged flow domain the band's worst aspect ratio goes 935 to 8.6 and its
99th percentile 116 to 2.0, and on case A's published domain 2176 to 3.7.

`SEAM_PINNED` is the opt-in strictness: seam vertices are fixed and seam edges
refuse split, collapse and flip alike, so the seam comes back vertex for vertex
identical. A caller that is going to weld something else to those vertices needs
exactly that, and so does one carrying a per-vertex array across the pass by
`GlobalNodeID`.

**Two settings here are not free parameters, and both were measured.** Smoothing
is uniform-weighted rather than g3Sharp's own `MeanValue` default, because
mean-value and cotangent weights both go negative on obtuse triangles: on case B's
28k-triangle domain the library default left 532 triangles under 20 degrees against
uniform weighting's 50. And every position change is gated on the `1e-12` cross-norm
that TetGen refuses a surface at,
because smoothing is the one step in the loop that can take a triangle to exactly
zero area -- 25 iterations at speed 0.1 produced 8 degenerate triangles on the real
flow domain without the gate.

Where this departs from g3Sharp, mostly because the surfaces here are worse
behaved than the library assumes. Each of these was measured against the
library's own rule, on case A's three surfaces and on a sphere and a disc
remeshed to half their edge length -- the last two because the real cases are
near-converged cleanup passes that perform only 30 to 330 flips, so a rule about
flipping barely registers on them:

- A constrained vertex is smoothed using **only its neighbours along the seam**,
  not its whole one-ring. g3Sharp smooths towards the one-ring centroid and lets
  the curve projection pull the result back, which works but spends the
  projection tolerance to buy nothing; using the seam's own neighbours relaxes
  the vertex spacing *along* the curve, which is what a resampled seam wants.
- A move that would degenerate or invert a triangle is reverted per vertex rather
  than clamped, and the revert is re-checked until the mesh is clean. That is the
  degeneracy gate above, applied continuously instead of as a final refusal.
- **Valence error is squared, where g3Sharp's is absolute.** The library scores a
  flip by `|valence - target|` summed over four vertices; `_valence_error` squares
  it. On a closed sphere, where there are no boundary vertices and the norm is
  therefore the only thing that differs, absolute error accepts 2931 flips against
  squared error's 9411 and leaves the worst aspect ratio at 2.66 against 1.55, the
  99th percentile at 2.62 against 1.43 -- and takes *longer*, 3.72 s against
  2.26 s, because the flips it refuses come back as splits and collapses. Squaring
  tolerates spreading one large valence outlier over several vertices; the absolute
  form scores that as no better and refuses it.
- **A boundary vertex's target valence is four**, where g3Sharp targets whatever
  the vertex already has -- which scores any change to it as worse, so the library
  never flips towards a better boundary. Four is the Botsch-Kobbelt value, so this
  is the one departure where g3Sharp is the one leaving the literature. It makes no
  difference on a disc (worst aspect 1.6834 either way) and a large one on the
  pinned patch, where the boundary *is* the seam: worst aspect 9.0 against 14.2,
  which is also the difference between landing under and over the aspect ratio of
  10 `quality.py` counts a triangle at.
- **Unverified, and recorded so it is not mistaken for a decision:** the order
  within a pass is collapse, then split, then flip, where g3Sharp's `ProcessEdge`
  is collapse, then *flip*, then split -- so the library tries to flip a too-long
  edge before splitting it. Measured, this is a wash: the library's order is better
  on the sphere's seam band (1.26 against 1.54) and worse on the disc (2.07 against
  1.68), identical on case A's capped domain, and reaches the same place with
  30% fewer splits. There is no evidence for either, so this stays as it is rather
  than churn on a coin flip. It matters to `QueuedRemesher`, which enqueues the
  neighbourhood of whatever an operation touched: a different order modifies
  different edges and so fills the queue differently.

One measured cost of the two valence departures, since they are a trade and not a
free win: on the patch they leave the *smallest* triangle of any variant tried,
0.0088 mm² against 0.0114 to 0.0129. They buy aspect ratio by spending triangle
area. That is nowhere near the `1e-12` gate, and aspect ratio is what every
downstream stage complains about, so it is the right side of the trade -- but it is
a trade.
"""

from __future__ import annotations

from math import inf, sqrt

import numpy as np
import vtk
from . import surfaces
from .conditioning import condition_surface
from .dynamic_mesh import DynamicMesh
from .quality import triangle_quality
from .surfaces import open_boundary_curves

# The cross-norm below which a triangle is degenerate. This is TetGen's threshold and the one
# `_validate_triangles` refuses a published surface at, so a remesher that produced anything
# under it would be handing the failure two steps downstream with a worse error message.
MINIMUM_TRIANGLE_CROSS = 1e-12
# Incremental split/collapse/flip/smooth sweeps. Ten is where the edge-length histogram stops
# moving on the surfaces here; twenty-five buys a little interior quality and starts creating
# degeneracy the gate then has to revert.
DEFAULT_ITERATIONS = 10
# Laplacian step per sweep. g3Sharp's own default, and the measured trade: smoothing earns most
# of the quality improvement and is also the only step that can collapse a triangle.
DEFAULT_SMOOTHING_SPEED = 0.1
# The Botsch-Kobbelt window around the target edge length. Splitting above 4/3 and collapsing
# below 4/5 is what makes the two operations stable against each other -- a narrower window has
# them undo one another for ever.
LONG_EDGE_FRACTION = 4.0 / 3.0
SHORT_EDGE_FRACTION = 4.0 / 5.0
# How sharply a seam may turn before that vertex is pinned rather than allowed to slide. A
# corner is not a resampling artefact, it is where two faces meet a third, and sliding a vertex
# through it moves the label boundary rather than re-discretising it.
DEFAULT_CORNER_ANGLE_DEGREES = 30.0
# Target valences a flip aims at: six in the interior of a triangulated surface, four on an open
# boundary.
INTERIOR_TARGET_VALENCE = 6
BOUNDARY_TARGET_VALENCE = 4
# No operation may create a triangle worse than this unless the triangles it replaced were
# already worse, in which case their own worst is the bound. `metrics.BAD_ASPECT_RATIO` is the
# same number, and it is here because the `1e-12` degeneracy gate is much too permissive to
# stand alone: a collapse that leaves three consecutive seam vertices carrying one triangle
# between them produces a sliver of aspect 5091 whose cross-norm is still 1e-3, and one of those
# survived the whole of case B's merged-domain remesh before this bound existed.
MAXIMUM_CREATED_ASPECT_RATIO = 10.0

SEAM_SLIDES = "slide"
SEAM_PINNED = "pin"

# What `Remesher.process_edge` did to an edge -- g3Sharp's `ProcessResult`. `QueuedRemesher`
# enqueues the neighbourhood of anything that is not `EDGE_UNCHANGED`.
EDGE_UNCHANGED = "unchanged"
EDGE_COLLAPSED = "collapsed"
EDGE_SPLIT = "split"
EDGE_FLIPPED = "flipped"

# When `QueuedRemesher`'s split-only prelude stops: once a pass splits fewer than this
# fraction of the edges, the mesh is close enough to the target that the full sweep -- which
# can also collapse, flip and smooth -- is the better thing to be running. g3Sharp's own 1%.
FAST_SPLIT_FRACTION = 0.01


# --- projection targets ------------------------------------------------------------------

class PolylineTarget:
    """Closest point on a polyline -- g3Sharp's `DCurveProjectionTarget`.

    The clamp at the segment ends is what stops a vertex sliding off the end of an open
    chain, so a chain between two pinned corners cannot leak past either of them.
    """

    # Queries are chunked because the closest-point test is a full outer product between
    # queries and segments, and a seam chain on a real domain can carry a few hundred of each.
    CHUNK = 256

    def __init__(self, points, closed=False):
        points = np.asarray(points, dtype=float)
        if len(points) < 2:
            raise ValueError("A polyline projection target needs at least two points.")
        if closed:
            points = np.vstack([points, points[0]])
        self.points = points
        self._starts = points[:-1]
        self._directions = points[1:] - points[:-1]
        self._lengths_squared = np.maximum(
            np.einsum("ij,ij->i", self._directions, self._directions), 1e-30)

    def project(self, point):
        return self.project_many(np.asarray(point, dtype=float).reshape((1, 3)))[0]

    def project_many(self, points):
        points = np.asarray(points, dtype=float)
        result = np.empty_like(points)
        for start in range(0, len(points), self.CHUNK):
            block = points[start:start + self.CHUNK]
            offsets = block[:, None, :] - self._starts[None, :, :]
            fractions = np.clip(
                np.einsum("ijk,jk->ij", offsets, self._directions)
                / self._lengths_squared[None, :], 0.0, 1.0)
            closest = self._starts[None, :, :] + fractions[:, :, None] * self._directions[None, :, :]
            gaps = block[:, None, :] - closest
            best = np.argmin(np.einsum("ijk,ijk->ij", gaps, gaps), axis=1)
            result[start:start + len(block)] = closest[np.arange(len(block)), best]
        return result

    def deviation(self, points):
        points = np.asarray(points, dtype=float)
        if not len(points):
            return np.zeros(0)
        return np.linalg.norm(points - self.project_many(points), axis=1)


class SurfaceTarget:
    """Closest point on the surface as it arrived -- g3Sharp's `MeshProjectionTarget`.

    This is what keeps a remesh from shrinking the shape: smoothing moves every vertex
    towards its neighbours' centroid, which on a curved surface is inwards, and the
    reprojection puts it back. Nothing else in the loop preserves geometry.
    """

    def __init__(self, surface):
        self._locator = vtk.vtkStaticCellLocator()
        self._locator.SetDataSet(surface)
        self._locator.BuildLocator()
        self._surface = surface
        # Held rather than made per call: a relax pass projects every vertex, and building
        # four VTK objects for each of fifteen thousand of them is measurable.
        self._closest = [0.0, 0.0, 0.0]
        self._cell_id = vtk.reference(0)
        self._sub_id = vtk.reference(0)
        self._squared_distance = vtk.reference(0.0)
        self._cell = vtk.vtkGenericCell()

    def project_many(self, points):
        points = np.asarray(points, dtype=float)
        result = np.empty_like(points)
        find = self._locator.FindClosestPoint
        closest = self._closest
        for index in range(len(points)):
            find([float(points[index, 0]), float(points[index, 1]), float(points[index, 2])],
                 closest, self._cell, self._cell_id, self._sub_id, self._squared_distance)
            result[index] = closest
        return result


# --- constraints -------------------------------------------------------------------------

class EdgeConstraint:
    """What may be done to one edge, and what curve its children belong to.

    `set_id` travels with the edge so a vertex inserted by a split inherits the chain it
    was inserted into. Without it a new seam vertex would carry a projection target but no
    identity, and a later collapse could not tell two seams apart.
    """

    __slots__ = ("can_split", "can_collapse", "can_flip", "target", "set_id")

    def __init__(self, can_split=True, can_collapse=True, can_flip=True,
                 target=None, set_id=None):
        self.can_split = bool(can_split)
        self.can_collapse = bool(can_collapse)
        self.can_flip = bool(can_flip)
        self.target = target
        self.set_id = set_id


class VertexConstraint:
    """Where one vertex may go. `fixed` is absolute; `target` is a curve it must stay on."""

    __slots__ = ("fixed", "target", "set_id")

    def __init__(self, fixed=False, target=None, set_id=None):
        self.fixed = bool(fixed)
        self.target = target
        self.set_id = set_id


class MeshConstraints:
    """Per-edge and per-vertex constraints, keyed the way the mesh keys them.

    Edges are keyed by the canonical vertex pair rather than by an edge id, which is what
    makes a split's bookkeeping trivial: the children of `(a, b)` are `(a, f)` and
    `(f, b)`, so the constraint is copied onto two keys the caller can name without asking
    the mesh for anything.
    """

    def __init__(self):
        self.edges = {}
        self.vertices = {}

    @staticmethod
    def key(first, second):
        return (first, second) if first < second else (second, first)

    def edge(self, first, second):
        return self.edges.get(self.key(first, second))

    def vertex(self, vertex):
        return self.vertices.get(vertex)

    def set_edge(self, key, constraint):
        self.edges[key] = constraint

    def set_vertex(self, vertex, constraint):
        self.vertices[vertex] = constraint

    def clear_edge(self, first, second):
        self.edges.pop(self.key(first, second), None)

    def clear_vertex(self, vertex):
        self.vertices.pop(vertex, None)


def _merged_edge_constraint(existing, incoming):
    """The constraint an edge carries once a collapse has merged another edge into it.

    Flags are intersected -- whatever either edge forbade stays forbidden -- and a
    projection target is kept if either had one. A collapse that merged a seam edge into a
    plain one has to leave a seam edge behind, or the chain the seam travels by is broken
    at exactly the vertex the collapse just created.
    """
    if existing is None:
        return incoming
    if incoming is None:
        return existing
    return EdgeConstraint(
        can_split=existing.can_split and incoming.can_split,
        can_collapse=existing.can_collapse and incoming.can_collapse,
        can_flip=existing.can_flip and incoming.can_flip,
        target=existing.target if existing.target is not None else incoming.target,
        set_id=existing.set_id if existing.set_id is not None else incoming.set_id,
    )


def constrained_edges(mesh):
    """Every edge that bounds a face label or the surface itself.

    Both belong in one set. A `ModelFaceID` boundary and an open mesh boundary are the same
    thing to a remesher -- a curve the surface is not allowed to lose -- and the only
    difference is that one has a triangle on each side.
    """
    result = []
    for first, second in mesh.edges():
        neighbours = mesh.edge_triangles(first, second)
        if len(neighbours) == 1:
            result.append((first, second))
        elif mesh.triangle_group(neighbours[0]) != mesh.triangle_group(neighbours[1]):
            result.append((first, second))
    return result


def seam_chains(mesh, corner_angle_degrees=DEFAULT_CORNER_ANGLE_DEGREES):
    """Split the constrained edges into chains between pinned vertices.

    A vertex is pinned when it is a junction -- fewer or more than two constrained edges
    meet there, so it is where three faces meet or where a chain ends -- or when the chain
    turns through more than `corner_angle_degrees`, which is a corner and not a
    discretisation artefact. Everything between two pinned vertices is one chain, free to
    be resampled along its own original polyline.

    Returns `(chains, pinned)`, where each chain is a dict of its ordered `vertices` and
    whether it is `closed`.
    """
    edges = constrained_edges(mesh)
    adjacency = {}
    for first, second in edges:
        adjacency.setdefault(first, []).append(second)
        adjacency.setdefault(second, []).append(first)

    positions = mesh.positions
    pinned = {vertex for vertex, neighbours in adjacency.items() if len(neighbours) != 2}
    cosine_limit = np.cos(np.radians(180.0 - float(corner_angle_degrees)))
    for vertex, neighbours in adjacency.items():
        if len(neighbours) != 2 or vertex in pinned:
            continue
        first = positions[neighbours[0]] - positions[vertex]
        second = positions[neighbours[1]] - positions[vertex]
        scale = np.linalg.norm(first) * np.linalg.norm(second)
        if scale <= 1e-30:
            pinned.add(vertex)
            continue
        # Straight through the vertex is a cosine of -1; a turn sharper than the limit takes
        # it above `cos(180 - limit)`.
        if float(first @ second) / scale > cosine_limit:
            pinned.add(vertex)

    visited = set()

    def walk(start, following):
        chain = [start]
        previous, current = start, following
        while True:
            chain.append(current)
            visited.add(MeshConstraints.key(previous, current))
            if current in pinned or current == start:
                return chain
            neighbours = adjacency[current]
            following = neighbours[0] if neighbours[1] == previous else neighbours[1]
            previous, current = current, following

    chains = []
    for vertex in sorted(pinned):
        for neighbour in adjacency[vertex]:
            if MeshConstraints.key(vertex, neighbour) in visited:
                continue
            chains.append({"vertices": walk(vertex, neighbour), "closed": False})
    for first, second in edges:
        if MeshConstraints.key(first, second) in visited:
            continue
        chain = walk(first, second)
        # A loop with no pinned vertex on it comes back to where it started, and the repeated
        # endpoint is the closure rather than a vertex.
        chains.append({"vertices": chain[:-1], "closed": True})
    return chains, pinned


def constrain_seams(mesh, constraints, seam=SEAM_SLIDES,
                    corner_angle_degrees=DEFAULT_CORNER_ANGLE_DEGREES):
    """Attach the seam constraints for one of the two modes.

    `SEAM_SLIDES` is g3Sharp's `PreserveBoundaryLoops`: every chain gets a
    `PolylineTarget` built from its polyline as it arrived, and its vertices and edges
    carry that target, so they may be resampled but never leave the curve.
    `SEAM_PINNED` is `FixAllGroupBoundaryEdges`: nothing on a seam moves at all.

    Returns the record of what was constrained, for the caller's log.
    """
    chains, pinned = seam_chains(mesh, corner_angle_degrees)
    positions = mesh.positions
    if seam == SEAM_PINNED:
        for first, second in constrained_edges(mesh):
            constraints.set_edge(MeshConstraints.key(first, second),
                                 EdgeConstraint(False, False, False))
            for vertex in (first, second):
                constraints.set_vertex(vertex, VertexConstraint(fixed=True))
        return {"seam": seam, "chains": len(chains), "pinned_vertices": len(constraints.vertices)}

    if seam != SEAM_SLIDES:
        raise ValueError(f"Unknown seam mode {seam!r}; expected {SEAM_SLIDES!r} or "
                         f"{SEAM_PINNED!r}.")

    for vertex in pinned:
        constraints.set_vertex(vertex, VertexConstraint(fixed=True))
    for index, chain in enumerate(chains):
        vertices = chain["vertices"]
        target = PolylineTarget(positions[vertices], closed=chain["closed"])
        edge_constraint = EdgeConstraint(can_split=True, can_collapse=True, can_flip=False,
                                         target=target, set_id=index)
        count = len(vertices) if chain["closed"] else len(vertices) - 1
        for step in range(count):
            constraints.set_edge(
                MeshConstraints.key(vertices[step], vertices[(step + 1) % len(vertices)]),
                edge_constraint)
        for vertex in vertices:
            if vertex in pinned:
                continue
            constraints.set_vertex(vertex, VertexConstraint(target=target, set_id=index))
    return {"seam": seam, "chains": len(chains), "pinned_vertices": len(pinned)}


# --- the remesher ------------------------------------------------------------------------

def _cross_norms(points, triangles):
    corners = points[triangles]
    return np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])


def _aspect_ratios(corners):
    """Circumradius over twice the inradius, for a stack of triangles.

    The same measure `metrics.triangle_quality` reports, so an operation judged by it is
    judged by the number the caller is going to read afterwards.

    Vectorised over a `(n, 3, 3)` corner array, and used where `n` is a whole surface --
    `face_band_quality`. The operation gates inside the refine pass ask the same question one
    to a dozen triangles at a time and use `_triangle_terms` instead, which is scalar; see
    that function for why, and for the measurement that says so.
    """
    corners = np.asarray(corners, dtype=float).reshape((-1, 3, 3))
    sides = np.linalg.norm(np.roll(corners, -1, axis=1) - corners, axis=2)
    double_areas = np.maximum(np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1), 1e-30)
    semiperimeters = np.maximum(0.5 * sides.sum(axis=1), 1e-30)
    return sides.prod(axis=1) * semiperimeters / double_areas ** 2 / 2.0


def _triangle_terms(first, second, third):
    """One triangle's cross product, that product's norm, and its aspect ratio.

    The same two measures as `_cross_norms` and `_aspect_ratios`, computed in scalar
    arithmetic for one triangle at a time, and this is the counterpart to the note on
    `_aspect_ratios` rather than a contradiction of it. That function is right that the
    refine pass asks a few hundred thousand times; what it gets wrong is the size of the
    ask. Split, collapse and flip judge one, two or a dozen triangles per call, and at that
    size a numpy call is almost all dispatch: 16.8 microseconds against 1.3 for the two
    triangles a flip weighs, because `roll`, `moveaxis` and `normalize_axis_tuple` do not
    care how short the array is. The vectorised pair stays for `face_band_quality` and
    `_refuse_degenerate`, which are handed a whole surface at once and where it is the right
    shape.

    The cross product is computed once and used for both answers. The numpy path computed it
    twice -- once for the degeneracy gate, once inside `_aspect_ratios`.

    `.tolist()` is what makes this worth doing. Unpacking a position straight out of the
    array leaves `numpy.float64` scalars, and arithmetic on those costs more than the numpy
    call it was meant to replace: 2.8 microseconds against 1.3 for Python floats.

    Returns `(cross, cross_norm, aspect_ratio)`, with `cross` a plain 3-tuple and
    `cross_norm` **not** floored at `1e-30` -- the degeneracy gate has to see a true zero.
    """
    ax, ay, az = first.tolist() if hasattr(first, "tolist") else first
    bx, by, bz = second.tolist() if hasattr(second, "tolist") else second
    cx, cy, cz = third.tolist() if hasattr(third, "tolist") else third
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    wx, wy, wz = cx - bx, cy - by, cz - bz
    # The three side lengths in the order `_aspect_ratios` takes them, so the products and
    # sums below round exactly as the vectorised version's do.
    side_a = sqrt(ux * ux + uy * uy + uz * uz)
    side_b = sqrt(wx * wx + wy * wy + wz * wz)
    side_c = sqrt(vx * vx + vy * vy + vz * vz)
    cross_x = uy * vz - uz * vy
    cross_y = uz * vx - ux * vz
    cross_z = ux * vy - uy * vx
    cross_norm = sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z)
    double_area = cross_norm if cross_norm > 1e-30 else 1e-30
    semiperimeter = 0.5 * (side_a + side_b + side_c)
    if semiperimeter <= 1e-30:
        semiperimeter = 1e-30
    aspect = (side_a * side_b * side_c * semiperimeter
              / (double_area * double_area) / 2.0)
    return (cross_x, cross_y, cross_z), cross_norm, aspect


def _farthest_distance(positions, vertices, origin):
    """How far the farthest of `vertices` is from `origin`.

    Scalar for the same reason as `_triangle_terms`: the collapse gate asks this of a one-ring,
    which is six vertices on a regular surface and a dozen at worst. The caller guarantees
    `vertices` is not empty; a zero here would read as "no neighbour is far away", which is the
    opposite of what an empty neighbourhood should mean, so it is not a case to fall through.
    """
    origin_x, origin_y, origin_z = (origin.tolist() if hasattr(origin, "tolist") else origin)
    worst = 0.0
    for vertex in vertices:
        x, y, z = positions[vertex].tolist()
        offset_x, offset_y, offset_z = x - origin_x, y - origin_y, z - origin_z
        distance = sqrt(offset_x * offset_x + offset_y * offset_y + offset_z * offset_z)
        if distance > worst:
            worst = distance
    return worst


def _cross_norm_and_aspect(corners):
    """The smallest cross-norm and the worst aspect ratio over a handful of triangles.

    `corners` is an iterable of anything that unpacks into three points -- a `(3, 3)` array
    row-wise, or a tuple of three positions -- which is what lets the split gate weigh a
    triangle whose middle corner is not a mesh vertex yet.
    """
    smallest_cross = inf
    worst_aspect = 0.0
    for first, second, third in corners:
        _, cross_norm, aspect = _triangle_terms(first, second, third)
        if cross_norm < smallest_cross:
            smallest_cross = cross_norm
        if aspect > worst_aspect:
            worst_aspect = aspect
    return smallest_cross, worst_aspect


class Remesher:
    """Incremental split/collapse/flip/smooth under constraints -- g3Sharp's `Remesher`."""

    def __init__(self, mesh, target_edge_length, *, constraints=None, surface_target=None,
                 smoothing_speed=DEFAULT_SMOOTHING_SPEED, enable_splits=True,
                 enable_collapses=True, enable_flips=True, enable_smoothing=True,
                 minimum_triangle_cross=MINIMUM_TRIANGLE_CROSS):
        target_edge_length = float(target_edge_length)
        if not np.isfinite(target_edge_length) or target_edge_length <= 0.0:
            raise ValueError("The target edge length must be a finite positive number.")
        self.mesh = mesh
        self.constraints = constraints if constraints is not None else MeshConstraints()
        self.surface_target = surface_target
        self.target_edge_length = target_edge_length
        self.minimum_edge_length = SHORT_EDGE_FRACTION * target_edge_length
        self.maximum_edge_length = LONG_EDGE_FRACTION * target_edge_length
        self.smoothing_speed = float(smoothing_speed)
        self.enable_splits = bool(enable_splits)
        self.enable_collapses = bool(enable_collapses)
        self.enable_flips = bool(enable_flips)
        self.enable_smoothing = bool(enable_smoothing)
        self.minimum_triangle_cross = float(minimum_triangle_cross)
        self.counts = {"splits": 0, "collapses": 0, "flips": 0, "reverted_moves": 0}

    def remesh(self, iterations=DEFAULT_ITERATIONS, on_iteration=None):
        """Sweep `iterations` times, reporting after each one if a caller asked.

        `on_iteration(done, total)` is called between passes and nowhere else, so it cannot
        change what the sweep decides. It exists because a sweep is seconds of pure Python
        with no I/O in it: a host with an event loop -- Slicer -- has no other moment to
        repaint a progress dialog or notice a Cancel, and a callback that raises here
        abandons the remesh cleanly, since nothing outside this object has been touched yet.
        """
        iterations = int(iterations)
        for index in range(iterations):
            self._refine_pass()
            self._relax_pass()
            if on_iteration is not None:
                on_iteration(index + 1, iterations)
        return self.counts

    # --- one pass over the edges ---------------------------------------------------------

    def edge_length(self, first, second):
        """The length of one edge.

        Scalar for the same reason as `_triangle_terms`: this runs once per edge per sweep --
        six hundred thousand times on the capped domain -- and one `np.linalg.norm` on a
        single 3-vector costs more than the whole subtraction.

        `mesh.position` per vertex rather than a hoisted `mesh.positions`: a split may have
        grown the position array, and a view taken before that would be pointing at the array
        it replaced.
        """
        mesh = self.mesh
        tail_x, tail_y, tail_z = mesh.position(first).tolist()
        head_x, head_y, head_z = mesh.position(second).tolist()
        gap_x, gap_y, gap_z = head_x - tail_x, head_y - tail_y, head_z - tail_z
        return sqrt(gap_x * gap_x + gap_y * gap_y + gap_z * gap_z)

    def process_edge(self, first, second):
        """Collapse, split or flip one edge, whichever applies -- g3Sharp's `ProcessEdge`.

        Returns which of them happened, because `QueuedRemesher` needs to know: it enqueues
        the neighbourhood of an edge that changed and leaves the rest alone. The whole gate
        lives here rather than in the sweep so both sweeps run the same one.

        The order is collapse, split, flip, which is not g3Sharp's; see the module docstring.
        """
        mesh = self.mesh
        if not mesh.has_edge(first, second):
            return EDGE_UNCHANGED
        length = self.edge_length(first, second)
        if (self.enable_collapses and length < self.minimum_edge_length
                and self._try_collapse(first, second)):
            return EDGE_COLLAPSED
        if (self.enable_splits and length > self.maximum_edge_length
                and self._try_split(first, second)):
            return EDGE_SPLIT
        if self.enable_flips and self._try_flip(first, second):
            return EDGE_FLIPPED
        return EDGE_UNCHANGED

    def _refine_pass(self):
        for first, second in self.mesh.edges():
            self.process_edge(first, second)

    def _on_edge_split(self, first, second, new_vertex):
        """Hook for a subclass that tracks what changed -- g3Sharp's `OnEdgeSplit`.

        Nothing here. `QueuedRemesher` uses it to queue the children of a split that are
        still too long, which is the one thing a split-only pass has to know and cannot
        recover afterwards.
        """

    # --- split ---------------------------------------------------------------------------

    def _try_split(self, first, second):
        mesh = self.mesh
        constraint = self.constraints.edge(first, second)
        if constraint is not None and not constraint.can_split:
            return False
        midpoint = 0.5 * (mesh.position(first) + mesh.position(second))
        if constraint is not None and constraint.target is not None:
            midpoint = constraint.target.project(midpoint)
        positions = mesh.positions
        for triangle in mesh.edge_triangles(first, second):
            apex = positions[mesh.third_vertex(triangle, first, second)]
            smallest_cross, worst_aspect = _cross_norm_and_aspect((
                (positions[first], midpoint, apex),
                (midpoint, positions[second], apex)))
            if smallest_cross <= self.minimum_triangle_cross:
                return False
            limit = max(MAXIMUM_CREATED_ASPECT_RATIO,
                        _cross_norm_and_aspect((positions[mesh.triangle(triangle)],))[1])
            if worst_aspect > limit:
                return False
        new_vertex, children = mesh.split_edge(first, second, midpoint)
        if constraint is not None:
            for child in children:
                self.constraints.set_edge(child, constraint)
            if constraint.target is not None:
                self.constraints.set_vertex(
                    new_vertex,
                    VertexConstraint(target=constraint.target, set_id=constraint.set_id))
        self.counts["splits"] += 1
        self._on_edge_split(first, second, new_vertex)
        return True

    # --- collapse ------------------------------------------------------------------------

    @staticmethod
    def _rank(constraint):
        """How strongly a vertex is held: fixed outranks constrained outranks free.

        A collapse may only remove a vertex whose rank is no higher than the one it is
        removed onto. That single rule is what keeps a seam from being dragged away by a
        collapse of an ordinary interior edge, and it is g3Sharp's "if one is fixed we can
        collapse the other into it" stated the other way round.
        """
        if constraint is None:
            return 0
        if constraint.fixed:
            return 2
        if constraint.target is not None or constraint.set_id is not None:
            return 1
        return 0

    def _try_collapse(self, first, second):
        mesh = self.mesh
        constraints = self.constraints
        edge_constraint = constraints.edge(first, second)
        if edge_constraint is not None and not edge_constraint.can_collapse:
            return False
        first_constraint = constraints.vertex(first)
        second_constraint = constraints.vertex(second)
        first_rank = self._rank(first_constraint)
        second_rank = self._rank(second_constraint)
        if first_rank == 2 and second_rank == 2:
            return False
        first_set = None if first_constraint is None else first_constraint.set_id
        second_set = None if second_constraint is None else second_constraint.set_id
        if first_set is not None and second_set is not None and first_set != second_set:
            return False

        options = []
        if second_rank <= first_rank:
            options.append((first, second, first_constraint, second_constraint))
        if first_rank <= second_rank:
            options.append((second, first, second_constraint, first_constraint))
        for keep, remove, keep_constraint, remove_constraint in options:
            if remove_constraint is not None and remove_constraint.target is not None:
                # A vertex living on a curve may only be removed onto a vertex of the same
                # curve, or onto a pinned corner of it. Anything else takes the curve with it.
                if keep_constraint is None:
                    continue
                if not (keep_constraint.fixed
                        or keep_constraint.target is remove_constraint.target):
                    continue
            if not mesh.collapse_would_be_valid(keep, remove):
                continue
            position = self._collapse_position(keep, remove, keep_constraint)
            if not self._collapse_is_sound(keep, remove, position):
                continue
            self._commit_collapse(keep, remove, position)
            return True
        return False

    def _collapse_position(self, keep, remove, keep_constraint):
        mesh = self.mesh
        if keep_constraint is not None and keep_constraint.fixed:
            return np.array(mesh.position(keep), dtype=float)
        midpoint = 0.5 * (mesh.position(keep) + mesh.position(remove))
        if keep_constraint is not None and keep_constraint.target is not None:
            return keep_constraint.target.project(midpoint)
        return midpoint

    def _collapse_is_sound(self, keep, remove, position):
        """Refuse a collapse that degenerates a triangle, folds one over, or overshoots.

        The long-edge test is what stops a run of collapses eating a coarse region: every
        collapse shortens the edge it is on and lengthens the ones around it, so without a
        ceiling the mesh drifts away from the target in the direction the pass happens to
        sweep. It is bounded by the neighbourhood's *own* longest edge as well as by the
        target, and that second term is not a softening -- it is what makes the test mean
        anything on the first pass.

        Measured, because the strict form looked right and was not. A sliver whose short edge
        is 0.14 mm between neighbours at 1.2 mm cannot be collapsed without producing an edge
        near 1.24, which is over a 0.85 mm target's 1.13 ceiling -- so the strict test refused
        **935 of the 1044** collapsible short edges on case B's patch face, and the
        slivers it was supposed to remove survived the whole remesh: 36 triangles over aspect
        10, worst 32.7. Refusing them was also pointless, since a 1.24 mm edge is over the
        split threshold and the next pass would have halved it. With the neighbourhood term the
        same face comes back with **nothing** over aspect 4 and a worst of 4.8, and
        case A's published domain goes from a worst aspect of 8.8 to 3.7.
        """
        mesh = self.mesh
        positions = mesh.positions
        removed = set(mesh.edge_triangles(keep, remove))
        affected = (set(mesh.vertex_triangles(keep)) | set(mesh.vertex_triangles(remove))
                    ) - removed
        if not affected:
            return False
        # Both one-rings are wanted twice -- once for the neighbourhood the collapse would
        # create and once for the one it already has -- so they are taken once.
        keep_ring = mesh.vertex_one_ring(keep)
        remove_ring = mesh.vertex_one_ring(remove)
        # A dozen triangles at most, which is `_triangle_terms` territory rather than numpy's.
        # The three tests below are the same conjunction the vectorised version took as three
        # `min`/`max` reductions; refusing at the first triangle that fails is the same answer.
        moved = position.tolist() if hasattr(position, "tolist") else list(position)
        worst_old_aspect = 0.0
        worst_new_aspect = 0.0
        for triangle in affected:
            corners = mesh.triangle(triangle)
            old_cross, _, old_aspect = _triangle_terms(
                positions[corners[0]], positions[corners[1]], positions[corners[2]])
            new_cross, new_cross_norm, new_aspect = _triangle_terms(*[
                moved if corner == keep or corner == remove else positions[corner]
                for corner in corners])
            if new_cross_norm <= self.minimum_triangle_cross:
                return False
            if (new_cross[0] * old_cross[0] + new_cross[1] * old_cross[1]
                    + new_cross[2] * old_cross[2]) <= 0.0:
                return False
            if old_aspect > worst_old_aspect:
                worst_old_aspect = old_aspect
            if new_aspect > worst_new_aspect:
                worst_new_aspect = new_aspect
        if worst_new_aspect > max(MAXIMUM_CREATED_ASPECT_RATIO, worst_old_aspect):
            return False
        neighbours = (keep_ring | remove_ring) - {keep, remove}
        if neighbours:
            longest = _farthest_distance(positions, neighbours, moved)
            existing = max(_farthest_distance(positions, keep_ring, positions[keep]),
                           _farthest_distance(positions, remove_ring, positions[remove]))
            # Same shape as the aspect bound below: an operation may not make the
            # neighbourhood worse than the target *or* than what it already was.
            if longest > max(self.maximum_edge_length, existing):
                return False
        return True

    def _commit_collapse(self, keep, remove, position):
        mesh = self.mesh
        constraints = self.constraints
        inherited = {}
        for neighbour in mesh.vertex_one_ring(remove):
            constraint = constraints.edge(remove, neighbour)
            if constraint is not None:
                inherited[neighbour] = constraint
            constraints.clear_edge(remove, neighbour)
        mesh.set_position(keep, position)
        mesh.collapse_edge(keep, remove)
        constraints.clear_vertex(remove)
        for neighbour, constraint in inherited.items():
            if neighbour == keep or not mesh.has_edge(keep, neighbour):
                continue
            key = MeshConstraints.key(keep, neighbour)
            constraints.set_edge(key, _merged_edge_constraint(constraints.edges.get(key),
                                                              constraint))
        self.counts["collapses"] += 1

    # --- flip ----------------------------------------------------------------------------

    def _valence_error(self, vertex, offset=0):
        target = (BOUNDARY_TARGET_VALENCE if self.mesh.vertex_is_boundary(vertex)
                  else INTERIOR_TARGET_VALENCE)
        return (self.mesh.vertex_valence(vertex) + offset - target) ** 2

    def _try_flip(self, first, second):
        mesh = self.mesh
        constraint = self.constraints.edge(first, second)
        if constraint is not None and not constraint.can_flip:
            return False
        if not mesh.flip_would_be_valid(first, second):
            return False
        left, right = mesh.opposite_vertices(first, second)
        before = (self._valence_error(first) + self._valence_error(second)
                  + self._valence_error(left) + self._valence_error(right))
        after = (self._valence_error(first, -1) + self._valence_error(second, -1)
                 + self._valence_error(left, 1) + self._valence_error(right, 1))
        if after >= before:
            return False
        positions = mesh.positions
        neighbours = mesh.edge_triangles(first, second)
        # Scalar throughout: two triangles before the flip and two after, which is the size
        # `_triangle_terms` exists for. `reference` is the summed normal of the pair the flip
        # would replace, and it is what the new pair has to agree with rather than fold over.
        reference_x = reference_y = reference_z = 0.0
        worst_old_aspect = 0.0
        for triangle in neighbours:
            corners = mesh.triangle(triangle)
            cross, _, aspect = _triangle_terms(positions[corners[0]], positions[corners[1]],
                                               positions[corners[2]])
            reference_x += cross[0]
            reference_y += cross[1]
            reference_z += cross[2]
            if aspect > worst_old_aspect:
                worst_old_aspect = aspect
        worst_new_aspect = 0.0
        for corners in mesh.flipped_triangles(first, second):
            cross, cross_norm, aspect = _triangle_terms(
                positions[corners[0]], positions[corners[1]], positions[corners[2]])
            if cross_norm <= self.minimum_triangle_cross:
                return False
            if (cross[0] * reference_x + cross[1] * reference_y
                    + cross[2] * reference_z) <= 0.0:
                return False
            if aspect > worst_new_aspect:
                worst_new_aspect = aspect
        # Valence alone is not enough to flip on, and this is the measurement that says so:
        # 117 valence-improving flips took a remeshed sphere's worst aspect ratio from 3.0 to
        # 195, and the churn they caused tripled the split and collapse counts. A flip is
        # allowed to rearrange the connectivity and not to spend triangle shape doing it.
        if worst_new_aspect > worst_old_aspect:
            return False
        mesh.flip_edge(first, second)
        self.counts["flips"] += 1
        return True

    # --- smooth and project --------------------------------------------------------------

    def _triangle_array(self):
        mesh = self.mesh
        return np.asarray([mesh.triangle(triangle) for triangle in mesh.alive_triangles()],
                          dtype=np.int64).reshape((-1, 3))

    def _neighbour_means(self, positions, edges, slots):
        """Uniform-weighted one-ring centroids, from an edge list."""
        if not len(edges):
            return np.zeros((slots, 3)), np.zeros(slots, dtype=np.int64)
        tails = np.concatenate([edges[:, 0], edges[:, 1]])
        heads = np.concatenate([edges[:, 1], edges[:, 0]])
        counts = np.bincount(tails, minlength=slots)
        sums = np.stack([np.bincount(tails, weights=positions[heads, axis], minlength=slots)
                         for axis in range(3)], axis=1)
        return sums / np.maximum(counts, 1)[:, None], counts

    def _relax_pass(self):
        """One smoothing step, with every moved vertex reprojected onto where it belongs.

        Smoothing and projection are one pass rather than two because they are one
        decision: the smoothed position is a proposal, the projection is what makes it
        admissible, and the degeneracy gate then judges the pair. Splitting them would
        commit an inadmissible position in between and gate it twice.
        """
        if not self.enable_smoothing and self.surface_target is None:
            return
        mesh = self.mesh
        slots = mesh.vertex_slots
        positions = np.array(mesh.positions, dtype=float)
        edges = np.asarray(mesh.edges(), dtype=np.int64).reshape((-1, 2))
        means, counts = self._neighbour_means(positions, edges, slots)

        seam_keys = [key for key in self.constraints.edges if mesh.has_edge(*key)]
        seam_edges = np.asarray(seam_keys, dtype=np.int64).reshape((-1, 2))
        seam_means, seam_counts = self._neighbour_means(positions, seam_edges, slots)

        speed = self.smoothing_speed if self.enable_smoothing else 0.0
        candidate = positions.copy()
        movable = counts > 0
        on_curve = np.full(slots, -1, dtype=np.int64)
        targets = []
        target_index = {}
        for vertex, constraint in self.constraints.vertices.items():
            if vertex >= slots or not mesh.vertex_is_alive(vertex):
                continue
            if constraint.fixed:
                movable[vertex] = False
            elif constraint.target is not None:
                index = target_index.get(id(constraint.target))
                if index is None:
                    index = len(targets)
                    target_index[id(constraint.target)] = index
                    targets.append(constraint.target)
                on_curve[vertex] = index

        # Along the seam, the useful relaxation is of the spacing between seam vertices, so
        # the centroid is taken over the seam's own neighbours. A vertex with fewer than two
        # of them is at a chain end and stays put.
        curve = movable & (on_curve >= 0)
        stranded = curve & (seam_counts < 2)
        movable[stranded] = False
        curve = curve & ~stranded
        plain = movable & (on_curve < 0)

        candidate[plain] = positions[plain] + speed * (means[plain] - positions[plain])
        candidate[curve] = positions[curve] + speed * (seam_means[curve] - positions[curve])
        if self.surface_target is not None:
            selection = np.where(plain)[0]
            if len(selection):
                candidate[selection] = self.surface_target.project_many(candidate[selection])
        for index, target in enumerate(targets):
            selection = np.where(curve & (on_curve == index))[0]
            if len(selection):
                candidate[selection] = target.project_many(candidate[selection])
        self._commit_positions(candidate, movable)

    def _commit_positions(self, candidate, movable):
        """Take the moves that leave every triangle valid, and revert the rest.

        Reverting is iterated because a mixed state is not implied valid by either end of
        it: dropping one vertex's move can leave a neighbour's move degenerate. It
        terminates because the moved set only shrinks, and the all-reverted state is the
        mesh as it was.
        """
        mesh = self.mesh
        triangles = self._triangle_array()
        original = np.array(mesh.positions, dtype=float)
        reference = _cross_norms(original, triangles)
        moved = np.array(movable, dtype=bool)
        for _ in range(8):
            points = np.where(moved[:, None], candidate, original)
            cross = _cross_norms(points, triangles)
            bad = ((np.linalg.norm(cross, axis=1) <= self.minimum_triangle_cross)
                   | (np.einsum("ij,ij->i", cross, reference) <= 0.0))
            if not bad.any():
                break
            reverted = np.unique(triangles[bad].ravel())
            self.counts["reverted_moves"] += int(moved[reverted].sum())
            moved[reverted] = False
        mesh.set_positions(np.where(moved[:, None], candidate, original))


class QueuedRemesher(Remesher):
    """The same remesher, visiting only the edges that could have changed -- `RemesherPro`.

    A subclass in g3Sharp and a subclass here, and the reason is worth stating: it inherits
    `process_edge` untouched, so every constraint, gate and bound above applies unaltered.
    The only thing it changes is *which* edges the sweep visits.

    **What it is for.** `Remesher` sweeps every edge every pass, and on the surfaces this
    cleanup remeshes almost every pass is wasted: case A's capped domain performs 85 to
    235 operations per pass against 61,000 edges -- 0.2% -- for all ten passes, and each pass
    costs the same as the first because the cost is the sweep and not the work. So the sweep
    is the thing to shrink.

    **How the queue is fed.** Two sources, and the second is the one that makes it converge:

    - An operation that succeeded enqueues the one-rings of the four vertices around the edge
      it changed. g3Sharp does the same and says of it "TODO: optimize the queuing here, are
      over-doing it!", which is fair -- it is a superset of what actually changed -- but a
      superset only costs a re-test, and missing an edge costs quality.
    - The relax pass enqueues every edge its smoothing took *outside* the length window.
      Smoothing is deliberately not queue-limited -- it moves every vertex, or the mesh would
      relax unevenly -- so without this the queue would go stale the moment a vertex moved.
      This is g3Sharp's `TrackedSmoothPass` and the same rule, `new_len < MinEdgeLength ||
      new_len > MaxEdgeLength`.

    **What it is worth, measured on the surfaces these passes were actually handed.**
    That qualifier is the whole of it: `intermediate/capped_domain.vtp` and
    `intermediate/initial_combined_mesh.vtp` are the *outputs* of the two passes, not their
    inputs, so measuring on them measures a remesh of an already-remeshed surface and flatters
    the queue badly. The real inputs, captured out of a run:

    | input | target / its own median | `Remesher` | this |
    |---|---|---|---|
    | capped domain, 47622 cells | 1.00 | 6.29 s, worst aspect 5.49 | 5.89 s (1.07x), **3.63** |
    | merged domain, 29354 cells | 1.22 | 2.28 s, worst aspect 2.95 | 2.02 s (1.13x), **2.56** |

    So the speed gain is **1.07 to 1.13x**, not the 1.4 to 1.6x the outputs suggested, and the
    real gain is triangle quality. The queue visits about five times fewer edges than the
    sweep and returns a tenth of that as wall clock, because building it costs what skipping
    costs: 2038 collapses enqueue on the order of 49,000 edges through their four one-rings
    each, dwarfing the 2,000 to 4,000 the relax pass contributes.

    **Use it when the target is within about a factor of two of the surface's own median edge,
    which is where both real passes sit** -- exactly 1.00 for the cleanup, since it targets
    that median by definition, and 1.22 for the flow domain's 0.85 mm against its 0.699 mm.
    Swept across a sphere, the queue is 1.04 to 1.09x faster and equal or better in quality
    from 1.0 to 2.0 times its own edge length, and outside that it loses: 0.82x and 0.81x the
    speed at 0.5 and 0.8, and worse quality at 3.0. Note that this is a rule about *edge
    length*, not triangle count -- the flow-domain pass halves its vertex count at a ratio of
    1.22, because coarsening goes as the square.

    The reason is structural and it is upstream's too: **a flip is judged on valence, and the
    queue cannot predict where valence will want one.** Length changes and topology changes
    are what fill it, so valence equalisation only propagates one ring per pass instead of
    sweeping the surface. On the sphere the full sweep performs 9411 flips and this performs
    6542, and running longer does not close it -- 6542 at ten iterations, 6590 at twenty-five,
    6683 at forty, against the full sweep's 9411 at ten. g3Sharp's own remedy is a default of
    25 iterations rather than 10, and measured here even that does not recover it: at 25 the
    queued sweep is 4.06 s and worst aspect 1.80 where the full sweep is 3.99 s and 1.50.

    **Two things g3Sharp does that this does not.** `FastestRemesh` disables surface
    projection on alternate iterations for speed; that is refused here, because
    `remesh_labelled_surface` is documented as unable to move the domain outward -- every
    vertex reprojected onto the surface as it arrived, measured at 0.0000 mm on all three
    example cases -- and half the passes not projecting puts that back in question for a
    speedup this does not need. And `SharpEdgeReprojectionRemesh` is not ported at all;
    nothing here wants triangles aligned to a target's face normals.

    **What it costs.** The result is not identical to `Remesher`'s. Visit order decides which
    of two competing operations wins, so the counts differ and so does the mesh. `Remesher`
    stays as it is, is still what the tests hold to its own trace, and is what this is judged
    against.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # `None` means "visit every edge", which is how g3Sharp's null `modified_edges`
        # reads. A pass consumes the queue and builds the next one, so the empty set and the
        # absent set have to stay distinguishable.
        self._queue = None
        self.counts["queued_passes"] = 0
        self.counts["edges_visited"] = 0

    # --- the queue -----------------------------------------------------------------------

    def reset_queue(self):
        """Forget the queue, so the next pass sweeps everything -- g3Sharp's `ResetQueue`.

        Called between the split-only prelude and the full passes, because the two enqueue
        different things and a queue built by one is not the right starting point for the
        other.
        """
        self._queue = None

    def _take_queue(self):
        """The edges this pass will visit, with a fresh empty queue installed behind them."""
        if self._queue is None:
            self._queue = set()
            return self.mesh.edges()
        edges = list(self._queue)
        self._queue = set()
        return edges

    def _queue_one_ring(self, vertex):
        """Enqueue every edge on a vertex -- g3Sharp's `queue_one_ring`.

        The liveness check is not defensive: after a collapse one of the four vertices named
        before the operation is dead, and after a flip `third_vertex` may have returned
        `INVALID` for a boundary edge that had only one triangle.
        """
        mesh = self.mesh
        if vertex is None or vertex < 0 or vertex >= mesh.vertex_slots:
            return
        if not mesh.vertex_is_alive(vertex):
            return
        queue = self._queue
        for neighbour in mesh.vertex_one_ring(vertex):
            queue.add((vertex, neighbour) if vertex < neighbour else (neighbour, vertex))

    def _on_edge_split(self, first, second, new_vertex):
        """Queue the children of a split that are still too long -- g3Sharp's `SplitF`.

        The length test is here rather than left to the next pass for the reason g3Sharp
        gives -- "because of overhead in ProcessEdge, it is worth it to do a distance-check
        here". A child that is already short enough will be refused by the full gate anyway,
        and refusing it costs more than measuring it.
        """
        if self._queue is None:
            return
        mesh = self.mesh
        maximum = self.maximum_edge_length
        queue = self._queue
        for neighbour in mesh.vertex_one_ring(new_vertex):
            if self.edge_length(new_vertex, neighbour) > maximum:
                queue.add((new_vertex, neighbour) if new_vertex < neighbour
                          else (neighbour, new_vertex))

    # --- the two kinds of pass -----------------------------------------------------------

    def fast_split_iteration(self):
        """One pass that only splits -- g3Sharp's `FastSplitIteration`. Returns the count.

        Collapses, flips and smoothing are off. While most edges are still longer than the
        target none of the three can do anything a later pass will not redo, and all of them
        cost: the flip gate alone is half a full pass. The enable flags are restored on the
        way out whatever happens, which is g3Sharp's `PushState`/`PopState`.
        """
        saved = (self.enable_collapses, self.enable_flips, self.enable_smoothing)
        self.enable_collapses = self.enable_flips = self.enable_smoothing = False
        try:
            splits = 0
            for first, second in self._take_queue():
                self.counts["edges_visited"] += 1
                if self.process_edge(first, second) is EDGE_SPLIT:
                    splits += 1
            return splits
        finally:
            self.enable_collapses, self.enable_flips, self.enable_smoothing = saved

    def remesh_iteration(self):
        """One full pass over the queue, then the tracked relax -- g3Sharp's `RemeshIteration`."""
        mesh = self.mesh
        for first, second in self._take_queue():
            if not mesh.has_edge(first, second):
                continue
            self.counts["edges_visited"] += 1
            # The four vertices whose one-rings an operation can change, named *before* it
            # runs: after a collapse one of them is gone, and after a flip the edge itself is.
            opposite = [mesh.third_vertex(triangle, first, second)
                        for triangle in mesh.edge_triangles(first, second)]
            if self.process_edge(first, second) is EDGE_UNCHANGED:
                continue
            self._queue_one_ring(first)
            self._queue_one_ring(second)
            for vertex in opposite:
                self._queue_one_ring(vertex)
        self.counts["queued_passes"] += 1
        self._tracked_relax_pass()

    def _tracked_relax_pass(self):
        """`_relax_pass`, then the edges its moves took outside the length window.

        Vectorised over every edge rather than walked per moved vertex, which is g3Sharp's
        shape but not its loop. That enqueues a superset: an edge whose endpoints did not move
        but which was already outside the window goes in too, and gets refused again for
        whatever reason it was refused before. One `einsum` over the whole edge array is
        cheaper than tracking which vertices moved, and it cannot miss one.
        """
        self._relax_pass()
        keys = self.mesh.edges()
        if not keys:
            return
        edges = np.asarray(keys, dtype=np.int64).reshape((-1, 2))
        positions = self.mesh.positions
        gaps = positions[edges[:, 0]] - positions[edges[:, 1]]
        lengths = np.sqrt(np.einsum("ij,ij->i", gaps, gaps))
        outside = np.flatnonzero((lengths < self.minimum_edge_length)
                                 | (lengths > self.maximum_edge_length))
        queue = self._queue
        for index in outside:
            queue.add(keys[index])

    # --- what the caller runs ------------------------------------------------------------

    def remesh(self, iterations=DEFAULT_ITERATIONS, fast_splits=True, on_iteration=None):
        """Split hard to reach the target, then sweep the queue -- g3Sharp's `FastestRemesh`.

        The relax pass still runs exactly `iterations` times, as it does in `Remesher`: the
        split-only prelude does not smooth, so adding it does not add smoothing passes. That
        keeps the one part of the loop that sets the final triangle quality on the same
        budget it had, and leaves the refine sweep as the only thing that changed.

        There is no early exit when the queue empties. It almost never does -- the relax pass
        refills it every time, which is the point of tracking it -- and stopping early would
        cut smoothing passes the quality numbers were measured with.
        """
        iterations = int(iterations)
        self.reset_queue()
        if fast_splits and self.enable_splits:
            for _ in range(iterations):
                splits = self.fast_split_iteration()
                edges = len(self.mesh.edges())
                if not edges or splits / edges < FAST_SPLIT_FRACTION:
                    break
            self.reset_queue()
        # Only the sweep is reported, not the split prelude: it exits early on its own, so a
        # caller counting its passes would draw a progress bar that stops part way and then
        # restarts. `Remesher.remesh` explains what the callback is for.
        for index in range(iterations):
            self.remesh_iteration()
            if on_iteration is not None:
                on_iteration(index + 1, iterations)
        return self.counts


# --- what a host calls --------------------------------------------------------------------

def face_band_quality(points, triangles, groups):
    """Aspect ratios of the triangles that touch a face seam, and of the ones that do not.

    `metrics.triangle_quality` splits on the *open* boundary, which on a closed flow domain is
    empty -- so its `seam_*` numbers come back as `nan` for exactly the surface whose seams
    matter most. This is the same split taken over `ModelFaceID` boundaries instead, and it is
    the number a full-surface remesh is judged by: on case B's merged domain the band's
    99th percentile went 115.8 to 1.75 while the interior's worst went 935 to 8.6.
    """
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    incident = {}
    for index, triangle in enumerate(triangles):
        for corner in range(3):
            first, second = int(triangle[corner]), int(triangle[(corner + 1) % 3])
            incident.setdefault((min(first, second), max(first, second)), []).append(index)
    band = set()
    for edge, cells in incident.items():
        if len(cells) == 1 or groups[cells[0]] != groups[cells[1]]:
            band.update(edge)
    touches = np.asarray([any(int(vertex) in band for vertex in triangle)
                          for triangle in triangles])
    aspect = _aspect_ratios(points[triangles])

    def spread(values):
        if not len(values):
            return {"p99": float("nan"), "maximum": float("nan")}
        return {"p99": float(np.percentile(values, 99)), "maximum": float(values.max())}

    return {
        "band_triangles": int(touches.sum()),
        "band_aspect_p99": spread(aspect[touches])["p99"],
        "band_aspect_maximum": spread(aspect[touches])["maximum"],
        "off_band_aspect_p99": spread(aspect[~touches])["p99"],
        "off_band_aspect_maximum": spread(aspect[~touches])["maximum"],
    }


def _refuse_degenerate(points, triangles, label):
    cross = np.linalg.norm(_cross_norms(points, triangles), axis=1)
    worst = float(cross.min())
    if worst <= MINIMUM_TRIANGLE_CROSS:
        count = int((cross <= MINIMUM_TRIANGLE_CROSS).sum())
        raise RuntimeError(
            f"{label} left {count} degenerate triangles, worst cross-norm {worst:.3g}, at or "
            f"under the {MINIMUM_TRIANGLE_CROSS:.0e} TetGen and the publication gate both "
            "refuse a surface at. Nothing downstream can take this surface, so it is refused "
            "here rather than two steps later."
        )
    return worst


def remesh_labelled_surface(surface, target_edge_length, *, seam=SEAM_SLIDES,
                            iterations=DEFAULT_ITERATIONS,
                            smoothing_speed=DEFAULT_SMOOTHING_SPEED,
                            enable_smoothing=True,
                            corner_angle_degrees=DEFAULT_CORNER_ANGLE_DEGREES,
                            surface_target=None, queued=False, on_iteration=None):
    """Remesh a `ModelFaceID`-labelled surface to a uniform edge length, seams intact.

    This is the once-per-run remesh of the fully merged flow domain, and it is the only
    place face-awareness pays. Every face comes back because the labels are the mesh's own
    triangle groups rather than something looked up afterwards, and the seams between them
    are resampled at the target edge length while staying on their original curves.

    Note what it does *not* do: it is not a repair. A surface that crosses itself goes in
    and comes out crossing itself, which is why `require_no_self_intersections` runs first,
    and a surface it cannot remesh without degenerating a triangle is refused outright.

    It does, however, *read* a surface that is not already a manifold triangle mesh, because a
    clip result routinely is not: caps arrive as single polygons and folded and duplicate cells
    arrive with them. `conditioning.condition_surface` removes those before the mesh is built
    and reports every one under `record["conditioning"]` -- see that module for why removing
    them is not the repair the paragraph above refuses, and why it cannot be done by running
    VTK's Clean over a labelled surface first.

    `on_iteration(done, total)` is handed straight to the remesher's sweep. It reports and
    does nothing else -- see `Remesher.remesh` for why the sweep is the one place a host with
    an event loop needs a word.

    `queued` runs `QueuedRemesher` instead, which is 1.4 to 1.6x faster on a surface already
    near its target and worse on one that is not -- see that class for the measurements. It is
    off by default because this function does not know which it has been handed, and because
    the numbers every caller's docstring quotes were measured with the full sweep; turning it
    on changes the surface, so it is the caller's call and not this one's.
    """
    # Everything below reads the *conditioned* surface, including the before/after quality and
    # boundary-loop comparisons and the projection target. A cell that is not a triangle is not
    # part of the surface being remeshed, so it must not be part of what the pass is judged by
    # either -- and on a clean input `condition_surface` hands back the same object it was
    # given, so this costs one vectorised audit and changes nothing.
    surface, conditioning = condition_surface(surface)
    points = surfaces.surface_points(surface)
    triangles = surfaces.triangle_indices(surface)
    labels = surfaces.face_ids(surface)
    groups = labels if labels is not None else np.zeros(len(triangles), dtype=np.int64)
    before = triangle_quality(points, triangles)
    band_before = face_band_quality(points, triangles, groups)
    boundary_loops_before = len(open_boundary_curves(surface))

    mesh = DynamicMesh.from_arrays(points, triangles, groups)
    constraints = MeshConstraints()
    seam_record = constrain_seams(mesh, constraints, seam=seam,
                                  corner_angle_degrees=corner_angle_degrees)
    target = surface_target if surface_target is not None else SurfaceTarget(surface)
    remesher = (QueuedRemesher if queued else Remesher)(
        mesh, target_edge_length, constraints=constraints, surface_target=target,
        smoothing_speed=smoothing_speed, enable_smoothing=enable_smoothing)
    remesher.remesh(iterations, on_iteration=on_iteration)

    deviations = [0.0]
    for vertex, constraint in constraints.vertices.items():
        if constraint.target is None or not mesh.vertex_is_alive(vertex):
            continue
        deviations.append(float(constraint.target.deviation(
            mesh.position(vertex).reshape((1, 3)))[0]))

    new_points, new_triangles, new_groups, _ = mesh.compact()
    worst_cross = _refuse_degenerate(new_points, new_triangles, "The remesh")
    remeshed = surfaces.polydata_from_arrays(new_points, new_triangles)
    if labels is not None:
        surfaces.stamp_face_ids(remeshed, new_groups,
                                surface.GetCellData().GetArray("ModelFaceID"))
    missing = sorted(set(groups.tolist()) - set(new_groups.tolist()))
    if missing:
        raise RuntimeError(
            f"The remesh lost ModelFaceID values {missing}, which every downstream stage "
            "extracts faces by. A face thin enough to be remeshed away is the usual cause; "
            "the target edge length is the thing to raise."
        )
    boundary_loops_after = len(open_boundary_curves(remeshed))
    if boundary_loops_after != boundary_loops_before:
        raise RuntimeError(
            f"The remesh took the surface from {boundary_loops_before} open boundary loops to "
            f"{boundary_loops_after}, so it tore or closed something. This is the failure "
            "`AGENTS.md` records against remeshing the flow domain, and it is refused rather "
            "than handed to TetGen."
            + (f" This surface also arrived carrying {conditioning['duplicate_points']} points "
               "sharing a position with another point exactly, which is a slit rather than a "
               "seam: the two sides of it are separate surface that the remesh is free to pull "
               "apart. Merging coincident points joins two sheets and so is not something "
               "conditioning will do behind you -- run Slicer's Surface Toolbox clean, or "
               "vtkCleanPolyData at tolerance 0, on the surface *before* it was labelled."
               if conditioning["duplicate_points"] else "")
        )
    after = triangle_quality(new_points, new_triangles)
    record = {
        "target_edge_length": float(target_edge_length),
        "iterations": int(iterations),
        "conditioning": conditioning,
        "points_before": int(len(points)),
        "points_after": int(len(new_points)),
        "triangles_before": int(len(triangles)),
        "triangles_after": int(len(new_triangles)),
        "seam_deviation": float(max(deviations)),
        "worst_cross_norm": worst_cross,
        "operations": dict(remesher.counts),
        "before": before,
        "after": after,
        "band_before": band_before,
        "band_after": face_band_quality(new_points, new_triangles, new_groups),
        **seam_record,
    }
    return {"surface": remeshed, "record": record}


# --- remeshing a deformed patch's interior, with the seam pinned --------------------------

# How far a seam vertex is allowed to have drifted before it is snapped back. Pinned vertices
# do not move at all, so this is not a tolerance being spent -- it is the check that the
# match found the right partner, and anything above it means the seam was resampled and the
# merge with the surrounding wall would no longer be watertight.
SEAM_DRIFT_TOLERANCE = 1e-6
# When a remesh is refused for having made the interior worse. There is a measurable way to do
# exactly that: ask for an edge length much finer than the pinned seam's own edges, and the
# ring bridging the two comes out as slivers.
#
# Both conditions have to hold, and the second is why: a *relative* test alone refuses a healthy
# remesh of an already-regular mesh, where 1.24 -> 1.58 is a real ratio and no kind of problem.
# 2.5 distinguishes a materially bad interior from ordinary variation in a healthy remesh.
MAXIMUM_INTERIOR_ASPECT_WORSENING = 1.25
ACCEPTABLE_INTERIOR_ASPECT_RATIO = 2.5
# The same pair for the band of triangles touching the pinned seam. It needs its own numbers
# because that band arrives bad -- the contour's discretisation is what sets it, and no remesh
# that pins the seam can improve it -- so the question is only whether the remesh made it very
# much worse. Measured on the patch as a run builds it: pinning against a target 0.7 of the
# seam's own edge length took the band from 50.6 to 21728, and against the patch's own edge
# length it does not move at all.
MAXIMUM_SEAM_ASPECT_WORSENING = 3.0
# The absolute floor the seam-band test needs beside its ratio, for the same reason the
# interior test has one: on an already-regular patch the band's ratio is a real number and no
# kind of problem. `metrics.BAD_ASPECT_RATIO` is where this repo already draws the line between
# a triangle worth reporting and one worth ignoring, so it is the same number.
BAD_SEAM_ASPECT_RATIO = 10.0


def _barycentric(point, corners):
    """Barycentric weights of a point on a triangle, clamped to the triangle.

    Written out rather than taken from `vtkTriangle` so the weights are guaranteed to be a
    partition of unity even for the near-degenerate seam triangles a cut contour carries,
    where the VTK helper's answer for a point off the triangle plane is not.
    """
    first = corners[1] - corners[0]
    second = corners[2] - corners[0]
    offset = point - corners[0]
    a, b, d = first @ first, first @ second, second @ second
    determinant = a * d - b * b
    if abs(determinant) <= 1e-30:
        return np.array([1.0, 0.0, 0.0])
    u = (d * (offset @ first) - b * (offset @ second)) / determinant
    v = (a * (offset @ second) - b * (offset @ first)) / determinant
    weights = np.clip(np.array([1.0 - u - v, u, v]), 0.0, 1.0)
    return weights / max(weights.sum(), 1e-30)


def _pull_back(reference, points):
    """Sample the reference surface's per-vertex values at each new vertex.

    Returned as a callable so the caller can carry any per-vertex quantity across the remesh --
    the session uses it for the origin positions `total_displacement` measures against, which
    would otherwise be indexed by a vertex numbering that no longer exists. Barycentric on the
    closest triangle rather than nearest-vertex, because the new vertices sit *between* the old
    ones by construction and a nearest-vertex map would quantise them onto the old ones.
    """
    reference_points = surfaces.surface_points(reference)
    reference_triangles = surfaces.triangle_indices(reference)
    locator = vtk.vtkCellLocator()
    locator.SetDataSet(reference)
    locator.BuildLocator()
    weights = np.zeros((len(points), 3), dtype=float)
    corners = np.zeros((len(points), 3), dtype=np.int64)
    closest = [0.0, 0.0, 0.0]
    cell_id = vtk.reference(0)
    sub_id = vtk.reference(0)
    squared_distance = vtk.reference(0.0)
    generic_cell = vtk.vtkGenericCell()
    for index, point in enumerate(points):
        locator.FindClosestPoint(
            [float(point[0]), float(point[1]), float(point[2])],
            closest, generic_cell, cell_id, sub_id, squared_distance)
        triangle = reference_triangles[int(cell_id)]
        corners[index] = triangle
        weights[index] = _barycentric(np.asarray(closest, dtype=float),
                                      reference_points[triangle])

    def sample(values):
        values = np.asarray(values, dtype=float)
        return np.einsum("ij,ij...->i...", weights, values[corners])

    return sample


def _seam_median_edge(points, triangles, seam_ids):
    """Median edge length over the triangles that touch the seam -- the pinned band's own.

    The interior cannot be remeshed much finer than this without the ring that bridges the two
    coming out as slivers, so it is what a refused target gets compared against.
    """
    seam = set(int(vertex) for vertex in seam_ids)
    lengths = []
    for triangle in triangles:
        if not any(int(vertex) in seam for vertex in triangle):
            continue
        for corner in range(3):
            lengths.append(float(np.linalg.norm(
                points[triangle[(corner + 1) % 3]] - points[triangle[corner]])))
    return float(np.median(lengths)) if lengths else float("nan")


def _matched_seam(remeshed_points, remeshed_seam, seam_points):
    """Pair every remeshed boundary vertex with the original seam vertex it came from.

    Nearest-neighbour, then checked for being one-to-one. A remesher that resampled the seam
    would show up here as either the wrong count or two boundary vertices claiming one original,
    and both have to be refusals: the watertight merge with the surrounding wall is keyed on
    these vertices matching the wall's exactly.
    """
    if len(remeshed_seam) != len(seam_points):
        raise RuntimeError(
            f"The remesh took the seam from {len(seam_points)} vertices to "
            f"{len(remeshed_seam)}; it has to leave the attachment contour alone."
        )
    distances = np.linalg.norm(
        remeshed_points[remeshed_seam][:, None, :] - seam_points[None, :, :], axis=2)
    partner = np.argmin(distances, axis=1)
    if len(set(partner.tolist())) != len(partner):
        raise RuntimeError(
            "Two remeshed boundary vertices matched the same original seam vertex, so the seam "
            "was resampled rather than held."
        )
    drift = float(distances[np.arange(len(partner)), partner].max())
    if drift > SEAM_DRIFT_TOLERANCE:
        raise RuntimeError(
            f"The remesh moved the seam by {drift:.3g}, above the {SEAM_DRIFT_TOLERANCE:.0e} "
            "the watertight merge with the surrounding wall allows."
        )
    return partner, drift


def remesh_patch_interior(patch, target_edge_length, *, iterations=DEFAULT_ITERATIONS,
                          smoothing_speed=DEFAULT_SMOOTHING_SPEED,
                          next_global_node_id=None, on_iteration=None):
    """Re-triangulate a deformed patch's interior to a uniform edge length, seam untouched.

    The rebase's mode: the same remesher as `remesh_labelled_surface`, with the patch's open
    boundary held as a pinned seam rather than a sliding one. Nothing is cut and nothing is
    stitched, and the seam vertices come back bit-identical -- they are snapped and verified
    anyway, as `condition_patch_interior` does for the same reason.

    Pinning is the right mode here and sliding is not available, for a reason worth stating
    because it looks like a missing option. A design session's patch is a *face of a merged
    surface*, and the merge with the surrounding wall welds vertices that coincide, so a seam
    resampled off the wall's vertices leaves the flow domain open. That is also why the
    interior remesh belongs after the merge and not before it: on the far side of the merge
    the same curve is an inter-face seam that both sides move together, which is what
    `remesh_labelled_surface` slides.

    What pinning cannot fix, and no seam-preserving remesh can, is the seam band itself.
    `triangle_quality`'s `seam_*` numbers are reported so it stays visible and gets fixed
    where it is made, upstream in the contour.

    Returns a dict carrying the remeshed `patch`, a `pull_back` callable that samples any
    per-vertex array of the input at the new vertices, and a record of what the remesh did.
    """
    points = surfaces.surface_points(patch)
    triangles = surfaces.triangle_indices(patch)
    seam_ids = surfaces.boundary_point_ids(triangles)
    loops = len(open_boundary_curves(patch))
    if loops != 1:
        raise RuntimeError(
            f"The patch has {loops} boundary loops, expected one to hold. A hole in it would "
            "leave the flow domain open after the merge, so this is not something to remesh past."
        )
    before = triangle_quality(points, triangles)
    edge_length = float(target_edge_length)
    if not np.isfinite(edge_length) or edge_length <= 0.0:
        raise ValueError("The target edge length must be a finite positive number.")

    labels = surfaces.face_ids(patch)
    groups = labels if labels is not None else np.zeros(len(triangles), dtype=np.int64)
    mesh = DynamicMesh.from_arrays(points, triangles, groups)
    constraints = MeshConstraints()
    constrain_seams(mesh, constraints, seam=SEAM_PINNED)
    remesher = Remesher(mesh, edge_length, constraints=constraints,
                        surface_target=SurfaceTarget(patch),
                        smoothing_speed=smoothing_speed)
    remesher.remesh(iterations, on_iteration=on_iteration)
    new_points, new_triangles, new_groups, _ = mesh.compact()
    _refuse_degenerate(new_points, new_triangles, "The interior remesh")

    new_seam = surfaces.boundary_point_ids(new_triangles)
    partner, drift = _matched_seam(new_points, new_seam, points[seam_ids])
    new_points[new_seam] = points[seam_ids][partner]

    surface = surfaces.polydata_from_arrays(new_points, new_triangles)
    if len(open_boundary_curves(surface)) != 1:
        raise RuntimeError(
            "The remesh left the patch with more than one boundary loop, which the merge with "
            "the surrounding wall cannot take."
        )

    # Face ids first: the merge extracts the patch by `ModelFaceID`, so a remeshed patch that
    # lost it would vanish from the surface it is supposed to be part of.
    if labels is not None:
        surfaces.stamp_face_ids(surface, new_groups, patch.GetCellData().GetArray("ModelFaceID"))

    # `GlobalNodeID` is what the harmonic area measurement is keyed on. The seam vertices keep
    # the ids they share with the wall; every interior vertex is numbered afresh from
    # `next_global_node_id`, which the caller takes from the *merged* surface so a fresh id
    # cannot collide with one of the wall's. Falling back to the patch's own maximum is only
    # right for a standalone patch, so it says so.
    original_ids = patch.GetPointData().GetArray("GlobalNodeID")
    identifiers = None
    first_new = None
    if original_ids is not None and original_ids.GetNumberOfTuples() == len(points):
        existing = np.asarray([int(original_ids.GetTuple1(index))
                               for index in range(len(points))], dtype=np.int64)
        first_new = (int(next_global_node_id) if next_global_node_id is not None
                     else int(existing.max()) + 1)
        identifiers = np.arange(first_new, first_new + len(new_points), dtype=np.int64)
        identifiers[new_seam] = existing[seam_ids][partner]
        if len(set(identifiers.tolist())) != len(identifiers):
            raise RuntimeError(
                f"Numbering the remeshed patch from {first_new} collided with an id the seam "
                "already carries; take the first new id from the merged surface's maximum."
            )
        # The same VTK array *type* as the patch arrived with, not merely the same name.
        # `vtkAppendPolyData` drops an array whose type differs between its inputs, so a
        # `vtkIntArray` here against the wall's `vtkIdTypeArray` would leave the merged surface
        # with no `GlobalNodeID` at all -- and the harmonic measurement, which is keyed on it,
        # refusing the rebased shape.
        array = original_ids.NewInstance()
        array.SetName("GlobalNodeID")
        array.SetNumberOfComponents(1)
        array.SetNumberOfTuples(len(identifiers))
        for index, value in enumerate(identifiers):
            array.SetTuple1(index, int(value))
        surface.GetPointData().AddArray(array)

    after = triangle_quality(new_points, new_triangles)
    seam_edge = _seam_median_edge(points, triangles, seam_ids)
    if (after["interior_aspect_maximum"]
            > MAXIMUM_INTERIOR_ASPECT_WORSENING * before["interior_aspect_maximum"]
            and after["interior_aspect_maximum"] > ACCEPTABLE_INTERIOR_ASPECT_RATIO):
        raise RuntimeError(
            f"The remesh took the interior's worst aspect ratio from "
            f"{before['interior_aspect_maximum']:.2f} to "
            f"{after['interior_aspect_maximum']:.2f}, so it made the mesh worse rather than "
            f"better. A target edge length of {edge_length:.3g} against a pinned seam whose own "
            f"edges are a median {seam_edge:.3g} is the usual cause: the seam is fixed, and "
            "the ring bridging it to a much finer interior comes out as slivers."
        )
    if (after["seam_aspect_maximum"]
            > MAXIMUM_SEAM_ASPECT_WORSENING * before["seam_aspect_maximum"]
            and after["seam_aspect_maximum"] > BAD_SEAM_ASPECT_RATIO):
        raise RuntimeError(
            f"The remesh took the seam band's worst aspect ratio from "
            f"{before['seam_aspect_maximum']:.2f} to {after['seam_aspect_maximum']:.2f}. That "
            f"band is pinned, so a remesh cannot improve it and must not wreck it: a target "
            f"edge length of {edge_length:.3g} against a pinned seam whose own edges are a "
            f"median {seam_edge:.3g} is the usual cause, and the target is the thing to raise."
        )
    record = {
        "target_edge_length": edge_length,
        "seam_median_edge": seam_edge,
        "iterations": int(iterations),
        "points_before": int(len(points)),
        "points_after": int(len(new_points)),
        "seam_vertices": int(len(seam_ids)),
        "seam_drift": drift,
        "first_new_global_node_id": first_new,
        "renumbered": identifiers is not None,
        "operations": dict(remesher.counts),
        "before": before,
        "after": after,
    }
    return {"patch": surface, "pull_back": _pull_back(patch, new_points), "record": record}
