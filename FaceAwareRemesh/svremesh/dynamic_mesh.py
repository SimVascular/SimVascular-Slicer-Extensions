"""A triangle mesh that can be split, collapsed and flipped in place.

This is the port of geometry3Sharp's `DMesh3`, cut down to what an incremental
remesher needs: positions, triangles, per-triangle groups, and the three
topological operations. `ModelFaceID` is carried as the triangle group, which is
the whole reason this exists -- a remesher that owns its own connectivity keeps
face labels by construction rather than by a proximity lookup after the fact.

Where it deliberately differs from `DMesh3`: g3Sharp stores an explicit edge
list, and every operation has to rewrite the edge records of the triangles it
touched by hand. Here edges live in a dictionary keyed by the vertex pair, and
each operation unregisters the triangles it is about to change and re-registers
what it produced. That is a few dictionary operations per call instead of a page
of index bookkeeping, and it buys something the remesher wants anyway: a
constraint attached to the *vertex pair* survives a split without being
renumbered, because the two children of edge `(a, b)` are `(a, f)` and `(f, b)`
and the caller can name them without asking the mesh for new edge ids.

One thing the dictionary does *not* give away for free, and it is stored back:
how many boundary edges a vertex is on. g3Sharp's explicit edge records answer
`vertex_is_boundary` and `vertex_valence` in constant time, and answering them by
walking the one-ring instead was 28% of a remesh -- the remesher asks millions of
times, because the flip rule weighs the valence of four vertices per candidate
edge and refuses most of them. So `_boundary_edge_count` is maintained by
`_register` and `_unregister`, which already visit exactly the edges whose
triangle count crosses one, and both queries are lookups again. It is derived
state, so it can drift; `_boundary_edges_agree` and `_valences_agree` recompute
it from scratch and are what the tests hold it to.

Deleted slots are kept rather than compacted, so ids stay stable while a remesh
runs. `compact()` is what produces the arrays a `vtkPolyData` is built from.

Only manifold, consistently wound triangle soup is supported. An edge with three
or more triangles is refused at construction: every operation below assumes one
or two, so a caller handing this a surface it has not checked gets the refusal
here rather than a wrong answer later.
"""

from __future__ import annotations

import numpy as np

INVALID = -1


def _key(first, second):
    """A canonical dictionary key for the undirected edge between two vertices."""
    return (first, second) if first < second else (second, first)


class NonManifoldMeshError(RuntimeError):
    """An edge carries more than two triangles, which no operation here can handle."""


class DynamicMesh:
    """A triangle mesh with edge split, collapse and flip.

    Vertices and triangles are addressed by id. Ids are never reused within a
    remesh, and a deleted vertex reads as dead through `vertex_is_alive` while a
    deleted triangle simply drops out of `alive_triangles`, so a caller iterating
    a snapshot of ids has to check.
    """

    def __init__(self):
        # Grown by doubling rather than by `vstack`, because a remesh splits thousands of
        # edges and reallocating the whole array on each one is the difference between
        # linear and quadratic.
        self._positions = np.zeros((16, 3), dtype=float)
        self._vertex_alive = []
        self._vertex_triangles = []
        self._triangles = []
        self._triangle_groups = []
        self._edge_triangles = {}
        # How many boundary edges -- edges carrying one triangle -- each vertex is on. This is
        # derived state, and the only reason it is stored rather than computed is that the
        # remesher asks for it constantly: `vertex_is_boundary` was 28% of a remesh when it
        # walked the one-ring to answer, because `_valence_error` asks it eight times per flip
        # attempt and most flip attempts are rejected. Maintained in `_register` and
        # `_unregister`, which already visit exactly the edges whose triangle count changes.
        # `_boundary_edges_agree` recomputes it from scratch and is what the tests check.
        self._boundary_edge_count = []

    # --- construction and export ---------------------------------------------------------

    @classmethod
    def from_arrays(cls, points, triangles, groups=None):
        """Build a mesh from a point array, a triangle index array and optional groups."""
        mesh = cls()
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points must be an (n, 3) array.")
        triangles = np.asarray(triangles, dtype=np.int64)
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("Triangles must be an (m, 3) array.")
        if groups is None:
            groups = np.zeros(len(triangles), dtype=np.int64)
        groups = np.asarray(groups, dtype=np.int64)
        if len(groups) != len(triangles):
            raise ValueError("There must be one group per triangle.")

        mesh._positions = np.zeros((max(16, 2 * len(points)), 3), dtype=float)
        mesh._positions[:len(points)] = points
        mesh._vertex_alive = [True] * len(points)
        mesh._vertex_triangles = [set() for _ in range(len(points))]
        mesh._boundary_edge_count = [0] * len(points)
        for triangle, group in zip(triangles, groups):
            mesh._append_triangle(int(triangle[0]), int(triangle[1]), int(triangle[2]),
                                  int(group))
        return mesh

    def compact(self):
        """Return `(points, triangles, groups, vertex_ids)` with the dead slots dropped.

        `vertex_ids` is the mesh id each row of `points` came from, which is what lets a
        caller carry a per-vertex array across the remesh for the vertices that survived
        it untouched.
        """
        alive_vertices = [index for index, alive in enumerate(self._vertex_alive) if alive]
        renumber = np.full(len(self._vertex_alive), INVALID, dtype=np.int64)
        renumber[alive_vertices] = np.arange(len(alive_vertices))
        triangles = []
        groups = []
        for triangle, group in zip(self._triangles, self._triangle_groups):
            if triangle is None:
                continue
            triangles.append([int(renumber[vertex]) for vertex in triangle])
            groups.append(int(group))
        return (
            self._positions[alive_vertices].copy(),
            np.asarray(triangles, dtype=np.int64).reshape((-1, 3)),
            np.asarray(groups, dtype=np.int64),
            np.asarray(alive_vertices, dtype=np.int64),
        )

    # --- state ---------------------------------------------------------------------------

    @property
    def positions(self):
        """A view of the positions, indexed by vertex id, dead slots included.

        A view rather than a copy so a vectorised smoothing pass can write straight into
        it; taken fresh each time, because splitting an edge may have reallocated.
        """
        return self._positions[:len(self._vertex_alive)]

    def position(self, vertex):
        return self._positions[vertex]

    def set_position(self, vertex, point):
        self._positions[vertex] = point

    def set_positions(self, points):
        """Overwrite every position at once, dead slots included."""
        points = np.asarray(points, dtype=float)
        if points.shape != (len(self._vertex_alive), 3):
            raise ValueError("The position array must cover every vertex slot.")
        self._positions[:len(self._vertex_alive)] = points

    def vertex_is_alive(self, vertex):
        return bool(self._vertex_alive[vertex])

    @property
    def vertex_slots(self):
        return len(self._vertex_alive)

    def alive_vertices(self):
        return [index for index, alive in enumerate(self._vertex_alive) if alive]

    def alive_triangles(self):
        return [index for index, triangle in enumerate(self._triangles) if triangle is not None]

    def triangle(self, triangle):
        return self._triangles[triangle]

    def triangle_group(self, triangle):
        return self._triangle_groups[triangle]

    def vertex_triangles(self, vertex):
        return self._vertex_triangles[vertex]

    def edges(self):
        """Every live edge, as a canonical vertex pair."""
        return list(self._edge_triangles.keys())

    # `_key` is inlined in the three edge lookups below rather than called. They are the
    # mesh's hottest methods -- a few million calls in a remesh between them -- and at that
    # count the function call costs more than the comparison it wraps.

    def edge_triangles(self, first, second):
        return self._edge_triangles.get(
            (first, second) if first < second else (second, first), ())

    def has_edge(self, first, second):
        return ((first, second) if first < second else (second, first)) in self._edge_triangles

    def edge_is_boundary(self, first, second):
        neighbours = self._edge_triangles.get(
            (first, second) if first < second else (second, first))
        return neighbours is not None and len(neighbours) == 1

    def vertex_is_boundary(self, vertex):
        """Whether the vertex is on the surface's own boundary.

        A lookup rather than a walk of the one-ring: `_boundary_edge_count` is maintained by
        `_register` and `_unregister`, and a vertex is on the boundary exactly when it is on
        at least one boundary edge.
        """
        return self._boundary_edge_count[vertex] > 0

    def vertex_one_ring(self, vertex):
        """The vertices sharing an edge with this one."""
        ring = set()
        triangles = self._triangles
        for triangle in self._vertex_triangles[vertex]:
            first, second, third = triangles[triangle]
            ring.add(first)
            ring.add(second)
            ring.add(third)
        ring.discard(vertex)
        return ring

    def vertex_valence(self, vertex):
        """How many vertices share an edge with this one.

        Counted rather than enumerated. On a manifold surface the triangles around a vertex
        form one fan, so an interior vertex has as many neighbours as triangles and a
        boundary vertex has one more -- the far end of the second boundary edge, which no
        triangle closes back to. `_valences_agree` checks that against
        `len(vertex_one_ring(...))` for every vertex, which is what the tests assert.
        """
        return len(self._vertex_triangles[vertex]) + (
            1 if self._boundary_edge_count[vertex] else 0)

    # --- checks on the derived state, for the tests ---------------------------------------

    def _boundary_edges_agree(self):
        """Whether `_boundary_edge_count` matches a count taken from scratch.

        Incremental bookkeeping drifts, and a missed transition in `_register` or
        `_unregister` would show up as a wrong answer from `vertex_is_boundary` long after
        the operation that caused it. This is the from-scratch definition, kept here so the
        tests can assert the two agree after a run of operations.
        """
        expected = [0] * len(self._vertex_alive)
        for (tail, head), neighbours in self._edge_triangles.items():
            if len(neighbours) == 1:
                expected[tail] += 1
                expected[head] += 1
        return list(self._boundary_edge_count) == expected

    def _valences_agree(self):
        """Whether the counted valence matches the enumerated one-ring, vertex for vertex."""
        return all(self.vertex_valence(vertex) == len(self.vertex_one_ring(vertex))
                   for vertex, alive in enumerate(self._vertex_alive) if alive
                   and self._vertex_triangles[vertex])

    def third_vertex(self, triangle, first, second):
        """The corner of a triangle that is neither of the two given."""
        for corner in self._triangles[triangle]:
            if corner != first and corner != second:
                return corner
        return INVALID

    def opposite_vertices(self, first, second):
        """The apexes of the one or two triangles on an edge."""
        return [self.third_vertex(triangle, first, second)
                for triangle in self._edge_triangles[_key(first, second)]]

    # --- registration --------------------------------------------------------------------

    def _append_triangle(self, first, second, third, group):
        triangle = len(self._triangles)
        self._triangles.append([first, second, third])
        self._triangle_groups.append(int(group))
        self._register(triangle)
        return triangle

    def _register(self, triangle):
        """Add a triangle to the vertex and edge indices.

        The locals below are bound once rather than looked up per corner, and `_key` is
        inlined, because between them these two methods run a few hundred thousand times in
        a remesh. The one-triangle and two-triangle branches are where
        `_boundary_edge_count` is maintained: an edge that has just reached one triangle is
        a boundary edge, and one that has just reached two is not.
        """
        corners = self._triangles[triangle]
        first, second, third = corners
        if first == second or second == third or first == third:
            raise ValueError(f"Triangle {triangle} repeats a vertex: {corners}.")
        vertex_triangles = self._vertex_triangles
        vertex_triangles[first].add(triangle)
        vertex_triangles[second].add(triangle)
        vertex_triangles[third].add(triangle)
        edge_triangles = self._edge_triangles
        boundary_count = self._boundary_edge_count
        for tail, head in ((first, second), (second, third), (third, first)):
            key = (tail, head) if tail < head else (head, tail)
            neighbours = edge_triangles.get(key)
            if neighbours is None:
                edge_triangles[key] = [triangle]
                boundary_count[tail] += 1
                boundary_count[head] += 1
                continue
            neighbours.append(triangle)
            if len(neighbours) == 2:
                boundary_count[tail] -= 1
                boundary_count[head] -= 1
            else:
                raise NonManifoldMeshError(
                    f"Edge {key} carries {len(neighbours)} triangles; this mesh only "
                    "handles manifold surfaces."
                )

    def _unregister(self, triangle):
        corners = self._triangles[triangle]
        first, second, third = corners
        vertex_triangles = self._vertex_triangles
        vertex_triangles[first].discard(triangle)
        vertex_triangles[second].discard(triangle)
        vertex_triangles[third].discard(triangle)
        edge_triangles = self._edge_triangles
        boundary_count = self._boundary_edge_count
        for tail, head in ((first, second), (second, third), (third, first)):
            key = (tail, head) if tail < head else (head, tail)
            neighbours = edge_triangles.get(key)
            if neighbours is None:
                continue
            neighbours.remove(triangle)
            if not neighbours:
                # The edge is gone, so it is no longer a boundary edge of either endpoint.
                del edge_triangles[key]
                boundary_count[tail] -= 1
                boundary_count[head] -= 1
            elif len(neighbours) == 1:
                boundary_count[tail] += 1
                boundary_count[head] += 1

    def _add_vertex(self, point):
        vertex = len(self._vertex_alive)
        if vertex >= len(self._positions):
            grown = np.zeros((2 * len(self._positions), 3), dtype=float)
            grown[:len(self._positions)] = self._positions
            self._positions = grown
        self._positions[vertex] = np.asarray(point, dtype=float)
        self._vertex_alive.append(True)
        self._vertex_triangles.append(set())
        self._boundary_edge_count.append(0)
        return vertex

    # --- split ---------------------------------------------------------------------------

    def split_edge(self, first, second, point):
        """Insert a vertex on an edge, splitting the one or two triangles on it.

        Returns `(new_vertex, [child_edges])`, where the child edges are the two halves of
        the edge that was split, in the order `(first, new)` then `(new, second)`. The
        caller needs them by name because an edge constraint is inherited by both halves.
        """
        key = _key(first, second)
        neighbours = list(self._edge_triangles[key])
        new_vertex = self._add_vertex(point)
        for triangle in neighbours:
            corners = list(self._triangles[triangle])
            index = next(corner for corner in range(3)
                         if {corners[corner], corners[(corner + 1) % 3]} == {first, second})
            tail = corners[index]
            head = corners[(index + 1) % 3]
            apex = corners[(index + 2) % 3]
            group = self._triangle_groups[triangle]
            self._unregister(triangle)
            self._triangles[triangle] = [tail, new_vertex, apex]
            self._register(triangle)
            self._append_triangle(new_vertex, head, apex, group)
        return new_vertex, [_key(first, new_vertex), _key(new_vertex, second)]

    # --- collapse ------------------------------------------------------------------------

    def collapse_would_be_valid(self, keep, remove):
        """Whether collapsing `remove` into `keep` leaves a manifold surface.

        The link condition is the substance of it: the one-rings of the two vertices may
        only meet at the apexes of the edge itself. Anything else and the collapse either
        folds two distinct parts of the surface onto each other or leaves a non-manifold
        edge behind, and the check is cheap next to finding that out downstream.

        The boundary rule covers two failures at once. A boundary vertex disappearing
        down an *interior* edge drags the boundary into the interior, and an interior edge
        joining two boundary vertices pinches the surface where they meet; refusing to
        remove a boundary vertex except along a boundary edge rules out both.
        """
        key = _key(keep, remove)
        neighbours = self._edge_triangles.get(key)
        if neighbours is None:
            return False
        if len(neighbours) not in (1, 2):
            return False
        apexes = {self.third_vertex(triangle, keep, remove) for triangle in neighbours}
        shared = self.vertex_one_ring(keep) & self.vertex_one_ring(remove)
        if shared != apexes:
            return False
        edge_is_boundary = len(neighbours) == 1
        if not edge_is_boundary and self.vertex_is_boundary(remove):
            return False
        # An isolated triangle -- one whose three edges are all boundary -- collapses to a
        # single edge and then to nothing, so it is refused rather than deleted silently.
        if edge_is_boundary:
            apex = next(iter(apexes))
            if (self.edge_is_boundary(keep, apex)
                    and self.edge_is_boundary(remove, apex)):
                return False
        return True

    def collapse_edge(self, keep, remove):
        """Collapse `remove` into `keep`. The caller checks validity first."""
        key = _key(keep, remove)
        for triangle in list(self._edge_triangles[key]):
            self._unregister(triangle)
            self._triangles[triangle] = None
        for triangle in list(self._vertex_triangles[remove]):
            corners = list(self._triangles[triangle])
            self._unregister(triangle)
            self._triangles[triangle] = [keep if corner == remove else corner
                                         for corner in corners]
            self._register(triangle)
        self._vertex_triangles[remove] = set()
        self._vertex_alive[remove] = False

    # --- flip ----------------------------------------------------------------------------

    def flip_would_be_valid(self, first, second):
        """Whether the interior edge between two triangles can be flipped to their apexes.

        Refused for a boundary edge, for an edge whose two triangles carry different
        groups -- flipping one would move a face label's own boundary, which is the thing
        this mesh exists to hold -- for an apex pair that is already joined, and for an
        endpoint of valence three in the interior, which the flip would strand.
        """
        key = _key(first, second)
        neighbours = self._edge_triangles.get(key)
        if neighbours is None or len(neighbours) != 2:
            return False
        if self._triangle_groups[neighbours[0]] != self._triangle_groups[neighbours[1]]:
            return False
        left = self.third_vertex(neighbours[0], first, second)
        right = self.third_vertex(neighbours[1], first, second)
        if left == right or self.has_edge(left, right):
            return False
        for endpoint in (first, second):
            if not self.vertex_is_boundary(endpoint) and self.vertex_valence(endpoint) <= 3:
                return False
        return True

    def flipped_triangles(self, first, second):
        """The two triangles a flip would produce, as vertex triples."""
        neighbours = self._edge_triangles[_key(first, second)]
        tail, head, left, right = self._flip_corners(neighbours, first, second)
        return [[tail, right, left], [right, head, left]]

    def _flip_corners(self, neighbours, first, second):
        """Orient the edge with the triangle that walks it forwards, and name the apexes.

        `tail -> head` is the edge as the first triangle winds it, `left` that triangle's
        apex and `right` the other's, so the quadrilateral is `tail, right, head, left`
        and the flipped diagonal joins `left` to `right`.
        """
        corners = self._triangles[neighbours[0]]
        index = next(corner for corner in range(3)
                     if {corners[corner], corners[(corner + 1) % 3]} == {first, second})
        tail = corners[index]
        head = corners[(index + 1) % 3]
        left = corners[(index + 2) % 3]
        right = self.third_vertex(neighbours[1], first, second)
        return tail, head, left, right

    def flip_edge(self, first, second):
        """Flip an interior edge onto the apexes of its two triangles."""
        key = _key(first, second)
        neighbours = list(self._edge_triangles[key])
        tail, head, left, right = self._flip_corners(neighbours, first, second)
        groups = [self._triangle_groups[triangle] for triangle in neighbours]
        for triangle in neighbours:
            self._unregister(triangle)
        self._triangles[neighbours[0]] = [tail, right, left]
        self._triangles[neighbours[1]] = [right, head, left]
        for triangle, group in zip(neighbours, groups):
            self._triangle_groups[triangle] = group
            self._register(triangle)
        return _key(left, right)
