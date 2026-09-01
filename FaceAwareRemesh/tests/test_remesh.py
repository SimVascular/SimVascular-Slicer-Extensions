"""The ported constrained remesher, and the dynamic mesh it runs on.

These are the tests for `svremesh/dynamic_mesh.py` and `svremesh/remesh.py` as algorithms.
Run them from this directory with the package importable:

    python -m unittest discover -s tests

Almost everything here is stated as an invariant of the *surface*, not of the code:
manifoldness, group survival, seam deviation, and the cross-norm no triangle may go under.
That is deliberate. The three topological operations are the part of a remesher that goes
subtly wrong -- a collapse that pinches a boundary, a flip that folds a triangle over -- and
the failures do not look like exceptions, they look like a surface a volume mesher refuses an
hour later. So the checks are made here, where the cause is still visible.
"""

import numpy.fft  # noqa: F401  - numpy must be imported before VTK

import math
import unittest
from collections import Counter

import numpy as np
import vtk

import svremesh
from svremesh import remesh, surfaces
from svremesh.dynamic_mesh import DynamicMesh, NonManifoldMeshError
from svremesh.quality import triangle_quality


def median_edge_length(surface):
    """The surface's median triangle edge length."""
    points = surfaces.surface_points(surface)
    triangles = surfaces.triangle_indices(surface)
    lengths = np.concatenate([
        np.linalg.norm(points[triangles[:, (corner + 1) % 3]] - points[triangles[:, corner]],
                       axis=1)
        for corner in range(3)
    ])
    return float(np.median(lengths))


def triangulated_disc(radius_mm, edge_mm):
    """Flat triangulated disc with roughly uniform edge length.

    Concentric rings spaced by `edge_mm`, staggered so the triangles come out near
    equilateral, then Delaunay in the plane. The point set is convex, so the
    triangulation's boundary is exactly the outer ring -- which is what gets pinned.
    """
    points = vtk.vtkPoints()
    points.SetDataTypeToDouble()
    points.InsertNextPoint(0.0, 0.0, 0.0)
    rings = max(int(round(radius_mm / edge_mm)), 2)
    for ring in range(1, rings + 1):
        radius = radius_mm * ring / rings
        count = max(int(round(2.0 * math.pi * radius / edge_mm)), 6)
        offset = 0.5 * (ring % 2) * 2.0 * math.pi / count
        for index in range(count):
            angle = offset + 2.0 * math.pi * index / count
            points.InsertNextPoint(radius * math.cos(angle),
                                   radius * math.sin(angle), 0.0)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    delaunay = vtk.vtkDelaunay2D()
    delaunay.SetInputData(polydata)
    delaunay.Update()
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(delaunay.GetOutputPort())
    triangles.Update()
    surface = vtk.vtkPolyData()
    surface.DeepCopy(triangles.GetOutput())
    return surface


def sphere(theta=16, phi=16, radius=10.0):
    source = vtk.vtkSphereSource()
    source.SetThetaResolution(theta)
    source.SetPhiResolution(phi)
    source.SetRadius(radius)
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(source.GetOutputPort())
    triangles.Update()
    surface = vtk.vtkPolyData()
    surface.DeepCopy(triangles.GetOutput())
    return surface


def banded_sphere(theta=24, phi=24, radius=10.0):
    """A sphere in two `ModelFaceID` bands split along a ring of mesh edges.

    Split by *vertex* height rather than by triangle centroid, so the boundary between the
    two labels runs along edges of the mesh and is a smooth circle. A centroid split gives a
    zigzag whose every vertex is a corner, which is a real case but a useless fixture: it
    pins the whole seam and neither mode can do anything with it.
    """
    surface = sphere(theta, phi, radius)
    points = surfaces.surface_points(surface)
    triangles = surfaces.triangle_indices(surface)
    upper = points[:, 2] > 0.0
    labels = np.where(upper[triangles].all(axis=1), 8, 2)
    array = vtk.vtkIntArray()
    array.SetName("ModelFaceID")
    array.SetNumberOfTuples(len(labels))
    for index, value in enumerate(labels):
        array.SetTuple1(index, int(value))
    surface.GetCellData().AddArray(array)
    return surface


def open_disc(radius=10.0, edge=1.0):
    """A flat disc with exactly one boundary loop, standing in for an unmerged patch.

    `triangulated_disc` and not `vtkDiskSource`: the latter at inner radius zero leaves a
    ring of coincident points at the centre that reads as a *second* boundary loop, which
    every check here about boundary loops would then be measuring rather than the remesh.
    """
    return triangulated_disc(radius, edge)


def mesh_of(surface, groups=None):
    return DynamicMesh.from_arrays(surfaces.surface_points(surface),
                                   surfaces.triangle_indices(surface), groups)


def polygon_capped_tube(radius=5.0, height=20.0, around=40, rings=20):
    """A tube whose two caps arrive as single `VTK_POLYGON` cells, labelled 2 and 3.

    Exactly what `vtkvmtkSimpleCapPolyData` -- the capper Slicer's **Clip Vessel** offers --
    hands over, and the reason it is worth a fixture here rather than only in
    `test_conditioning`: a cap that arrives as one polygon is *fanned* into triangles before
    the mesh is built, and the fan is a ring of obtuse triangles whose long edge is a chord of
    the rim. Whether the remesher can get out of that shape is a question about the remesher.

    The radius is several times any target these tests use, so the cap genuinely has to be
    refined rather than being one triangle wide already.
    """
    points, wall, groups = [], [], []
    for ring in range(rings + 1):
        z = -0.5 * height + height * ring / rings
        for index in range(around):
            angle = 2.0 * math.pi * index / around
            points.append((radius * math.cos(angle), radius * math.sin(angle), z))
    for ring in range(rings):
        for index in range(around):
            first, second = ring * around + index, ring * around + (index + 1) % around
            third = (ring + 1) * around + index
            fourth = (ring + 1) * around + (index + 1) % around
            wall += [[first, second, fourth], [first, fourth, third]]
            groups += [1, 1]

    surface = surfaces.polydata_from_arrays(points, np.asarray(wall, dtype=np.int64))
    cells = vtk.vtkCellArray()
    for triangle in wall:
        cells.InsertNextCell(3)
        for vertex in triangle:
            cells.InsertCellPoint(int(vertex))
    for rim, reverse in ((list(range(around)), False),
                         (list(range(rings * around, (rings + 1) * around)), True)):
        cells.InsertNextCell(len(rim))
        for vertex in (rim if reverse else reversed(rim)):
            cells.InsertCellPoint(int(vertex))
    surface.SetPolys(cells)
    surfaces.stamp_face_ids(surface, np.asarray(groups + [2, 3], dtype=np.int64))
    return surface


def face_edge_lengths(surface, face):
    """Every edge length of the triangles carrying one `ModelFaceID`."""
    points = surfaces.surface_points(surface)
    triangles = surfaces.triangle_indices(surface)[surfaces.face_ids(surface) == face]
    return np.concatenate([
        np.linalg.norm(points[triangles[:, (corner + 1) % 3]] - points[triangles[:, corner]],
                       axis=1)
        for corner in range(3)])


def bad_edge_counts(points, triangles):
    """Free and non-manifold edge counts, the way a watertightness check asks."""
    counts = Counter()
    for triangle in triangles:
        for corner in range(3):
            first, second = int(triangle[corner]), int(triangle[(corner + 1) % 3])
            counts[(min(first, second), max(first, second))] += 1
    return (sum(1 for count in counts.values() if count == 1),
            sum(1 for count in counts.values() if count > 2))


class DynamicMeshTests(unittest.TestCase):
    """The three operations, judged by whether the surface is still a surface afterwards."""

    def test_the_arrays_survive_a_round_trip_with_their_groups(self):
        surface = banded_sphere(theta=8, phi=8)
        labels = surfaces.face_ids(surface)
        mesh = mesh_of(surface, labels)

        points, triangles, groups, ids = mesh.compact()

        self.assertTrue(np.allclose(points, surfaces.surface_points(surface)))
        self.assertTrue(np.array_equal(triangles, surfaces.triangle_indices(surface)))
        self.assertTrue(np.array_equal(groups, labels))
        self.assertTrue(np.array_equal(ids, np.arange(len(points))))

    def test_an_edge_with_three_triangles_is_refused_at_construction(self):
        """Every operation here assumes one or two, so a third is not something to handle."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
        triangles = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])

        with self.assertRaises(NonManifoldMeshError):
            DynamicMesh.from_arrays(points, triangles)

    def test_splitting_every_edge_of_a_closed_surface_leaves_it_closed(self):
        mesh = mesh_of(sphere(8, 8))
        before = len(mesh.alive_triangles())

        for first, second in mesh.edges():
            mesh.split_edge(first, second,
                            0.5 * (mesh.position(first) + mesh.position(second)))

        points, triangles, _, _ = mesh.compact()
        self.assertEqual(len(triangles), 4 * before)
        self.assertEqual(bad_edge_counts(points, triangles), (0, 0))

    def test_a_split_gives_both_halves_the_parent_triangle_group(self):
        surface = banded_sphere(theta=8, phi=8)
        mesh = mesh_of(surface, surfaces.face_ids(surface))
        first, second = mesh.edges()[0]
        parent = {mesh.triangle_group(triangle)
                  for triangle in mesh.edge_triangles(first, second)}

        mesh.split_edge(first, second, 0.5 * (mesh.position(first) + mesh.position(second)))

        _, _, groups, _ = mesh.compact()
        self.assertEqual(set(groups.tolist()) & parent, parent)

    def test_splitting_a_boundary_edge_makes_two_triangles_out_of_one(self):
        mesh = mesh_of(open_disc())
        boundary = next((first, second) for first, second in mesh.edges()
                        if mesh.edge_is_boundary(first, second))
        before = len(mesh.alive_triangles())

        mesh.split_edge(*boundary, 0.5 * (mesh.position(boundary[0])
                                          + mesh.position(boundary[1])))

        self.assertEqual(len(mesh.alive_triangles()), before + 1)

    def test_collapsing_every_edge_it_will_take_leaves_a_closed_surface(self):
        """The link condition is what this is really checking, and it is checked by its
        consequence: a collapse that violated it would leave a non-manifold edge behind."""
        mesh = mesh_of(sphere(12, 12))
        collapsed = 0
        for first, second in mesh.edges():
            if mesh.has_edge(first, second) and mesh.collapse_would_be_valid(first, second):
                mesh.collapse_edge(first, second)
                collapsed += 1

        self.assertGreater(collapsed, 0)
        points, triangles, _, _ = mesh.compact()
        self.assertEqual(bad_edge_counts(points, triangles), (0, 0))

    def test_a_boundary_vertex_cannot_disappear_down_an_interior_edge(self):
        """It would drag the open boundary into the interior, which is the shape of the
        failure that breaks a watertight merge."""
        mesh = mesh_of(open_disc())
        boundary, interior = next(
            (vertex, neighbour)
            for vertex in mesh.alive_vertices() if mesh.vertex_is_boundary(vertex)
            for neighbour in mesh.vertex_one_ring(vertex)
            if not mesh.vertex_is_boundary(neighbour))

        self.assertFalse(mesh.collapse_would_be_valid(interior, boundary))
        self.assertTrue(mesh.collapse_would_be_valid(boundary, interior))

    def test_an_edge_whose_apexes_are_already_joined_is_not_flipped(self):
        """Flipping it would make a duplicate edge, and the mesh would stop being a mesh."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0],
                           [0.5, -1.0, 0.0]])
        mesh = DynamicMesh.from_arrays(points, np.array([[0, 1, 2], [1, 0, 3]]))
        mesh_with_bridge = DynamicMesh.from_arrays(
            np.vstack([points, [[0.5, 0.0, 1.0]]]),
            np.array([[0, 1, 2], [1, 0, 3], [2, 3, 4]]))

        self.assertTrue(mesh.flip_would_be_valid(0, 1))
        self.assertFalse(mesh_with_bridge.flip_would_be_valid(0, 1))

    def test_an_edge_between_two_face_labels_is_not_flipped(self):
        """A flip there would move the label boundary rather than re-triangulate it, and
        holding face labels is the whole reason this mesh carries groups."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0],
                           [0.5, -1.0, 0.0]])
        same = DynamicMesh.from_arrays(points, np.array([[0, 1, 2], [1, 0, 3]]), [7, 7])
        split = DynamicMesh.from_arrays(points, np.array([[0, 1, 2], [1, 0, 3]]), [7, 9])

        self.assertTrue(same.flip_would_be_valid(0, 1))
        self.assertFalse(split.flip_would_be_valid(0, 1))

    def test_a_boundary_edge_is_not_flipped(self):
        mesh = mesh_of(open_disc())
        boundary = next((first, second) for first, second in mesh.edges()
                        if mesh.edge_is_boundary(first, second))

        self.assertFalse(mesh.flip_would_be_valid(*boundary))

    def test_flipping_twice_comes_back_to_where_it_started(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0],
                           [0.5, -1.0, 0.0]])
        mesh = DynamicMesh.from_arrays(points, np.array([[0, 1, 2], [1, 0, 3]]))

        new_edge = mesh.flip_edge(0, 1)
        self.assertEqual(new_edge, (2, 3))
        mesh.flip_edge(*new_edge)

        self.assertTrue(mesh.has_edge(0, 1))
        self.assertEqual(len(mesh.alive_triangles()), 2)


class DerivedStateTests(unittest.TestCase):
    """`vertex_is_boundary` and `vertex_valence` answer from a counter, not from a walk.

    `_boundary_edge_count` is maintained by `_register` and `_unregister` rather than
    recomputed, which is what makes those two queries O(1) -- and incremental bookkeeping is
    the kind of thing that drifts. A missed transition would not raise; it would make
    `vertex_is_boundary` wrong somewhere, and the flip and collapse rules that ask it would
    quietly start making different decisions. So these check the counter against the
    from-scratch definition, and the valence against the one-ring it used to enumerate.
    """

    def test_a_closed_surface_has_no_boundary_vertex_and_an_open_one_has_a_ring(self):
        closed = mesh_of(sphere(theta=8, phi=8))
        self.assertTrue(closed._boundary_edges_agree())
        self.assertFalse(any(closed.vertex_is_boundary(vertex)
                             for vertex in closed.alive_vertices()))

        disc = mesh_of(open_disc(radius=4.0, edge=1.0))
        self.assertTrue(disc._boundary_edges_agree())
        boundary = [vertex for vertex in disc.alive_vertices()
                    if disc.vertex_is_boundary(vertex)]
        self.assertTrue(boundary)
        # The open boundary is one loop, so it has as many vertices as it has edges.
        loop_edges = [key for key, triangles in disc._edge_triangles.items()
                      if len(triangles) == 1]
        self.assertEqual(len(boundary), len(loop_edges))

    def test_the_counted_valence_matches_the_enumerated_one_ring(self):
        for surface in (sphere(theta=8, phi=8), open_disc(radius=4.0, edge=1.0)):
            mesh = mesh_of(surface)
            self.assertTrue(mesh._valences_agree())
            for vertex in mesh.alive_vertices():
                self.assertEqual(mesh.vertex_valence(vertex),
                                 len(mesh.vertex_one_ring(vertex)))

    def test_the_counter_survives_a_run_of_splits_collapses_and_flips(self):
        """The test that would catch a missed transition in `_register`/`_unregister`.

        A fixed pseudo-random walk over the three operations on an *open* surface, because
        the boundary is what the counter is about and a closed surface never exercises the
        one-triangle branch after construction. The counter is checked against a from-scratch
        recount after every operation, so a drift is reported at the operation that caused it
        rather than at the end.
        """
        mesh = mesh_of(open_disc(radius=5.0, edge=1.2))
        generator = np.random.default_rng(20260824)
        performed = {"splits": 0, "collapses": 0, "flips": 0}
        for step in range(400):
            edges = mesh.edges()
            if not edges:
                break
            first, second = edges[int(generator.integers(len(edges)))]
            choice = step % 3
            if choice == 0:
                midpoint = 0.5 * (mesh.position(first) + mesh.position(second))
                mesh.split_edge(first, second, midpoint)
                performed["splits"] += 1
            elif choice == 1 and mesh.collapse_would_be_valid(first, second):
                mesh.collapse_edge(first, second)
                performed["collapses"] += 1
            elif choice == 2 and mesh.flip_would_be_valid(first, second):
                mesh.flip_edge(first, second)
                performed["flips"] += 1
            else:
                continue
            self.assertTrue(mesh._boundary_edges_agree(),
                            f"boundary counts drifted at step {step} on {(first, second)}")
            self.assertTrue(mesh._valences_agree(),
                            f"valences drifted at step {step} on {(first, second)}")
        # A walk that never managed one of the three would be checking less than it looks.
        for operation, count in performed.items():
            self.assertGreater(count, 0, f"the walk performed no {operation}")

    def test_a_vertex_left_with_no_triangles_reads_as_off_the_boundary(self):
        """The state a collapse leaves behind. The counter has to come back to zero.

        `collapse_edge` unregisters every triangle around the removed vertex before marking
        it dead, so the count reaches zero through those unregistrations rather than by being
        cleared -- and if it did not, the dead slot would read as a boundary vertex and
        `vertex_valence` would report one neighbour it does not have.
        """
        mesh = mesh_of(open_disc(radius=5.0, edge=1.2))
        # An interior edge whose removed end is also interior -- the only kind
        # `collapse_would_be_valid` will take, since a boundary vertex may only go down a
        # boundary edge.
        collapsible = next((keep, remove) for keep, remove in mesh.edges()
                           if mesh.collapse_would_be_valid(keep, remove)
                           and not mesh.vertex_is_boundary(remove))
        keep, remove = collapsible
        self.assertGreater(mesh.vertex_valence(remove), 0)

        mesh.collapse_edge(keep, remove)

        self.assertFalse(mesh.vertex_is_alive(remove))
        self.assertEqual(mesh._boundary_edge_count[remove], 0)
        self.assertFalse(mesh.vertex_is_boundary(remove))
        self.assertEqual(mesh.vertex_valence(remove), 0)
        self.assertTrue(mesh._boundary_edges_agree())

    def test_collapsing_a_boundary_edge_leaves_the_boundary_one_vertex_shorter(self):
        """The transition the counter is most likely to get wrong.

        Removing a boundary vertex along a boundary edge merges two boundary edges into one,
        so the kept vertex stays on the boundary and the loop loses exactly one vertex. A
        counter that missed the merge would leave the kept vertex reading as interior, and
        `flip_would_be_valid` would then start allowing flips that strand it.
        """
        mesh = mesh_of(open_disc(radius=5.0, edge=1.2))
        before = sum(1 for vertex in mesh.alive_vertices() if mesh.vertex_is_boundary(vertex))
        keep, remove = next((keep, remove) for keep, remove in mesh.edges()
                            if mesh.edge_is_boundary(keep, remove)
                            and mesh.collapse_would_be_valid(keep, remove))

        mesh.collapse_edge(keep, remove)

        self.assertTrue(mesh._boundary_edges_agree())
        self.assertTrue(mesh.vertex_is_boundary(keep))
        after = sum(1 for vertex in mesh.alive_vertices() if mesh.vertex_is_boundary(vertex))
        self.assertEqual(after, before - 1)


class ScalarGeometryTests(unittest.TestCase):
    """The scalar helpers the operation gates use, pinned to the vectorised ones.

    `_triangle_terms` and `_farthest_distance` exist only because a numpy call on two
    triangles is almost all dispatch overhead. They are not a different measure, and the
    point of these tests is that they never become one: the split, collapse and flip gates
    all compare against thresholds calibrated on `_aspect_ratios`, so a helper that
    disagreed even in the last bits would move those decisions.
    """

    def test_the_aspect_ratio_matches_the_vectorised_one_exactly(self):
        generator = np.random.default_rng(11)
        corners = generator.normal(size=(600, 3, 3))
        # Slivers and near-degenerate triangles included on purpose: they are where the two
        # formulations could disagree, and they are what the gates are made of.
        corners[:200, 2] = corners[:200, 0] + 1e-9 * corners[:200, 1]
        expected = remesh._aspect_ratios(corners)
        for index, triangle in enumerate(corners):
            _, _, aspect = remesh._triangle_terms(*triangle)
            self.assertEqual(aspect, float(expected[index]),
                             f"triangle {index} disagrees with _aspect_ratios")

    def test_the_cross_product_and_its_norm_match_numpy(self):
        generator = np.random.default_rng(12)
        corners = generator.normal(size=(300, 3, 3))
        expected = remesh._cross_norms(corners.reshape((-1, 3)),
                                       np.arange(900).reshape((300, 3)))
        for index, triangle in enumerate(corners):
            cross, norm, _ = remesh._triangle_terms(*triangle)
            self.assertTrue(np.array_equal(np.asarray(cross), expected[index]))
            self.assertEqual(norm, float(np.linalg.norm(expected[index])))

    def test_a_degenerate_triangle_reports_a_zero_cross_norm_rather_than_the_floor(self):
        """The degeneracy gate compares against `1e-12`, so it has to see a true zero.

        The aspect ratio floors the area at `1e-30` to stay finite; the cross-norm must not,
        or a collapsed triangle would report `1e-30` and pass a gate set at `1e-12`.
        """
        flat = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

        _, norm, aspect = remesh._triangle_terms(*flat)

        self.assertEqual(norm, 0.0)
        self.assertTrue(np.isfinite(aspect))

    def test_the_farthest_distance_matches_the_vectorised_norm(self):
        generator = np.random.default_rng(13)
        positions = generator.normal(size=(40, 3))
        vertices = [3, 7, 11, 19, 28]
        origin = positions[5]

        expected = float(np.linalg.norm(positions[vertices] - origin[None, :], axis=1).max())

        self.assertEqual(remesh._farthest_distance(positions, vertices, origin), expected)

    def test_the_helpers_take_a_point_that_is_not_a_row_of_the_mesh(self):
        """The split gate weighs a triangle whose middle corner is a midpoint, not a vertex."""
        corners = (np.array([0.0, 0.0, 0.0]), (0.5, 0.25, 0.0), [1.0, 0.0, 0.0])

        _, norm, aspect = remesh._triangle_terms(*corners)

        expected = remesh._aspect_ratios(np.array([[0.0, 0.0, 0.0], [0.5, 0.25, 0.0],
                                                   [1.0, 0.0, 0.0]]))
        self.assertEqual(aspect, float(expected[0]))
        self.assertGreater(norm, 0.0)


class PolylineTargetTests(unittest.TestCase):
    """g3Sharp's `DCurveProjectionTarget`: the thing that makes a sliding seam legal."""

    def test_a_point_off_the_line_lands_on_its_foot(self):
        target = remesh.PolylineTarget([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

        self.assertTrue(np.allclose(target.project([4.0, 3.0, 0.0]), [4.0, 0.0, 0.0]))

    def test_the_ends_clamp_rather_than_extrapolate(self):
        """What stops a vertex sliding off the end of a chain between two pinned corners."""
        target = remesh.PolylineTarget([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

        self.assertTrue(np.allclose(target.project([-5.0, 1.0, 0.0]), [0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(target.project([15.0, 1.0, 0.0]), [10.0, 0.0, 0.0]))

    def test_a_closed_curve_projects_onto_the_segment_that_closes_it(self):
        square = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        target = remesh.PolylineTarget(square, closed=True)

        self.assertTrue(np.allclose(target.project([-0.5, 0.5, 0.0]), [0.0, 0.5, 0.0]))

    def test_deviation_is_the_distance_a_constraint_is_verified_by(self):
        target = remesh.PolylineTarget([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

        self.assertAlmostEqual(
            float(target.deviation(np.array([[5.0, 2.0, 0.0]]))[0]), 2.0)


class SeamChainTests(unittest.TestCase):
    """Which curves get constrained, and where they are cut into chains."""

    def test_a_face_boundary_and_an_open_boundary_are_both_constrained(self):
        """They are the same thing to a remesher -- a curve the surface may not lose -- and the
        only difference is whether there is a triangle on each side."""
        banded = banded_sphere(theta=24, phi=24)
        closed = remesh.constrained_edges(mesh_of(banded, surfaces.face_ids(banded)))
        opened = remesh.constrained_edges(mesh_of(open_disc()))

        self.assertGreater(len(closed), 0)
        self.assertGreater(len(opened), 0)

    def test_a_smooth_closed_seam_is_one_chain_with_nothing_pinned(self):
        banded = banded_sphere(theta=24, phi=24)
        mesh = mesh_of(banded, surfaces.face_ids(banded))

        chains, pinned = remesh.seam_chains(mesh)

        self.assertEqual(len(chains), 1)
        self.assertTrue(chains[0]["closed"])
        self.assertEqual(pinned, set())

    def test_a_corner_sharper_than_the_threshold_is_pinned_and_cuts_the_chain(self):
        """A corner is where a label boundary turns, not a resampling artefact, so sliding a
        vertex through it would move the boundary rather than re-discretise it."""
        mesh = mesh_of(open_disc(radius=5.0, edge=1.0))

        loose, _ = remesh.seam_chains(mesh, corner_angle_degrees=179.0)
        tight, pinned = remesh.seam_chains(mesh, corner_angle_degrees=1.0)

        self.assertEqual(len(loose), 1)
        self.assertGreater(len(tight), 1)
        self.assertGreater(len(pinned), 0)

    def test_pinning_fixes_every_seam_vertex_and_forbids_every_seam_operation(self):
        banded = banded_sphere(theta=24, phi=24)
        mesh = mesh_of(banded, surfaces.face_ids(banded))
        constraints = remesh.MeshConstraints()

        remesh.constrain_seams(mesh, constraints, seam=remesh.SEAM_PINNED)

        self.assertTrue(all(entry.fixed for entry in constraints.vertices.values()))
        self.assertTrue(all(not entry.can_split and not entry.can_collapse
                            and not entry.can_flip
                            for entry in constraints.edges.values()))

    def test_sliding_gives_every_seam_vertex_a_curve_and_forbids_only_the_flip(self):
        banded = banded_sphere(theta=24, phi=24)
        mesh = mesh_of(banded, surfaces.face_ids(banded))
        constraints = remesh.MeshConstraints()

        remesh.constrain_seams(mesh, constraints, seam=remesh.SEAM_SLIDES)

        self.assertTrue(all(entry.target is not None and not entry.fixed
                            for entry in constraints.vertices.values()))
        self.assertTrue(all(entry.can_split and entry.can_collapse and not entry.can_flip
                            for entry in constraints.edges.values()))

    def test_an_unknown_seam_mode_is_refused_rather_than_defaulted(self):
        mesh = mesh_of(sphere(8, 8))

        with self.assertRaisesRegex(ValueError, "Unknown seam mode"):
            remesh.constrain_seams(mesh, remesh.MeshConstraints(), seam="whatever")


class RemesherTests(unittest.TestCase):
    """The incremental loop, on a surface whose right answer is known analytically."""

    def remeshed(self, surface, target, **kwargs):
        mesh = mesh_of(surface)
        remesher = remesh.Remesher(mesh, target,
                                   surface_target=remesh.SurfaceTarget(surface), **kwargs)
        remesher.remesh(remesh.DEFAULT_ITERATIONS)
        return mesh, remesher

    def test_the_vertex_count_lands_near_what_the_target_edge_length_implies(self):
        """A uniform triangulation of area A at edge L has about 2A/(sqrt(3) L^2) vertices, so
        this is the one statement about a remesher that does not need a baseline to compare
        with. On a radius-10 sphere at a 1 mm target that is 1451, and the loop lands at 1445.
        """
        surface = sphere(32, 32)
        expected = 2.0 * (4.0 * np.pi * 100.0) / (np.sqrt(3.0) * 1.0 ** 2)

        mesh, _ = self.remeshed(surface, 1.0)

        points, _, _, _ = mesh.compact()
        self.assertLess(abs(len(points) - expected) / expected, 0.15)

    def test_the_triangles_come_back_near_equilateral(self):
        surface = sphere(32, 32)

        mesh, _ = self.remeshed(surface, 1.0)

        points, triangles, _, _ = mesh.compact()
        quality = triangle_quality(points, triangles)
        self.assertLess(quality["aspect_maximum"], 2.0)
        self.assertAlmostEqual(quality["median_edge"], 1.0, delta=0.15)

    def test_it_stays_on_the_surface_it_started_from(self):
        """Smoothing moves every vertex towards its neighbours' centroid, which on a curved
        surface is inwards. The projection target is the only thing that puts it back, so the
        area is what says whether it is working."""
        surface = sphere(32, 32)

        mesh, _ = self.remeshed(surface, 1.0)

        points, triangles, _, _ = mesh.compact()
        self.assertAlmostEqual(triangle_quality(points, triangles)["area"],
                               4.0 * np.pi * 100.0, delta=0.02 * 4.0 * np.pi * 100.0)

    def test_the_surface_is_still_closed_and_manifold(self):
        surface = sphere(24, 24)

        mesh, _ = self.remeshed(surface, 1.0)

        points, triangles, _, _ = mesh.compact()
        self.assertEqual(bad_edge_counts(points, triangles), (0, 0))

    def test_no_triangle_ends_up_under_the_cross_norm_tetgen_refuses_at(self):
        """The gate that has to hold whatever else does. Smoothing is the one step in the loop
        that can take a triangle to exactly zero area, and a surface carrying one is refused by
        `_validate_triangles` and by TetGen alike."""
        surface = sphere(24, 24)

        mesh, _ = self.remeshed(surface, 1.0)

        points, triangles, _, _ = mesh.compact()
        crosses = np.linalg.norm(remesh._cross_norms(points, triangles), axis=1)
        self.assertGreater(float(crosses.min()), remesh.MINIMUM_TRIANGLE_CROSS)

    def test_a_flip_that_would_spend_triangle_shape_is_refused(self):
        """Valence alone is not enough to flip on, and the measurement is in the code: 117
        valence-improving flips took a remeshed sphere's worst aspect ratio from 3.0 to 195.
        With the quality bound in place a pristine sphere gets no flips at all."""
        surface = sphere(32, 32)
        mesh = mesh_of(surface)
        remesher = remesh.Remesher(mesh, 1.121, enable_splits=False, enable_collapses=False,
                                   enable_smoothing=False)

        remesher.remesh(5)

        self.assertEqual(remesher.counts["flips"], 0)

    @staticmethod
    def sliver_fan(apex, rim):
        """A 0.1-long edge with two apexes across it at `apex` and two rim vertices at `rim`.

        Collapsing the short edge to its midpoint leaves the apex edges as long as they were
        -- the apexes sit over the middle of it -- and lengthens the two rim edges by 0.05. So
        `apex` sets what the neighbourhood's worst edge already is and `rim` sets what the
        collapse would make it, which is exactly the pair the guard has to tell apart.
        Everything but the rim is interior, so nothing topological can refuse the collapse.
        """
        points = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0],
                           [0.05, apex, 0.0], [0.05, -apex, 0.0],
                           [-rim, 0.0, 0.0], [rim + 0.1, 0.0, 0.0]])
        triangles = np.array([[0, 1, 2], [1, 0, 3], [0, 2, 4], [0, 4, 3],
                              [1, 5, 2], [1, 3, 5]])
        return DynamicMesh.from_arrays(points, triangles)

    def test_a_sliver_between_coarse_neighbours_is_still_collapsed(self):
        """The long-edge guard must not be the reason a sliver survives.

        This is the failure the guard's neighbourhood term exists for, as a unit. The
        neighbourhood already carries a 1.3 edge, well over a 0.85 target's 1.13 ceiling, and
        the collapse does not lengthen it -- so a guard that only knows the target refuses a
        collapse that makes nothing worse. On `example_2`'s patch face the strict form refused
        935 of 1044 collapsible short edges and left 36 triangles over aspect 10 behind.
        """
        mesh = self.sliver_fan(apex=1.3, rim=0.9)
        remesher = remesh.Remesher(mesh, 0.85, enable_splits=False, enable_flips=False,
                                   enable_smoothing=False)

        self.assertTrue(remesher._try_collapse(0, 1))

    def test_a_collapse_that_coarsens_past_the_neighbourhood_is_still_refused(self):
        """The other half of the same guard, on the same fixture with the rim pushed out.

        Every collapse shortens the edge it is on and lengthens the ones around it, so without
        a ceiling the mesh drifts away from the target in whichever direction the pass happens
        to sweep. Here the rim is at 1.5 and the collapse would take it to 1.55, past both the
        target's ceiling and the neighbourhood's own worst.
        """
        mesh = self.sliver_fan(apex=0.5, rim=1.5)
        remesher = remesh.Remesher(mesh, 0.85, enable_splits=False, enable_flips=False,
                                   enable_smoothing=False)

        self.assertFalse(remesher._try_collapse(0, 1))

    def test_an_impossible_target_edge_length_is_refused_up_front(self):
        mesh = mesh_of(sphere(8, 8))

        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite positive"):
                remesh.Remesher(mesh, bad)

    def test_smoothing_can_be_turned_off_and_the_projection_still_runs(self):
        surface = sphere(24, 24)

        mesh, _ = self.remeshed(surface, 1.0, enable_smoothing=False)

        points, triangles, _, _ = mesh.compact()
        self.assertEqual(bad_edge_counts(points, triangles), (0, 0))
        self.assertAlmostEqual(triangle_quality(points, triangles)["area"],
                               4.0 * np.pi * 100.0, delta=0.03 * 4.0 * np.pi * 100.0)


class LabelledSurfaceRemeshTests(unittest.TestCase):
    """`remesh_labelled_surface`: the once-per-run remesh of the merged flow domain."""

    def test_every_face_comes_back_and_the_array_keeps_its_type(self):
        """`ModelFaceID` is what every downstream stage extracts faces by, and
        `vtkAppendPolyData` drops an array whose type differs between its inputs -- so the
        type is copied from the input rather than chosen."""
        surface = banded_sphere()

        outcome = remesh.remesh_labelled_surface(surface, 1.2)

        array = outcome["surface"].GetCellData().GetArray("ModelFaceID")
        self.assertIsInstance(array, vtk.vtkIntArray)
        self.assertEqual(
            {int(array.GetTuple1(index)) for index in range(array.GetNumberOfTuples())},
            {2, 8})

    def test_the_sliding_seam_is_resampled_and_stays_on_its_own_curve(self):
        """The whole point of the sliding mode: the seam's *discretisation* is free and its
        *geometry* is not. So the vertex count on it changes and the deviation does not."""
        surface = banded_sphere()
        before = len(remesh.constrained_edges(mesh_of(surface, surfaces.face_ids(surface))))

        outcome = remesh.remesh_labelled_surface(surface, 1.2, seam=remesh.SEAM_SLIDES)

        after = len(remesh.constrained_edges(
            mesh_of(outcome["surface"], surfaces.face_ids(outcome["surface"]))))
        self.assertNotEqual(after, before)
        self.assertLess(outcome["record"]["seam_deviation"], 1e-9)

    def test_the_pinned_seam_comes_back_vertex_for_vertex(self):
        surface = banded_sphere()
        mesh = mesh_of(surface, surfaces.face_ids(surface))
        seam = np.asarray(sorted({vertex for edge in remesh.constrained_edges(mesh)
                                  for vertex in edge}))
        seam_points = mesh.positions[seam]

        outcome = remesh.remesh_labelled_surface(surface, 1.2, seam=remesh.SEAM_PINNED)

        remeshed = mesh_of(outcome["surface"], surfaces.face_ids(outcome["surface"]))
        kept = np.asarray(sorted({vertex for edge in remesh.constrained_edges(remeshed)
                                  for vertex in edge}))
        self.assertEqual(len(kept), len(seam))
        distances = np.linalg.norm(
            remeshed.positions[kept][:, None, :] - seam_points[None, :, :], axis=2)
        self.assertLess(float(distances.min(axis=1).max()), 1e-12)

    def test_the_seam_band_is_where_the_two_modes_differ(self):
        """Pinning holds the seam's vertices and pays for it in the band bridging them to a
        remeshed interior; sliding resamples them and does not. Both are correct behaviour,
        which is why this is a comparison rather than a threshold -- and it is why Stage 2
        slides the merged domain's seams and a rebase pins the patch's.
        """
        surface = banded_sphere()

        slid = remesh.remesh_labelled_surface(surface, 1.2, seam=remesh.SEAM_SLIDES)
        pinned = remesh.remesh_labelled_surface(surface, 1.2, seam=remesh.SEAM_PINNED)

        self.assertLess(slid["record"]["band_after"]["band_aspect_maximum"],
                        pinned["record"]["band_after"]["band_aspect_maximum"])

    def test_a_cap_that_arrived_as_one_polygon_is_refined_and_not_left_as_its_fan(self):
        """The regression a clipped vessel showed as "one outlet was skipped".

        A cap arrives as a single polygon and is fanned into obtuse triangles whose long edge
        spans the rim. Halving such an edge makes both children *worse* than their parent
        before the flips and the smoothing that follow make anything better -- so a split gate
        on aspect ratio refuses every split across the cap, and the fan survives the whole
        remesh while the wall beside it comes out at the target. Nothing else in the loop can
        shorten those edges, which is why `_try_split` weighs degeneracy and not shape.

        Stated as an edge length rather than a triangle count: the fan's chords are the width
        of the cap, so the thing that distinguishes a refined cap from a fan is that its
        longest edge is near the target instead of near the diameter.
        """
        surface = polygon_capped_tube()

        outcome = remesh.remesh_labelled_surface(surface, 1.0)

        for face in (2, 3):
            lengths = face_edge_lengths(outcome["surface"], face)
            self.assertLess(float(lengths.max()), remesh.LONG_EDGE_FRACTION * 1.0 + 1e-9,
                            f"face {face} still carries an edge the length of the fan's chords")
            self.assertLess(abs(float(np.median(lengths)) - 1.0), 0.2)

    def test_an_over_long_edge_is_split_even_where_that_makes_the_shape_worse(self):
        """The gate above, as the one decision it turns on, so a bound reintroduced on split
        fails here rather than only on the cap.

        A very obtuse triangle whose long edge is over the split threshold: halving it raises
        the worst aspect ratio, and it has to happen anyway, because the alternative is an
        edge that stays over the threshold for ever.
        """
        points = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [9.5, 1.0, 0.0],
                             [9.5, -1.0, 0.0]], dtype=float)
        triangles = np.asarray([[0, 1, 2], [0, 3, 1]], dtype=np.int64)
        midpoint = [5.0, 0.0, 0.0]
        parent = float(remesh._aspect_ratios(points[triangles]).max())
        halves = float(remesh._aspect_ratios(np.asarray([
            [points[0], midpoint, points[2]], [midpoint, points[1], points[2]]])).max())
        # The case the gate turned on: the parent is a perfectly ordinary triangle, so its own
        # aspect ratio does not raise the bound, and halving its long edge lands well over it.
        self.assertLess(parent, remesh.MAXIMUM_CREATED_ASPECT_RATIO)
        self.assertGreater(halves, remesh.MAXIMUM_CREATED_ASPECT_RATIO)
        remesher = remesh.Remesher(DynamicMesh.from_arrays(points, triangles), 1.0,
                                   enable_collapses=False, enable_flips=False,
                                   enable_smoothing=False)

        self.assertTrue(remesher._try_split(0, 1))

    def test_an_unlabelled_surface_is_remeshed_without_being_given_labels(self):
        """Nothing in the pipeline hands it one, but the group machinery must not invent an
        array a caller would then have to strip."""
        outcome = remesh.remesh_labelled_surface(sphere(24, 24), 1.2)

        self.assertIsNone(outcome["surface"].GetCellData().GetArray("ModelFaceID"))

    def test_an_open_surface_keeps_the_number_of_boundary_loops_it_arrived_with(self):
        """Tearing the surface open is the failure `AGENTS.md` records against remeshing the
        flow domain, earned on MMG doing exactly this, so it is checked and not assumed."""
        disc = open_disc()
        self.assertEqual(len(remesh.open_boundary_curves(disc)), 1)

        outcome = remesh.remesh_labelled_surface(disc, 1.2)

        self.assertEqual(len(remesh.open_boundary_curves(outcome["surface"])), 1)

    def test_the_sweep_reports_each_pass_without_changing_what_it_produces(self):
        """`on_iteration` exists so Slicer can repaint and poll Cancel during a sweep that is
        seconds of pure Python with no I/O in it. It is a report, so the surface it reports on
        has to be the surface produced without it -- vertex for vertex, not merely similar."""
        surface = banded_sphere()
        seen = []

        quiet = remesh.remesh_labelled_surface(surface, 1.2)
        reported = remesh.remesh_labelled_surface(
            surface, 1.2, on_iteration=lambda done, total: seen.append((done, total)))

        total = remesh.DEFAULT_ITERATIONS
        self.assertEqual(seen, [(index + 1, total) for index in range(total)])
        np.testing.assert_array_equal(
            surfaces.surface_points(reported["surface"]),
            surfaces.surface_points(quiet["surface"]))
        self.assertEqual(reported["record"]["operations"], quiet["record"]["operations"])

    def test_a_callback_that_raises_abandons_the_sweep_where_it_stood(self):
        """The frontend cancels by raising out of the callback, so the sweep must not swallow
        it or finish anyway. Nothing outside the remesher has been touched at that point."""
        class Cancelled(RuntimeError):
            pass

        def cancel_after_three(done, total):
            if done == 3:
                raise Cancelled()

        with self.assertRaises(Cancelled):
            remesh.remesh_labelled_surface(
                banded_sphere(), 1.2, on_iteration=cancel_after_three)


class QueuedRemesherTests(unittest.TestCase):
    """The queued sweep -- g3Sharp's `RemesherPro`.

    It inherits `process_edge`, so every gate and constraint `RemesherTests` checks applies
    here unaltered and is not re-checked. What is checked is the queue: that it does not lose
    the surface, that it visits far fewer edges than the full sweep, and that the invariants
    which do not depend on visit order still hold. Quality is deliberately *not* asserted
    against the full sweep's, because it is worse on a surface far from its target and that is
    documented rather than fixed.
    """

    def test_it_visits_far_fewer_edges_than_the_full_sweep_would(self):
        """The whole point, so it is measured rather than assumed.

        At the surface's own median edge, which is the case this class is for and the case
        both pipeline callers are in. The edge count barely moves on a converged remesh, so
        `edges * iterations` is a fair stand-in for what the full sweep would have visited.
        On a surface *far* from its target the comparison goes the other way -- splitting
        grows the mesh and every operation seeds four one-rings -- which is why the class
        docstring says not to use it there.
        """
        surface = banded_sphere(theta=24, phi=24)
        mesh = mesh_of(surface, surfaces.face_ids(surface))
        constraints = remesh.MeshConstraints()
        remesh.constrain_seams(mesh, constraints)
        remesher = remesh.QueuedRemesher(
            mesh, float(median_edge_length(surface)), constraints=constraints,
            surface_target=remesh.SurfaceTarget(surface))
        full_sweep = len(mesh.edges()) * remesh.DEFAULT_ITERATIONS

        remesher.remesh(remesh.DEFAULT_ITERATIONS)

        self.assertLess(remesher.counts["edges_visited"], full_sweep)
        self.assertEqual(remesher.counts["queued_passes"], remesh.DEFAULT_ITERATIONS)

    def test_it_reports_the_sweep_and_not_the_split_prelude(self):
        """`remesh_flow_domain` is the queued caller, so it needs the same progress the full
        sweep gives -- and it needs one bar that runs once, not a split prelude that exits
        early and then restarts the count."""
        seen = []

        remesh.remesh_labelled_surface(
            banded_sphere(theta=20, phi=20), 0.9, queued=True,
            on_iteration=lambda done, total: seen.append((done, total)))

        total = remesh.DEFAULT_ITERATIONS
        self.assertEqual(seen, [(index + 1, total) for index in range(total)])

    def test_the_surface_is_still_closed_and_manifold_afterwards(self):
        surface = banded_sphere(theta=20, phi=20)
        outcome = remesh.remesh_labelled_surface(surface, 0.9, queued=True)
        points = surfaces.surface_points(outcome["surface"])
        triangles = surfaces.triangle_indices(outcome["surface"])

        free, non_manifold = bad_edge_counts(points, triangles)

        self.assertEqual(free, 0)
        self.assertEqual(non_manifold, 0)

    def test_every_face_comes_back_and_the_seam_stays_on_its_curve(self):
        """Order of visiting may change; losing a face or letting a seam drift may not."""
        surface = banded_sphere(theta=20, phi=20)

        outcome = remesh.remesh_labelled_surface(surface, 0.9, queued=True)

        self.assertEqual(set(surfaces.face_ids(outcome["surface"]).tolist()), {2, 8})
        self.assertLess(outcome["record"]["seam_deviation"], 1e-6)

    def test_no_triangle_ends_up_under_the_cross_norm_tetgen_refuses_at(self):
        surface = banded_sphere(theta=20, phi=20)

        outcome = remesh.remesh_labelled_surface(surface, 0.9, queued=True)

        points = surfaces.surface_points(outcome["surface"])
        triangles = surfaces.triangle_indices(outcome["surface"])
        crosses = np.linalg.norm(remesh._cross_norms(points, triangles), axis=1)
        self.assertGreater(float(crosses.min()), remesh.MINIMUM_TRIANGLE_CROSS)

    def test_an_open_surface_keeps_its_one_boundary_loop(self):
        disc = open_disc()

        outcome = remesh.remesh_labelled_surface(disc, 1.2, queued=True)

        self.assertEqual(len(remesh.open_boundary_curves(outcome["surface"])), 1)

    def test_the_split_only_prelude_only_splits(self):
        """`fast_split_iteration` turns three things off and has to turn them back on.

        A leaked `enable_collapses = False` would silently disable collapsing for the rest of
        the run, which would look like a quality problem rather than a bug.
        """
        surface = sphere(theta=12, phi=12)
        mesh = mesh_of(surface)
        remesher = remesh.QueuedRemesher(mesh, 1.0)
        remesher.reset_queue()

        splits = remesher.fast_split_iteration()

        self.assertGreater(splits, 0)
        self.assertEqual(remesher.counts["collapses"], 0)
        self.assertEqual(remesher.counts["flips"], 0)
        self.assertTrue(remesher.enable_collapses)
        self.assertTrue(remesher.enable_flips)
        self.assertTrue(remesher.enable_smoothing)

    def test_a_fresh_queue_sweeps_everything_and_a_used_one_does_not(self):
        """`None` and the empty set have to stay distinguishable, as upstream's null does."""
        surface = sphere(theta=12, phi=12)
        mesh = mesh_of(surface)
        remesher = remesh.QueuedRemesher(mesh, 1.0)

        remesher.reset_queue()
        self.assertEqual(len(remesher._take_queue()), len(mesh.edges()))
        # That call installed an empty queue, so the next pass has nothing to visit.
        self.assertEqual(remesher._take_queue(), [])

    def test_the_relax_pass_requeues_the_edges_it_pushed_out_of_the_window(self):
        """Without this the queue goes stale the moment smoothing moves a vertex."""
        surface = banded_sphere(theta=20, phi=20)
        mesh = mesh_of(surface, surfaces.face_ids(surface))
        constraints = remesh.MeshConstraints()
        remesh.constrain_seams(mesh, constraints)
        remesher = remesh.QueuedRemesher(
            mesh, 0.9, constraints=constraints,
            surface_target=remesh.SurfaceTarget(surface))
        remesher._queue = set()

        remesher._tracked_relax_pass()

        self.assertTrue(remesher._queue)
        for first, second in remesher._queue:
            length = remesher.edge_length(first, second)
            self.assertTrue(length < remesher.minimum_edge_length
                            or length > remesher.maximum_edge_length,
                            f"edge {(first, second)} is inside the window and was queued")

    def test_a_collapse_does_not_queue_the_vertex_it_removed(self):
        """`_queue_one_ring` is handed four vertices named before the operation ran, and after
        a collapse one of them is dead. Queueing its edges would put a dead key in the queue.
        """
        mesh = mesh_of(open_disc(radius=5.0, edge=1.2))
        remesher = remesh.QueuedRemesher(mesh, 1.2)
        remesher._queue = set()
        keep, remove = next((keep, remove) for keep, remove in mesh.edges()
                            if mesh.collapse_would_be_valid(keep, remove)
                            and not mesh.vertex_is_boundary(remove))
        mesh.collapse_edge(keep, remove)

        remesher._queue_one_ring(remove)
        remesher._queue_one_ring(keep)

        self.assertFalse(any(remove in key for key in remesher._queue))
        for first, second in remesher._queue:
            self.assertTrue(mesh.has_edge(first, second))

    def test_it_agrees_with_the_full_sweep_on_a_surface_already_at_its_target(self):
        """The case it is for. On a converged surface the two should land in the same place.

        Not identical -- visit order decides which of two competing operations wins -- but
        within a few percent on the numbers the pipeline gates on, which is the claim that
        justifies using it on the capped and merged domains.
        """
        surface = banded_sphere(theta=24, phi=24)
        target = float(median_edge_length(surface))

        full = remesh.remesh_labelled_surface(surface, target)["record"]
        fast = remesh.remesh_labelled_surface(banded_sphere(theta=24, phi=24), target,
                                              queued=True)["record"]

        self.assertAlmostEqual(fast["triangles_after"], full["triangles_after"],
                               delta=0.05 * full["triangles_after"])
        self.assertLess(fast["after"]["aspect_maximum"],
                        1.3 * full["after"]["aspect_maximum"])
        self.assertLess(fast["seam_deviation"], 1e-6)


class SharedFaceAwarePassTests(unittest.TestCase):
    """`remesh_preserving_faces`: the pass every host reaches.

    The arithmetic above is `remesh_labelled_surface`'s and is tested there. What this adds is
    the part a host needs and a library call does not: the log lines, the per-face cell counts,
    and `ModelFaceID` left as the active scalars so the result draws coloured by face the moment
    it lands in a scene. Every frontend goes through this one function, so a surface remeshed in
    any of them comes out the same.
    """

    def test_the_faces_survive_and_are_counted(self):
        surface = banded_sphere()
        before = Counter(int(surface.GetCellData().GetArray("ModelFaceID").GetTuple1(cell))
                         for cell in range(surface.GetNumberOfCells()))

        remeshed, record = svremesh.remesh_preserving_faces(
            surface, 1.2, log=lambda message: None)

        self.assertEqual(set(record["faces"]), set(before))
        self.assertEqual(sum(record["faces"].values()), remeshed.GetNumberOfCells())
        self.assertEqual(record["faces"],
                         svremesh.face_cell_counts(remeshed))

    def test_the_result_draws_coloured_by_face_without_a_host_setting_it_up(self):
        """The active scalars, which is the one piece of display state the pass owns. A host that
        had to set this itself is a host that can forget to."""
        remeshed, _ = svremesh.remesh_preserving_faces(
            banded_sphere(), 1.2, log=lambda message: None)

        self.assertEqual(remeshed.GetCellData().GetScalars().GetName(), "ModelFaceID")

    def test_an_unlabelled_surface_is_remeshed_as_one_face_rather_than_refused(self):
        """The labels are what it preserves, not what it requires -- an operator pointing the
        module at an ordinary unlabelled model gets a remesh, not an error."""
        remeshed, record = svremesh.remesh_preserving_faces(
            sphere(), 1.2, log=lambda message: None)

        self.assertEqual(record["faces"], {})
        self.assertGreater(remeshed.GetNumberOfCells(), 0)

    def test_omitted_options_are_the_remeshers_own_defaults(self):
        """`None` means "whatever `remesh_labelled_surface` considers right" rather than a second
        set of numbers here that could drift from it."""
        quiet = dict(log=lambda message: None)
        through = svremesh.remesh_preserving_faces(
            banded_sphere(), 1.2, seam=None, iterations=None, corner_angle_degrees=None,
            **quiet)[1]
        direct = remesh.remesh_labelled_surface(banded_sphere(), 1.2)["record"]

        for key in ("points_after", "triangles_after", "chains", "pinned_vertices"):
            self.assertEqual(through[key], direct[key], key)

    def test_it_reports_what_the_pass_came_to(self):
        """The host shows these to the operator, so a silent pass is a bug."""
        lines = []
        svremesh.remesh_preserving_faces(
            banded_sphere(), 1.2, log=lines.append, describe="'a labelled sphere'")

        joined = "\n".join(lines)
        self.assertIn("a labelled sphere", joined)
        self.assertIn("seam chains", joined)
        self.assertIn("Faces after the remesh", joined)

if __name__ == "__main__":
    unittest.main()
