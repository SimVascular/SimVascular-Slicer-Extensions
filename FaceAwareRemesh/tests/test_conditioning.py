"""Reading a surface a clip produced, and what may and may not be done to it on the way in.

These are the tests for `svremesh/conditioning.py`. Run them from the parent directory with
the package importable:

    PYTHONPATH=. python -m unittest discover -s tests

The cases here are the cells a clip actually leaves behind, and they are written as claims
about the *surface* rather than about the counts: that removing a fold leaves the surrounding
triangles exactly as they were, that a cap arriving as one polygon comes back as triangles
still labelled as that cap, and that a surface which needed nothing is handed back untouched.
The counts are checked too, but they are the report and not the property.

The reason this file is separate from `test_remesh.py` is that its subject is the *input*. Every
test in there begins from a surface this package could have produced itself; every test in here
begins from one it could not.
"""

import numpy.fft  # noqa: F401  - numpy must be imported before VTK

import unittest
from collections import Counter

import numpy as np
import vtk

from svremesh import conditioning, remesh, surfaces


def labelled_tube(rings=8, around=16, height=8.0, radius=2.0):
    """A closed tube: wall labelled 1, and two fan-triangulated caps labelled 2 and 3.

    The shape a `ClipVessel` result has -- a wall with labelled cap faces on it -- because that
    is the surface these defects arrive on and the labels are what has to survive them.
    """
    points, triangles, groups = [], [], []
    for ring in range(rings + 1):
        height_at = -0.5 * height + height * ring / rings
        for index in range(around):
            angle = 2.0 * np.pi * index / around
            points.append((radius * np.cos(angle), radius * np.sin(angle), height_at))

    def wall(ring, index):
        return ring * around + index % around

    for ring in range(rings):
        for index in range(around):
            first, second = wall(ring, index), wall(ring, index + 1)
            third, fourth = wall(ring + 1, index), wall(ring + 1, index + 1)
            triangles += [[first, second, fourth], [first, fourth, third]]
            groups += [1, 1]
    for ring, face, flip in ((0, 2, True), (rings, 3, False)):
        centre = len(points)
        points.append((0.0, 0.0, -0.5 * height if ring == 0 else 0.5 * height))
        for index in range(around):
            first, second = wall(ring, index), wall(ring, index + 1)
            triangles.append([centre, second, first] if flip else [centre, first, second])
            groups.append(face)
    return (np.asarray(points, dtype=float), np.asarray(triangles, dtype=np.int64),
            np.asarray(groups, dtype=np.int64))


def surface_of(points, triangles, groups=None):
    surface = surfaces.polydata_from_arrays(points, triangles)
    if groups is not None:
        surfaces.stamp_face_ids(surface, groups)
    return surface


def sorted_triangles(surface):
    """The surface's triangles as a comparable multiset of sorted corner triples."""
    triangles = surfaces.triangle_indices(surface)
    return Counter(tuple(sorted(triangle)) for triangle in triangles.tolist())


def labels_by_triangle(surface):
    """`{sorted corners: label}`, which is the mapping conditioning must not disturb."""
    triangles = surfaces.triangle_indices(surface)
    labels = surfaces.face_ids(surface)
    return {tuple(sorted(triangle)): int(label)
            for triangle, label in zip(triangles.tolist(), labels.tolist())}


class CleanInputTests(unittest.TestCase):
    """A surface that needed nothing must not be touched, or every existing measurement in
    this package is measured against a different surface than it was."""

    def test_a_clean_surface_comes_back_as_the_very_same_object(self):
        points, triangles, groups = labelled_tube()
        surface = surface_of(points, triangles, groups)

        conditioned, record = conditioning.condition_surface(surface)

        self.assertIs(conditioned, surface)
        self.assertEqual(record["triangles_out"], len(triangles))

    def test_a_clean_surface_reports_every_count_as_zero(self):
        """This is the property a caller wanting the older strict behaviour tests instead of
        needing a flag, so it has to hold exactly and not merely nearly."""
        points, triangles, groups = labelled_tube()

        _, record = conditioning.condition_surface(surface_of(points, triangles, groups))

        self.assertEqual(
            {key: value for key, value in record.items() if key != "cells_in"
             and key != "triangles_out"},
            {"polygons_fanned": 0, "stray_cells": 0, "folded_triangles": 0,
             "duplicated_triangles": 0, "duplicate_points": 0, "unused_points": 0})
        self.assertIsNone(conditioning.describe(record))


class FoldedCellTests(unittest.TestCase):
    """`[a, b, a]`: the cell in the traceback from remeshing a freshly clipped model."""

    def test_a_folded_cell_is_removed_and_the_surface_around_it_is_untouched(self):
        """The whole claim this module rests on. A cell repeating a point id contributes a
        second copy of one edge and no area, so taking it out has to leave every genuine
        triangle exactly where it was -- not nearly, and not renumbered."""
        points, triangles, groups = labelled_tube()
        clean = surface_of(points, triangles, groups)
        folded = np.vstack([triangles, [[triangles[0][0], triangles[0][1], triangles[0][0]]]])

        conditioned, record = conditioning.condition_surface(
            surface_of(points, folded, np.append(groups, 1)))

        self.assertEqual(record["folded_triangles"], 1)
        self.assertEqual(sorted_triangles(conditioned), sorted_triangles(clean))
        self.assertEqual(labels_by_triangle(conditioned), labels_by_triangle(clean))

    def test_every_way_a_cell_can_repeat_a_point_is_caught(self):
        """The repeat can sit on any of the three corner pairs, and a cell can be all one
        point. Catching two of the three would leave `from_arrays` raising on the third."""
        points, triangles, groups = labelled_tube()
        first, second = int(triangles[0][0]), int(triangles[0][1])
        folds = [[first, first, second], [first, second, second], [first, second, first],
                 [first, first, first]]

        _, record = conditioning.condition_surface(surface_of(
            points, np.vstack([triangles, folds]), np.append(groups, [1] * len(folds))))

        self.assertEqual(record["folded_triangles"], len(folds))

    def test_a_folded_cell_no_longer_reaches_the_mesh(self):
        """The regression proper: this is the traceback the module was reported for."""
        points, triangles, groups = labelled_tube()
        folded = np.vstack([triangles, [[triangles[0][0], triangles[0][1], triangles[0][0]]]])
        surface = surface_of(points, folded, np.append(groups, 1))

        outcome = remesh.remesh_labelled_surface(surface, 0.9)

        self.assertEqual(outcome["record"]["conditioning"]["folded_triangles"], 1)
        self.assertEqual(set(surfaces.face_ids(outcome["surface"]).tolist()), {1, 2, 3})

    def test_a_surface_of_nothing_but_folded_cells_is_refused_and_says_why(self):
        points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        folds = np.asarray([[0, 1, 0], [1, 2, 1], [2, 0, 2]])

        with self.assertRaises(RuntimeError) as refused:
            conditioning.condition_surface(surface_of(points, folds))

        self.assertIn("no surface here to remesh", str(refused.exception))


class DuplicateCellTests(unittest.TestCase):
    def test_a_repeated_triangle_is_removed_and_the_first_of_the_set_is_kept(self):
        points, triangles, groups = labelled_tube()
        clean = surface_of(points, triangles, groups)
        repeated = np.vstack([triangles, [triangles[3]], [triangles[3]]])

        conditioned, record = conditioning.condition_surface(
            surface_of(points, repeated, np.append(groups, [1, 1])))

        self.assertEqual(record["duplicated_triangles"], 2)
        self.assertEqual(sorted_triangles(conditioned), sorted_triangles(clean))

    def test_a_fin_counts_as_a_duplicate_however_it_is_wound(self):
        """The form a clip leaves is the same three vertices wound both ways. To the edges
        either side of it that is one triangle twice, so winding cannot be what decides."""
        points, triangles, groups = labelled_tube()
        reversed_copy = [list(reversed(triangles[5].tolist()))]

        _, record = conditioning.condition_surface(surface_of(
            points, np.vstack([triangles, reversed_copy]), np.append(groups, 1)))

        self.assertEqual(record["duplicated_triangles"], 1)

    def test_the_label_kept_is_the_one_the_surface_listed_first(self):
        """Which copy survives is arbitrary only until the copies are labelled differently,
        and then it decides which face a triangle belongs to. It has to be defined."""
        points, triangles, groups = labelled_tube()
        target = tuple(sorted(triangles[7].tolist()))

        conditioned, _ = conditioning.condition_surface(surface_of(
            points, np.vstack([triangles, [triangles[7]]]), np.append(groups, 99)))

        self.assertEqual(labels_by_triangle(conditioned)[target], int(groups[7]))


class PolygonAndStrayCellTests(unittest.TestCase):
    """What VMTK's cappers and VTK's Clean leave in a surface besides triangles."""

    def test_a_cap_arriving_as_one_polygon_is_fanned_and_keeps_that_cap_s_label(self):
        """`vtkvmtkSimpleCapPolyData`, which Slicer's Clip Vessel offers, emits each cap as a
        single `VTK_POLYGON`. Fanning it must not turn the cap into part of the wall."""
        points, triangles, groups = labelled_tube()
        wall = [triangle for triangle, group in zip(triangles.tolist(), groups.tolist())
                if group != 3]
        wall_groups = [group for group in groups.tolist() if group != 3]
        cells = vtk.vtkCellArray()
        for triangle in wall:
            cells.InsertNextCell(3)
            for vertex in triangle:
                cells.InsertCellPoint(int(vertex))
        rim = list(range(8 * 16, 9 * 16))
        cells.InsertNextCell(len(rim))
        for vertex in reversed(rim):
            cells.InsertCellPoint(vertex)
        surface = surfaces.polydata_from_arrays(points, np.asarray(wall))
        surface.SetPolys(cells)
        surfaces.stamp_face_ids(surface, np.asarray(wall_groups + [3]))

        conditioned, record = conditioning.condition_surface(surface)

        self.assertEqual(record["polygons_fanned"], 1)
        labels = surfaces.face_ids(conditioned)
        self.assertEqual(int((labels == 3).sum()), len(rim) - 2)
        self.assertEqual({5}, {conditioned.GetCellType(cell)
                               for cell in range(conditioned.GetNumberOfCells())})

    def test_a_line_cell_left_behind_by_clean_is_dropped_rather_than_read_as_a_triangle(self):
        """`vtkCleanPolyData` converts a fold to a `VTK_LINE` and leaves it in the surface,
        which used to reach `triangle_indices` and raise there."""
        points, triangles, groups = labelled_tube()
        surface = surface_of(points, triangles, groups)
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(2)
        lines.InsertCellPoint(0)
        lines.InsertCellPoint(1)
        surface.SetLines(lines)
        # A polydata orders its cells verts, lines, polys, so the label array has to grow at
        # the front to stay aligned -- which is exactly the trap this module documents.
        surfaces.stamp_face_ids(surface, np.append([1], groups))

        conditioned, record = conditioning.condition_surface(surface)

        self.assertEqual(record["stray_cells"], 1)
        self.assertEqual(sorted_triangles(conditioned),
                         sorted_triangles(surface_of(points, triangles, groups)))

    def test_a_surface_with_no_triangles_at_all_is_refused_and_says_why(self):
        points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        surface = surfaces.polydata_from_arrays(points, np.zeros((0, 3), dtype=np.int64))
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(2)
        lines.InsertCellPoint(0)
        lines.InsertCellPoint(1)
        surface.SetLines(lines)

        with self.assertRaises(RuntimeError) as refused:
            conditioning.condition_surface(surface)

        self.assertIn("no surface here to remesh", str(refused.exception))


class JunctionTests(unittest.TestCase):
    def test_three_sheets_on_one_edge_are_refused_rather_than_read(self):
        """What is left after the non-cells are gone is a modelling decision, not a repair:
        which two of the three sheets the edge belongs to is not recoverable from the surface.
        It is refused as a `RuntimeError` so a frontend reports it rather than tracing back, and
        it is counted, because one junction from a clip that strayed and a hundred from the
        wrong model being selected are not the same message."""
        points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                             [0.0, -1.0, 0.0], [0.0, 0.0, 1.0],
                             [5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [5.0, 1.0, 0.0],
                             [5.0, -1.0, 0.0], [5.0, 0.0, 1.0]])
        sheets = np.asarray([[0, 1, 2], [0, 1, 3], [0, 1, 4],
                             [5, 6, 7], [5, 6, 8], [5, 6, 9]])

        with self.assertRaises(RuntimeError) as refused:
            conditioning.condition_surface(surface_of(points, sheets))

        self.assertIn("2 non-manifold junctions", str(refused.exception))

    def test_a_lone_junction_is_reported_in_the_singular(self):
        """One is the common case -- a clip that strayed into one branch -- so it is the
        wording an operator actually reads, and "1 junctions" is how a message loses trust."""
        points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                             [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        sheets = np.asarray([[0, 1, 2], [0, 1, 3], [0, 1, 4]])

        with self.assertRaises(RuntimeError) as refused:
            conditioning.condition_surface(surface_of(points, sheets))

        self.assertIn("1 non-manifold junction left", str(refused.exception))

    def test_a_junction_made_only_of_folds_is_not_reported_as_one(self):
        """The order matters: the folds have to go first, or the second copy of an edge each
        one contributes counts towards the junction test and a clean surface is refused."""
        points, triangles, groups = labelled_tube()
        first, second = int(triangles[0][0]), int(triangles[0][1])
        folded = np.vstack([triangles, [[first, second, first]], [[first, second, second]]])

        _, record = conditioning.condition_surface(
            surface_of(points, folded, np.append(groups, [1, 1])))

        self.assertEqual(record["folded_triangles"], 2)


class ReportTests(unittest.TestCase):
    def test_converting_and_removing_are_reported_as_the_different_things_they_are(self):
        """A cap that arrived as one polygon is converted and keeps its area and its label; a
        fold is removed and had neither. Reporting both as removed is how an operator comes to
        believe a cap was thrown away."""
        record = {"polygons_fanned": 2, "stray_cells": 0, "folded_triangles": 3,
                  "duplicated_triangles": 1}

        described = conditioning.describe(record)

        self.assertIn("fanned 2 polygons or strips into triangles", described)
        self.assertIn("removed 3 folded triangles and 1 duplicate triangle", described)
        self.assertLess(described.index("fanned"), described.index("removed"))

    def test_nothing_to_report_reads_as_nothing_rather_than_as_an_empty_sentence(self):
        self.assertIsNone(conditioning.describe(
            {"polygons_fanned": 0, "stray_cells": 0, "folded_triangles": 0,
             "duplicated_triangles": 0}))

    def test_coincident_points_are_counted_and_left_where_they_are(self):
        """Merging them joins two sheets of surface, which is the repair this package does not
        do -- so the count exists to explain the boundary-loop refusal it causes, and the
        points come back untouched."""
        points, triangles, groups = labelled_tube()
        doubled = np.vstack([points, [points[0]]])

        conditioned, record = conditioning.condition_surface(
            surface_of(doubled, triangles, groups))

        self.assertEqual(record["duplicate_points"], 1)
        self.assertEqual(conditioned.GetNumberOfPoints(), len(doubled))


if __name__ == "__main__":
    unittest.main()
