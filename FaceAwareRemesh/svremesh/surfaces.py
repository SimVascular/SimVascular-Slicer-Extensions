"""VTK polydata helpers the remesher reads a surface through, and writes one back with.

Three groups, and the split between them is the one the remesher cares about: the first three
ask VTK about a surface's edges, the next four turn a `vtkPolyData` into the plain numpy arrays
`DynamicMesh` is built from, and the last two turn them back. Nothing here knows what the
surface is of.
"""

from __future__ import annotations

import numpy as np
import vtk


def feature_edges(polydata: vtk.vtkPolyData, mode: str) -> vtk.vtkPolyData:
    """Extract boundary or non-manifold edges from a surface."""
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(polydata)
    edges.BoundaryEdgesOff()
    edges.NonManifoldEdgesOff()
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    if mode == "boundary":
        edges.BoundaryEdgesOn()
    elif mode == "nonmanifold":
        edges.NonManifoldEdgesOn()
    else:
        raise ValueError("mode must be 'boundary' or 'nonmanifold'")
    edges.Update()
    output = vtk.vtkPolyData()
    output.DeepCopy(edges.GetOutput())
    return output


def count_feature_edges(polydata: vtk.vtkPolyData, mode: str) -> int:
    """Count boundary or non-manifold edge cells."""
    return feature_edges(polydata, mode).GetNumberOfCells()


def open_boundary_curves(polydata: vtk.vtkPolyData) -> list[np.ndarray]:
    """Return connected free-edge curves as ordered coordinate arrays."""
    stripper = vtk.vtkStripper()
    stripper.SetInputData(feature_edges(polydata, "boundary"))
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()

    curves = []
    edge_polydata = stripper.GetOutput()
    point_ids = vtk.vtkIdList()
    edge_polydata.GetLines().InitTraversal()
    while edge_polydata.GetLines().GetNextCell(point_ids):
        coordinates = np.asarray(
            [
                edge_polydata.GetPoint(point_ids.GetId(index))
                for index in range(point_ids.GetNumberOfIds())
            ],
            dtype=float,
        )
        if (
            len(coordinates) > 1
            and np.linalg.norm(coordinates[0] - coordinates[-1]) < 1e-8
        ):
            coordinates = coordinates[:-1]
        if len(coordinates) >= 3:
            curves.append(coordinates)
    return curves


def triangle_indices(surface):
    triangles = []
    for cell_id in range(surface.GetNumberOfCells()):
        cell = surface.GetCell(cell_id)
        if cell.GetNumberOfPoints() != 3:
            raise ValueError("The surface must contain only triangles.")
        triangles.append([cell.GetPointId(corner) for corner in range(3)])
    if not triangles:
        raise ValueError("The surface contains no triangles.")
    return np.asarray(triangles, dtype=np.int64)

def boundary_point_ids(triangles):
    """Vertices on the open boundary, which is what gets pinned."""
    counts = {}
    for triangle in triangles:
        for corner in range(3):
            first = int(triangle[corner])
            second = int(triangle[(corner + 1) % 3])
            edge = (min(first, second), max(first, second))
            counts[edge] = counts.get(edge, 0) + 1
    boundary_edges = [edge for edge, count in counts.items() if count == 1]
    boundary = sorted({point for edge in boundary_edges for point in edge})
    if len(boundary) < 3:
        raise ValueError("The surface must have an open boundary to pin.")
    return np.asarray(boundary, dtype=np.int64)

def surface_points(surface):
    return np.asarray(
        [surface.GetPoint(index) for index in range(surface.GetNumberOfPoints())],
        dtype=float,
    )


def face_ids(surface, name="ModelFaceID"):
    """The per-cell face labels as an integer array, or `None` if the surface carries none.

    A label array whose length disagrees with the cell count is reported as absent rather than
    raised on, because that is what a filter which changed the cell count and left the array
    behind produces, and there is no index by which such an array can be read.
    """
    array = surface.GetCellData().GetArray(name)
    if array is None or array.GetNumberOfTuples() != surface.GetNumberOfCells():
        return None
    return np.asarray([int(array.GetTuple1(index)) for index in range(array.GetNumberOfTuples())],
                      dtype=np.int64)


def polydata_from_arrays(points, triangles):
    """A triangulated `vtkPolyData` from a point array and a triangle index array."""
    vtk_points = vtk.vtkPoints()
    vtk_points.SetDataTypeToDouble()
    for point in np.asarray(points, dtype=float):
        vtk_points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
    cells = vtk.vtkCellArray()
    for triangle in np.asarray(triangles, dtype=np.int64):
        cells.InsertNextCell(3)
        for vertex in triangle:
            cells.InsertCellPoint(int(vertex))
    surface = vtk.vtkPolyData()
    surface.SetPoints(vtk_points)
    surface.SetPolys(cells)
    return surface


def stamp_face_ids(surface, values, like=None):
    """Attach `ModelFaceID` onto a surface built from arrays.

    The array *type* is copied from the input rather than chosen, because
    `vtkAppendPolyData` drops an array whose type differs between its inputs, and this one
    is what every downstream stage extracts faces by.
    """
    array = like.NewInstance() if like is not None else vtk.vtkIntArray()
    array.SetName("ModelFaceID")
    array.SetNumberOfComponents(1)
    array.SetNumberOfTuples(len(values))
    for index, value in enumerate(values):
        array.SetTuple1(index, int(value))
    surface.GetCellData().RemoveArray("ModelFaceID")
    surface.GetCellData().AddArray(array)
    surface.GetCellData().SetActiveScalars("ModelFaceID")
    return surface
