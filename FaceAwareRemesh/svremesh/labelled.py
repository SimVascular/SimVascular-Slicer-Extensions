"""The pass a host calls, and the reporting around it.

`remesh.remesh_labelled_surface` is the arithmetic. What is added here is everything a host
shows an operator: the log line, the per-face cell counts, and making `ModelFaceID` the active
scalars so the result draws coloured by face the moment it lands in a scene.

Callers should come through `remesh_preserving_faces` rather than going under it. Two frontends
reaching past it to the remesher is how they start producing different surfaces from one input --
the defaults here are deliberately `None`, meaning "whatever the remesher considers right",
rather than a second set of numbers that could drift from it.
"""

from __future__ import annotations

from . import conditioning, remesh


def remesh_preserving_faces(surface, target_edge_length, *, seam=None, iterations=None,
                            smoothing_speed=None, enable_smoothing=True,
                            corner_angle_degrees=None, queued=False, log=print,
                            on_iteration=None, describe="the surface"):
    """Remesh a `ModelFaceID`-labelled surface to a uniform edge length, seams intact.

    Returns `(surface, record)`. The record is the remesher's own, plus `faces`, which is cells
    per `ModelFaceID` after the pass. A surface carrying no `ModelFaceID` is remeshed as one
    face and reports `{}` -- the labels are what this preserves, not what it requires.

    A surface that had to be conditioned to be read at all says so on its own line before the
    remesh line, rather than inside it. An operator who has just clipped a model needs to know
    that cells were removed from it whatever else the pass reports, and burying that in a
    sentence about aspect ratios is how it gets missed.
    """
    options = {"queued": bool(queued), "enable_smoothing": bool(enable_smoothing),
               "on_iteration": on_iteration}
    for name, value in (("seam", seam), ("iterations", iterations),
                        ("smoothing_speed", smoothing_speed),
                        ("corner_angle_degrees", corner_angle_degrees)):
        if value is not None:
            options[name] = value
    outcome = remesh.remesh_labelled_surface(surface, float(target_edge_length), **options)
    record = outcome["record"]
    conditioned = conditioning.describe(record["conditioning"])
    if conditioned is not None:
        log(f"Conditioned {describe} before remeshing it: {conditioned}. Nothing was moved and "
            "no two points were joined.")
    log(f"Remeshed {describe}: {record['points_before']} -> {record['points_after']} "
        f"vertices over {record['chains']} seam chains ({record['pinned_vertices']} corners "
        f"pinned), median edge {record['after']['median_edge']:.3g}, seam band's worst aspect "
        f"ratio {record['band_before']['band_aspect_maximum']:.1f} -> "
        f"{record['band_after']['band_aspect_maximum']:.1f}, smallest triangle "
        f"{record['before']['minimum_area']:.2e} -> {record['after']['minimum_area']:.2e}, "
        f"seams held to {record['seam_deviation']:.1e}.")
    remeshed = assign_active_face_scalars(outcome["surface"])
    faces = face_cell_counts(remeshed)
    log(f"Faces after the remesh: {dict(sorted(faces.items())) or 'none labelled'}.")
    record["faces"] = faces
    return remeshed, record


def face_cell_counts(polydata):
    """Cells per ModelFaceID, for the log lines that report what a step produced."""
    face_ids = polydata.GetCellData().GetArray("ModelFaceID")
    if face_ids is None:
        return {}
    counts = {}
    for cell_id in range(polydata.GetNumberOfCells()):
        value = int(face_ids.GetTuple1(cell_id))
        counts[value] = counts.get(value, 0) + 1
    return counts


def assign_active_face_scalars(polydata):
    """Make `ModelFaceID` the active scalars, which is what a face-coloured display reads by."""
    if polydata.GetCellData().GetArray("ModelFaceID") is not None:
        polydata.GetCellData().SetActiveScalars("ModelFaceID")
    return polydata
