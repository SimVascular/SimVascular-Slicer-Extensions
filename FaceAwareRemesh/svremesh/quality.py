"""Triangle shape, size and crease statistics, split by whether a triangle touches the seam.

One function. It is here rather than in a general metrics module because the remesher judges
every operation by it and reports it before and after every pass, so it travels with the
remesher rather than with whatever else a host measures.
"""

from __future__ import annotations

import numpy as np



# Aspect ratio above which a triangle is bad enough to be worth counting rather than averaging.
# 10 is not a mesher's threshold -- it is where the two populations on a stitched surface
# separate. Measured on a clinical heart case, every triangle above 10 touched the seam a patch
# had been stitched along and the interior sat below 2, so the count is a count of seam slivers.
BAD_ASPECT_RATIO = 10.0


def triangle_quality(points, triangles) -> dict:
    """Triangle shape, size and crease statistics, split by whether a triangle touches the seam.

    The split is the point. Aspect ratio here is the circumradius over twice the inradius, which
    is 1 for an equilateral triangle and grows without bound as one collapses, and on a stitched
    surface it has two populations with nothing in between: the interior sits under 2, and the
    seam band carries everything above 10. A single number mixing those populations reports the
    seam's defect as though it were the mesh's, and reports every interior remesh as a failure,
    since no remesh that keeps the seam exact can touch the seam band. So `interior_*` is what a
    remesh is judged by and `seam_*` is what gets reported upstream.

    `minimum_area` and `median_area` are absolute, in the surface's own units squared, because
    what stops a long editing session is a triangle collapsing towards zero area rather than a
    ratio.
    """
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    corners = points[triangles]
    first = np.linalg.norm(corners[:, 1] - corners[:, 0], axis=1)
    second = np.linalg.norm(corners[:, 2] - corners[:, 1], axis=1)
    third = np.linalg.norm(corners[:, 0] - corners[:, 2], axis=1)
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    double_areas = np.linalg.norm(cross, axis=1)
    safe_areas = np.maximum(double_areas, 1e-30)
    semiperimeter = np.maximum(0.5 * (first + second + third), 1e-30)
    # circumradius / (2 * inradius), with inradius = area / semiperimeter.
    aspect = (first * second * third) / (2.0 * safe_areas) / (
        2.0 * safe_areas / (2.0 * semiperimeter))

    incident = {}
    for cell_id, triangle in enumerate(triangles):
        for corner in range(3):
            first_id, second_id = int(triangle[corner]), int(triangle[(corner + 1) % 3])
            incident.setdefault((min(first_id, second_id), max(first_id, second_id)),
                                []).append(cell_id)
    seam = {point for edge, cells in incident.items() if len(cells) == 1 for point in edge}
    touches_seam = np.asarray(
        [any(int(vertex) in seam for vertex in triangle) for triangle in triangles])
    interior = ~touches_seam

    normals = cross / safe_areas[:, None]
    creases = np.asarray([
        np.degrees(np.arccos(np.clip(float(np.dot(normals[cells[0]], normals[cells[1]])),
                                     -1.0, 1.0)))
        for cells in incident.values() if len(cells) == 2
    ] or [0.0], dtype=float)

    def spread(values):
        if not len(values):
            return {"p99": float("nan"), "maximum": float("nan")}
        return {"p99": float(np.percentile(values, 99)), "maximum": float(values.max())}

    return {
        "triangles": int(len(triangles)),
        "aspect_p99": spread(aspect)["p99"],
        "aspect_maximum": spread(aspect)["maximum"],
        "interior_aspect_p99": spread(aspect[interior])["p99"],
        "interior_aspect_maximum": spread(aspect[interior])["maximum"],
        "seam_aspect_p99": spread(aspect[touches_seam])["p99"],
        "seam_aspect_maximum": spread(aspect[touches_seam])["maximum"],
        "bad_triangles": int((aspect > BAD_ASPECT_RATIO).sum()),
        "bad_interior_triangles": int((aspect[interior] > BAD_ASPECT_RATIO).sum()),
        "minimum_area": float(0.5 * double_areas.min()),
        "median_area": float(np.median(0.5 * double_areas)),
        "area": float(0.5 * double_areas.sum()),
        "median_edge": float(np.median(np.concatenate([first, second, third]))),
        "crease_p99": float(np.percentile(creases, 99)),
        "crease_maximum": float(creases.max()),
    }
