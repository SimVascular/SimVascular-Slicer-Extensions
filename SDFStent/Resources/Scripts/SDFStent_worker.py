#!/usr/bin/env python3
"""
Standalone worker for SDFStent - runs svmorph stent deployment in an external Python environment.

Usage: python SDFStent_worker.py <workdir>

Expects in workdir:
  surface_input.vtp    - vessel surface polydata (mm)
  centerline_input.vtp - centerline polydata (mm)
  params.json          - computation parameters

Writes to workdir:
  surface_output.vtp    - deployed surface polydata (mm)
  centerline_output.vtp - deployed centerline polydata (mm)
  stent_axis.json       - resampled stent axis points (cm)
  result.json           - {"success": bool, "error": str, "actualRadius": float}
"""

import json
import logging
import pathlib
import sys
import traceback

MM_TO_CM = 0.1
CM_TO_MM = 10.0


def _read_vtp(path):
    import vtk
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def _write_vtp(poly_data, path):
    import vtk
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly_data)
    writer.Write()


def _add_displacement_array(poly_data, current_pts_cm, initial_pts_cm):
    """Add a "Displacement" point data array (in mm) that points from each deployed point position
    back to its initial (undeployed) position."""
    import numpy as np
    from vtk.util.numpy_support import numpy_to_vtk
    displacements_mm = np.ascontiguousarray(
        (np.asarray(initial_pts_cm) - np.asarray(current_pts_cm)) * CM_TO_MM, dtype=np.float32)
    displacement_array = numpy_to_vtk(displacements_mm, deep=True)
    displacement_array.SetName("Displacement")
    poly_data.GetPointData().AddArray(displacement_array)
    poly_data.GetPointData().SetActiveVectors("Displacement")


def _add_strain_arrays(poly_data, current_pts, initial_pts):
    """Add per-triangle Green-Lagrange principal strain and area strain cell data arrays
    ("PrincipalStrainMax", "PrincipalStrainMin", "AreaStrain") for the surface deformation from
    the initial to the current point positions. No-op for non-triangle meshes."""
    import numpy as np
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    number_of_triangles = poly_data.GetNumberOfPolys()
    if number_of_triangles == 0:
        return
    connectivity = vtk_to_numpy(poly_data.GetPolys().GetConnectivityArray())
    if len(connectivity) != 3 * number_of_triangles:
        return
    triangles = connectivity.reshape(-1, 3)

    initial = np.asarray(initial_pts, dtype=float)
    current = np.asarray(current_pts, dtype=float)
    rest_edge1 = initial[triangles[:, 1]] - initial[triangles[:, 0]]
    rest_edge2 = initial[triangles[:, 2]] - initial[triangles[:, 0]]
    current_edge1 = current[triangles[:, 1]] - current[triangles[:, 0]]
    current_edge2 = current[triangles[:, 2]] - current[triangles[:, 0]]

    def tangent_components(edge1, edge2):
        # 2D components [[a, b], [0, d]] of the two edge vectors in the triangle's tangent basis
        normal = np.cross(edge1, edge2)
        double_area = np.linalg.norm(normal, axis=1)
        u = edge1 / np.maximum(np.linalg.norm(edge1, axis=1), 1e-12)[:, None]
        v = np.cross(normal / np.maximum(double_area, 1e-12)[:, None], u)
        a = np.einsum("ij,ij->i", edge1, u)
        b = np.einsum("ij,ij->i", edge2, u)
        d = np.einsum("ij,ij->i", edge2, v)
        return a, b, d, double_area

    a, b, d, rest_double_area = tangent_components(rest_edge1, rest_edge2)
    p, q, r, _ = tangent_components(current_edge1, current_edge2)

    valid = (rest_double_area > 1e-12) & (np.abs(a) > 1e-12) & (np.abs(d) > 1e-12)
    a = np.where(valid, a, 1.0)
    d = np.where(valid, d, 1.0)
    # Deformation gradient F = [[p, q], [0, r]] @ inv([[a, b], [0, d]]) (upper triangular)
    f11 = p / a
    f12 = (q * a - p * b) / (a * d)
    f22 = r / d
    # Green-Lagrange strain tensor E = (F^T F - I) / 2, principal strains from its eigenvalues
    e11 = 0.5 * (f11 * f11 - 1.0)
    e12 = 0.5 * (f11 * f12)
    e22 = 0.5 * (f12 * f12 + f22 * f22 - 1.0)
    strain_mean = 0.5 * (e11 + e22)
    strain_radius = np.sqrt((0.5 * (e11 - e22)) ** 2 + e12 ** 2)
    arrays = {
        "PrincipalStrainMax": np.where(valid, strain_mean + strain_radius, 0.0),
        "PrincipalStrainMin": np.where(valid, strain_mean - strain_radius, 0.0),
        "AreaStrain": np.where(valid, f11 * f22 - 1.0, 0.0),
    }
    for array_name, values in arrays.items():
        strain_array = numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float32), deep=True)
        strain_array.SetName(array_name)
        poly_data.GetCellData().AddArray(strain_array)
    poly_data.GetCellData().SetActiveScalars("PrincipalStrainMax")


def _scaled_polydata(poly_data, scale):
    import vtk
    transform = vtk.vtkTransform()
    transform.Scale(scale, scale, scale)
    f = vtk.vtkTransformPolyDataFilter()
    f.SetTransform(transform)
    f.SetInputData(poly_data)
    f.Update()
    result = vtk.vtkPolyData()
    result.DeepCopy(f.GetOutput())
    return result


def main(workdir):
    workdir = pathlib.Path(workdir)
    result = {"success": False, "error": "", "actualRadius": 0.0}

    try:
        print("Worker started, reading parameters...", flush=True)
        with open(workdir / "params.json") as f:
            params = json.load(f)

        target_radius_cm = float(params["targetRadius"]) * MM_TO_CM
        start_radius_cm = float(params["startRadius"]) * MM_TO_CM
        stent_length_cm = float(params["stentLength"]) * MM_TO_CM
        start_point_id = int(params["startPointId"])
        verbose_logging = bool(params.get("verboseLogging", False))
        enable_snapshots = bool(params.get("enableSnapshots", False))
        compute_output_arrays = bool(params.get("computeOutputModelArrays", False))
        save_step_cm = float(params.get("saveStep", 1.0)) * MM_TO_CM if enable_snapshots else None

        print("Loading vtk... (it may take a few minutes)", flush=True)
        import vtk  # noqa: F401 — ensures vtk is loaded before the first _read_vtp call
        print("Reading input geometry...", flush=True)
        surface_pd_mm = _read_vtp(workdir / "surface_input.vtp")
        centerline_pd_mm = _read_vtp(workdir / "centerline_input.vtp")
        surface_pd_cm = _scaled_polydata(surface_pd_mm, MM_TO_CM)
        centerline_pd_cm = _scaled_polydata(centerline_pd_mm, MM_TO_CM)

        print("Loading svmorph (jax startup may take a minute on first run)...", flush=True)
        from svmorph.core import deformation, geometry, mesh_data
        from svmorph.core.defaults import FORESHORTENING_PERCENTAGE
        from svmorph.core.units import L, set_unit_scale
        from svmorph.logging import setup_logging as svmorph_setup_logging, TIMING
        from svmorph.scripts import common
        from svmorph.visualization import vtk_io

        set_unit_scale(1.0)
        svmorph_setup_logging(TIMING if verbose_logging else logging.INFO)

        data = vtk_io.extract_mesh_arrays(surface_pd_cm, centerline_pd_cm)
        parent_tip_map, segment_base_mask = vtk_io.build_parent_tip_map(centerline_pd_cm)
        tangents = vtk_io.extract_centerline_tangents(centerline_pd_cm)
        inscribed_sphere_radii = vtk_io.extract_inscribed_sphere_radii(centerline_pd_cm)
        ctx = common.SimulationContext(
            data=data,
            parent_tip_map=parent_tip_map,
            segment_base_mask=segment_base_mask,
            tangents=tangents,
            inscribed_sphere_radii=inscribed_sphere_radii,
            surface_pd=surface_pd_cm,
            centerline_pd=centerline_pd_cm,
        )

        initial_surface_pts = ctx.data["points"]["surface"].copy()
        initial_centerline_pts = ctx.data["points"]["centerline"].copy()

        deformation.set_node_indices(ctx.data, [start_point_id])
        deformation.set_force_center(ctx.data, start_point_id)

        # Foreshortening convention from svmorph:
        # https://github.com/SimVascular/svMorph/blob/main/svmorph/core/defaults.py
        deployed_length = stent_length_cm * (1 - FORESHORTENING_PERCENTAGE)
        axis_pts = geometry.resample_stent_axis(
            ctx.data["points"]["centerline_points_view_np"],
            ctx.parent_tip_map,
            ctx.segment_base_mask,
            start_point_id,
            deployed_length,
            0.1 * L(),
            sampling_direction=-1,
        )

        mesh_data.compute_material_constants(1.0, 0.2)
        smoothing_k = 0.01 * L()
        cur_R = start_radius_cm - smoothing_k

        snapshotMgr = None
        if enable_snapshots:
            snapshotMgr = common.SnapshotManager(
                start_value=start_radius_cm,
                target_value=target_radius_cm,
                save_step=save_step_cm,
                out_mesh_path=str(workdir / "deployed_surface.vtp"),
                out_cl_path=str(workdir / "deployed_centerline.vtp"),
                surface_pd=ctx.surface_pd,
                centerline_pd=ctx.centerline_pd,
                data=ctx.data,
            )

        iteration = 0
        displayed_R = start_radius_cm
        while True:
            surf_disp, cl_disp, dR = deformation.compute_sdf_contact_displacements(
                ctx.data,
                axis_pts,
                s=-1.0,
                target_stent_radius=target_radius_cm,
                current_stent_radius=cur_R,
            )
            if cur_R + dR > target_radius_cm:
                print(f"Step {iteration:3d}: next increment would overshoot target -- done.", flush=True)
                break

            mesh_data.apply_displacements(ctx.data, surf_disp, "surface")
            mesh_data.apply_displacements(ctx.data, cl_disp, "centerline")
            cur_R += dR
            iteration += 1
            displayed_R = cur_R + smoothing_k

            print(f"Step {iteration:3d}: dR={dR:.5f}  R={displayed_R:.5f}", flush=True)

            if snapshotMgr:
                snapshotMgr.check_and_save(displayed_R)

        vtk_io.sync_polydata(ctx.surface_pd, ctx.data, "surface")
        vtk_io.sync_polydata(ctx.centerline_pd, ctx.data, "centerline")

        output_surface_mm = _scaled_polydata(ctx.surface_pd, CM_TO_MM)
        output_centerline_mm = _scaled_polydata(ctx.centerline_pd, CM_TO_MM)
        if compute_output_arrays:
            _add_displacement_array(output_surface_mm, ctx.data["points"]["surface"], initial_surface_pts)
            _add_displacement_array(output_centerline_mm, ctx.data["points"]["centerline"], initial_centerline_pts)
            _add_strain_arrays(output_surface_mm, ctx.data["points"]["surface"], initial_surface_pts)

        _write_vtp(output_surface_mm, workdir / "surface_output.vtp")
        _write_vtp(output_centerline_mm, workdir / "centerline_output.vtp")

        import numpy as np
        with open(workdir / "stent_axis.json", "w") as f:
            json.dump(np.asarray(axis_pts, dtype=float).tolist(), f)

        result["success"] = True
        result["actualRadius"] = displayed_R * CM_TO_MM

    except Exception as e:
        result["error"] = str(e)
        print(f"ERROR: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()

    with open(workdir / "result.json", "w") as f:
        json.dump(result, f)

    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <workdir>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
