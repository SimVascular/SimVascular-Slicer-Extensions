"""Remesh a surface model to a uniform edge length without losing its face labels.

Slicer's own remeshers -- and every clustering remesher -- treat a surface
as one sheet of triangles. A surface that carries `ModelFaceID` is not one sheet: it is several
faces meeting along seams, and those seams are what everything downstream selects by. Remesh it
without knowing that and the labels come back scrambled, the seams drift off their curves, or a
thin face disappears entirely and a later step cannot find the face it needs.

What this does instead is remesh the *labelled* surface. The labels are the mesh's own triangle
groups rather than something looked up afterwards, so every face comes back; the seams between
faces are constrained to their original curves and resampled along them at the target edge
length; and the corners where three faces meet are pinned. On the merged flow domain of a clinical
heart case at a 0.85 mm target that takes the seam band's worst aspect ratio from 935 to 8.6
and its 99th percentile from 116 to 2.0, with every face still present.

Note what it does **not** do. It is not a repair: a surface that crosses itself goes in and comes
out crossing itself. It is not a decimator you can point anywhere -- it drives toward a uniform
edge length, so a surface whose features are finer than the target loses them. And it refuses
rather than degrading: a pass that would leave a degenerate triangle, tear an open boundary, or
remesh a face away is reported and the input is left alone.

## Where the geometry is

All of it is in `svremesh`, the package beside this file, which imports nothing from Slicer and
is pip-installable on its own. This file is an MRML adapter and nothing else: it copies the
selected model's polydata, calls `svremesh.remesh_preserving_faces`, and puts the result in a
node.

The split is deliberate. Any other host calling the same function gets the same surface out,
and this module is the operator's own access to it -- to remesh an anatomy before working on
it, to clean up a merged surface, or to bring a labelled surface from anywhere else to a
uniform edge length.

This is the module to reach for after [Paint Model](../Docs/PaintModel.md) has partitioned a
model into face groups: `ModelFaceID` is the array both modules work in terms of, and it is the
labelling SimVascular's meshing and boundary-condition setup depend on.
"""

import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)

import svremesh
from svremesh import face_cell_counts, remesh_preserving_faces

# The seam policies the remesher offers, in the order the combo lists them. "Slide" resamples
# each seam along its own original curve, which is what fixes the seam band; "Pin" holds every
# seam vertex exactly where it is, which is what a caller needs when something else is going to
# be welded to those vertices afterwards. Slide is the default because pinning a fine seam next
# to a coarsened interior is where the worst triangles in this repo's history came from -- it
# took one seam band's worst aspect ratio from 50.6 to 21728.
SEAM_POLICIES = (
    ("Slide along the seam curve (recommended)", svremesh.SEAM_SLIDES),
    ("Pin every seam vertex", svremesh.SEAM_PINNED),
)

# A qualitative palette rather than a continuous rainbow: ModelFaceID is a category, and the
# same integer must read as the same face on both sides of the input/output comparison.
FACE_COLORS = (
    (0.1216, 0.4667, 0.7059),
    (1.0000, 0.4980, 0.0549),
    (0.1725, 0.6275, 0.1725),
    (0.8392, 0.1529, 0.1569),
    (0.5804, 0.4039, 0.7412),
    (0.5490, 0.3373, 0.2941),
    (0.8902, 0.4667, 0.7608),
    (0.4980, 0.4980, 0.4980),
    (0.7373, 0.7412, 0.1333),
    (0.0902, 0.7451, 0.8118),
)


class FaceAwareRemesh(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "Face-Aware Remesh"
        self.parent.categories = ["Surface Models"]
        self.parent.dependencies = []
        self.parent.contributors = ["Aaron Brown"]
        self.parent.helpText = (
            "Remesh a surface model to a uniform edge length while keeping its ModelFaceID "
            "labels, the seams between them, and the corners where they meet. Use it after "
            "Paint Model has partitioned a model into face groups."
        )
        self.parent.acknowledgementText = (
            "The remesher is a port of geometry3Sharp's (gradientspace, Boost licence)."
        )


class FaceAwareRemeshLogic(ScriptedLoadableModuleLogic):
    """An MRML adapter over `svremesh.remesh_preserving_faces`, and nothing else.

    Every geometric decision belongs to the library. What is here is the node handling: taking a
    copy of the input so a refused pass leaves the scene untouched, and writing the result into
    whichever node the operator asked for.
    """

    def remesh(self, inputModel, targetEdgeLength, *, seam=svremesh.SEAM_SLIDES, iterations=None,
               passes=1, enableSmoothing=True, cornerAngleDegrees=None, queued=False,
               log=print, onIteration=None):
        """`(surface, record)` for one model node, without touching the node itself.

        The copy is deliberate and is the whole safety property of this module: the remesher
        refuses a pass it cannot complete cleanly, and a refusal has to leave the operator with
        exactly the model they started with rather than a half-remeshed one.
        """
        polydata = inputModel.GetPolyData()
        if polydata is None or polydata.GetNumberOfCells() == 0:
            raise RuntimeError(
                f"'{inputModel.GetName()}' has no surface in it, so there is nothing to remesh.")
        copied = vtk.vtkPolyData()
        copied.DeepCopy(polydata)
        passes = int(passes)
        if passes < 1:
            raise ValueError("Remesh passes must be at least one.")

        initialPoints = copied.GetNumberOfPoints()
        initialTriangles = copied.GetNumberOfCells()
        passRecords = []
        surface = copied
        for passIndex in range(passes):
            def reportIteration(completed, total, index=passIndex):
                if onIteration is not None:
                    onIteration(index * total + completed, passes * total)

            description = f"'{inputModel.GetName()}'"
            if passes > 1:
                description += f" pass {passIndex + 1} of {passes}"
            surface, record = remesh_preserving_faces(
                surface, targetEdgeLength, seam=seam, iterations=iterations,
                enable_smoothing=enableSmoothing, corner_angle_degrees=cornerAngleDegrees,
                queued=queued, log=log, on_iteration=reportIteration,
                describe=description)
            passRecords.append(record)

        record = dict(passRecords[-1])
        record["passes"] = passes
        record["initial_points"] = initialPoints
        record["initial_triangles"] = initialTriangles
        record["operations"] = {
            operation: sum(result["operations"][operation] for result in passRecords)
            for operation in passRecords[0]["operations"]
        }
        # Only the first pass sees the operator's own surface -- every later one is handed the
        # previous pass's output, which this package built and which is clean by construction.
        # So the counts describing the *input* come from the first pass and the counts of what
        # was acted on are summed: a later pass finding anything to condition would mean the
        # remesher had produced it, and that must show here rather than be overwritten.
        first, last = passRecords[0]["conditioning"], passRecords[-1]["conditioning"]
        record["conditioning"] = {
            **first,
            "triangles_out": last["triangles_out"],
            **{key: sum(result["conditioning"][key] for result in passRecords)
               for key in ("polygons_fanned", "stray_cells", "folded_triangles",
                           "duplicated_triangles")},
        }
        return surface, record

    def faceReport(self, surface):
        """Cells per `ModelFaceID`, as the line the panel shows."""
        faces = face_cell_counts(surface)
        if not faces:
            return "No ModelFaceID on this surface: it was remeshed as one face."
        return "Faces: " + ", ".join(
            f"{face_id} ({count} cells)" for face_id, count in sorted(faces.items()))


class FaceAwareRemeshWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()
        self.logic = FaceAwareRemeshLogic()
        self._faceColorNode = None
        self._loadUI()
        self._configureUI()

    def _loadUI(self):
        """Load the Designer form and expose its named controls."""
        self.uiWidget = slicer.util.loadUI(self.resourcePath("UI/FaceAwareRemesh.ui"))
        self.layout.addWidget(self.uiWidget)
        self.ui = slicer.util.childWidgetVariables(self.uiWidget)
        self.uiWidget.setMRMLScene(slicer.mrmlScene)
        for name in ("inputSelector", "outputSelector", "targetEdgeLength", "iterations",
                     "passes", "seamCombo", "smoothingToggle", "cornerAngle", "queuedToggle",
                     "advancedBox", "applyButton", "viewToggle", "report", "log"):
            setattr(self, name, getattr(self.ui, name))

    def _configureUI(self):
        for selector in (self.inputSelector, self.outputSelector):
            selector.setMRMLScene(slicer.mrmlScene)
        for label, policy in SEAM_POLICIES:
            self.seamCombo.addItem(label, policy)
        self.seamCombo.currentIndex = 0
        self.applyButton.clicked.connect(self.onApply)
        self.viewToggle.toggled.connect(self.onViewToggled)
        self.inputSelector.currentNodeChanged.connect(self.onInputChanged)
        self.outputSelector.currentNodeChanged.connect(self.onOutputChanged)
        self.onInputChanged()

    def _syncApplyButton(self, *_):
        node = self.inputSelector.currentNode()
        self.applyButton.enabled = node is not None
        self.report.text = "" if node is None else self.logic.faceReport(node.GetPolyData())

    @staticmethod
    def _outputName(inputModel):
        """The name a fresh output model gets, whichever route created it."""
        return f"{inputModel.GetName()} remeshed"

    def _syncOutputName(self):
        """Offer that name in the output selector too, not only on Apply.

        `baseName` is what the selector's own "Create new Model as..." uses. Without this the
        two routes to a fresh output node disagree -- the selector called it "Model" and Apply
        called it "<input> remeshed" -- and which name a result ended up under depended on which
        one the operator happened to take.
        """
        node = self.inputSelector.currentNode()
        self.outputSelector.baseName = (
            "Remeshed model" if node is None else self._outputName(node))

    @staticmethod
    def _hasSurface(node):
        return (node is not None and node.GetPolyData() is not None
                and node.GetPolyData().GetNumberOfCells() > 0)

    def _faceIds(self):
        ids = set()
        for selector in (self.inputSelector, self.outputSelector):
            node = selector.currentNode()
            if self._hasSurface(node):
                ids.update(face_cell_counts(node.GetPolyData()))
        return sorted(ids)

    def _faceColors(self, faceIds):
        """One scene color node shared by input and output, keyed by face ID."""
        if self._faceColorNode is None or self._faceColorNode.GetScene() is None:
            self._faceColorNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLProceduralColorNode", "ModelFaceID colors")
            self._faceColorNode.SetHideFromEditors(True)
        colors = self._faceColorNode.GetColorTransferFunction()
        colors.RemoveAllPoints()
        for index, faceId in enumerate(faceIds):
            color = FACE_COLORS[index % len(FACE_COLORS)]
            colors.AddRGBPoint(float(faceId), *color)
        # A one-face surface still needs a non-zero color-node range.
        if len(faceIds) == 1:
            faceId = float(faceIds[0])
            color = FACE_COLORS[0]
            colors.AddRGBPoint(faceId - 0.5, *color)
            colors.AddRGBPoint(faceId + 0.5, *color)
        return self._faceColorNode

    def _colorModelsByFaceId(self):
        faceIds = self._faceIds()
        colorNode = self._faceColors(faceIds) if faceIds else None
        for selector in (self.inputSelector, self.outputSelector):
            node = selector.currentNode()
            if not self._hasSurface(node):
                continue
            node.CreateDefaultDisplayNodes()
            display = node.GetDisplayNode()
            display.SetEdgeVisibility(True)
            array = node.GetPolyData().GetCellData().GetArray("ModelFaceID")
            if array is None:
                display.SetScalarVisibility(False)
                continue
            node.GetPolyData().GetCellData().SetActiveScalars("ModelFaceID")
            display.SetActiveScalarName("ModelFaceID")
            display.SetActiveAttributeLocation(vtk.vtkAssignAttribute.CELL_DATA)
            display.SetAndObserveColorNodeID(colorNode.GetID())
            display.SetScalarRange(float(min(faceIds)), float(max(faceIds)))
            display.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange)
            display.SetScalarVisibility(True)

    def _syncViewToggle(self, showOutput=None):
        inputModel = self.inputSelector.currentNode()
        outputModel = self.outputSelector.currentNode()
        available = (self._hasSurface(inputModel) and self._hasSurface(outputModel)
                     and inputModel is not outputModel)
        self.viewToggle.enabled = available
        if showOutput is None:
            showOutput = bool(self.viewToggle.checked and available)
        else:
            showOutput = bool(showOutput and available)
        blocked = self.viewToggle.blockSignals(True)
        self.viewToggle.checked = showOutput
        self.viewToggle.text = "Show input" if showOutput else "Show output"
        self.viewToggle.blockSignals(blocked)
        if inputModel is not None:
            inputModel.CreateDefaultDisplayNodes()
            inputModel.GetDisplayNode().SetVisibility(not showOutput)
        if outputModel is not None and outputModel is not inputModel:
            outputModel.CreateDefaultDisplayNodes()
            outputModel.GetDisplayNode().SetVisibility(showOutput)

    def onInputChanged(self, *_):
        self._syncApplyButton()
        self._syncOutputName()
        self._colorModelsByFaceId()
        self._syncViewToggle(showOutput=False)

    def onOutputChanged(self, *_):
        self._colorModelsByFaceId()
        self._syncViewToggle()

    def onViewToggled(self, showOutput):
        self._syncViewToggle(showOutput=showOutput)

    def appendLog(self, message):
        self.log.appendPlainText(str(message))
        slicer.app.processEvents()

    def onApply(self):
        """Remesh the selected model, and report what the pass came to either way.

        A refusal is shown in the log rather than raised at the operator as a traceback, because
        every refusal the remesher makes is actionable: the target edge length is almost always
        the thing to change, and the message says so.

        `ValueError` is caught beside `RuntimeError` and for the same reason. The library raises
        it for a surface it cannot read at all rather than one it cannot remesh -- a target edge
        length that is not a positive number, or a `vtkPolyData` whose arrays do not line up --
        and an operator who has just clipped a model and pressed Apply is owed the same sentence
        in the log for that as for any other refusal, not a traceback in the Python console.
        """
        inputModel = self.inputSelector.currentNode()
        if inputModel is None:
            return
        outputModel = self.outputSelector.currentNode()
        if outputModel is None:
            outputModel = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLModelNode", self._outputName(inputModel))
            outputModel.CreateDefaultDisplayNodes()
            self.outputSelector.setCurrentNode(outputModel)

        dialog = slicer.util.createProgressDialog(
            parent=self.parent, windowTitle="Face-aware remesh", maximum=100)

        def onIteration(completed, total):
            dialog.value = int(100 * completed / max(total, 1))
            dialog.setLabelText(f"Sweep {completed} of {total}")
            slicer.app.processEvents()

        try:
            surface, record = self.logic.remesh(
                inputModel, self.targetEdgeLength.value,
                seam=self.seamCombo.currentData,
                iterations=int(self.iterations.value),
                passes=int(self.passes.value),
                enableSmoothing=bool(self.smoothingToggle.checked),
                cornerAngleDegrees=self.cornerAngle.value,
                queued=bool(self.queuedToggle.checked),
                log=self.appendLog, onIteration=onIteration)
        except (RuntimeError, ValueError) as refusal:
            self.appendLog(f"Refused: {refusal}")
            self.report.text = "Refused; the model is unchanged."
            return
        finally:
            dialog.close()

        outputModel.SetAndObservePolyData(surface)
        outputModel.CreateDefaultDisplayNodes()
        self._colorModelsByFaceId()
        self._syncViewToggle(showOutput=True)
        conditioned = svremesh.describe_conditioning(record["conditioning"])
        self.report.text = (
            f"{record['passes']} {'pass' if record['passes'] == 1 else 'passes'}, "
            f"{record['initial_points']} -> {record['points_after']} vertices, "
            f"median edge {record['after']['median_edge']:.3g} mm, "
            f"seam band's worst aspect {record['band_after']['band_aspect_maximum']:.1f}, "
            f"seams held to {record['seam_deviation']:.1e} mm. "
            + self.logic.faceReport(surface)
            # Named in the report and not only in the log, because it is the one thing here an
            # operator may want to go back and fix upstream rather than accept.
            + ("" if conditioned is None else f" Input conditioned first: {conditioned}."))


class FaceAwareRemeshTest(ScriptedLoadableModuleTest):
    """A smoke test under Slicer. The arithmetic is covered headlessly in
    `tests/test_remesh.py`; what cannot be covered there is the node handling."""

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_a_labelled_sphere_keeps_both_of_its_faces()

    def test_a_labelled_sphere_keeps_both_of_its_faces(self):
        source = vtk.vtkSphereSource()
        source.SetThetaResolution(24)
        source.SetPhiResolution(24)
        source.Update()
        surface = source.GetOutput()
        # Two faces split at the equator, so there is a seam to hold.
        ids = vtk.vtkIntArray()
        ids.SetName("ModelFaceID")
        ids.SetNumberOfTuples(surface.GetNumberOfCells())
        for cellId in range(surface.GetNumberOfCells()):
            centre = [0.0, 0.0, 0.0]
            points = surface.GetCell(cellId).GetPoints()
            for pointId in range(points.GetNumberOfPoints()):
                point = points.GetPoint(pointId)
                centre = [centre[axis] + point[axis] / 3.0 for axis in range(3)]
            ids.SetTuple1(cellId, 2 if centre[2] >= 0.0 else 3)
        surface.GetCellData().AddArray(ids)

        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "labelled sphere")
        node.SetAndObservePolyData(surface)

        remeshed, record = FaceAwareRemeshLogic().remesh(node, 0.08, log=lambda message: None)

        self.assertEqual(set(record["faces"]), {2, 3})
        self.assertGreater(record["points_after"], 0)
        # The input node is untouched, which is the module's safety property.
        self.assertEqual(node.GetPolyData().GetNumberOfPoints(), surface.GetNumberOfPoints())
