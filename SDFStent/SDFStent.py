import importlib.util
import logging
import os
import pathlib
import shutil
import tempfile
import time
from typing import Annotated, Optional

import qt
import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLMarkupsCurveNode, vtkMRMLMarkupsFiducialNode, vtkMRMLMarkupsNode, vtkMRMLModelNode, vtkMRMLNode, vtkMRMLSegmentationNode, vtkMRMLTransformNode


class SDFStent(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Virtual Stent (SDFStent)")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "SimVascular")]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Andras Lasso (PerkLab, Queen's University)",
            "Jeff Bohan Li (Cardiovascular Biomechanics Computation Lab, Stanford)",
            "Matthew A. Jolley (CHOP)",
            "Alison Marsden (Cardiovascular Biomechanics Computation Lab, Stanford)" ]
        self.parent.helpText = _("""
Expand vessel to simulate stent deployment.
Centerline is required as input, which can be generated using SlicerVMTK extension's Extract Centerline module.
See full documentation at {link}.
""".format(link="<a href=\"https://github.com/SimVascular/SlicerSimVascular\">https://github.com/SimVascular/SlicerSimVascular</a>"))
        self.parent.acknowledgementText = _("""
Developed in collaboration with the SlicerHeart project.
""")
        
        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)

#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # TemplateKey1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="SimVascular",
        sampleName="Vessel01",
        thumbnailFileName=os.path.join(iconsPath, "Vessel01.jpg"),
        uris=["https://github.com/SimVascular/SlicerSimVascular/releases/download/testing-data/Vessel01_Segmentation.seg.nrrd",
              "https://github.com/SimVascular/SlicerSimVascular/releases/download/testing-data/Vessel01_Centerline2.mrk.json"],
        checksums=["SHA256:a9071c6e5e37267720c9c6c3963d3a2b12a3ae1f017eebc3354c4f669629cf00",
                   "SHA256:0ba09b93cb50942677d68084c3f823667fb541bc334b767586023c4e417070e8"],
        fileNames=["Vessel01.seg.nrrd", "Vessel01.mrk.json"],
        nodeNames=["Vessel01 Segmentation", "Vessel01 Centerline"],
    )

    # Stent mesh derived from Open Stent Design, Craig Bonsignore, Nitinol Devices & Components (NDC),
    # http://nitinol.com (https://github.com/cbonsig/open-stent), CC BY-SA 3.0.
    # Constrained configuration, outer radius 3.0 mm, length 47.6 mm, centered at the origin, long axis along Z.
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        category="SimVascular",
        sampleName="Stent_30x476",
        thumbnailFileName=os.path.join(iconsPath, "Stent_30x476.jpg"),
        uris=["https://github.com/SimVascular/SlicerSimVascular/releases/download/testing-data/Stent_30x476.ply"],
        checksums=["SHA256:660960d1a3d95dade0000fb595098f4e71c7fb9bfa0f33380a66719a311936d6"],
        fileNames=["Stent_30x476.ply"],
        nodeNames=["Stent_30x476"],
    )


@parameterNodeWrapper
class SDFStentParameterNode:
    inputVesselSegmentation: Optional[vtkMRMLSegmentationNode] = None
    inputVesselSegmentId: str = ""
    inputCenterlineCurve: Optional[vtkMRMLNode] = None
    centerPointMarkup: Optional[vtkMRMLMarkupsFiducialNode] = None
    targetRadius: Annotated[float, WithinRange(0.0001, 30.0)] = 10.0
    startRadius: Annotated[float, WithinRange(0.0001, 30.0)] = 5.0
    stentLength: Annotated[float, WithinRange(0.0001, 100.0)] = 30.0
    enableSnapshots: bool = False
    verboseLogging: bool = False
    computeOutputModelArrays: bool = False
    saveStep: Annotated[float, WithinRange(0.0001, 100.0)] = 1.0
    preserveTemporaryFiles: bool = False
    outputMeshFileName: str = "deployed_surface.vtp"
    outputCenterlineFileName: str = "deployed_centerline.vtp"
    outputSurfaceModel: Optional[vtkMRMLModelNode] = None
    outputCenterlineModel: Optional[vtkMRMLModelNode] = None
    outputStentTransform: Optional[vtkMRMLTransformNode] = None
    outputStraightStentModel: Optional[vtkMRMLModelNode] = None
    actualRadius: Annotated[float, WithinRange(0.0, 30.0)] = 0.0


class SDFStentWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)

        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self._centerPointMarkupNode = None
        self._isProcessing = False
        self._svmorphEnsured = False

    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SDFStent.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = SDFStentLogic()

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.ui.updateButton.clicked.connect(self.onApplyButton)
        self.ui.updateButton.checkBoxToggled.connect(self._onUpdateButtonToggled)
        self.ui.inputSurfaceSelector.currentNodeChanged.connect(self._onInputSurfaceChanged)
        self.ui.inputSurfaceSelector.currentNodeChanged.connect(self._checkCanApply)
        self.ui.inputSurfaceSelector.currentSegmentChanged.connect(self._onInputSurfaceChanged)
        self.ui.inputSurfaceSelector.currentSegmentChanged.connect(self._checkCanApply)
        self.ui.inputCenterlineSelector.currentNodeChanged.connect(self._onInputCenterlineNodeChanged)
        self.ui.inputCenterlineSelector.currentNodeChanged.connect(self._checkCanApply)
        self.ui.centerPointMarkupSelector.currentNodeChanged.connect(self._onCenterPointMarkupSelectorChanged)
        self.ui.centerPointMarkupSelector.currentNodeChanged.connect(self._checkCanApply)
        self.ui.enableSnapshotsCheckBox.connect("toggled(bool)", self.onSnapshotToggleChanged)
        self.ui.startPointPlaceWidget.setMRMLScene(slicer.mrmlScene)
        self.ui.startPointPlaceWidget.placeMultipleMarkups = slicer.qSlicerMarkupsPlaceWidget.ForcePlaceSingleMarkup
        self.ui.startPointPlaceWidget.deleteAllControlPointsOptionVisible = False
        self.ui.startPointPlaceWidget.placeButton().show()
        self.ui.startPointPlaceWidget.deleteButton().show()
        self.ui.startPointPlaceWidget.connect("activeMarkupsFiducialPlaceModeChanged(bool)", self._checkCanApply)
        self.ui.logPlainTextEdit.setMaximumBlockCount(2000)
        self.ui.statusLabel.text = _("Idle")

        self.initializeParameterNode()

    def cleanup(self) -> None:
        self.removeObservers()
        # Disconnect the parameter node GUI connection, otherwise a reloaded module would leave
        # this widget instance behind, still reacting to parameter node changes
        self.setParameterNode(None)

    def enter(self) -> None:
        self.initializeParameterNode()

    def exit(self) -> None:
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._onParameterNodeModified)
            self._setCenterPointMarkupNode(None)

    def onSceneStartClose(self, caller, event) -> None:
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        self.setParameterNode(self.logic.getParameterNode())

        if not self._parameterNode.inputVesselSegmentation:
            firstSegmentationNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
            if firstSegmentationNode:
                self._parameterNode.inputVesselSegmentation = firstSegmentationNode

        if not self.ui.inputSurfaceSelector.currentNode() and self._parameterNode.inputVesselSegmentation:
            self.ui.inputSurfaceSelector.setCurrentNode(self._parameterNode.inputVesselSegmentation)

        if not self.ui.inputCenterlineSelector.currentNode() and self._parameterNode.inputCenterlineCurve:
            self.ui.inputCenterlineSelector.setCurrentNode(self._parameterNode.inputCenterlineCurve)

        if self.ui.inputSurfaceSelector.currentNode() and not self.ui.inputSurfaceSelector.currentSegmentID():
            segmentation = self.ui.inputSurfaceSelector.currentNode().GetSegmentation()
            if segmentation and segmentation.GetNumberOfSegments() > 0:
                self.ui.inputSurfaceSelector.setCurrentSegmentID(segmentation.GetNthSegmentID(0))

        if not self._parameterNode.inputCenterlineCurve:
            firstCurveNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsCurveNode")
            if firstCurveNode:
                self._parameterNode.inputCenterlineCurve = firstCurveNode
            modelCount = slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLModelNode")
            if not self._parameterNode.inputCenterlineCurve and modelCount > 1:
                self._parameterNode.inputCenterlineCurve = slicer.mrmlScene.GetNthNodeByClass(1, "vtkMRMLModelNode")

        self._ensureCenterPointMarkupNode()

        self.onSnapshotToggleChanged(self.ui.enableSnapshotsCheckBox.checked)

    def _onInputSurfaceChanged(self, *unused) -> None:
        if self._parameterNode:
            self._parameterNode.inputVesselSegmentation = self.ui.inputSurfaceSelector.currentNode()
            self._parameterNode.inputVesselSegmentId = self.ui.inputSurfaceSelector.currentSegmentID() or ""

    def _onInputCenterlineNodeChanged(self, node: vtkMRMLNode | None) -> None:
        if self._parameterNode:
            self._parameterNode.inputCenterlineCurve = node

    def _onCenterPointMarkupSelectorChanged(self, node: vtkMRMLNode | None) -> None:
        if not self._parameterNode:
            return
        centerPointMarkupNode = node if (node and node.IsA("vtkMRMLMarkupsFiducialNode")) else None
        self._parameterNode.centerPointMarkup = centerPointMarkupNode
        if centerPointMarkupNode:
            self._configureCenterPointMarkupNode(centerPointMarkupNode)
            if self.ui.startPointPlaceWidget.currentNode() != centerPointMarkupNode:
                self.ui.startPointPlaceWidget.setCurrentNode(centerPointMarkupNode)
            self._setCenterPointMarkupNode(centerPointMarkupNode)
        else:
            self._setCenterPointMarkupNode(None)

    def _ensureCenterPointMarkupNode(self) -> vtkMRMLMarkupsFiducialNode | None:
        if not self._parameterNode or slicer.mrmlScene.IsClosing():
            return None

        centerPointMarkupNode = self._parameterNode.centerPointMarkup

        if (not centerPointMarkupNode) or (centerPointMarkupNode.GetScene() is None):
            centerPointMarkupNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "CenterPoint")
            self._parameterNode.centerPointMarkup = centerPointMarkupNode

        self._configureCenterPointMarkupNode(centerPointMarkupNode)
        if self.ui.startPointPlaceWidget.currentNode() != centerPointMarkupNode:
            self.ui.startPointPlaceWidget.setCurrentNode(centerPointMarkupNode)
        self._setCenterPointMarkupNode(centerPointMarkupNode)
        return centerPointMarkupNode

    def _configureCenterPointMarkupNode(self, centerPointMarkupNode: vtkMRMLMarkupsFiducialNode | None) -> None:
        if not centerPointMarkupNode:
            return
        centerPointMarkupNode.SetMaximumNumberOfControlPoints(1)
        if not centerPointMarkupNode.GetDisplayNode():
            centerPointMarkupNode.CreateDefaultDisplayNodes()
        while centerPointMarkupNode.GetNumberOfControlPoints() > 1:
            centerPointMarkupNode.RemoveNthControlPoint(centerPointMarkupNode.GetNumberOfControlPoints() - 1)

    def _setCenterPointMarkupNode(self, centerPointMarkupNode: vtkMRMLMarkupsFiducialNode | None) -> None:
        if self._centerPointMarkupNode:
            if self.hasObserver(self._centerPointMarkupNode, vtk.vtkCommand.ModifiedEvent, self._onCenterPointMarkupModified):
                self.removeObserver(self._centerPointMarkupNode, vtk.vtkCommand.ModifiedEvent, self._onCenterPointMarkupModified)
            if self.hasObserver(self._centerPointMarkupNode, vtkMRMLMarkupsNode.PointModifiedEvent, self._onCenterPointMarkupModified):
                self.removeObserver(self._centerPointMarkupNode, vtkMRMLMarkupsNode.PointModifiedEvent, self._onCenterPointMarkupModified)
        self._centerPointMarkupNode = centerPointMarkupNode
        if self._centerPointMarkupNode:
            self.addObserver(self._centerPointMarkupNode, vtk.vtkCommand.ModifiedEvent, self._onCenterPointMarkupModified)
            self.addObserver(self._centerPointMarkupNode, vtkMRMLMarkupsNode.PointModifiedEvent, self._onCenterPointMarkupModified)

    def _onCenterPointMarkupModified(self, caller=None, event=None) -> None:
        self._configureCenterPointMarkupNode(self._centerPointMarkupNode)
        self._checkCanApply()
        self._autoUpdateIfEnabled()

    def _onParameterNodeModified(self, caller=None, event=None) -> None:
        self._checkCanApply()
        self._autoUpdateIfEnabled()

    def _onUpdateButtonToggled(self, checked: bool) -> None:
        if checked:
            self._autoUpdateIfEnabled()

    def _autoUpdateIfEnabled(self) -> None:
        if self.ui.updateButton.checkState == qt.Qt.Checked and not self._isProcessing and self.ui.updateButton.enabled:
            self.onApplyButton()

    def setParameterNode(self, inputParameterNode: SDFStentParameterNode | None) -> None:
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._onParameterNodeModified)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._onParameterNodeModified)
            if self.ui.inputCenterlineSelector.currentNode() != self._parameterNode.inputCenterlineCurve:
                self.ui.inputCenterlineSelector.setCurrentNode(self._parameterNode.inputCenterlineCurve)
            self._ensureCenterPointMarkupNode()
            self._checkCanApply()
        else:
            self._setCenterPointMarkupNode(None)

    def onSnapshotToggleChanged(self, enabled: bool) -> None:
        self.ui.saveStepSpinBox.enabled = enabled
        self.ui.labelSaveStep.enabled = enabled
        self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._isProcessing:
            self.ui.updateButton.enabled = True
            self.ui.updateButton.text = _("Cancel")
            self.ui.updateButton.toolTip = _("Cancel running stent deployment.")
            return

        self.ui.updateButton.text = _("Update")
        if not self._parameterNode:
            self.ui.updateButton.enabled = False
            self.ui.updateButton.toolTip = _("Parameter node is not available.")
            return

        inputVesselSegmentation = self.ui.inputSurfaceSelector.currentNode()
        inputVesselSegmentId = self.ui.inputSurfaceSelector.currentSegmentID()
        inputCenterlineCurve = self.ui.inputCenterlineSelector.currentNode()
        # Only check the center point here; this method runs on every GUI/parameter node change
        # (including during scene close), so it must not create any nodes.
        centerPointMarkupNode = self._parameterNode.centerPointMarkup
        hasCenterPoint = bool(centerPointMarkupNode and centerPointMarkupNode.GetScene()
                              and centerPointMarkupNode.GetNumberOfControlPoints() > 0)

        disabledReason = None
        if not inputVesselSegmentation:
            disabledReason = _("Select an input surface segmentation.")
        elif not inputVesselSegmentId:
            disabledReason = _("Select an input surface segment.")
        elif not inputCenterlineCurve:
            disabledReason = _("Select an input centerline model/curve.")
        elif not hasCenterPoint:
            disabledReason = _("Place one center point.")
        elif self._parameterNode.targetRadius <= 0.0:
            disabledReason = _("Target radius must be > 0.")
        elif self._parameterNode.startRadius <= 0.0:
            disabledReason = _("Start radius must be > 0.")
        elif self._parameterNode.stentLength <= 0.0:
            disabledReason = _("Stent length must be > 0.")
        elif self._parameterNode.startRadius >= self._parameterNode.targetRadius:
            disabledReason = _("Start radius must be smaller than target radius.")
        elif self._parameterNode.enableSnapshots and self._parameterNode.saveStep <= 0.0:
            disabledReason = _("Save step must be > 0 when snapshots are enabled.")

        canApply = disabledReason is None
        self.ui.updateButton.enabled = canApply
        self.ui.updateButton.toolTip = disabledReason if disabledReason else _("Run stent deployment and load output models.")

    def _setStatus(self, text: str) -> None:
        self.ui.statusLabel.text = text
        slicer.app.processEvents()

    def _appendLog(self, text: str) -> None:
        if not text:
            return
        self.ui.logPlainTextEdit.appendPlainText(text)
        scrollbar = self.ui.logPlainTextEdit.verticalScrollBar()
        maximumValue = scrollbar.maximum() if callable(getattr(scrollbar, "maximum", None)) else scrollbar.maximum
        scrollbar.setValue(maximumValue)
        slicer.app.processEvents()

    def _handleProcessMessage(self, text: str, isError: bool = False) -> None:
        lines = [line for line in text.replace("\r", "\n").split("\n") if line.strip()]
        for line in lines:
            logLine = f"[stderr] {line}" if isError else line
            self._appendLog(logLine)
            if "->  R =" in line:
                self._setStatus(_("Running: {line}").format(line=line[:120]))

    def _ensureSvmorphInstalled(self) -> None:
        if self.logic._useExternalPythonEnv():
            # Using external Python environment, no need to install svmorph in it
            return

        if self._svmorphEnsured:
            return

        if importlib.util.find_spec("svmorph"):
            self._svmorphEnsured = True
            return

        self._appendLog(_("svmorph is not installed. Installing svmorph..."))
        self._setStatus(_("Installing svmorph..."))
        try:
            slicer.util.pip_install("svmorph")
        except Exception as exc:
            raise RuntimeError(_("Failed to install required dependency 'svmorph'.")) from exc

        if not importlib.util.find_spec("svmorph"):
            raise RuntimeError(_("Failed to verify 'svmorph' installation."))

        self._appendLog(_("svmorph installation completed."))
        self._svmorphEnsured = True

    def onApplyButton(self) -> None:
        if self._isProcessing:
            self.logic.requestCancel()
            self.ui.updateButton.enabled = False
            self.ui.updateButton.toolTip = _("Cancelling deployment...")
            self._setStatus(_("Cancelling..."))
            return

        with slicer.util.tryWithErrorDisplay(_("Failed to deploy stent."), waitCursor=True):
            self._isProcessing = True
            self._checkCanApply()
            self.ui.logPlainTextEdit.clear()
            self._setStatus(_("Starting deployment..."))
            targetRadiusAtStart = float(self.ui.targetRadiusSpinBox.value)
            startRadiusAtStart = float(self.ui.startRadiusSpinBox.value)
            stentLengthAtStart = float(self.ui.stentLengthSpinBox.value)
            cancelledByUser = False
            try:
                self._ensureSvmorphInstalled()
                centerPointMarkupNode = self.ui.startPointPlaceWidget.currentNode()
                centerPointPositionAtStart = [0.0, 0.0, 0.0]
                if centerPointMarkupNode and centerPointMarkupNode.GetNumberOfControlPoints() > 0:
                    centerPointMarkupNode.GetNthControlPointPositionWorld(0, centerPointPositionAtStart)
                if self._parameterNode and centerPointMarkupNode:
                    referencedCenterPointMarkupNode = self._parameterNode.centerPointMarkup
                    if referencedCenterPointMarkupNode != centerPointMarkupNode:
                        self._parameterNode.centerPointMarkup = centerPointMarkupNode
                    self._configureCenterPointMarkupNode(centerPointMarkupNode)
                    self._setCenterPointMarkupNode(centerPointMarkupNode)

                # Input selectors that are not bound to the parameter node via the .ui file are synced here
                self._onInputSurfaceChanged()
                self._onInputCenterlineNodeChanged(self.ui.inputCenterlineSelector.currentNode())

                self.logic.process(processMessageCallback=self._handleProcessMessage)

                actualRadius = self._parameterNode.actualRadius if self._parameterNode else 0.0
                self._setStatus(_("Completed (R={radius:.2f} mm)").format(radius=actualRadius))
            except SDFStentRestartError:
                self._appendLog(_("Parameters changed, restarting deployment..."))
                self._setStatus(_("Restarting..."))
            except SDFStentCancelledError:
                cancelledByUser = True
                self._appendLog(_("Deployment cancelled by user."))
                self._setStatus(_("Cancelled"))
            except Exception:
                self._setStatus(_("Failed"))
                raise
            finally:
                self._isProcessing = False
                self._checkCanApply()
                if not cancelledByUser:
                    currentCenterPointPosition = [0.0, 0.0, 0.0]
                    currentCenterPointNode = self.ui.startPointPlaceWidget.currentNode()
                    if currentCenterPointNode and currentCenterPointNode.GetNumberOfControlPoints() > 0:
                        currentCenterPointNode.GetNthControlPointPositionWorld(0, currentCenterPointPosition)
                    inputsChanged = (
                        float(self.ui.targetRadiusSpinBox.value) != targetRadiusAtStart
                        or float(self.ui.startRadiusSpinBox.value) != startRadiusAtStart
                        or float(self.ui.stentLengthSpinBox.value) != stentLengthAtStart
                        or currentCenterPointPosition != centerPointPositionAtStart
                    )
                    if inputsChanged:
                        if self.ui.updateButton.enabled:
                            self.onApplyButton()
                        else:
                            self._autoUpdateIfEnabled()


class SDFStentCancelledError(RuntimeError):
    pass


class SDFStentRestartError(RuntimeError):
    pass


class SDFStentLogic(ScriptedLoadableModuleLogic):
    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)
        self._cancelRequested = False
        self._deploymentState = None
        self._temporaryCenterlineCurveNode = None
        self._temporaryCenterlineCurveSource = None
        self._mmToCm = 0.1
        self._cmToMm = 10.0

    def requestCancel(self) -> None:
        self._cancelRequested = True

    def getParameterNode(self):
        return SDFStentParameterNode(super().getParameterNode())

    def _scaledPolyData(self, polyData: vtk.vtkPolyData, scaleFactor: float) -> vtk.vtkPolyData:
        transform = vtk.vtkTransform()
        transform.Scale(float(scaleFactor), float(scaleFactor), float(scaleFactor))
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transform)
        transformFilter.SetInputData(polyData)
        transformFilter.Update()
        scaledPolyData = vtk.vtkPolyData()
        scaledPolyData.DeepCopy(transformFilter.GetOutput())
        return scaledPolyData

    def _getSegmentClosedSurfacePolyData(self, segmentationNode: vtkMRMLSegmentationNode, segmentId: str) -> vtk.vtkPolyData:
        if not segmentationNode:
            raise ValueError("Input surface segmentation node is required")
        if not segmentId:
            raise ValueError("Input surface segment is required")

        segmentation = segmentationNode.GetSegmentation()
        if not segmentation:
            raise ValueError("Selected segmentation node has no segmentation data")
        if not segmentation.GetSegment(segmentId):
            raise ValueError("Selected segment was not found in input segmentation")

        if not segmentationNode.CreateClosedSurfaceRepresentation():
            raise RuntimeError("Failed to create closed surface representation for selected segment")

        polyData = vtk.vtkPolyData()
        success = segmentationNode.GetClosedSurfaceRepresentation(segmentId, polyData)
        if not success or not polyData or polyData.GetNumberOfPoints() == 0:
            raise RuntimeError("Selected segment has no closed surface representation")

        return polyData

    def _ensureOutputModelNode(self, node: vtkMRMLModelNode | None, name: str) -> vtkMRMLModelNode:
        if node:
            return node
        return slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)

    def _getCenterlinePolyData(self, centerlineNode: vtkMRMLNode) -> vtk.vtkPolyData:
        if not centerlineNode:
            raise ValueError("Input centerline node is required")

        centerlinePolyData = vtk.vtkPolyData()
        if centerlineNode.IsA("vtkMRMLModelNode"):
            modelNode = centerlineNode
            modelPolyData = modelNode.GetPolyData()
            if not modelPolyData or modelPolyData.GetNumberOfPoints() == 0:
                raise ValueError("Input centerline model has no polydata")
            centerlinePolyData.DeepCopy(modelPolyData)
        elif centerlineNode.IsA("vtkMRMLMarkupsCurveNode"):
            curveNode = centerlineNode
            curvePolyData = curveNode.GetCurveWorld()
            if not curvePolyData or curvePolyData.GetNumberOfPoints() == 0:
                raise ValueError("Input centerline curve has no curve points")
            centerlinePolyData.DeepCopy(curvePolyData)
        else:
            raise ValueError("Input centerline node must be a model or markups curve node")

        if centerlinePolyData.GetNumberOfPoints() == 0:
            raise ValueError("Input centerline has no polydata")
        return centerlinePolyData

    def _principalStrainArrays(self, polyData: vtk.vtkPolyData, currentPoints, initialPoints):
        """Per-triangle Green-Lagrange principal strains and area strain of the surface deformation
        from the initial to the current point positions. Returns (max, min, area) strain arrays,
        or None if the mesh is not a pure triangle mesh. Strains are dimensionless; rigid
        translation and rotation yield zero strain."""
        import numpy as np
        from vtk.util.numpy_support import vtk_to_numpy

        numberOfTriangles = polyData.GetNumberOfPolys()
        if numberOfTriangles == 0:
            return None
        connectivity = vtk_to_numpy(polyData.GetPolys().GetConnectivityArray())
        if len(connectivity) != 3 * numberOfTriangles:
            return None
        triangles = connectivity.reshape(-1, 3)

        initial = np.asarray(initialPoints, dtype=float)
        current = np.asarray(currentPoints, dtype=float)
        restEdge1 = initial[triangles[:, 1]] - initial[triangles[:, 0]]
        restEdge2 = initial[triangles[:, 2]] - initial[triangles[:, 0]]
        currentEdge1 = current[triangles[:, 1]] - current[triangles[:, 0]]
        currentEdge2 = current[triangles[:, 2]] - current[triangles[:, 0]]

        def tangentComponents(edge1, edge2):
            # 2D components [[a, b], [0, d]] of the two edge vectors in the triangle's tangent basis
            normal = np.cross(edge1, edge2)
            doubleArea = np.linalg.norm(normal, axis=1)
            u = edge1 / np.maximum(np.linalg.norm(edge1, axis=1), 1e-12)[:, None]
            v = np.cross(normal / np.maximum(doubleArea, 1e-12)[:, None], u)
            a = np.einsum("ij,ij->i", edge1, u)
            b = np.einsum("ij,ij->i", edge2, u)
            d = np.einsum("ij,ij->i", edge2, v)
            return a, b, d, doubleArea

        a, b, d, restDoubleArea = tangentComponents(restEdge1, restEdge2)
        p, q, r, unusedCurrentDoubleArea = tangentComponents(currentEdge1, currentEdge2)

        valid = (restDoubleArea > 1e-12) & (np.abs(a) > 1e-12) & (np.abs(d) > 1e-12)
        a = np.where(valid, a, 1.0)
        d = np.where(valid, d, 1.0)
        # Deformation gradient F = [[p, q], [0, r]] @ inv([[a, b], [0, d]]) (upper triangular)
        F11 = p / a
        F12 = (q * a - p * b) / (a * d)
        F22 = r / d
        # Green-Lagrange strain tensor E = (F^T F - I) / 2, principal strains from its eigenvalues
        E11 = 0.5 * (F11 * F11 - 1.0)
        E12 = 0.5 * (F11 * F12)
        E22 = 0.5 * (F12 * F12 + F22 * F22 - 1.0)
        strainMean = 0.5 * (E11 + E22)
        strainRadius = np.sqrt((0.5 * (E11 - E22)) ** 2 + E12 ** 2)
        principalStrainMax = np.where(valid, strainMean + strainRadius, 0.0)
        principalStrainMin = np.where(valid, strainMean - strainRadius, 0.0)
        areaStrain = np.where(valid, F11 * F22 - 1.0, 0.0)
        return principalStrainMax, principalStrainMin, areaStrain

    def _updateOrAddArray(self, attributeData, arrayName: str, valuesFloat32) -> None:
        """Overwrite the values of the existing array with this name if its type and shape match
        (avoids reallocating arrays on repeated updates); otherwise add a new array."""
        from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
        numberOfComponents = valuesFloat32.shape[1] if valuesFloat32.ndim > 1 else 1
        existingArray = attributeData.GetArray(arrayName)
        if (existingArray is not None
                and existingArray.GetDataType() == vtk.VTK_FLOAT
                and existingArray.GetNumberOfComponents() == numberOfComponents
                and existingArray.GetNumberOfTuples() == valuesFloat32.shape[0]):
            vtk_to_numpy(existingArray)[:] = valuesFloat32
            existingArray.Modified()
        else:
            newArray = numpy_to_vtk(valuesFloat32, deep=True)
            newArray.SetName(arrayName)
            attributeData.AddArray(newArray)

    def _outputPolyData(self, polyDataCm: vtk.vtkPolyData, currentPointsCm, initialPointsCm, computeArrays: bool = False,
                        existingPolyDataMm: vtk.vtkPolyData | None = None) -> vtk.vtkPolyData:
        """Return polydata scaled to mm. If computeArrays is enabled, a "Displacement" point data
        array (in mm) is added that points from each deployed point position back to its initial
        (undeployed) position, and for triangle meshes "PrincipalStrainMax"/"PrincipalStrainMin"/
        "AreaStrain" cell data arrays are added that characterize how much the surface stretches
        (unlike displacement, strain is not affected by translation or rotation of the vessel wall).
        If existingPolyDataMm (typically the output model node's current mesh) is provided and its
        point and cell counts match, its points and data arrays are updated in place instead of
        allocating a new polydata; only pass a mesh whose topology is known to match polyDataCm."""
        import numpy as np
        from vtk.util.numpy_support import vtk_to_numpy

        updateInPlace = (existingPolyDataMm is not None
                         and existingPolyDataMm.GetNumberOfPoints() == polyDataCm.GetNumberOfPoints()
                         and existingPolyDataMm.GetNumberOfCells() == polyDataCm.GetNumberOfCells())
        if updateInPlace:
            outputPolyData = existingPolyDataMm
            vtk_to_numpy(outputPolyData.GetPoints().GetData())[:] = np.asarray(currentPointsCm) * self._cmToMm
            outputPolyData.GetPoints().GetData().Modified()
        else:
            outputPolyData = self._scaledPolyData(polyDataCm, self._cmToMm)

        if computeArrays:
            displacementsMm = np.ascontiguousarray(
                (np.asarray(initialPointsCm) - np.asarray(currentPointsCm)) * self._cmToMm, dtype=np.float32)
            self._updateOrAddArray(outputPolyData.GetPointData(), "Displacement", displacementsMm)
            outputPolyData.GetPointData().SetActiveVectors("Displacement")
            strainArrays = self._principalStrainArrays(outputPolyData, currentPointsCm, initialPointsCm)
        else:
            # Drop a stale displacement array that may remain on a reused polydata after arrays are disabled
            outputPolyData.GetPointData().RemoveArray("Displacement")
            strainArrays = None
        if strainArrays is not None:
            for arrayName, values in zip(("PrincipalStrainMax", "PrincipalStrainMin", "AreaStrain"), strainArrays):
                self._updateOrAddArray(outputPolyData.GetCellData(), arrayName, np.ascontiguousarray(values, dtype=np.float32))
            outputPolyData.GetCellData().SetActiveScalars("PrincipalStrainMax")
        else:
            # Drop stale strain arrays that may remain on a reused polydata after strain computation is disabled
            for arrayName in ("PrincipalStrainMax", "PrincipalStrainMin", "AreaStrain"):
                outputPolyData.GetCellData().RemoveArray(arrayName)
        if updateInPlace:
            outputPolyData.Modified()
        return outputPolyData

    def _straightStentPolyData(self, radiusMm: float, lengthMm: float) -> vtk.vtkPolyData:
        """Cylinder surface with the given radius and length, centered at the origin, long axis along Z."""
        line = vtk.vtkLineSource()
        line.SetPoint1(0.0, 0.0, -0.5 * lengthMm)
        line.SetPoint2(0.0, 0.0, 0.5 * lengthMm)
        line.SetResolution(50)
        tube = vtk.vtkTubeFilter()
        tube.SetInputConnection(line.GetOutputPort())
        tube.SetRadius(radiusMm)
        tube.SetNumberOfSides(36)
        tube.CappingOff()
        tube.Update()
        tubePolyData = vtk.vtkPolyData()
        tubePolyData.DeepCopy(tube.GetOutput())
        return tubePolyData

    def _centerlineCurveNodeFromInput(self, centerlineNode: vtkMRMLNode) -> vtkMRMLMarkupsCurveNode:
        """Return a markups curve node for the input centerline: the node itself if it is already a curve
        node, otherwise a hidden temporary linear curve node created from the model's polyline points
        (cached until the input model changes)."""
        if centerlineNode.IsA("vtkMRMLMarkupsCurveNode"):
            return centerlineNode
        cacheKey = (centerlineNode.GetID(), centerlineNode.GetMTime())
        if (self._temporaryCenterlineCurveNode is not None
                and self._temporaryCenterlineCurveNode.GetScene() is not None
                and self._temporaryCenterlineCurveSource == cacheKey):
            return self._temporaryCenterlineCurveNode
        if self._temporaryCenterlineCurveNode is not None and self._temporaryCenterlineCurveNode.GetScene() is not None:
            slicer.mrmlScene.RemoveNode(self._temporaryCenterlineCurveNode)
        curveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", "SDFStent centerline")
        curveNode.SetCurveTypeToLinear()
        curveNode.SetHideFromEditors(True)
        curveNode.SetSaveWithScene(False)
        curveNode.CreateDefaultDisplayNodes()
        curveNode.GetDisplayNode().SetVisibility(False)
        _, centerlinePoints = self._polylinePointIdsAndPointsFromPolyData(self._getCenterlinePolyData(centerlineNode))
        controlPoints = vtk.vtkPoints()
        for point in centerlinePoints:
            controlPoints.InsertNextPoint(point)
        curveNode.SetControlPointPositionsWorld(controlPoints)
        self._temporaryCenterlineCurveNode = curveNode
        self._temporaryCenterlineCurveSource = cacheKey
        return curveNode

    def _updateStentTransformNode(self, transformNode: vtkMRMLTransformNode, centerlineCurveNode: vtkMRMLMarkupsCurveNode,
                                  stentAxisStartMm, stentAxisEndMm, capsuleRadiusMm: float, startRadiusMm: float, stentLengthMm: float) -> None:
        """Set transformNode to a rigid+bspline combination that moves and warps the straight stent model
        (cylinder of start radius/stent length, centered at the origin, long axis along Z) to the deployed
        stent. The rigid transform maps the model center to the stent center with the model Z axis along
        the centerline tangent; the bspline transform (applied before the rigid transform, in the model
        coordinate system) bends the tube along the centerline and expands it to the deployed radius
        (with axial foreshortening). Beyond the stent length the warp follows the centerline continuation
        without radial expansion or foreshortening, with a smooth transition near the stent ends that
        follows the spherical end cap profile of the deployed stent. Positions and parallel transport
        frames along the centerline are provided by the curve node."""
        import numpy as np
        from vtk.util.numpy_support import numpy_to_vtk

        curveLength = centerlineCurveNode.GetCurveLengthWorld()
        startIndex = centerlineCurveNode.GetClosestCurvePointIndexToPositionWorld(list(stentAxisStartMm))
        endIndex = centerlineCurveNode.GetClosestCurvePointIndexToPositionWorld(list(stentAxisEndMm))
        directionSign = 1.0 if endIndex >= startIndex else -1.0
        arcAtAxisStart = centerlineCurveNode.GetCurveLengthBetweenStartEndPointsWorld(0, startIndex)
        stentAxisLength = centerlineCurveNode.GetCurveLengthBetweenStartEndPointsWorld(min(startIndex, endIndex), max(startIndex, endIndex))

        def frameAt(stentArcPosition):
            # Position and parallel transport frame at an arc position measured along the stent axis
            # direction from the stent axis start. The returned frame is right-handed with the tangent
            # pointing along increasing arc position; positions beyond the curve ends are extended linearly.
            curveArcPosition = arcAtAxisStart + directionSign * stentArcPosition
            clampedArcPosition = min(max(curveArcPosition, 0.0), curveLength)
            pointIndex = centerlineCurveNode.GetCurvePointIndexAlongCurveWorld(0, clampedArcPosition)
            curvePointToWorld = vtk.vtkMatrix4x4()
            centerlineCurveNode.GetCurvePointToWorldTransformAtPointIndex(pointIndex, curvePointToWorld)
            position = np.array([curvePointToWorld.GetElement(row, 3) for row in range(3)])
            normal = np.array([curvePointToWorld.GetElement(row, 0) for row in range(3)])
            binormal = np.array([curvePointToWorld.GetElement(row, 1) for row in range(3)])
            tangent = np.array([curvePointToWorld.GetElement(row, 2) for row in range(3)])
            exactPosition = [0.0, 0.0, 0.0]
            if centerlineCurveNode.GetPositionAlongCurveWorld(exactPosition, 0, clampedArcPosition):
                position = np.array(exactPosition)
            position = position + (curveArcPosition - clampedArcPosition) * tangent
            return position, normal, binormal * directionSign, tangent * directionSign

        # Rigid transform: model center (origin) to stent center, model Z axis along the centerline tangent
        midPosition, midNormal, midBinormal, midTangent = frameAt(0.5 * stentAxisLength)
        rotation = np.column_stack((midNormal, midBinormal, midTangent))
        rigidMatrix = vtk.vtkMatrix4x4()
        for row in range(3):
            for col in range(3):
                rigidMatrix.SetElement(row, col, rotation[row, col])
            rigidMatrix.SetElement(row, 3, midPosition[row])
        rigidTransform = vtk.vtkTransform()
        rigidTransform.SetMatrix(rigidMatrix)

        # BSpline transform, defined in the model coordinate system: displacement at model point p is
        # chosen so that rigid(bspline(p)) lands on the corresponding point along the deployed stent
        radialExtent = 2.0 * max(startRadiusMm, 1.0)
        axialMargin = capsuleRadiusMm + 0.2 * stentLengthMm
        axialExtent = 0.5 * stentLengthMm + axialMargin
        axialSpacing = stentLengthMm / 25.0
        axialNodeCount = int(np.ceil(2.0 * axialExtent / axialSpacing)) + 1
        gridX = np.linspace(-radialExtent, radialExtent, 13)
        gridY = np.linspace(-radialExtent, radialExtent, 13)
        gridZ = np.linspace(-axialExtent, axialExtent, axialNodeCount)

        # The frame and radial scale only depend on z, so compute them once per grid plane.
        # Axial mapping: foreshortened inside the stent, arc-length preserving beyond the stent ends.
        # Radial scale: deployed radius inside the stent, following the spherical end cap profile near
        # the stent ends, and no expansion (scale 1) beyond that; blended with a smoothstep so the
        # transition is smooth while staying exact inside and far outside the stent.
        halfLength = 0.5 * stentLengthMm
        transitionWidth = 0.5 * startRadiusMm
        positions = np.zeros((axialNodeCount, 3))
        normals = np.zeros((axialNodeCount, 3))
        binormals = np.zeros((axialNodeCount, 3))
        radialScales = np.zeros(axialNodeCount)
        for zIndex, z in enumerate(gridZ):
            if z < -halfLength:
                stentArcPosition = z + halfLength
            elif z > halfLength:
                stentArcPosition = stentAxisLength + (z - halfLength)
            else:
                stentArcPosition = (z + halfLength) / stentLengthMm * stentAxisLength
            positions[zIndex], normals[zIndex], binormals[zIndex], _ = frameAt(stentArcPosition)
            overhang = max(0.0, -stentArcPosition, stentArcPosition - stentAxisLength)
            capsuleProfileRadius = (max(capsuleRadiusMm ** 2 - overhang ** 2, 0.0)) ** 0.5
            blend = min(max((capsuleProfileRadius - (startRadiusMm - transitionWidth)) / (2.0 * transitionWidth), 0.0), 1.0)
            blend = blend * blend * (3.0 - 2.0 * blend)  # smoothstep
            radialScales[zIndex] = (startRadiusMm + blend * max(capsuleProfileRadius - startRadiusMm, 0.0)) / startRadiusMm

        gx, gy, gz = np.meshgrid(gridX, gridY, gridZ, indexing="ij")
        gridPoints = np.stack((gx.ravel(order="F"), gy.ravel(order="F"), gz.ravel(order="F")), axis=1)
        zIndices = np.arange(len(gridPoints)) // (len(gridX) * len(gridY))
        warpedPoints = positions[zIndices] + (gridPoints[:, 0:1] * normals[zIndices]
                                              + gridPoints[:, 1:2] * binormals[zIndices]) * radialScales[zIndices][:, None]
        displacements = (warpedPoints - midPosition) @ rotation - gridPoints

        coefficientImage = vtk.vtkImageData()
        coefficientImage.SetDimensions(len(gridX), len(gridY), len(gridZ))
        coefficientImage.SetOrigin(gridX[0], gridY[0], gridZ[0])
        coefficientImage.SetSpacing(gridX[1] - gridX[0], gridY[1] - gridY[0], gridZ[1] - gridZ[0])
        coefficientArray = numpy_to_vtk(np.ascontiguousarray(displacements, dtype=np.float64), deep=True)
        coefficientArray.SetName("Displacement")
        coefficientImage.GetPointData().SetScalars(coefficientArray)
        # vtkOrientedBSplineTransform (not plain vtkBSplineTransform) is required for the transform
        # node to be convertible to an ITK transform when saving to file
        bsplineTransform = slicer.vtkOrientedBSplineTransform()
        bsplineTransform.SetCoefficientData(coefficientImage)
        bsplineTransform.SetBorderModeToEdge()

        combinedTransform = vtk.vtkGeneralTransform()
        combinedTransform.Concatenate(rigidTransform)
        combinedTransform.Concatenate(bsplineTransform)  # applied first (in model coordinate system)
        transformNode.SetAndObserveTransformToParent(combinedTransform)

    def _updateOptionalStentOutputs(self, parameterNode: SDFStentParameterNode, axisPointsCm, centerlineNode: vtkMRMLNode, capsuleRadiusCm: float, startRadiusMm: float, stentLengthMm: float) -> None:
        """Update the optional straight stent model and stent transform outputs (skipped when not selected)."""
        straightStentNode = parameterNode.outputStraightStentModel
        if straightStentNode:
            self._setDisplayedPolyData(straightStentNode, self._straightStentPolyData(startRadiusMm, stentLengthMm),
                                       defaultOpacity=0.8, defaultColor=(0.8, 0.8, 0.9))
        transformNode = parameterNode.outputStentTransform
        if not transformNode:
            return
        import numpy as np
        axisPointsMm = np.asarray(axisPointsCm, dtype=float) * self._cmToMm
        centerlineCurveNode = self._centerlineCurveNodeFromInput(centerlineNode)
        self._updateStentTransformNode(transformNode, centerlineCurveNode, axisPointsMm[0], axisPointsMm[-1],
                                       capsuleRadiusCm * self._cmToMm, startRadiusMm, stentLengthMm)
        if straightStentNode and straightStentNode.GetParentTransformNode() != transformNode:
            straightStentNode.SetAndObserveTransformNodeID(transformNode.GetID())

    def _setModelNodePolyData(self, node: vtkMRMLModelNode, polyData: vtk.vtkPolyData) -> None:
        """Set the polydata on the model node, unless the node already holds this same object
        (mesh updated in place, with Modified() invoked on the changed arrays and the polydata).
        For in-place changes the display nodes' automatic ("data") scalar range is not recomputed
        by any observer, so refresh it here; everything else is updated by the displayable
        managers in response to the mesh modification event."""
        if node.GetPolyData() is not polyData:
            node.SetAndObservePolyData(polyData)
            return
        for displayNodeIndex in range(node.GetNumberOfDisplayNodes()):
            displayNode = node.GetNthDisplayNode(displayNodeIndex)
            if displayNode:
                displayNode.UpdateScalarRange()

    def _setDisplayedPolyData(self, node: vtkMRMLModelNode, polyData: vtk.vtkPolyData, defaultOpacity: float | None = None, defaultColor: tuple[float, float, float] | None = None) -> None:
        self._setModelNodePolyData(node, polyData)
        displayNode = node.GetDisplayNode()
        if not displayNode:
            node.CreateDefaultDisplayNodes()
            displayNode = node.GetDisplayNode()
            if defaultOpacity is not None:
                displayNode.SetOpacity(defaultOpacity)
            if defaultColor is not None:
                displayNode.SetColor(defaultColor[0], defaultColor[1], defaultColor[2])

    def _polylinePointIdsAndPointsFromPolyData(self, polyData: vtk.vtkPolyData) -> tuple[list[int], list[tuple[float, float, float]]]:
        if not polyData or polyData.GetNumberOfPoints() < 2:
            return [], []

        lines = polyData.GetLines()
        idList = vtk.vtkIdList()
        bestPointIds = None
        bestPointCount = 0
        if lines:
            lines.InitTraversal()
            while lines.GetNextCell(idList):
                pointCount = idList.GetNumberOfIds()
                if pointCount > bestPointCount:
                    bestPointCount = pointCount
                    bestPointIds = [idList.GetId(i) for i in range(pointCount)]

        points = polyData.GetPoints()
        if bestPointIds and len(bestPointIds) >= 2:
            return bestPointIds, [points.GetPoint(pointId) for pointId in bestPointIds]
        pointIds = list(range(polyData.GetNumberOfPoints()))
        return pointIds, [points.GetPoint(pointId) for pointId in pointIds]

    def _startPointIdFromCenterAndLength(self, polyData: vtk.vtkPolyData, centerWorldPosition: list[float], stentLength: float) -> int:
        pointIds, points = self._polylinePointIdsAndPointsFromPolyData(polyData)
        if len(points) < 1:
            raise ValueError("Input centerline has no points")
        if len(points) == 1:
            return int(pointIds[0])

        cumulativeDistances = [0.0]
        for pointIndex in range(1, len(points)):
            cumulativeDistances.append(cumulativeDistances[-1] + vtk.vtkMath.Distance2BetweenPoints(points[pointIndex - 1], points[pointIndex]) ** 0.5)

        centerIndex = min(
            range(len(points)),
            key=lambda index: vtk.vtkMath.Distance2BetweenPoints(points[index], centerWorldPosition),
        )

        targetDistanceFromStart = cumulativeDistances[centerIndex] + max(float(stentLength) * 0.5, 0.0)
        if targetDistanceFromStart >= cumulativeDistances[-1]:
            return int(pointIds[-1])

        for pointIndex in range(centerIndex, len(cumulativeDistances) - 1):
            if cumulativeDistances[pointIndex] <= targetDistanceFromStart <= cumulativeDistances[pointIndex + 1]:
                segmentLength = cumulativeDistances[pointIndex + 1] - cumulativeDistances[pointIndex]
                if segmentLength <= 1e-12:
                    return int(pointIds[pointIndex])
                interpolation = (targetDistanceFromStart - cumulativeDistances[pointIndex]) / segmentLength
                targetLocalIndex = pointIndex + 1 if interpolation >= 0.5 else pointIndex
                return int(pointIds[targetLocalIndex])

        return int(pointIds[-1])

    def process(self, processMessageCallback=None) -> tuple[vtkMRMLModelNode, vtkMRMLModelNode]:
        """Run stent deployment. All inputs and parameters are taken from the module's parameter node."""
        parameterNode = self.getParameterNode()
        inputVesselSegmentation = parameterNode.inputVesselSegmentation
        inputVesselSegmentId = parameterNode.inputVesselSegmentId
        inputCenterlineCurve = parameterNode.inputCenterlineCurve
        centerPointMarkup = parameterNode.centerPointMarkup
        targetRadius = float(parameterNode.targetRadius)
        startRadius = float(parameterNode.startRadius)
        stentLength = float(parameterNode.stentLength)
        enableSnapshots = bool(parameterNode.enableSnapshots)
        verboseLogging = bool(parameterNode.verboseLogging)
        computeArrays = bool(parameterNode.computeOutputModelArrays)
        saveStep = float(parameterNode.saveStep)
        preserveTemporaryFiles = bool(parameterNode.preserveTemporaryFiles)

        if not inputVesselSegmentation or not inputVesselSegmentId or not inputCenterlineCurve:
            raise ValueError("Input surface segmentation/segment and centerline node are required")
        if not centerPointMarkup or centerPointMarkup.GetNumberOfControlPoints() < 1:
            raise ValueError("One center point fiducial is required")
        startRadiusAtStart = startRadius
        if startRadius >= targetRadius:
            logging.warning("Start radius is greater than or equal to target radius. Deployment will start at target radius.")
            startRadius = targetRadius * 0.99
        if enableSnapshots and saveStep <= 0.0:
            raise ValueError("Save step must be > 0 when snapshots are enabled")

        self._cancelRequested = False

        targetRadiusCm = float(targetRadius) * self._mmToCm
        startRadiusCm = float(startRadius) * self._mmToCm
        stentLengthCm = float(stentLength) * self._mmToCm
        saveStepCm = float(saveStep) * self._mmToCm if enableSnapshots else None

        outputSurfaceNode = self._ensureOutputModelNode(parameterNode.outputSurfaceModel, "deployed_surface")
        outputCenterlineNode = self._ensureOutputModelNode(parameterNode.outputCenterlineModel, "deployed_centerline")
        parameterNode.outputSurfaceModel = outputSurfaceNode
        parameterNode.outputCenterlineModel = outputCenterlineNode

        inputSurfacePolyData = self._getSegmentClosedSurfacePolyData(inputVesselSegmentation, inputVesselSegmentId)
        inputCenterlinePolyData = self._getCenterlinePolyData(inputCenterlineCurve)
        inputSurfacePolyDataCm = self._scaledPolyData(inputSurfacePolyData, self._mmToCm)
        inputCenterlinePolyDataCm = self._scaledPolyData(inputCenterlinePolyData, self._mmToCm)

        if not inputSurfacePolyDataCm or inputSurfacePolyDataCm.GetNumberOfPoints() == 0:
            raise ValueError("Input surface model has no polydata")

        centerPointPositionWorld = [0.0, 0.0, 0.0]
        centerPointMarkup.GetNthControlPointPositionWorld(0, centerPointPositionWorld)
        centerPointPositionWorldCm = [c * self._mmToCm for c in centerPointPositionWorld]
        startPointId = self._startPointIdFromCenterAndLength(inputCenterlinePolyDataCm, centerPointPositionWorldCm, stentLengthCm)

        if self._useExternalPythonEnv():
            pythonExePath = self._getMacOSExternalPythonPath(processMessageCallback)
            return self._processWithExternalPython(
                pythonCmd=["arch", "-arm64", pythonExePath],
                inputSurfacePolyDataMm=inputSurfacePolyData,
                inputCenterlinePolyDataMm=inputCenterlinePolyData,
                inputCenterlineCurveNode=inputCenterlineCurve,
                startPointId=startPointId,
                targetRadius=targetRadius,
                startRadius=startRadius,
                stentLength=stentLength,
                enableSnapshots=enableSnapshots,
                verboseLogging=verboseLogging,
                computeOutputModelArrays=computeArrays,
                saveStep=saveStep,
                preserveTemporaryFiles=preserveTemporaryFiles,
                outputSurfaceModel=outputSurfaceNode,
                outputCenterlineModel=outputCenterlineNode,
                processMessageCallback=processMessageCallback,
            )

        # --- In-process path: import svmorph and run ---
        import sys
        moduleDir = os.path.dirname(__file__)
        if moduleDir not in sys.path:
            sys.path.insert(0, moduleDir)

        from svmorph.core import deformation, geometry, mesh_data
        from svmorph.core.defaults import FORESHORTENING_PERCENTAGE
        from svmorph.core.units import L, set_unit_scale
        from svmorph.logging import setup_logging as svmorph_setup_logging, TIMING
        from svmorph.scripts import common
        from svmorph.visualization import vtk_io

        set_unit_scale(1.0)  # working in cm
        svmorph_setup_logging(TIMING if verboseLogging else logging.INFO)

        # --- Determine whether we can reuse the cached deployment state ---

        stateKey = (
            inputVesselSegmentation.GetID(),
            inputVesselSegmentId,
            inputCenterlineCurve.GetID(),
            inputCenterlineCurve.GetMTime(),
            startPointId,
            round(startRadiusCm, 6),
            round(stentLengthCm, 6),
        )

        state = self._deploymentState
        if state and state["key"] == stateKey:
            # Reuse existing simulation context
            ctx = state["ctx"]
            axis_pts = state["axis_pts"]
            smoothing_k = state["smoothing_k"]
            ptCache = state["ptCache"]  # list of (displayed_R, surf_pts, cl_pts)

            lastCachedR = ptCache[-1][0]
            if targetRadiusCm <= lastCachedR:
                # Target is within cached range: serve the closest cached state.
                bestIdx = min(range(len(ptCache)), key=lambda i: abs(ptCache[i][0] - targetRadiusCm))
                bestR, surfPts, clPts = ptCache[bestIdx]
                ctx.data["points"]["surface"][:] = surfPts
                ctx.data["points"]["centerline"][:] = clPts
                vtk_io.sync_polydata(ctx.surface_pd, ctx.data, "surface")
                vtk_io.sync_polydata(ctx.centerline_pd, ctx.data, "centerline")
                with slicer.util.RenderBlocker():
                    outputSurfacePolyData = self._outputPolyData(ctx.surface_pd, surfPts, ptCache[0][1], computeArrays,
                                                                 existingPolyDataMm=outputSurfaceNode.GetPolyData())
                    outputCenterlinePolyData = self._outputPolyData(ctx.centerline_pd, clPts, ptCache[0][2], computeArrays,
                                                                    existingPolyDataMm=outputCenterlineNode.GetPolyData())
                    self._setDisplayedPolyData(outputSurfaceNode, outputSurfacePolyData, defaultOpacity=0.5, defaultColor=(1.0, 0.5, 0.0))
                    self._setDisplayedPolyData(outputCenterlineNode, outputCenterlinePolyData)
                    self._updateOptionalStentOutputs(parameterNode, axis_pts, inputCenterlineCurve, bestR, startRadius, stentLength)
                parameterNode.actualRadius = bestR * self._cmToMm
                logging.info(f"Served from cache at R={bestR:.4f} cm (target={targetRadiusCm:.4f} cm)")
                return outputSurfaceNode, outputCenterlineNode

            # Target exceeds cache: restore to last cached state and continue from there.
            lastR, lastSurfPts, lastClPts = ptCache[-1]
            ctx.data["points"]["surface"][:] = lastSurfPts
            ctx.data["points"]["centerline"][:] = lastClPts
            cur_R = lastR - smoothing_k
            logging.info(f"Continuing deployment from R={lastR:.4f} cm to target R={targetRadiusCm:.4f} cm")
        else:
            # Fresh start: build simulation context from input polydata
            data = vtk_io.extract_mesh_arrays(inputSurfacePolyDataCm, inputCenterlinePolyDataCm)
            parent_tip_map, segment_base_mask = vtk_io.build_parent_tip_map(inputCenterlinePolyDataCm)
            tangents = vtk_io.extract_centerline_tangents(inputCenterlinePolyDataCm)
            inscribed_sphere_radii = vtk_io.extract_inscribed_sphere_radii(inputCenterlinePolyDataCm)
            ctx = common.SimulationContext(
                data=data,
                parent_tip_map=parent_tip_map,
                segment_base_mask=segment_base_mask,
                tangents=tangents,
                inscribed_sphere_radii=inscribed_sphere_radii,
                surface_pd=inputSurfacePolyDataCm,
                centerline_pd=inputCenterlinePolyDataCm,
            )

            deformation.set_node_indices(ctx.data, [startPointId])
            deformation.set_force_center(ctx.data, startPointId)

            # Foreshortening convention from svmorph:
            # https://github.com/SimVascular/svMorph/blob/main/svmorph/core/defaults.py
            deployed_length = stentLengthCm * (1 - FORESHORTENING_PERCENTAGE)
            axis_pts = geometry.resample_stent_axis(
                ctx.data["points"]["centerline_points_view_np"],
                ctx.parent_tip_map,
                ctx.segment_base_mask,
                startPointId,
                deployed_length,
                0.1 * L(),
                sampling_direction=-1,
            )

            mesh_data.compute_material_constants(1.0, 0.2)
            smoothing_k = 0.01 * L()
            cur_R = startRadiusCm - smoothing_k

            # Save the initial (undeployed) state as the first cache entry
            ptCache = [(startRadiusCm, ctx.data["points"]["surface"].copy(), ctx.data["points"]["centerline"].copy())]
            state = {
                "key": stateKey,
                "ctx": ctx,
                "axis_pts": axis_pts,
                "smoothing_k": smoothing_k,
                "ptCache": ptCache,
            }
            self._deploymentState = state

            # Show the initial (undeployed) surface immediately (with zero displacements)
            with slicer.util.RenderBlocker():
                outputSurfacePolyData = self._outputPolyData(ctx.surface_pd, ptCache[0][1], ptCache[0][1], computeArrays)
                outputCenterlinePolyData = self._outputPolyData(ctx.centerline_pd, ptCache[0][2], ptCache[0][2], computeArrays)
                self._setDisplayedPolyData(outputSurfaceNode, outputSurfacePolyData, defaultOpacity=0.5, defaultColor=(1.0, 0.5, 0.0))
                self._setDisplayedPolyData(outputCenterlineNode, outputCenterlinePolyData)
                self._updateOptionalStentOutputs(parameterNode, axis_pts, inputCenterlineCurve, startRadiusCm, startRadius, stentLength)
            ptCache = state["ptCache"]
            logging.info(f"Starting fresh deployment: start_R={startRadiusCm:.4f} cm, target_R={targetRadiusCm:.4f} cm")

        # --- Deployment loop ---
        workDir = tempfile.mkdtemp(prefix="SDFStent_") if (enableSnapshots or preserveTemporaryFiles) else None
        shouldDeleteTemporaryFiles = not preserveTemporaryFiles
        try:
            snapshotMgr = None
            if workDir and enableSnapshots:
                workDirPath = pathlib.Path(workDir)
                snapshotMgr = common.SnapshotManager(
                    start_value=startRadiusCm,
                    target_value=targetRadiusCm,
                    save_step=saveStepCm,
                    out_mesh_path=str(workDirPath / "deployed_surface.vtp"),
                    out_cl_path=str(workDirPath / "deployed_centerline.vtp"),
                    surface_pd=ctx.surface_pd,
                    centerline_pd=ctx.centerline_pd,
                    data=ctx.data,
                )

            startTime = qt.QDateTime.currentDateTimeUtc()
            iteration = 0
            t0 = time.time()
            while True:
                if self._cancelRequested:
                    raise SDFStentCancelledError("Deployment cancelled by user")

                currentCenterPos = [0.0, 0.0, 0.0]
                if centerPointMarkup.GetNumberOfControlPoints() > 0:
                    centerPointMarkup.GetNthControlPointPositionWorld(0, currentCenterPos)
                if (
                    float(parameterNode.startRadius) != startRadiusAtStart
                    or float(parameterNode.stentLength) != stentLength
                    or currentCenterPos != centerPointPositionWorld
                ):
                    raise SDFStentRestartError("Parameters changed during deployment")

                latestTargetCm = float(parameterNode.targetRadius) * self._mmToCm
                if latestTargetCm > targetRadiusCm:
                    targetRadiusCm = latestTargetCm

                surf_disp, cl_disp, dR = deformation.compute_sdf_contact_displacements(
                    ctx.data,
                    axis_pts,
                    s=-1.0,
                    target_stent_radius=targetRadiusCm,
                    current_stent_radius=cur_R,
                )
                if cur_R + dR > targetRadiusCm:
                    logging.info("Next increment would overshoot target -- done.")
                    break

                mesh_data.apply_displacements(ctx.data, surf_disp, "surface")
                mesh_data.apply_displacements(ctx.data, cl_disp, "centerline")
                cur_R += dR
                iteration += 1
                displayed_R = cur_R + smoothing_k

                ptCache.append((displayed_R, ctx.data["points"]["surface"].copy(), ctx.data["points"]["centerline"].copy()))

                msg = f"Step {iteration:3d}: dR={dR:.5f}  R={displayed_R:.5f}"
                logging.info(msg)
                if processMessageCallback:
                    processMessageCallback(msg, False)

                if snapshotMgr:
                    snapshotMgr.check_and_save(displayed_R)

                # Real-time update of output models
                vtk_io.sync_polydata(ctx.surface_pd, ctx.data, "surface")
                vtk_io.sync_polydata(ctx.centerline_pd, ctx.data, "centerline")
                with slicer.util.RenderBlocker():
                    self._setModelNodePolyData(
                        outputSurfaceNode,
                        self._outputPolyData(ctx.surface_pd, ctx.data["points"]["surface"], ptCache[0][1], computeArrays,
                                             existingPolyDataMm=outputSurfaceNode.GetPolyData()))
                    self._setModelNodePolyData(
                        outputCenterlineNode,
                        self._outputPolyData(ctx.centerline_pd, ctx.data["points"]["centerline"], ptCache[0][2], computeArrays,
                                             existingPolyDataMm=outputCenterlineNode.GetPolyData()))
                    self._updateOptionalStentOutputs(parameterNode, axis_pts, inputCenterlineCurve, displayed_R, startRadius, stentLength)
                slicer.app.processEvents()

            elapsed = time.time() - t0
            logging.info(f"Deployment complete: {iteration} steps in {elapsed:.2f} s")

            # Final update (covers the case where the loop exited on the first check)
            vtk_io.sync_polydata(ctx.surface_pd, ctx.data, "surface")
            vtk_io.sync_polydata(ctx.centerline_pd, ctx.data, "centerline")
            with slicer.util.RenderBlocker():
                self._setModelNodePolyData(
                    outputSurfaceNode,
                    self._outputPolyData(ctx.surface_pd, ctx.data["points"]["surface"], ptCache[0][1], computeArrays,
                                         existingPolyDataMm=outputSurfaceNode.GetPolyData()))
                self._setModelNodePolyData(
                    outputCenterlineNode,
                    self._outputPolyData(ctx.centerline_pd, ctx.data["points"]["centerline"], ptCache[0][2], computeArrays,
                                         existingPolyDataMm=outputCenterlineNode.GetPolyData()))
                self._updateOptionalStentOutputs(parameterNode, axis_pts, inputCenterlineCurve, ptCache[-1][0], startRadius, stentLength)

            parameterNode.actualRadius = ptCache[-1][0] * self._cmToMm
            elapsedMs = startTime.msecsTo(qt.QDateTime.currentDateTimeUtc())
            logging.info(f"SDFStent completed in {elapsedMs / 1000.0:.2f} seconds")
            return outputSurfaceNode, outputCenterlineNode
        finally:
            if workDir:
                if shouldDeleteTemporaryFiles:
                    shutil.rmtree(workDir, ignore_errors=True)
                else:
                    logging.info(f"Preserved temporary files in: {workDir}")
                    if processMessageCallback:
                        processMessageCallback(f"Preserved temporary files in: {workDir}", False)


    def _useExternalPythonEnv(self) -> bool:
        # jax is not available in on Apple Silicon in x86_64 executables (using Rosetta2 translation).
        # Therefore, for these cases we need to use an external Python environment for the computations.
        import sys
        if sys.platform != "darwin":
            return False
        import subprocess
        result = subprocess.run(["sysctl", "-n", "sysctl.proc_translated"], capture_output=True, text=True)
        return result.stdout.strip() == "1"

    def _getMacOSExternalPythonPath(self, processMessageCallback=None) -> str:
        import subprocess
        envDir = pathlib.Path.home() / ".SlicerSimVascularPythonEnv"
        pythonExe = envDir / "bin" / "python3"

        if pythonExe.exists():
            test = subprocess.run(
                ["arch", "-arm64", str(pythonExe), "-c", "pass"],
                capture_output=True, env=slicer.util.startupEnvironment(),
            )
            if test.returncode != 0:
                msg = "Removing x86 Python environment and recreating with ARM Python..."
                logging.info(msg)
                if processMessageCallback:
                    processMessageCallback(msg, False)
                shutil.rmtree(str(envDir), ignore_errors=True)

        if not pythonExe.exists():
            msg = f"Creating ARM Python environment at {envDir}..."
            logging.info(msg)
            if processMessageCallback:
                processMessageCallback(msg, False)
            result = subprocess.run(
                ["arch", "-arm64", "/usr/bin/python3", "-m", "venv", str(envDir)],
                capture_output=True, text=True, env=slicer.util.startupEnvironment(),
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to create ARM Python virtual environment at {envDir}.\n{result.stderr}\n"
                    "Make sure /usr/bin/python3 supports ARM (install Xcode Command Line Tools)."
                )
            logging.info(f"ARM Python environment created at {envDir}")
        return str(pythonExe)

    def _ensureExternalDependencies(self, pythonCmd: list, env: dict, processMessageCallback) -> None:
        import subprocess
        required = [("svmorph", "svmorph")]
        for importName, pipName in required:
            check = subprocess.run(
                pythonCmd + ["-c", f"import {importName}"],
                capture_output=True, text=True, env=env,
            )
            if check.returncode == 0:
                continue
            msg = f"Installing {pipName} in external Python environment..."
            logging.info(msg)
            if processMessageCallback:
                processMessageCallback(msg, False)
            proc = subprocess.Popen(
                pythonCmd + ["-m", "pip", "install", pipName],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            for line in proc.stdout:
                line = line.rstrip()
                if line and processMessageCallback:
                    processMessageCallback(line, False)
                slicer.app.processEvents()
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"Failed to install '{pipName}' in the external Python environment.")
            logging.info(f"Successfully installed {pipName}.")
            if processMessageCallback:
                processMessageCallback(f"Successfully installed {pipName}.", False)

    def _processWithExternalPython(
        self,
        pythonCmd: list,
        inputSurfacePolyDataMm: vtk.vtkPolyData,
        inputCenterlinePolyDataMm: vtk.vtkPolyData,
        inputCenterlineCurveNode: vtkMRMLNode,
        startPointId: int,
        targetRadius: float,
        startRadius: float,
        stentLength: float,
        enableSnapshots: bool,
        verboseLogging: bool,
        computeOutputModelArrays: bool,
        saveStep: float,
        preserveTemporaryFiles: bool,
        outputSurfaceModel: vtkMRMLModelNode,
        outputCenterlineModel: vtkMRMLModelNode,
        processMessageCallback=None,
    ) -> tuple[vtkMRMLModelNode, vtkMRMLModelNode]:
        import json
        import subprocess

        workDir = tempfile.mkdtemp(prefix="SDFStent_")
        shouldDeleteTemporaryFiles = not preserveTemporaryFiles
        try:
            workDirPath = pathlib.Path(workDir)

            surfaceWriter = vtk.vtkXMLPolyDataWriter()
            surfaceWriter.SetFileName(str(workDirPath / "surface_input.vtp"))
            surfaceWriter.SetInputData(inputSurfacePolyDataMm)
            surfaceWriter.Write()

            centerlineWriter = vtk.vtkXMLPolyDataWriter()
            centerlineWriter.SetFileName(str(workDirPath / "centerline_input.vtp"))
            centerlineWriter.SetInputData(inputCenterlinePolyDataMm)
            centerlineWriter.Write()

            params = {
                "targetRadius": targetRadius,
                "startRadius": startRadius,
                "stentLength": stentLength,
                "startPointId": startPointId,
                "verboseLogging": verboseLogging,
                "enableSnapshots": enableSnapshots,
                "computeOutputModelArrays": computeOutputModelArrays,
                "saveStep": saveStep,
            }
            with open(workDirPath / "params.json", "w") as f:
                json.dump(params, f)

            env = slicer.util.startupEnvironment()
            self._ensureExternalDependencies(pythonCmd, env, processMessageCallback)

            if processMessageCallback:
                processMessageCallback("Launching ARM Python worker...", False)
            workerScript = os.path.join(os.path.dirname(__file__), "Resources", "Scripts", "SDFStent_worker.py")
            proc = subprocess.Popen(
                pythonCmd + [workerScript, workDir],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )

            for line in proc.stdout:
                line = line.rstrip()
                if self._cancelRequested:
                    proc.terminate()
                    proc.wait()
                    raise SDFStentCancelledError("Deployment cancelled by user")
                if line and processMessageCallback:
                    processMessageCallback(line, False)
                slicer.app.processEvents()

            proc.wait()

            if self._cancelRequested:
                raise SDFStentCancelledError("Deployment cancelled by user")

            resultPath = workDirPath / "result.json"
            if resultPath.exists():
                with open(resultPath) as f:
                    result = json.load(f)
                if not result.get("success"):
                    raise RuntimeError(f"Worker failed: {result.get('error', 'unknown error')}")
                self.getParameterNode().actualRadius = float(result.get("actualRadius", 0.0))
            else:
                raise RuntimeError(f"Worker exited with code {proc.returncode} and produced no result file")

            surfaceReader = vtk.vtkXMLPolyDataReader()
            surfaceReader.SetFileName(str(workDirPath / "surface_output.vtp"))
            surfaceReader.Update()

            centerlineReader = vtk.vtkXMLPolyDataReader()
            centerlineReader.SetFileName(str(workDirPath / "centerline_output.vtp"))
            centerlineReader.Update()

            self._setDisplayedPolyData(outputSurfaceModel, surfaceReader.GetOutput(), defaultOpacity=0.5, defaultColor=(1.0, 0.5, 0.0))
            self._setDisplayedPolyData(outputCenterlineModel, centerlineReader.GetOutput())

            stentAxisPath = workDirPath / "stent_axis.json"
            if stentAxisPath.exists():
                with open(stentAxisPath) as f:
                    axisPointsCm = json.load(f)
                self._updateOptionalStentOutputs(self.getParameterNode(), axisPointsCm, inputCenterlineCurveNode,
                                                 float(result.get("actualRadius", 0.0)) * self._mmToCm,
                                                 startRadius, stentLength)
            return outputSurfaceModel, outputCenterlineModel
        finally:
            if shouldDeleteTemporaryFiles:
                shutil.rmtree(workDir, ignore_errors=True)
            else:
                logging.info(f"Preserved temporary files in: {workDir}")
                if processMessageCallback:
                    processMessageCallback(f"Preserved temporary files in: {workDir}", False)


class SDFStentTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_SDFStent_deployVessel01()

    def test_SDFStent_deployVessel01(self):
        import numpy as np
        from vtk.util.numpy_support import vtk_to_numpy

        if not importlib.util.find_spec("svmorph"):
            self.delayDisplay("Installing svmorph...")
            slicer.util.pip_install("svmorph")

        self.delayDisplay("Loading Vessel01 sample data")
        import SampleData
        loadedNodes = SampleData.SampleDataLogic().downloadSamples("Vessel01")
        segmentationNode = next(node for node in loadedNodes if node.IsA("vtkMRMLSegmentationNode"))
        centerlineNode = next(node for node in loadedNodes if node.IsA("vtkMRMLMarkupsCurveNode"))
        self.assertGreater(segmentationNode.GetSegmentation().GetNumberOfSegments(), 0)
        segmentId = segmentationNode.GetSegmentation().GetNthSegmentID(0)
        segmentationNode.GetDisplayNode().SetOpacity3D(0.5)

        logic = SDFStentLogic()
        parameterNode = logic.getParameterNode()
        parameterNode.inputVesselSegmentation = segmentationNode
        parameterNode.inputVesselSegmentId = segmentId
        parameterNode.inputCenterlineCurve = centerlineNode

        # Use the current center point if one is already placed; otherwise place one in the
        # straight vessel section
        curvePoints = slicer.util.arrayFromMarkupsCurvePoints(centerlineNode, world=True)
        self.assertGreater(len(curvePoints), 2)
        centerPointNode = parameterNode.centerPointMarkup
        if (not centerPointNode) or (centerPointNode.GetScene() is None):
            centerPointNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "CenterPoint")
            parameterNode.centerPointMarkup = centerPointNode
        if centerPointNode.GetNumberOfControlPoints() == 0:
            centerPointNode.AddControlPointWorld(-32.3214, -54.5704, 113.5213)
        centerPosition = [0.0, 0.0, 0.0]
        centerPointNode.GetNthControlPointPositionWorld(0, centerPosition)
        centerPosition = np.array(centerPosition)

        inputSurfacePolyData = logic._getSegmentClosedSurfacePolyData(segmentationNode, segmentId)
        inputSurfacePoints = vtk_to_numpy(inputSurfacePolyData.GetPoints().GetData())
        inputRadiusAtCenter = np.min(np.linalg.norm(inputSurfacePoints - centerPosition, axis=1))

        parameterNode.outputStentTransform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", "StentTransform")
        parameterNode.outputStraightStentModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "StraightStent")

        # The target radius must exceed the local vessel radius so that the deployment visibly expands the vessel
        parameterNode.startRadius = 3.0
        parameterNode.targetRadius = 9.0
        parameterNode.stentLength = 45.0
        parameterNode.computeOutputModelArrays = True
        targetRadius = parameterNode.targetRadius
        self.assertLess(inputRadiusAtCenter, targetRadius)

        self.delayDisplay("Deploying stent")
        outputSurfaceNode, outputCenterlineNode = logic.process()

        # Output surface and centerline are valid meshes with the same topology as the input
        outputSurfacePolyData = outputSurfaceNode.GetPolyData()
        outputCenterlinePolyData = outputCenterlineNode.GetPolyData()
        self.assertEqual(outputSurfacePolyData.GetNumberOfPoints(), inputSurfacePolyData.GetNumberOfPoints())
        self.assertEqual(outputSurfacePolyData.GetNumberOfCells(), inputSurfacePolyData.GetNumberOfCells())
        self.assertGreater(outputCenterlinePolyData.GetNumberOfPoints(), 2)

        # Reported deployed radius reached the target
        self.assertAlmostEqual(logic.getParameterNode().actualRadius, targetRadius, delta=0.5)

        # The vessel wall around the center point expanded to the target radius
        outputSurfacePoints = vtk_to_numpy(outputSurfacePolyData.GetPoints().GetData())
        outputCenterlinePoints = vtk_to_numpy(outputCenterlinePolyData.GetPoints().GetData())
        deployedCenterPosition = outputCenterlinePoints[np.argmin(np.linalg.norm(outputCenterlinePoints - centerPosition, axis=1))]
        outputRadiusAtCenter = np.min(np.linalg.norm(outputSurfacePoints - deployedCenterPosition, axis=1))
        self.assertGreater(outputRadiusAtCenter, inputRadiusAtCenter + 1.0)
        self.assertAlmostEqual(outputRadiusAtCenter, targetRadius, delta=1.0)

        # Surface far away from the stented region (beyond the stent, its end caps, and the
        # deformation influence region) must remain unchanged
        surfaceDisplacements = np.linalg.norm(outputSurfacePoints - inputSurfacePoints, axis=1)
        farFromStentMask = np.linalg.norm(inputSurfacePoints - centerPosition, axis=1) > 0.5 * parameterNode.stentLength + 20.0
        self.assertLess(np.max(surfaceDisplacements[farFromStentMask]), 0.1)

        # Displacement point data arrays must point from the deployed positions back to the original positions
        surfaceDisplacementArray = outputSurfacePolyData.GetPointData().GetArray("Displacement")
        self.assertIsNotNone(surfaceDisplacementArray)
        self.assertEqual(surfaceDisplacementArray.GetNumberOfComponents(), 3)
        self.assertEqual(surfaceDisplacementArray.GetNumberOfTuples(), outputSurfacePolyData.GetNumberOfPoints())
        np.testing.assert_allclose(vtk_to_numpy(surfaceDisplacementArray), inputSurfacePoints - outputSurfacePoints, atol=0.01)
        centerlineDisplacementArray = outputCenterlinePolyData.GetPointData().GetArray("Displacement")
        self.assertIsNotNone(centerlineDisplacementArray)
        self.assertEqual(centerlineDisplacementArray.GetNumberOfComponents(), 3)
        self.assertEqual(centerlineDisplacementArray.GetNumberOfTuples(), outputCenterlinePolyData.GetNumberOfPoints())

        # Principal strain cell data: expansion inside the stented region, no strain far from it
        principalStrainMax = vtk_to_numpy(outputSurfacePolyData.GetCellData().GetArray("PrincipalStrainMax"))
        self.assertEqual(len(principalStrainMax), outputSurfacePolyData.GetNumberOfCells())
        self.assertIsNotNone(outputSurfacePolyData.GetCellData().GetArray("PrincipalStrainMin"))
        self.assertIsNotNone(outputSurfacePolyData.GetCellData().GetArray("AreaStrain"))
        surfaceTriangles = vtk_to_numpy(outputSurfacePolyData.GetPolys().GetConnectivityArray()).reshape(-1, 3)
        cellDistancesFromCenter = np.linalg.norm(inputSurfacePoints[surfaceTriangles].mean(axis=1) - centerPosition, axis=1)
        self.assertGreater(np.mean(principalStrainMax[cellDistancesFromCenter < 10.0]), 0.1)
        self.assertLess(np.max(np.abs(principalStrainMax[cellDistancesFromCenter > 0.5 * parameterNode.stentLength + 20.0])), 0.02)

        # Straight stent model: cylinder with start radius and stent length, centered at the origin, long axis along Z
        straightStentPolyData = parameterNode.outputStraightStentModel.GetPolyData()
        straightStentPoints = vtk_to_numpy(straightStentPolyData.GetPoints().GetData())
        np.testing.assert_allclose(np.linalg.norm(straightStentPoints[:, :2], axis=1), parameterNode.startRadius, atol=0.01)
        self.assertAlmostEqual(np.min(straightStentPoints[:, 2]), -0.5 * parameterNode.stentLength, delta=0.01)
        self.assertAlmostEqual(np.max(straightStentPoints[:, 2]), 0.5 * parameterNode.stentLength, delta=0.01)

        # Stent transform: must move the model center onto the centerline, close to where the center
        # point (which may have been placed off the centerline) is snapped to the centerline
        stentTransform = parameterNode.outputStentTransform.GetTransformToParent()
        transformedCenter = np.array(stentTransform.TransformPoint([0.0, 0.0, 0.0]))
        self.assertLess(np.min(np.linalg.norm(curvePoints - transformedCenter, axis=1)), 1.5)
        nearestCurvePointToCenter = curvePoints[np.argmin(np.linalg.norm(curvePoints - centerPosition, axis=1))]
        self.assertLess(np.linalg.norm(transformedCenter - nearestCurvePointToCenter), 0.1 * parameterNode.stentLength + 2.0)
        warpedRadii = [np.min(np.linalg.norm(curvePoints - np.array(stentTransform.TransformPoint(p)), axis=1))
                       for p in straightStentPoints[::20]]
        np.testing.assert_allclose(warpedRadii, targetRadius, atol=1.5)

        # Beyond the stent length the transform must follow the centerline without radial expansion
        for zBeyondStent in (-0.5 * parameterNode.stentLength - 12.0, 0.5 * parameterNode.stentLength + 12.0):
            transformedBeyond = np.array(stentTransform.TransformPoint([parameterNode.startRadius, 0.0, zBeyondStent]))
            distanceToCenterline = np.min(np.linalg.norm(curvePoints - transformedBeyond, axis=1))
            self.assertLess(abs(distanceToCenterline - parameterNode.startRadius), 1.5)

        # Apply the stent transform to a realistic stent mesh (derived from Open Stent Design)
        self.delayDisplay("Loading Stent_30x476 sample stent mesh")
        stentMeshNode = SampleData.SampleDataLogic().downloadSamples("Stent_30x476")[0]
        stentMeshNode.SetAndObserveTransformNodeID(parameterNode.outputStentTransform.GetID())
        stentMeshPoints = vtk_to_numpy(stentMeshNode.GetPolyData().GetPoints().GetData())
        self.assertGreater(len(stentMeshPoints), 1000)
        # points well inside the stented section must be expanded by the target/start radius ratio
        insideStentPoints = stentMeshPoints[np.abs(stentMeshPoints[:, 2]) < 0.5 * parameterNode.stentLength - 5.0][::200]
        expectedStentRadii = np.linalg.norm(insideStentPoints[:, :2], axis=1) * targetRadius / parameterNode.startRadius
        warpedStentRadii = np.array([np.min(np.linalg.norm(curvePoints - np.array(stentTransform.TransformPoint(p)), axis=1))
                                     for p in insideStentPoints])
        np.testing.assert_allclose(warpedStentRadii, expectedStentRadii, atol=1.5)

        # Show the realistic stent mesh instead of the straight stent template
        parameterNode.outputStraightStentModel.GetDisplayNode().SetVisibility(False)

        # The stent transform must be savable to file (requires ITK-convertible transform components)
        transformFilePath = os.path.join(slicer.app.temporaryPath, "SDFStentTest_StentTransform.h5")
        self.assertTrue(slicer.util.saveNode(parameterNode.outputStentTransform, transformFilePath))
        os.remove(transformFilePath)

        self.delayDisplay("SDFStent Vessel01 deployment test passed")
