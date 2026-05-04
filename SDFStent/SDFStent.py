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

from slicer import vtkMRMLMarkupsCurveNode, vtkMRMLMarkupsFiducialNode, vtkMRMLMarkupsNode, vtkMRMLModelNode, vtkMRMLNode, vtkMRMLSegmentationNode


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
              "https://github.com/SimVascular/SlicerSimVascular/releases/download/testing-data/Vessel01_Centerline.mrk.json"],
        checksums=["SHA256:a9071c6e5e37267720c9c6c3963d3a2b12a3ae1f017eebc3354c4f669629cf00",
                   "SHA256:16faa9c7afb819dc419708edaca37eeb603fbb0fa2840c5548605b3b798fdc36"],
        fileNames=["Vessel01.seg.nrrd", "Vessel01.mrk.json"],
        nodeNames=["Vessel01 Segmentation", "Vessel01 Centerline"],
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
    saveStep: Annotated[float, WithinRange(0.0001, 100.0)] = 1.0
    preserveTemporaryFiles: bool = False
    outputMeshFileName: str = "deployed_surface.vtp"
    outputCenterlineFileName: str = "deployed_centerline.vtp"
    outputSurfaceModel: Optional[vtkMRMLModelNode] = None
    outputCenterlineModel: Optional[vtkMRMLModelNode] = None
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
        self.ui.inputSurfaceSelector.currentNodeChanged.connect(self._checkCanApply)
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
        if not self._parameterNode:
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
        centerPointMarkupNode = self._ensureCenterPointMarkupNode()
        hasCenterPoint = bool(centerPointMarkupNode and centerPointMarkupNode.GetNumberOfControlPoints() > 0)

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

                outputSurfaceModelNode = self.ui.outputSurfaceSelector.currentNode()
                if outputSurfaceModelNode is None:
                    outputSurfaceModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "deployed_surface")
                    self.ui.outputSurfaceSelector.setCurrentNode(outputSurfaceModelNode)

                outputCenterlineModelNode = self.ui.outputCenterlineSelector.currentNode()
                if outputCenterlineModelNode is None:
                    outputCenterlineModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "deployed_centerline")
                    self.ui.outputCenterlineSelector.setCurrentNode(outputCenterlineModelNode)

                outputSurfaceNode, outputCenterlineNode = self.logic.process(
                    inputVesselSegmentation=self.ui.inputSurfaceSelector.currentNode(),
                    inputVesselSegmentId=self.ui.inputSurfaceSelector.currentSegmentID(),
                    inputCenterlineCurve=self.ui.inputCenterlineSelector.currentNode(),
                    centerPointMarkup=centerPointMarkupNode,
                    targetRadius=targetRadiusAtStart,
                    startRadius=float(self.ui.startRadiusSpinBox.value),
                    stentLength=float(self.ui.stentLengthSpinBox.value),
                    enableSnapshots=bool(self.ui.enableSnapshotsCheckBox.checked),
                    verboseLogging=bool(self.ui.verboseLoggingCheckBox.checked),
                    saveStep=float(self.ui.saveStepSpinBox.value),
                    preserveTemporaryFiles=bool(self.ui.preserveTempFilesCheckBox.checked),
                    outputSurfaceModel=outputSurfaceModelNode,
                    outputCenterlineModel=outputCenterlineModelNode,
                    processMessageCallback=self._handleProcessMessage,
                )
                if outputSurfaceNode:
                    self.ui.outputSurfaceSelector.setCurrentNode(outputSurfaceNode)
                if outputCenterlineNode:
                    self.ui.outputCenterlineSelector.setCurrentNode(outputCenterlineNode)

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

    def _setDisplayedPolyData(self, node: vtkMRMLModelNode, polyData: vtk.vtkPolyData, defaultOpacity: float | None = None, defaultColor: tuple[float, float, float] | None = None) -> None:
        node.SetAndObservePolyData(polyData)
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

    def process(
        self,
        inputVesselSegmentation: vtkMRMLSegmentationNode,
        inputVesselSegmentId: str,
        inputCenterlineCurve: vtkMRMLNode,
        centerPointMarkup: vtkMRMLMarkupsFiducialNode,
        targetRadius: float,
        startRadius: float,
        stentLength: float,
        enableSnapshots: bool,
        verboseLogging: bool,
        saveStep: float,
        preserveTemporaryFiles: bool,
        outputSurfaceModel: vtkMRMLModelNode | None = None,
        outputCenterlineModel: vtkMRMLModelNode | None = None,
        processMessageCallback=None,
    ) -> tuple[vtkMRMLModelNode, vtkMRMLModelNode]:
        if not inputVesselSegmentation or not inputVesselSegmentId or not inputCenterlineCurve:
            raise ValueError("Input surface segmentation/segment and centerline node are required")
        if not centerPointMarkup or centerPointMarkup.GetNumberOfControlPoints() < 1:
            raise ValueError("One center point fiducial is required")
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

        import sys
        moduleDir = os.path.dirname(__file__)
        if moduleDir not in sys.path:
            sys.path.insert(0, moduleDir)

        from svmorph.core import deformation, geometry, mesh_data
        from svmorph.core.units import L, set_unit_scale
        from svmorph.logging import setup_logging as svmorph_setup_logging, TIMING
        from svmorph.scripts import common
        from svmorph.visualization import vtk_io

        set_unit_scale(1.0)  # working in cm
        svmorph_setup_logging(TIMING if verboseLogging else logging.INFO)

        outputSurfaceNodeName = "deployed_surface"
        outputCenterlineNodeName = "deployed_centerline"
        outputSurfaceNode = self._ensureOutputModelNode(outputSurfaceModel, outputSurfaceNodeName)
        outputCenterlineNode = self._ensureOutputModelNode(outputCenterlineModel, outputCenterlineNodeName)

        # --- Determine whether we can reuse the cached deployment state ---
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
                outputSurfacePolyData = self._scaledPolyData(ctx.surface_pd, self._cmToMm)
                outputCenterlinePolyData = self._scaledPolyData(ctx.centerline_pd, self._cmToMm)
                self._setDisplayedPolyData(outputSurfaceNode, outputSurfacePolyData, defaultOpacity=0.5, defaultColor=(1.0, 0.5, 0.0))
                self._setDisplayedPolyData(outputCenterlineNode, outputCenterlinePolyData)
                self.getParameterNode().actualRadius = bestR * self._cmToMm
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

            foreshortening = 0.1
            deployed_length = stentLengthCm * (1 - foreshortening)
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

            # Show the initial (undeployed) surface immediately
            outputSurfacePolyData = self._scaledPolyData(ctx.surface_pd, self._cmToMm)
            outputCenterlinePolyData = self._scaledPolyData(ctx.centerline_pd, self._cmToMm)
            self._setDisplayedPolyData(outputSurfaceNode, outputSurfacePolyData, defaultOpacity=0.5, defaultColor=(1.0, 0.5, 0.0))
            self._setDisplayedPolyData(outputCenterlineNode, outputCenterlinePolyData)
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

            parameterNode = self.getParameterNode()
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
                    float(parameterNode.startRadius) != startRadius
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
                outputSurfaceNode.SetAndObservePolyData(self._scaledPolyData(ctx.surface_pd, self._cmToMm))
                outputCenterlineNode.SetAndObservePolyData(self._scaledPolyData(ctx.centerline_pd, self._cmToMm))
                slicer.app.processEvents()

            elapsed = time.time() - t0
            logging.info(f"Deployment complete: {iteration} steps in {elapsed:.2f} s")

            # Final update (covers the case where the loop exited on the first check)
            vtk_io.sync_polydata(ctx.surface_pd, ctx.data, "surface")
            vtk_io.sync_polydata(ctx.centerline_pd, ctx.data, "centerline")
            outputSurfaceNode.SetAndObservePolyData(self._scaledPolyData(ctx.surface_pd, self._cmToMm))
            outputCenterlineNode.SetAndObservePolyData(self._scaledPolyData(ctx.centerline_pd, self._cmToMm))

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


class SDFStentTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_SDFStent_smoke()

    def test_SDFStent_smoke(self):
        self.delayDisplay("Starting SDFStent smoke test")

        surfaceNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "surface")
        centerlineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "centerline")
        sphere = vtk.vtkSphereSource()
        sphere.Update()
        line = vtk.vtkLineSource()
        line.Update()
        surfaceNode.SetAndObservePolyData(sphere.GetOutput())
        centerlineNode.SetAndObservePolyData(line.GetOutput())

        import sys
        moduleDir = os.path.dirname(slicer.util.modulePath("SDFStent"))
        if moduleDir not in sys.path:
            sys.path.insert(0, moduleDir)
        from svmorph.scripts import deploy_stent
        self.assertTrue(hasattr(deploy_stent, "main"))
        logic = SDFStentLogic()
        self.assertIsNotNone(logic)

        self.delayDisplay("SDFStent smoke test passed")
