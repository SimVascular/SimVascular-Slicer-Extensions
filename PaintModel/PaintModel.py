import logging
import math
import colorsys
import functools

import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

#
# PaintModel
#

class PaintModel(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Paint Model"
    self.parent.categories = ["Surface Models"]
    self.parent.dependencies = []
    self.parent.contributors = ["Aaron Brown (Stanford)"]
    self.parent.helpText = """
Paint interactively on a surface model to partition it into face groups, then
refine, merge, and export those groups. Brush-select cells with the S/D hotkeys,
create groups automatically from a dihedral-angle threshold, expand a selection
to whole groups, carve a new group from the current selection, and export the
result (a "ModelFaceID" cell array) as a .vtp file.
"""
    self.parent.acknowledgementText = """
The automatic face-grouping workflow (partition by angle threshold, then
merge small groups, then manually paint or edit groups) is inspired by the
face-group tools in Autodesk Meshmixer.
"""

#
# PaintModelWidget
#

class PaintModelWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)
    self.logic = None
    self._faceGroupSelection = set()
    self._brushObserverTags = []
    self._brushInteractorCallbacks = []
    self._brushInteractionEnabled = False
    self._brushDragging = False
    self._brushMode = None
    self._brushLastLocalPosition = None
    self._brushHotkeysDown = set()
    self._brushTopologyNodeID = None
    self._brushCellNeighbors = None
    self._brushCellCentroids = None
    self._brushCursorSphere = None
    self._brushCursorModelNode = None
    self._selectionModelNode = None
    self._selectionDisplayModelNode = None
    self._faceGroupColorNode = None
    self._faceGroupColorsByModel = {}

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    self.logic = PaintModelLogic()

    # Paint Model operates in-place on the selected model by adding/editing
    # ModelFaceID cell data. The UI is built entirely in Python since this
    # module has no destructive filter pipeline to share a form with.
    self.setupFaceGroupsUI()

    # These connections ensure that helper nodes are cleared out when the scene closes.
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    self.layout.addStretch(1)
    self.setBrushEnabled(True)

  def cleanup(self):
    """
    Called when the application closes and the module widget is destroyed.
    """
    self.removeObservers()
    self.setBrushEnabled(False)
    self.removeFaceGroupHelperNodes()

  def enter(self):
    """
    Called each time the user opens this module.
    """
    self.setBrushEnabled(True)

  def exit(self):
    """
    Called each time the user opens a different module.
    """
    self.setBrushEnabled(False)

  def onSceneStartClose(self, caller=None, event=None):
    self.setBrushEnabled(False)
    self._faceGroupSelection.clear()
    self._brushTopologyNodeID = None
    self._brushCellNeighbors = None
    self._brushCellCentroids = None
    self._selectionModelNode = None
    self._selectionDisplayModelNode = None
    self._faceGroupColorNode = None
    self._brushCursorModelNode = None
    self._brushCursorSphere = None
    self._faceGroupColorsByModel = {}

  def onSceneEndClose(self, caller=None, event=None):
    if self.parent.isEntered:
      self.setBrushEnabled(True)

  def setupFaceGroupsUI(self):
    """Build the Paint Model panel."""
    modelSection = ctk.ctkCollapsibleButton()
    modelSection.text = "Create face groups"
    self.layout.addWidget(modelSection)
    form = qt.QFormLayout(modelSection)

    self.faceGroupModelSelector = slicer.qMRMLNodeComboBox()
    self.faceGroupModelSelector.nodeTypes = ["vtkMRMLModelNode"]
    self.faceGroupModelSelector.noneEnabled = True
    self.faceGroupModelSelector.addEnabled = False
    self.faceGroupModelSelector.removeEnabled = False
    self.faceGroupModelSelector.setMRMLScene(slicer.mrmlScene)
    self.faceGroupModelSelector.toolTip = "Model whose polygons will receive ModelFaceID values"
    form.addRow("Model:", self.faceGroupModelSelector)

    self.faceGroupAngleSlider = ctk.ctkSliderWidget()
    self.faceGroupAngleSlider.minimum = 0.0
    self.faceGroupAngleSlider.maximum = 180.0
    self.faceGroupAngleSlider.value = 45.0
    self.faceGroupAngleSlider.decimals = 1
    self.faceGroupAngleSlider.singleStep = 1.0
    self.faceGroupAngleSlider.suffix = " deg"
    self.faceGroupAngleSlider.toolTip = "Adjacent polygons remain in a group when their normal angle is at most this value"
    form.addRow("Edge angle threshold:", self.faceGroupAngleSlider)

    self.faceGroupSizeSpinBox = qt.QSpinBox()
    self.faceGroupSizeSpinBox.minimum = 1
    self.faceGroupSizeSpinBox.maximum = 100000000
    self.faceGroupSizeSpinBox.value = 25
    self.faceGroupSizeSpinBox.toolTip = "Groups smaller than this polygon count are merged into the neighboring group with the longest shared boundary"
    form.addRow("Minimum size (cells):", self.faceGroupSizeSpinBox)

    self.createFaceGroupsButton = qt.QPushButton("Create face groups")
    self.createFaceGroupsButton.toolTip = "Partition the model using the angle threshold, then merge groups below the size threshold"
    form.addRow(self.createFaceGroupsButton)

    editSection = ctk.ctkCollapsibleButton()
    editSection.text = "Paint and edit selection"
    self.layout.addWidget(editSection)
    editForm = qt.QFormLayout(editSection)

    self.brushRadiusSpinBox = ctk.ctkDoubleSpinBox()
    self.brushRadiusSpinBox.minimum = 0.1
    self.brushRadiusSpinBox.maximum = 1000.0
    self.brushRadiusSpinBox.value = 5.0
    self.brushRadiusSpinBox.decimals = 1
    self.brushRadiusSpinBox.singleStep = 0.5
    self.brushRadiusSpinBox.suffix = " mm"
    self.brushRadiusSpinBox.toolTip = "Surface-connected brush radius used by the S and D hotkeys"
    editForm.addRow("Brush radius:", self.brushRadiusSpinBox)

    self.hotkeyInstructionsLabel = qt.QLabel(
      "Hotkeys: hold S + left-drag to select; hold D + left-drag to deselect; "
      "G expand to groups; I invert; N new group; C clear")
    self.hotkeyInstructionsLabel.wordWrap = True
    editForm.addRow(self.hotkeyInstructionsLabel)

    editRow = qt.QHBoxLayout()
    self.expandFaceGroupButton = qt.QPushButton("Expand to groups")
    self.newFaceGroupButton = qt.QPushButton("New group from selection")
    editRow.addWidget(self.expandFaceGroupButton)
    editRow.addWidget(self.newFaceGroupButton)
    editForm.addRow(editRow)

    clearExportRow = qt.QHBoxLayout()
    self.clearFaceGroupSelectionButton = qt.QPushButton("Clear selection")
    self.invertFaceGroupSelectionButton = qt.QPushButton("Invert selection")
    self.exportFaceGroupsButton = qt.QPushButton("Export .vtp...")
    clearExportRow.addWidget(self.clearFaceGroupSelectionButton)
    clearExportRow.addWidget(self.invertFaceGroupSelectionButton)
    clearExportRow.addWidget(self.exportFaceGroupsButton)
    editForm.addRow(clearExportRow)

    self.faceGroupStatusLabel = qt.QLabel("No cells selected")
    self.faceGroupStatusLabel.wordWrap = True
    editForm.addRow(self.faceGroupStatusLabel)

    self.createFaceGroupsButton.connect('clicked()', self.onCreateFaceGroups)
    self.expandFaceGroupButton.connect('clicked()', self.onExpandFaceGroups)
    self.newFaceGroupButton.connect('clicked()', self.onNewFaceGroupFromSelection)
    self.clearFaceGroupSelectionButton.connect('clicked()', self.clearFaceGroupSelection)
    self.invertFaceGroupSelectionButton.connect('clicked()', self.onInvertFaceGroupSelection)
    self.exportFaceGroupsButton.connect('clicked()', self.onExportFaceGroups)
    self.faceGroupModelSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.onFaceGroupModelChanged)

  def faceGroupModelNode(self):
    return self.faceGroupModelSelector.currentNode() if self.faceGroupModelSelector else None

  def onFaceGroupModelChanged(self, node=None):
    self._faceGroupSelection.clear()
    self._brushTopologyNodeID = None
    self._buildBrushTopology()
    if self._brushInteractionEnabled:
      self.setBrushEnabled(True)
    self.updateSelectionOverlay()

  def onCreateFaceGroups(self):
    modelNode = self.faceGroupModelNode()
    if not modelNode or not modelNode.GetPolyData():
      slicer.util.errorDisplay("Select a model in the Create face groups panel.")
      return
    qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
    try:
      groupCount = self.logic.createFaceGroups(
        modelNode, float(self.faceGroupAngleSlider.value), int(self.faceGroupSizeSpinBox.value))
      self._faceGroupSelection.clear()
      self.showFaceGroupColors(modelNode, groupCount)
      self.updateSelectionOverlay()
      self.faceGroupStatusLabel.text = f"Created {groupCount} face groups on {modelNode.GetPolyData().GetNumberOfCells()} cells"
    except Exception as exc:
      slicer.util.errorDisplay("Failed to create face groups: " + str(exc))
      logging.exception("Face-group creation failed")
    finally:
      qt.QApplication.restoreOverrideCursor()

  def showFaceGroupColors(self, modelNode, groupCount):
    if not modelNode.GetDisplayNode():
      modelNode.CreateDefaultDisplayNodes()
    colorNode = self._faceGroupColorNodeForModel(modelNode, create=True)
    self._configureFaceGroupColorTable(colorNode, groupCount, modelNode=modelNode)
    self._faceGroupColorNode = colorNode
    displayNode = modelNode.GetDisplayNode()
    displayNode.SetAndObserveColorNodeID(colorNode.GetID())
    # ModelFaceID is cell data. Setting the association explicitly is essential:
    # otherwise a display node that previously colored by point data may ignore it.
    displayNode.SetActiveScalar(PaintModelLogic.FACE_GROUP_ARRAY_NAME, vtk.vtkAssignAttribute.CELL_DATA)
    displayNode.SetScalarRangeFlag(displayNode.UseManualScalarRange)
    displayNode.SetScalarVisibility(True)
    displayNode.SetScalarRange(0, max(1, groupCount))
    displayNode.SetVisibility(True)
    modelNode.Modified()

  def _faceGroupColorNodeForModel(self, modelNode, create=False):
    displayNode = modelNode.GetDisplayNode() if modelNode else None
    colorNode = None
    if displayNode and displayNode.GetColorNodeID():
      candidate = slicer.mrmlScene.GetNodeByID(displayNode.GetColorNodeID())
      isFaceGroupPalette = candidate and (
        candidate.GetAttribute("PaintModel.FaceGroupColors") == "1" or
        "FaceGroupColors" in (candidate.GetName() or ""))
      if candidate and candidate.IsA("vtkMRMLColorTableNode") and isFaceGroupPalette:
        colorNode = candidate
    if not colorNode and create:
      colorNode = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLColorTableNode", modelNode.GetName() + " FaceGroupColors")
      colorNode.SetAttribute("PaintModel.FaceGroupColors", "1")
    return colorNode

  @staticmethod
  def _isSelectionHighlightColor(rgb):
    highlight = (1.0, 0.75, 0.05)
    return math.sqrt(sum((rgb[i] - highlight[i]) ** 2 for i in range(3))) < 0.18

  def _persistentFaceGroupColor(self, modelNode, groupId):
    modelPalette = self._faceGroupColorsByModel.setdefault(modelNode.GetID(), {})
    if groupId in modelPalette:
      return modelPalette[groupId]

    # Start with a readable categorical palette. Yellow is intentionally absent
    # because it is reserved exclusively for the active selection.
    categoricalColors = [
      (0.22, 0.45, 0.95),  # blue
      (0.95, 0.40, 0.10),  # orange
      (0.20, 0.72, 0.35),  # green
      (0.62, 0.35, 0.88),  # purple
      (0.05, 0.70, 0.82),  # cyan
      (0.90, 0.20, 0.22),  # red
      (0.90, 0.35, 0.68),  # pink
      (0.08, 0.58, 0.52),  # teal
      (0.58, 0.38, 0.22),  # brown
      (0.55, 0.58, 0.65),  # gray-blue
    ]
    if groupId <= len(categoricalColors):
      rgb = categoricalColors[groupId - 1]
    else:
      # Generate additional stable colors while skipping the yellow hue band.
      hue = (groupId * 0.61803398875) % 1.0
      while 0.08 <= hue <= 0.18:
        hue = (hue + 0.2360679775) % 1.0
      rgb = colorsys.hsv_to_rgb(hue, 0.68, 0.92)
    modelPalette[groupId] = rgb
    return rgb

  def _configureFaceGroupColorTable(self, colorNode, groupCount, highlightValue=None, modelNode=None):
    """Preserve existing group colors, add new colors, and reserve yellow for selection."""
    modelPalette = self._faceGroupColorsByModel.setdefault(modelNode.GetID(), {}) if modelNode else {}
    lookupTable = colorNode.GetLookupTable()
    if lookupTable:
      tableSize = lookupTable.GetNumberOfTableValues()
      for groupId in range(1, min(groupCount, tableSize - 1) + 1):
        existing = lookupTable.GetTableValue(groupId)
        rgb = tuple(float(existing[i]) for i in range(3))
        if groupId not in modelPalette and not self._isSelectionHighlightColor(rgb):
          modelPalette[groupId] = rgb

    colorNode.SetTypeToUser()
    highestValue = max(groupCount, highlightValue if highlightValue is not None else 0)
    colorNode.SetNumberOfColors(max(2, highestValue + 1))
    colorNode.SetColor(0, "Unassigned", 0.25, 0.25, 0.25, 1.0)
    for groupId in range(1, groupCount + 1):
      rgb = self._persistentFaceGroupColor(modelNode, groupId) if modelNode else colorsys.hsv_to_rgb(
        (groupId * 0.61803398875) % 1.0, 0.65, 0.95)
      colorNode.SetColor(groupId, f"Face group {groupId}", rgb[0], rgb[1], rgb[2], 1.0)
    if highlightValue is not None:
      colorNode.SetColor(highlightValue, "Selected faces", 1.0, 0.75, 0.05, 1.0)
    colorNode.SetHideFromEditors(True)

  def setBrushEnabled(self, enabled):
    self._brushInteractionEnabled = bool(enabled)
    self._removeBrushObservers()
    if not enabled:
      return
    layoutManager = slicer.app.layoutManager()
    if not layoutManager:
      return
    for viewIndex in range(layoutManager.threeDViewCount):
      threeDView = layoutManager.threeDWidget(viewIndex).threeDView()
      interactor = threeDView.interactor()
      callbacks = []
      for vtkEvent, eventName in (
          (vtk.vtkCommand.KeyPressEvent, "keyPress"),
          (vtk.vtkCommand.KeyReleaseEvent, "keyRelease"),
          (vtk.vtkCommand.LeftButtonPressEvent, "press"),
          (vtk.vtkCommand.MouseMoveEvent, "move"),
          (vtk.vtkCommand.LeftButtonReleaseEvent, "release"),
          (vtk.vtkCommand.LeaveEvent, "leave")):
        callback = functools.partial(self.onBrushInteractorEvent, threeDView, eventName)
        callbacks.append(callback)
        tag = interactor.AddObserver(vtkEvent, callback, 1.0)
        self._brushObserverTags.append((interactor, tag))
      # VTK stores Python callbacks weakly in some builds, so retain them explicitly.
      self._brushInteractorCallbacks.extend(callbacks)

  def _removeBrushObservers(self):
    for interactor, tag in self._brushObserverTags:
      try:
        interactor.RemoveObserver(tag)
      except Exception:
        pass
    self._brushObserverTags = []
    self._brushInteractorCallbacks = []
    self._brushDragging = False
    self._brushMode = None
    self._brushLastLocalPosition = None
    self._brushHotkeysDown.clear()
    self._restoreCameraLeftDrag()
    self._refreshBrushCursor(None)

  def _setCameraLeftDragSuppressed(self, suppressed):
    layoutManager = slicer.app.layoutManager()
    if not layoutManager:
      return
    for viewIndex in range(layoutManager.threeDViewCount):
      threeDView = layoutManager.threeDWidget(viewIndex).threeDView()
      viewNode = threeDView.mrmlViewNode()
      cameraManager = slicer.app.applicationLogic().GetViewDisplayableManagerByClassName(
        viewNode, "vtkMRMLCameraDisplayableManager")
      if not cameraManager or not cameraManager.GetCameraWidget():
        continue
      cameraWidget = cameraManager.GetCameraWidget()
      if suppressed:
        cameraWidget.SetEventTranslationClickAndDrag(
          cameraWidget.WidgetStateIdle, vtk.vtkCommand.LeftButtonPressEvent,
          vtk.vtkEvent.NoModifier, cameraWidget.WidgetStateIdle,
          vtk.vtkWidgetEvent.NoEvent, vtk.vtkWidgetEvent.NoEvent)
      else:
        cameraWidget.SetEventTranslationClickAndDrag(
          cameraWidget.WidgetStateIdle, vtk.vtkCommand.LeftButtonPressEvent,
          vtk.vtkEvent.NoModifier, cameraWidget.WidgetStateRotate,
          cameraWidget.WidgetEventRotateStart, cameraWidget.WidgetEventRotateEnd)

  def _restoreCameraLeftDrag(self):
    self._setCameraLeftDragSuppressed(False)

  def _buildBrushTopology(self):
    modelNode = self.faceGroupModelNode() if hasattr(self, 'faceGroupModelSelector') else None
    if not modelNode or not modelNode.GetPolyData():
      self._brushCellNeighbors = None
      self._brushCellCentroids = None
      return
    if self._brushTopologyNodeID == modelNode.GetID() and self._brushCellNeighbors is not None:
      return
    polyData = modelNode.GetPolyData()
    neighbors = [set() for unused in range(polyData.GetNumberOfCells())]
    centroids = []
    edgeCells = {}
    pointIds = vtk.vtkIdList()
    for cellId in range(polyData.GetNumberOfCells()):
      polyData.GetCellPoints(cellId, pointIds)
      ids = [pointIds.GetId(i) for i in range(pointIds.GetNumberOfIds())]
      points = [polyData.GetPoint(pointId) for pointId in ids]
      centroids.append(tuple(sum(point[axis] for point in points) / len(points) for axis in range(3)))
      for i, pointId in enumerate(ids):
        edgeCells.setdefault(tuple(sorted((pointId, ids[(i + 1) % len(ids)]))), []).append(cellId)
    for cells in edgeCells.values():
      for cellId in cells:
        neighbors[cellId].update(other for other in cells if other != cellId)
    self._brushTopologyNodeID = modelNode.GetID()
    self._brushCellNeighbors = neighbors
    self._brushCellCentroids = centroids

  def _brushCellsInsideSphere(self, seedCellId, brushCenterWorld, radiusMm):
    """Return the surface-connected cells whose centroids are inside the brush sphere."""
    self._buildBrushTopology()
    if self._brushCellNeighbors is None or seedCellId < 0:
      return set()

    modelNode = self.faceGroupModelNode()
    modelToWorld = vtk.vtkGeneralTransform()
    slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
      modelNode.GetParentTransformNode(), None, modelToWorld)
    return PaintModelLogic.surfaceConnectedCellsInsideSphere(
      self._brushCellNeighbors, self._brushCellCentroids, seedCellId,
      brushCenterWorld, radiusMm, modelToWorld.TransformPoint)

  def _pickModelCell(self, threeDView, x, y):
    modelNode = self.faceGroupModelNode()
    if not modelNode or not modelNode.GetDisplayNode():
      return None
    renderer = threeDView.renderWindow().GetRenderers().GetFirstRenderer()
    modelManager = slicer.app.applicationLogic().GetViewDisplayableManagerByClassName(
      threeDView.mrmlViewNode(), "vtkMRMLModelDisplayableManager")
    actor = modelManager.GetActorByID(modelNode.GetDisplayNode().GetID()) if modelManager else None
    if not actor:
      return None
    picker = vtk.vtkCellPicker()
    picker.SetTolerance(0.0005)
    picker.PickFromListOn()
    picker.AddPickList(actor)
    if not picker.Pick(x, y, 0.0, renderer) or picker.GetActor() != actor or picker.GetCellId() < 0:
      return None
    worldToModel = vtk.vtkGeneralTransform()
    slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, modelNode.GetParentTransformNode(), worldToModel)
    return int(picker.GetCellId()), tuple(worldToModel.TransformPoint(picker.GetPickPosition())), tuple(picker.GetPickPosition())

  def _paintBrush(self, threeDView, caller):
    picked = self._pickModelCell(threeDView, *caller.GetEventPosition())
    if not picked:
      self._refreshBrushCursor(None)
      return
    seedCellId, localPosition, worldPosition = picked
    self._refreshBrushCursor(worldPosition)
    radius = float(self.brushRadiusSpinBox.value)
    selected = self._brushCellsInsideSphere(seedCellId, worldPosition, radius)
    if self._brushMode == "select":
      self._faceGroupSelection.update(selected)
    else:
      self._faceGroupSelection.difference_update(selected)
    self._brushLastLocalPosition = localPosition
    self.updateSelectionOverlay()

  def onBrushInteractorEvent(self, threeDView, eventName, caller, event):
    if not self._brushInteractionEnabled:
      return
    key = (caller.GetKeySym() or "").lower() if eventName.startswith("key") else ""
    if eventName == "keyPress":
      if key in ("s", "d"):
        self._brushMode = "select" if key == "s" else "deselect"
        self._brushHotkeysDown.add(key)
        self._setCameraLeftDragSuppressed(True)
      elif key not in self._brushHotkeysDown:
        self._brushHotkeysDown.add(key)
        if key == "g":
          self.onExpandFaceGroups()
        elif key == "i":
          self.onInvertFaceGroupSelection()
        elif key == "n":
          self.onNewFaceGroupFromSelection()
        elif key == "c":
          self.clearFaceGroupSelection()
      return
    if eventName == "keyRelease":
      self._brushHotkeysDown.discard(key)
      if key in ("s", "d"):
        self._brushMode = "select" if "s" in self._brushHotkeysDown else ("deselect" if "d" in self._brushHotkeysDown else None)
        if self._brushMode is None:
          self._brushDragging = False
          self._restoreCameraLeftDrag()
          self._refreshBrushCursor(None)
      return
    if self._brushMode is None:
      return
    if eventName == "press":
      self._brushDragging = True
      self._paintBrush(threeDView, caller)
    elif eventName == "move":
      if self._brushDragging:
        self._paintBrush(threeDView, caller)
      else:
        picked = self._pickModelCell(threeDView, *caller.GetEventPosition())
        self._refreshBrushCursor(picked[2] if picked else None)
    elif eventName in ("release", "leave"):
      self._brushDragging = False
      self._brushLastLocalPosition = None
      self._refreshBrushCursor(None)

  def _refreshBrushCursor(self, worldPosition):
    if worldPosition is None:
      if self._brushCursorModelNode:
        self._brushCursorModelNode.SetAndObservePolyData(vtk.vtkPolyData())
      return
    if not self._brushCursorModelNode:
      self._brushCursorModelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "Paint Model brush cursor")
      self._brushCursorModelNode.CreateDefaultDisplayNodes()
      self._brushCursorModelNode.SetHideFromEditors(True)
      display = self._brushCursorModelNode.GetDisplayNode()
      display.SetColor(0.0, 0.9, 1.0)
      display.SetOpacity(0.3)
      display.SetScalarVisibility(False)
      self._brushCursorSphere = vtk.vtkSphereSource()
      self._brushCursorSphere.SetThetaResolution(24)
      self._brushCursorSphere.SetPhiResolution(24)
    self._brushCursorSphere.SetCenter(*worldPosition)
    self._brushCursorSphere.SetRadius(float(self.brushRadiusSpinBox.value))
    self._brushCursorSphere.Update()
    self._brushCursorModelNode.SetAndObservePolyData(self._brushCursorSphere.GetOutput())

  def updateSelectionOverlay(self):
    modelNode = self.faceGroupModelNode() if hasattr(self, 'faceGroupModelSelector') else None
    if self._selectionDisplayModelNode and self._selectionDisplayModelNode != modelNode:
      self._clearSelectionDisplay(self._selectionDisplayModelNode)
    if self._selectionModelNode:
      # Remove overlays created by earlier versions of the module. The selection is
      # now rendered directly on the source actor, which eliminates z-fighting.
      slicer.mrmlScene.RemoveNode(self._selectionModelNode)
      self._selectionModelNode = None
    if not modelNode or not modelNode.GetPolyData() or not self._faceGroupSelection:
      if modelNode:
        self._clearSelectionDisplay(modelNode)
      if hasattr(self, 'faceGroupStatusLabel'):
        self.faceGroupStatusLabel.text = "No cells selected"
      return

    polyData = modelNode.GetPolyData()
    faceIds = polyData.GetCellData().GetArray(PaintModelLogic.FACE_GROUP_ARRAY_NAME)
    groupCount = max(
      [int(faceIds.GetTuple1(cellId)) for cellId in range(faceIds.GetNumberOfTuples())] or [0]) if faceIds else 0
    highlightValue = groupCount + 1
    displayIds = vtk.vtkIntArray()
    displayIds.SetName(PaintModelLogic.SELECTION_DISPLAY_ARRAY_NAME)
    displayIds.SetNumberOfTuples(polyData.GetNumberOfCells())
    for cellId in range(polyData.GetNumberOfCells()):
      baseValue = int(faceIds.GetTuple1(cellId)) if faceIds else 0
      displayIds.SetValue(cellId, highlightValue if cellId in self._faceGroupSelection else baseValue)
    polyData.GetCellData().RemoveArray(PaintModelLogic.SELECTION_DISPLAY_ARRAY_NAME)
    polyData.GetCellData().AddArray(displayIds)

    if not modelNode.GetDisplayNode():
      modelNode.CreateDefaultDisplayNodes()
    self._faceGroupColorNode = self._faceGroupColorNodeForModel(modelNode, create=True)
    self._configureFaceGroupColorTable(
      self._faceGroupColorNode, groupCount, highlightValue, modelNode=modelNode)
    displayNode = modelNode.GetDisplayNode()
    displayNode.SetAndObserveColorNodeID(self._faceGroupColorNode.GetID())
    displayNode.SetActiveScalar(PaintModelLogic.SELECTION_DISPLAY_ARRAY_NAME, vtk.vtkAssignAttribute.CELL_DATA)
    displayNode.SetScalarRangeFlag(displayNode.UseManualScalarRange)
    displayNode.SetScalarRange(0, highlightValue)
    displayNode.SetScalarVisibility(True)
    self._selectionDisplayModelNode = modelNode
    polyData.Modified()
    modelNode.Modified()
    self.faceGroupStatusLabel.text = f"{len(self._faceGroupSelection)} cells selected"
    slicer.app.processEvents()

  def _clearSelectionDisplay(self, modelNode):
    if not modelNode or not modelNode.GetPolyData():
      return
    polyData = modelNode.GetPolyData()
    polyData.GetCellData().RemoveArray(PaintModelLogic.SELECTION_DISPLAY_ARRAY_NAME)
    faceIds = polyData.GetCellData().GetArray(PaintModelLogic.FACE_GROUP_ARRAY_NAME)
    displayNode = modelNode.GetDisplayNode()
    if displayNode:
      if faceIds and faceIds.GetNumberOfTuples() == polyData.GetNumberOfCells():
        groupCount = max([int(faceIds.GetTuple1(i)) for i in range(faceIds.GetNumberOfTuples())] or [0])
        # Selection adds one extra yellow lookup-table entry. Remove that entry
        # before restoring ModelFaceID; otherwise VTK rescales IDs 0..N across
        # N+2 colors, shifting existing group colors and mapping group N to yellow.
        colorNode = self._faceGroupColorNodeForModel(modelNode, create=True)
        self._configureFaceGroupColorTable(colorNode, groupCount, modelNode=modelNode)
        displayNode.SetAndObserveColorNodeID(colorNode.GetID())
        self._faceGroupColorNode = colorNode
        displayNode.SetActiveScalar(PaintModelLogic.FACE_GROUP_ARRAY_NAME, vtk.vtkAssignAttribute.CELL_DATA)
        displayNode.SetScalarRange(0, max(1, groupCount))
        displayNode.SetScalarVisibility(True)
      else:
        displayNode.SetScalarVisibility(False)
    if self._selectionDisplayModelNode == modelNode:
      self._selectionDisplayModelNode = None
    polyData.Modified()
    modelNode.Modified()

  def clearFaceGroupSelection(self):
    self._faceGroupSelection.clear()
    self.updateSelectionOverlay()

  def onInvertFaceGroupSelection(self):
    modelNode = self.faceGroupModelNode()
    if not modelNode or not modelNode.GetPolyData():
      return
    cellCount = modelNode.GetPolyData().GetNumberOfCells()
    self._faceGroupSelection = {
      cellId for cellId in range(cellCount) if cellId not in self._faceGroupSelection}
    self.updateSelectionOverlay()

  def onExpandFaceGroups(self):
    modelNode = self.faceGroupModelNode()
    if not modelNode or not self._faceGroupSelection:
      return
    array = modelNode.GetPolyData().GetCellData().GetArray(PaintModelLogic.FACE_GROUP_ARRAY_NAME)
    if not array:
      slicer.util.errorDisplay("Create face groups before expanding the selection.")
      return
    groupIds = {int(array.GetTuple1(cellId)) for cellId in self._faceGroupSelection}
    self._faceGroupSelection = {cellId for cellId in range(modelNode.GetPolyData().GetNumberOfCells()) if int(array.GetTuple1(cellId)) in groupIds}
    self.updateSelectionOverlay()

  def onNewFaceGroupFromSelection(self):
    modelNode = self.faceGroupModelNode()
    if not modelNode or not self._faceGroupSelection:
      slicer.util.errorDisplay("Brush-select at least one cell first.")
      return
    newId = self.logic.createGroupFromSelection(modelNode, self._faceGroupSelection)
    array = modelNode.GetPolyData().GetCellData().GetArray(PaintModelLogic.FACE_GROUP_ARRAY_NAME)
    maxId = max(int(array.GetTuple1(i)) for i in range(array.GetNumberOfTuples()))
    self.showFaceGroupColors(modelNode, maxId)
    self.faceGroupStatusLabel.text = f"Created face group {newId} from {len(self._faceGroupSelection)} selected cells"
    self.updateSelectionOverlay()

  def onExportFaceGroups(self):
    modelNode = self.faceGroupModelNode()
    if not modelNode:
      slicer.util.errorDisplay("Select a face-group model first.")
      return
    filePath = qt.QFileDialog.getSaveFileName(slicer.util.mainWindow(), "Export model with face groups", modelNode.GetName() + ".vtp", "VTK PolyData (*.vtp)")
    if not filePath:
      return
    if not filePath.lower().endswith(".vtp"):
      filePath += ".vtp"
    try:
      self.logic.exportFaceGroups(modelNode, filePath)
      self.faceGroupStatusLabel.text = "Exported " + filePath
    except Exception as exc:
      slicer.util.errorDisplay("Failed to export face groups: " + str(exc))

  def removeFaceGroupHelperNodes(self):
    self._clearSelectionDisplay(self._selectionDisplayModelNode)
    for node in (self._selectionModelNode, self._faceGroupColorNode, self._brushCursorModelNode):
      if node and slicer.mrmlScene.IsNodePresent(node):
        slicer.mrmlScene.RemoveNode(node)
    self._selectionModelNode = None
    self._selectionDisplayModelNode = None
    self._faceGroupColorNode = None
    self._brushCursorModelNode = None
    self._brushCursorSphere = None

#
# PaintModelLogic
#

class PaintModelLogic(ScriptedLoadableModuleLogic):
  """This class implements all the actual computation done by the module,
  independent of the widget, so it can be used in batch mode.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  FACE_GROUP_ARRAY_NAME = "ModelFaceID"
  SELECTION_DISPLAY_ARRAY_NAME = "_PaintModelSelectionDisplay"

  def __init__(self):
    ScriptedLoadableModuleLogic.__init__(self)

  @staticmethod
  def surfaceConnectedCellsInsideSphere(
      cellNeighbors, cellCentroids, seedCellId, brushCenter, radius, transformPoint=None):
    """Find the seed-connected triangle centroids inside an Euclidean sphere."""
    if seedCellId < 0 or seedCellId >= len(cellNeighbors):
      return set()
    if transformPoint is None:
      transformPoint = lambda point: point
    radiusSquared = float(radius) ** 2

    def centroidIsInsideSphere(cellId):
      centroid = transformPoint(cellCentroids[cellId])
      return sum(
        (centroid[axis] - brushCenter[axis]) ** 2 for axis in range(3)) <= radiusSquared

    # Always include the cell under the cursor. Its centroid can be outside a
    # small sphere when the picked triangle is large, even though the picked
    # surface point is at the center of the sphere.
    selected = {seedCellId}
    visited = {seedCellId}
    stack = [seedCellId]
    while stack:
      cellId = stack.pop()
      for neighbor in cellNeighbors[cellId]:
        if neighbor in visited:
          continue
        visited.add(neighbor)
        if centroidIsInsideSphere(neighbor):
          selected.add(neighbor)
          stack.append(neighbor)
    return selected

  @staticmethod
  def _cellNormal(polyData, cellId):
    cell = polyData.GetCell(cellId)
    count = cell.GetNumberOfPoints()
    if count < 3:
      return (0.0, 0.0, 0.0)
    # Newell's method supports triangles and general planar polygons.
    normal = [0.0, 0.0, 0.0]
    for index in range(count):
      p = polyData.GetPoint(cell.GetPointId(index))
      q = polyData.GetPoint(cell.GetPointId((index + 1) % count))
      normal[0] += (p[1] - q[1]) * (p[2] + q[2])
      normal[1] += (p[2] - q[2]) * (p[0] + q[0])
      normal[2] += (p[0] - q[0]) * (p[1] + q[1])
    length = math.sqrt(sum(value * value for value in normal))
    return tuple(value / length for value in normal) if length else (0.0, 0.0, 0.0)

  @staticmethod
  def createFaceGroups(modelNode, edgeAngleThreshold=45.0, minimumGroupSize=25):
    """Create ModelFaceID by region growing across edges below a dihedral-angle threshold.

    Regions smaller than ``minimumGroupSize`` cells are repeatedly merged into the
    adjacent region with which they share the most mesh edges. Disconnected small
    components are retained because there is no geometrically valid neighbor.
    """
    polyData = modelNode.GetPolyData()
    if not polyData or polyData.GetNumberOfCells() == 0:
      raise ValueError("The selected model has no surface cells")
    cellCount = polyData.GetNumberOfCells()
    normals = [PaintModelLogic._cellNormal(polyData, cellId) for cellId in range(cellCount)]
    bounds = polyData.GetBounds()
    diagonal = math.sqrt((bounds[1]-bounds[0])**2 + (bounds[3]-bounds[2])**2 + (bounds[5]-bounds[4])**2)
    coordinateTolerance = max(diagonal * 1e-9, 1e-12)
    pointKeys = {}
    def pointKey(pointId):
      if pointId not in pointKeys:
        point = polyData.GetPoint(pointId)
        pointKeys[pointId] = tuple(int(round(value / coordinateTolerance)) for value in point)
      return pointKeys[pointId]
    edgeCells = {}
    for cellId in range(cellCount):
      cell = polyData.GetCell(cellId)
      for edgeIndex in range(cell.GetNumberOfEdges()):
        edge = cell.GetEdge(edgeIndex)
        a, b = pointKey(edge.GetPointId(0)), pointKey(edge.GetPointId(1))
        edgeCells.setdefault(tuple(sorted((a, b))), []).append(cellId)
    neighbors = [set() for _ in range(cellCount)]
    cosineThreshold = math.cos(math.radians(max(0.0, min(180.0, edgeAngleThreshold))))
    for cells in edgeCells.values():
      for firstIndex in range(len(cells)):
        for secondIndex in range(firstIndex + 1, len(cells)):
          first, second = cells[firstIndex], cells[secondIndex]
          dot = sum(normals[first][axis] * normals[second][axis] for axis in range(3))
          if dot >= cosineThreshold:
            neighbors[first].add(second)
            neighbors[second].add(first)

    groupOfCell = [-1] * cellCount
    groupCount = 0
    for seed in range(cellCount):
      if groupOfCell[seed] >= 0:
        continue
      stack = [seed]
      groupOfCell[seed] = groupCount
      while stack:
        current = stack.pop()
        for neighbor in neighbors[current]:
          if groupOfCell[neighbor] < 0:
            groupOfCell[neighbor] = groupCount
            stack.append(neighbor)
      groupCount += 1

    minimumGroupSize = max(1, int(minimumGroupSize))
    while True:
      members = {}
      for cellId, groupId in enumerate(groupOfCell):
        members.setdefault(groupId, []).append(cellId)
      smallGroups = sorted((len(cells), groupId) for groupId, cells in members.items() if len(cells) < minimumGroupSize)
      mergedAny = False
      for unusedSize, smallGroup in smallGroups:
        if smallGroup not in members:
          continue
        boundaryCounts = {}
        for cells in edgeCells.values():
          groups = {groupOfCell[cellId] for cellId in cells}
          if smallGroup in groups:
            for adjacentGroup in groups:
              if adjacentGroup != smallGroup:
                boundaryCounts[adjacentGroup] = boundaryCounts.get(adjacentGroup, 0) + 1
        if not boundaryCounts:
          continue
        targetGroup = max(boundaryCounts, key=lambda groupId: (boundaryCounts[groupId], len(members.get(groupId, []))))
        for cellId in members[smallGroup]:
          groupOfCell[cellId] = targetGroup
        members.setdefault(targetGroup, []).extend(members[smallGroup])
        del members[smallGroup]
        mergedAny = True
      if not mergedAny:
        break

    oldToNew = {oldId: newId + 1 for newId, oldId in enumerate(sorted(set(groupOfCell)))}
    faceIds = vtk.vtkIntArray()
    faceIds.SetName(PaintModelLogic.FACE_GROUP_ARRAY_NAME)
    faceIds.SetNumberOfTuples(cellCount)
    for cellId, oldId in enumerate(groupOfCell):
      faceIds.SetValue(cellId, oldToNew[oldId])
    polyData.GetCellData().RemoveArray(PaintModelLogic.FACE_GROUP_ARRAY_NAME)
    polyData.GetCellData().AddArray(faceIds)
    polyData.GetCellData().SetActiveScalars(PaintModelLogic.FACE_GROUP_ARRAY_NAME)
    polyData.Modified()
    modelNode.Modified()
    return len(oldToNew)

  @staticmethod
  def createGroupFromSelection(modelNode, selectedCellIds):
    polyData = modelNode.GetPolyData()
    array = polyData.GetCellData().GetArray(PaintModelLogic.FACE_GROUP_ARRAY_NAME)
    if not array:
      array = vtk.vtkIntArray()
      array.SetName(PaintModelLogic.FACE_GROUP_ARRAY_NAME)
      array.SetNumberOfTuples(polyData.GetNumberOfCells())
      array.Fill(1)
      polyData.GetCellData().AddArray(array)
    newGroupId = max([int(array.GetTuple1(i)) for i in range(array.GetNumberOfTuples())] or [0]) + 1
    for cellId in selectedCellIds:
      if 0 <= cellId < polyData.GetNumberOfCells():
        array.SetTuple1(cellId, newGroupId)
    array.Modified()
    polyData.GetCellData().SetActiveScalars(PaintModelLogic.FACE_GROUP_ARRAY_NAME)
    polyData.Modified()
    modelNode.Modified()
    return newGroupId

  @staticmethod
  def extractCells(polyData, cellIds):
    ids = vtk.vtkIdTypeArray()
    for cellId in cellIds:
      if 0 <= cellId < polyData.GetNumberOfCells():
        ids.InsertNextValue(cellId)
    selectionNode = vtk.vtkSelectionNode()
    selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
    selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
    selectionNode.SetSelectionList(ids)
    selection = vtk.vtkSelection()
    selection.AddNode(selectionNode)
    extract = vtk.vtkExtractSelection()
    extract.SetInputData(0, polyData)
    extract.SetInputData(1, selection)
    geometry = vtk.vtkGeometryFilter()
    geometry.SetInputConnection(extract.GetOutputPort())
    geometry.Update()
    output = vtk.vtkPolyData()
    output.DeepCopy(geometry.GetOutput())
    return output

  @staticmethod
  def exportFaceGroups(modelNode, filePath):
    polyData = modelNode.GetPolyData()
    array = polyData.GetCellData().GetArray(PaintModelLogic.FACE_GROUP_ARRAY_NAME)
    if not array or array.GetNumberOfTuples() != polyData.GetNumberOfCells():
      raise ValueError("The model does not have a valid ModelFaceID cell array")
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filePath)
    exportPolyData = vtk.vtkPolyData()
    exportPolyData.DeepCopy(polyData)
    exportPolyData.GetCellData().RemoveArray(PaintModelLogic.SELECTION_DISPLAY_ARRAY_NAME)
    exportFaceIds = exportPolyData.GetCellData().GetArray(PaintModelLogic.FACE_GROUP_ARRAY_NAME)
    sourceIds = sorted({int(exportFaceIds.GetTuple1(i)) for i in range(exportFaceIds.GetNumberOfTuples())})
    consecutiveId = {sourceId: index + 1 for index, sourceId in enumerate(sourceIds)}
    for cellId in range(exportFaceIds.GetNumberOfTuples()):
      exportFaceIds.SetTuple1(cellId, consecutiveId[int(exportFaceIds.GetTuple1(cellId))])
    exportFaceIds.Modified()
    exportPolyData.GetCellData().SetActiveScalars(PaintModelLogic.FACE_GROUP_ARRAY_NAME)
    writer.SetInputData(exportPolyData)
    writer.SetDataModeToBinary()
    if writer.Write() != 1:
      raise IOError("VTK could not write " + filePath)

#
# PaintModelTest
#

class PaintModelTest(ScriptedLoadableModuleTest):
  """
  This is the test case for scripted module PaintModel.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """Reset the state - typically at the beginning of the test.
    """
    slicer.mrmlScene.Clear()

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_PaintModelCreateFaceGroups()

  def test_PaintModelCreateFaceGroups(self):
    """Create a simple cube-like polydata and confirm createFaceGroups partitions
    it into flat faces and that createGroupFromSelection / exportFaceGroups work.
    """
    self.delayDisplay("Starting test_PaintModelCreateFaceGroups")

    cube = vtk.vtkCubeSource()
    cube.Update()

    modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "PaintModelTestCube")
    modelNode.SetAndObservePolyData(cube.GetOutput())
    modelNode.CreateDefaultDisplayNodes()

    logic = PaintModelLogic()
    selected = logic.surfaceConnectedCellsInsideSphere(
      [{1, 2}, {0}, {0}],
      [(0.0, 0.0, 0.0), (-0.6, 0.0, 0.0), (1.2, 0.0, 0.0)],
      0, (0.9, 0.0, 0.0), 1.0)
    self.assertEqual(
      selected, {0, 2},
      "The seed and in-sphere neighbor should be selected, but an out-of-sphere neighbor must not be")

    groupCount = logic.createFaceGroups(modelNode, edgeAngleThreshold=1.0, minimumGroupSize=1)
    self.assertEqual(groupCount, 6, "A cube with a tight angle threshold should yield 6 face groups")

    array = modelNode.GetPolyData().GetCellData().GetArray(PaintModelLogic.FACE_GROUP_ARRAY_NAME)
    self.assertIsNotNone(array)
    self.assertEqual(array.GetNumberOfTuples(), modelNode.GetPolyData().GetNumberOfCells())

    firstFaceCells = {i for i in range(array.GetNumberOfTuples()) if int(array.GetTuple1(i)) == 1}
    self.assertTrue(len(firstFaceCells) > 0)
    newGroupId = logic.createGroupFromSelection(modelNode, firstFaceCells)
    self.assertEqual(newGroupId, groupCount + 1)

    import tempfile, os
    outputPath = os.path.join(tempfile.gettempdir(), "PaintModelTestCube.vtp")
    logic.exportFaceGroups(modelNode, outputPath)
    self.assertTrue(os.path.exists(outputPath))
    os.remove(outputPath)

    self.delayDisplay("test_PaintModelCreateFaceGroups passed")
