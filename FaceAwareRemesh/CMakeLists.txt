#-----------------------------------------------------------------------------
set(MODULE_NAME FaceAwareRemesh)

#-----------------------------------------------------------------------------
# `svremesh` is the remesher; this module is an MRML adapter over it. Listing the package's
# files here installs them beside the module, which is already on Slicer's Python path -- so
# neither a binary install nor a development checkout on Additional Module Paths needs pip.
# `FaceAwareRemesh/pyproject.toml` covers the other direction: importing it outside Slicer.
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  svremesh/__init__.py
  svremesh/conditioning.py
  svremesh/dynamic_mesh.py
  svremesh/labelled.py
  svremesh/quality.py
  svremesh/remesh.py
  svremesh/surfaces.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/UI/${MODULE_NAME}.ui
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
