cmake_minimum_required(VERSION 3.20.6...3.22.6 FATAL_ERROR)

project(SimVascular)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/SimVascular/SlicerSimVascular")
set(EXTENSION_CONTRIBUTORS "Alison Marsden (Cardiovascular Biomechanics Computation Lab, Stanford University), Bohan Jeff Li (Cardiovascular Biomechanics Computation Lab, Stanford University)")
set(EXTENSION_DESCRIPTION "SimVascular extensions developed by Marsden lab members and associates for 3D Slicer.")
set(EXTENSION_ICONURL "https://github.com/SimVascular/SlicerSimVascular/raw/master/SimVascular.png")
# TODO: Update with real screenshot URL
set(EXTENSION_SCREENSHOTURLS "https://github.com/SimVascular/SlicerSimVascular/raw/master/Docs/DeployStent01.jpg https://github.com/SimVascular/SlicerSimVascular/raw/master/Docs/DeployStent02.jpg")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(DeployStent)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
