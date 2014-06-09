cmake_minimum_required (VERSION 2.6)
project(CatalystMiniFERefAdaptor)

SET(ADAPTOR_LIBRARY_NAME CatalystMiniFERefAdaptor)

find_package(ParaView REQUIRED COMPONENTS vtkPVCatalyst vtkPVPythonCatalyst)
include(${PARAVIEW_USE_FILE})

add_definitions("-DUSE_CATALYST")

set(PATH_TO_MINIFE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src" CACHE PATH "Path To directory containing miniFE/src")
include_directories(${PATH_TO_MINIFE_SRC_DIR})

set(PATH_TO_MINIFE_UTILS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../utils " CACHE PATH "Path To directory containing miniFE/utils")
include_directories(${PATH_TO_MINIFE_UTILS_DIR})

add_library(${ADAPTOR_LIBRARY_NAME} SHARED catalyst_adapter.cpp)
target_link_libraries(${ADAPTOR_LIBRARY_NAME} vtkPVCatalyst vtkPVPythonCatalyst)

