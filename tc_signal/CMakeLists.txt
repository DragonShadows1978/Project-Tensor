# tc_signal: TensorCUDA add-on for signal ops (P1T-M2: Conv1d, ConvTranspose1d). Separate .so (_tc_signal); it does
# not link, include or compile anything from tensor_cuda/ (core). It shares core's CUDA runtime by linking the SAME
# shared libcudart.so.12 (CUDA 12.6), so one runtime instance, one context and one default memory pool serve both.
cmake_minimum_required(VERSION 3.18)
project(tc_signal LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Disable LTO/IPO, as core does (tensor_cuda/CMakeLists.txt): pybind11 adds -flto in Release unless this is defined,
# and nvcc's device-link LTO bytecode version does not match the host gcc's.
if(NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
endif()

# Shared CUDA runtime (never cudart_static): one cudart instance with core.
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

# Numerics contract: no fast-math anywhere in this package, no per-thread default stream.
foreach(_flags CMAKE_CUDA_FLAGS CMAKE_CUDA_FLAGS_RELEASE CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE)
  if("${${_flags}}" MATCHES "fast[-_]math|default-stream")
    message(FATAL_ERROR "tc_signal: ${_flags} ('${${_flags}}') carries a forbidden flag (fast-math or default-stream)")
  endif()
endforeach()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(CUDAToolkit REQUIRED)

# pybind11 v2.12.0, the same as core. Offline builds pass -DFETCHCONTENT_SOURCE_DIR_PYBIND11=<pybind11 source>
# (build.sh does this from PYBIND11_SOURCE_DIR); FetchContent then uses that tree and never touches the network.
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(_tc_signal
  src/conv.cu
  src/dlpack_operand.cpp
  src/bindings.cpp
  src/lstm.cu
  src/lstm_bindings.cpp
)

target_include_directories(_tc_signal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(_tc_signal PRIVATE CUDA::cudart CUDA::cublas)

# RUNPATH = <real toolkit root>/lib64, the same string the production core module carries
# (/usr/local/cuda-12.6/lib64, from its scripts/apa_*_build.sh link line), so `ldd` resolves libcudart.so.12 to the
# same path for both modules. CMake's automatic build RPATH would record the symlink-resolved
# <root>/targets/x86_64-linux/lib instead (the same file, a different path string), so it is skipped here.
get_filename_component(_tcs_cuda_root "${CUDAToolkit_BIN_DIR}/.." REALPATH)
set(_tcs_cuda_rpath "${_tcs_cuda_root}/lib64")
if(NOT EXISTS "${_tcs_cuda_rpath}/libcudart.so.12")
  set(_tcs_cuda_rpath "${CUDAToolkit_LIBRARY_DIR}")
endif()
message(STATUS "tc_signal: RUNPATH ${_tcs_cuda_rpath}")
target_link_options(_tc_signal PRIVATE "LINKER:-rpath,${_tcs_cuda_rpath}")

set_target_properties(_tc_signal PROPERTIES
  SKIP_BUILD_RPATH ON
  CUDA_RUNTIME_LIBRARY Shared
  # Drop the built module straight into the Python package directory.
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tc_signal
)
