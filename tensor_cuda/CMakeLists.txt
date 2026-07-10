cmake_minimum_required(VERSION 3.18)
project(tensor_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Disable LTO/IPO. pybind11 enables -flto for Release builds, but nvcc's
# device-link object carries the CUDA toolkit's LTO bytecode version (e.g. 12.0)
# while the host gcc expects its own (e.g. 13.1), so the final link dies with a
# "bytecode stream ... LTO version" mismatch. Setting this BEFORE pybind11 is
# configured makes pybind11 skip its -flto injection entirely (it only adds LTO
# when CMAKE_INTERPROCEDURAL_OPTIMIZATION is unset). Override on the command line
# if your gcc and CUDA toolkit LTO versions actually match.
if(NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
endif()

# RTX 3070 is sm_86; include a spread of common archs. Override with
#   -DCMAKE_CUDA_ARCHITECTURES=86
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(CUDAToolkit REQUIRED)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(_tensor_cuda
  src/kernels.cu
  src/dda.cu
  src/matmul.cu
  src/conv.cu
  src/autograd.cpp
  src/ops.cpp
  src/bindings.cpp
)

target_include_directories(_tensor_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(_tensor_cuda PRIVATE CUDA::cudart CUDA::cublas)
set_target_properties(_tensor_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  # Drop the built module straight into the Python package directory.
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tensor_cuda
)
