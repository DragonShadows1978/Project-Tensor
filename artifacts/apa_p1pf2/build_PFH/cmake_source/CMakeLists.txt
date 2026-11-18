cmake_minimum_required(VERSION 3.18)
project(tensor_cuda LANGUAGES CXX CUDA)

option(TC_APA_GEMM_DEBUG_ASSERTS
  "Enable device-side bounds assertions in the APAMQ GEMM path" OFF)

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
  /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1pf2/tensor_cuda/src/kernels.cu
  /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1pf2/tensor_cuda/src/paint_raster.cu
  /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1pf2/tensor_cuda/src/paint_bake.cu
  /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1pf2/tensor_cuda/src/paint_inpaint.cu
  /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1pf2/tensor_cuda/src/dda.cu
  /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1pf2/tensor_cuda/src/terrain.cu
  /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1pf2/tensor_cuda/src/terrain_objects.cu
  /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1pf2/tensor_cuda/src/matmul.cu
  /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1pf2/tensor_cuda/src/gemm_apa.cu
  /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1pf2/tensor_cuda/src/conv.cu
  /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1pf2/tensor_cuda/src/autograd.cpp
  /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1pf2/tensor_cuda/src/ops.cpp
  /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1pf2/tensor_cuda/src/bindings.cpp
)

target_include_directories(_tensor_cuda PRIVATE /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1pf2/tensor_cuda/include)
target_link_libraries(_tensor_cuda PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt)
if(TC_APA_GEMM_DEBUG_ASSERTS)
  target_compile_definitions(_tensor_cuda PRIVATE TC_APA_GEMM_DEBUG_ASSERTS=1)
endif()
set_target_properties(_tensor_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  # Drop the built module straight into the Python package directory.
  LIBRARY_OUTPUT_DIRECTORY /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1pf2/artifacts/apa_p1pf2/build_PFH/lib
)

# APA-P1 port: Git three-way merge (2005), survey S5 dry run; inherited
# Project-Tensor SPQ1/SPQ4/SPQ6/EB2b/EB3/EB4 (2026), unchanged numerics.
# AB-1 main feature definitions are applied consistently to host bindings and
# CUDA. Production's initializer is retained. APA-P1-D (R-G): the packed build
# ships ON, since packed SP keys are the default representation; S3's test
# hooks (TC_APA_P1_TEST_HOOKS) stay off in the default build.
option(TC_APA_PACKED "Build packed APA SP/SP-tk8 decode entries (default ON)" ON)
if(TC_APA_PACKED)
  target_compile_definitions(_tensor_cuda PRIVATE
    TC_APA_PACKED=1 TC_APA_EB2B=1 TC_APA_EB3=1 TC_APA_EB4=1
    APA_EB2B_MUTANT=0 APA_FXTC1_VARIANT_DEFAULT=2)
endif()

# Test-only hook; never enabled by the shipped CMake default.
target_compile_definitions(_tensor_cuda PRIVATE TC_APA_P1_TEST_HOOKS=1)
