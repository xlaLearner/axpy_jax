cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)
message(STATUS "Using CMake version: " ${CMAKE_VERSION})

# Find pybind11
set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/lib)

# CPU op library
pybind11_add_module(axpy_cpu ${CMAKE_CURRENT_LIST_DIR}/lib/axpy_cpu.cc)
install(TARGETS axpy_cpu LIBRARY DESTINATION .)

# Include the CUDA extensions if possible
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  pybind11_add_module(
    axpy_gpu
    ${CMAKE_CURRENT_LIST_DIR}/lib/axpy_gpu_kernel.cc.cu
    ${CMAKE_CURRENT_LIST_DIR}/lib/axpy_gpu.cc)
  install(TARGETS axpy_gpu LIBRARY DESTINATION .)
else()
  message(STATUS "Building without CUDA")
endif()
