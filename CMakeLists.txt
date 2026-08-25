cmake_minimum_required(VERSION 3.25)

project(mlx_nf4_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

find_package(
  Python 3.10
  COMPONENTS Interpreter Development.Module
  REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

add_library(mlx_nf4_native)
target_sources(mlx_nf4_native PUBLIC ${CMAKE_CURRENT_LIST_DIR}/csrc/mlx_nf4/nf4.cpp)
target_include_directories(mlx_nf4_native PUBLIC ${CMAKE_CURRENT_LIST_DIR}/csrc)
target_link_libraries(mlx_nf4_native PUBLIC mlx)

if(MLX_BUILD_METAL)
  mlx_build_metallib(
    TARGET
    mlx_nf4_metallib
    TITLE
    mlx_nf4
    SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/metal/mlx_nf4/nf4_quantized.metal
    INCLUDE_DIRS
    ${CMAKE_CURRENT_LIST_DIR}/metal
    ${MLX_INCLUDE_DIRS}
    DEPS
    ${CMAKE_CURRENT_LIST_DIR}/metal/mlx_nf4/nf4.h
    ${CMAKE_CURRENT_LIST_DIR}/metal/mlx_nf4/nf4_quantized.h
    OUTPUT_DIRECTORY
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

  add_dependencies(mlx_nf4_native mlx_nf4_metallib)
endif()

nanobind_add_module(
  _ext
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_LIST_DIR}/csrc/mlx_nf4/bindings.cpp)
target_link_libraries(_ext PRIVATE mlx_nf4_native)

if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
