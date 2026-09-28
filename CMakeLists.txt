cmake_minimum_required(VERSION 3.20)
project(anukulan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(ANUKULAN_ENABLE_CUDA "Build the CUDA backend" OFF)

if(ANUKULAN_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Covers the cards actually available on the free tiers: P100 is 60,
    # T4 is 75, A100 is 80, L4 and 4090 are 89.
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 89)
  endif()
endif()
option(ANUKULAN_WERROR "Treat warnings as errors" ON)

add_library(anukulan
  src/anukulan/sparse.cpp
  src/anukulan/backend.cpp
  src/anukulan/model.cpp
  src/anukulan/mps_reader.cpp
  src/anukulan/standard_form.cpp
  src/anukulan/scaling.cpp
  src/anukulan/pdhg.cpp
  src/anukulan/branch_and_bound.cpp
  src/anukulan/qp.cpp
  src/anukulan/cuts.cpp
  src/anukulan/lu.cpp
  src/anukulan/simplex.cpp
  src/anukulan/crossover.cpp
  src/anukulan/presolve.cpp
  src/anukulan/ldl.cpp
  src/anukulan/threading.cpp
  src/anukulan/threaded_backend.cpp
  src/anukulan/cholesky.cpp
)
target_include_directories(anukulan PUBLIC src)

# std::thread needs the platform's thread library linked, and nothing more. No
# OpenMP: the reductions have to be bit-identical, which means owning the
# partition, and Apple Clang ships without libomp so requiring it would break
# the one thing the build promises - clone it and run cmake. See
# src/anukulan/threading.hpp.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(anukulan PUBLIC Threads::Threads)

if(MSVC)
  target_compile_options(anukulan PRIVATE /W4)
else()
  target_compile_options(anukulan PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion>)
  # nvcc must not be handed the C++ warning set at all.
  target_compile_options(anukulan PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-w>)
  if(ANUKULAN_WERROR)
    target_compile_options(anukulan PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
  endif()
endif()

if(ANUKULAN_ENABLE_CUDA)
  target_sources(anukulan PRIVATE src/anukulan/cuda_backend.cu)
  target_compile_definitions(anukulan PUBLIC ANUKULAN_WITH_CUDA)
  set_target_properties(anukulan PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  # The strict warning set is ours, for our C++. nvcc forwards host flags to the
  # host compiler while also generating its own code, and -Wshadow fires inside
  # the CUDA headers rather than in anything we wrote - which is why a first
  # attempt at building this on Kaggle had to delete -Werror wholesale. Scoping
  # the flags to C++ sources keeps them on for the code they were meant for.
  set_source_files_properties(src/anukulan/cuda_backend.cu PROPERTIES
                              COMPILE_OPTIONS "-w")
endif()

add_executable(anukulan-cli app/anukulan_cli.cpp)
target_link_libraries(anukulan-cli PRIVATE anukulan)
set_target_properties(anukulan-cli PROPERTIES OUTPUT_NAME anukulan)

enable_testing()
add_subdirectory(tests)
