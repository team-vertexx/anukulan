cmake_minimum_required(VERSION 3.20)
project(sankhya LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(SANKHYA_ENABLE_CUDA "Build the CUDA backend" OFF)

if(SANKHYA_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Covers the cards actually available on the free tiers: P100 is 60,
    # T4 is 75, A100 is 80, L4 and 4090 are 89.
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 89)
  endif()
endif()
option(SANKHYA_WERROR "Treat warnings as errors" ON)

add_library(sankhya
  src/sankhya/sparse.cpp
  src/sankhya/backend.cpp
  src/sankhya/model.cpp
  src/sankhya/mps_reader.cpp
  src/sankhya/standard_form.cpp
  src/sankhya/scaling.cpp
  src/sankhya/pdhg.cpp
  src/sankhya/branch_and_bound.cpp
  src/sankhya/qp.cpp
  src/sankhya/cuts.cpp
  src/sankhya/lu.cpp
  src/sankhya/simplex.cpp
)
target_include_directories(sankhya PUBLIC src)

if(MSVC)
  target_compile_options(sankhya PRIVATE /W4)
else()
  target_compile_options(sankhya PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion>)
  # nvcc must not be handed the C++ warning set at all.
  target_compile_options(sankhya PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-w>)
  if(SANKHYA_WERROR)
    target_compile_options(sankhya PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
  endif()
endif()

if(SANKHYA_ENABLE_CUDA)
  target_sources(sankhya PRIVATE src/sankhya/cuda_backend.cu)
  target_compile_definitions(sankhya PUBLIC SANKHYA_WITH_CUDA)
  set_target_properties(sankhya PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  # The strict warning set is ours, for our C++. nvcc forwards host flags to the
  # host compiler while also generating its own code, and -Wshadow fires inside
  # the CUDA headers rather than in anything we wrote - which is why a first
  # attempt at building this on Kaggle had to delete -Werror wholesale. Scoping
  # the flags to C++ sources keeps them on for the code they were meant for.
  set_source_files_properties(src/sankhya/cuda_backend.cu PROPERTIES
                              COMPILE_OPTIONS "-w")
endif()

add_executable(sankhya-cli app/sankhya_cli.cpp)
target_link_libraries(sankhya-cli PRIVATE sankhya)
set_target_properties(sankhya-cli PROPERTIES OUTPUT_NAME sankhya)

enable_testing()
add_subdirectory(tests)
