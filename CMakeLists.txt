cmake_minimum_required(VERSION 3.20)
project(sankhya LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(SANKHYA_ENABLE_CUDA "Build the CUDA backend" OFF)
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
)
target_include_directories(sankhya PUBLIC src)

if(MSVC)
  target_compile_options(sankhya PRIVATE /W4)
else()
  target_compile_options(sankhya PRIVATE -Wall -Wextra -Wpedantic -Wshadow
                                         -Wconversion -Wsign-conversion)
  if(SANKHYA_WERROR)
    target_compile_options(sankhya PRIVATE -Werror)
  endif()
endif()

add_executable(sankhya-cli app/sankhya_cli.cpp)
target_link_libraries(sankhya-cli PRIVATE sankhya)
set_target_properties(sankhya-cli PROPERTIES OUTPUT_NAME sankhya)

enable_testing()
add_subdirectory(tests)
