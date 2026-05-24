cmake_minimum_required(VERSION 3.20)
project(nbody-sim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# Fetch dependencies
include(FetchContent)

FetchContent_Declare(nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.8.1
)
FetchContent_MakeAvailable(Catch2)

# Source files
set(SIM_SOURCES
  src/config/Config.cpp
  src/disk/DiskInit.cpp
  src/physics/BHTree.cu
  src/physics/Integrator.cu
  src/analysis/Aggregate.cpp
  src/io/OutputWriter.cpp
)

# Main executable
add_executable(nbody-sim src/main.cpp ${SIM_SOURCES})
target_include_directories(nbody-sim PRIVATE include)
target_link_libraries(nbody-sim PRIVATE nlohmann_json::nlohmann_json)
set_target_properties(nbody-sim PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES "86;89"
)

# Test executable
set(TEST_SOURCES
  tests/test_config.cpp
  tests/test_disk_init.cpp
  tests/test_output.cpp
  tests/test_integrator.cu
  tests/test_bhtree.cu
  tests/test_collision.cu
  tests/test_aggregate.cpp
)
add_executable(nbody-tests ${TEST_SOURCES} ${SIM_SOURCES})
target_include_directories(nbody-tests PRIVATE include)
target_link_libraries(nbody-tests PRIVATE Catch2::Catch2WithMain nlohmann_json::nlohmann_json)
set_target_properties(nbody-tests PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES "86;89"
)

include(CTest)
include(Catch)
catch_discover_tests(nbody-tests)
