cmake_minimum_required(VERSION 3.16)
project(microkernel VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Library
add_library(microkernel
    src/tensor.cpp
    src/dtype.cpp
)

target_include_directories(microkernel
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# GoogleTest via FetchContent
include(FetchContent)
set(GTEST_VERSION v1.17.0)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/${GTEST_VERSION}.zip
)
# Prevent GTest from installing alongside the project
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Tests
enable_testing()
include(GoogleTest)

add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE microkernel GTest::gtest GTest::gtest_main)
gtest_discover_tests(test_tensor)
