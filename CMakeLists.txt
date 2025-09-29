cmake_minimum_required(VERSION 3.16)
project(CLAHE_Modern VERSION 1.0)

# Set C++ standard to C++17 for modern features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV 4.x
find_package(OpenCV 4.0 REQUIRED)

# Add optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

# Create library for modern CLAHE implementation
add_library(clahe_modern STATIC
    src/clahe_modern.cpp
)

target_link_libraries(clahe_modern ${OpenCV_LIBS})

# Create demo executable
add_executable(clahe_demo
    src/demo.cpp
)

target_link_libraries(clahe_demo clahe_modern ${OpenCV_LIBS})

# Create benchmark executable
add_executable(clahe_benchmark
    src/benchmark_main.cpp
    src/benchmark.cpp
)

target_link_libraries(clahe_benchmark clahe_modern ${OpenCV_LIBS})

# Set output directories
set_target_properties(clahe_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(clahe_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
