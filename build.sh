#!/bin/bash

# Modern CLAHE Build Script     echo "Executables created:"
    echo "  - bin/clahe_demo: Test application with comparison features"
    echo "  - bin/clahe_harness: Benchmarking harness (15-30 sec runtime)"
    echo ""
    echo "To test CLAHE:"
    echo "  ./build/bin/clahe_demo test_input.png output.png --metrics"
    echo ""
    echo "To benchmark (15-30 seconds):"
    echo "  ./build/bin/clahe_harness --iterations 266"CV 4.x Support
echo "=== Building Modern CLAHE Project ==="

# Check if OpenCV is installed
if ! pkg-config --exists opencv4; then
    echo "Warning: OpenCV 4.x not found via pkg-config"
    echo "Please ensure OpenCV 4.x is installed:"
    echo "  macOS: brew install opencv"
    echo "  Ubuntu: sudo apt install libopencv-dev"
    echo ""
fi

# Create and enter build directory
echo "Creating build directory..."
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_STANDARD=17 \
      -DOpenCV_DIR=/usr/local/lib/cmake/opencv4 \
      ..

if [ $? -ne 0 ]; then
    echo "CMake configuration failed!"
    echo "Trying alternative OpenCV paths..."
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_STANDARD=17 \
          ..
fi

# Build the project
echo "Building project..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Build Successful! ==="
    echo "Executables created:"
    echo "  - bin/clahe_demo: Main demo application"
    echo "  - bin/clahe_benchmark: Performance benchmarking suite"
    echo ""
    echo "To run demo:"
    echo "  ./build/bin/clahe_demo input_image.png output_image.png --metrics"
    echo ""
    echo "To run benchmark:"
    echo "  ./build/bin/clahe_benchmark --quick"
else
    echo "Build failed!"
    exit 1
fi
