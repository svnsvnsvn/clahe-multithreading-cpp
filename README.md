# CLAHE OpenCV 4.x Port

Port of Contrast Limited Adaptive Histogram Equalization from legacy OpenCV 1.x to OpenCV 4.x. Foundation for multithreading implementation.

## TODO
- Implement multithreading in compute_histograms(), clip_histograms(), interpolate_image()
- Benchmark single-threaded vs multithreaded performance

## Quick Start

### 1. Install Dependencies
```bash
# macOS
brew install cmake opencv

# Ubuntu/Linux  
sudo apt install cmake libopencv-dev
```

### 2. Build
```bash
./build.sh
```

### 3. Test It Works
```bash
# Quick test with included image
./build/bin/clahe_demo test_input.png output.png --metrics

# Quick benchmark test (a few seconds)
./build/bin/clahe_harness --iterations 10 --image-size 1024x1024
```

### 4. Run Performance Benchmarks
```bash
# 15-second benchmark
./build/bin/clahe_harness --iterations 266

# 30-second benchmark  
./build/bin/clahe_harness --iterations 533

# Test with your own images
./build/bin/clahe_harness --input your_image.png --iterations 50
```
```

## What Changed from Original Code

### Before (Legacy - `legacy/clahe.cpp`)
- OpenCV 1.x `IplImage*` API
- Manual memory management with malloc/free
- Sequential processing only
- Division operations in tight loops

### After (Modern - `src/clahe_modern.cpp`)
- OpenCV 4.x `cv::Mat` instead of `IplImage*`
- RAII memory management replaces malloc/free
- Performance profiling support

## Changes Made

1. Memory: Pre-allocated arrays instead of malloc/free calls
2. Cache: Block-based processing instead of pixel-by-pixel access
3. Math: Bit shifts for power-of-2 divisions where possible
4. API: Modern C++17 error handling and RAII

## Project Files

```
src/clahe_modern.cpp    # Main implementation (single-threaded)
include/clahe_modern.h  # Modern API header
src/demo.cpp           # Test application with OpenCV comparison
src/clahe_harness.cpp  # Benchmarking harness (15-30 sec runtime)
legacy/               # Original code for reference
test_input.png        # Test image (xray)
```

## Usage Examples

```cpp
#include "clahe_modern.h"

// Simple usage
cv::Mat input = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);
cv::Mat output = clahe::apply_clahe(input, 2.0, 8, 8);

// Advanced configuration
clahe::CLAHEConfig config;
config.clip_limit = 3.0;
config.grid_width = 16;
config.grid_height = 16;

clahe::ModernCLAHE processor(config);
cv::Mat result = processor.process(input);
```

## Benchmarking Harness

For reliable performance testing with scalable runtimes:

```bash
# Quick verification (2-3 seconds)
./build/bin/clahe_harness --iterations 10 --image-size 1024x1024

# Standard benchmarks
./build/bin/clahe_harness --iterations 266    # ~15 seconds
./build/bin/clahe_harness --iterations 533    # ~30 seconds

# Test different configurations
./build/bin/clahe_harness --grid-size 32x32 --iterations 150
./build/bin/clahe_harness --image-size 8192x8192 --iterations 50

# Use custom images
./build/bin/clahe_harness --input test_input.png --iterations 100
```

**Key Features:**
- Generates synthetic test images automatically
- Progress indicators for long runs
- Automatic runtime scaling estimates
- Memory usage reporting
- Works with any input image size


These functions are the bottlenecks that need parallelization.

## Troubleshooting

**Build fails?** Make sure you have OpenCV 4.x: `pkg-config --modversion opencv4`

**Demo crashes?** Verify your input image exists and is readable by OpenCV

**Poor performance?** Enable Release mode: `cmake -DCMAKE_BUILD_TYPE=Release`