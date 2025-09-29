# Quick Setup for Team Members

## What This Project Is
Modern C++17 port of CLAHE algorithm from old OpenCV 1.x to OpenCV 4.x

## Setup (Takes 5 minutes)

### 1. Clone and Install
```bash
git clone <repository-url>
cd CLAHE

# Install dependencies (choose your OS)
brew install cmake opencv          # macOS
sudo apt install cmake libopencv-dev  # Ubuntu
```

### 2. Build
```bash
./build.sh
```

### 3. Test It Works
```bash
./build/bin/clahe_demo test_input.png output.png --metrics
```

## What Changed From Original Code

- **Old**: `IplImage*` + manual `malloc`/`free` 
- **New**: `cv::Mat` + modern C++17 RAII

**Main files to look at:**
- `src/clahe_modern.cpp` - New implementation (single-threaded, ready for your multithreading)
- `legacy/clahe.cpp` - Original code for comparison

## Your Job: Add Multithreading

Look for these functions in `src/clahe_modern.cpp`:
- `compute_histograms()` - Parallelize the grid regions
- `clip_histograms()` - Each histogram independent  
- `interpolate_image()` - Each subregion independent

## Common Issues

**"cmake: command not found"** → Install cmake first

**"OpenCV not found"** → Make sure you have OpenCV 4.x, check with: `pkg-config --modversion opencv4`

**Build works but demo crashes?** → Make sure `test_input.png` exists in the directory