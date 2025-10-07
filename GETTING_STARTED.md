# Quick Setup for Team Members

## What This Project Is
Modern C++17 port of CLAHE algorithm from old OpenCV 1.x to OpenCV 4.x

## Setup (Takes 5 minutes)

### 1. Clone and Install
```bash
git clone <repository-url>
cd CLAHE
```

**Install dependencies (choose your OS):**

**macOS:**
```bash
brew install cmake opencv
```

**Ubuntu/Linux:**
```bash
sudo apt install cmake libopencv-dev
```

**Windows:**
1. **Install Visual Studio 2019/2022** with C++ Desktop Development workload
2. **Install CMake:** Download from https://cmake.org/download/ or use chocolatey:
   ```powershell
   choco install cmake
   ```
3. **Install OpenCV:** 
   - Download pre-built binaries from https://opencv.org/releases/
   - Extract to `C:\opencv`
   - Add `C:\opencv\build\x64\vc16\bin` to your PATH environment variable
   - Or use vcpkg:
     ```powershell
     git clone https://github.com/Microsoft/vcpkg.git
     cd vcpkg
     .\bootstrap-vcpkg.bat
     .\vcpkg install opencv4[contrib]:x64-windows
     ```

### 2. Build

**macOS/Linux:**
```bash
./build.sh
```

**Windows (Command Prompt/PowerShell):**
```powershell
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 ..
cmake --build . --config Release
```

**Windows (if using vcpkg):**
```powershell
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake ..
cmake --build . --config Release
```

### 3. Test It Works

**macOS/Linux:**
```bash
./build/bin/clahe_demo test_input.png output.png --metrics
./build/bin/clahe_harness --iterations 10 --image-size 1024x1024
```

**Windows:**
```powershell
.\build\Release\clahe_demo.exe test_input.png output.png --metrics
.\build\Release\clahe_harness.exe --iterations 10 --image-size 1024x1024
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

**"OpenCV not found"** → Make sure you have OpenCV 4.x installed:
- **macOS/Linux:** Check with `pkg-config --modversion opencv4`
- **Windows:** Verify OpenCV bin directory is in PATH, or use vcpkg integration

**Build works but demo crashes?** → Make sure `test_input.png` exists in the directory

**Windows-specific issues:**
- **"MSVCR140.dll missing"** → Install Visual C++ Redistributable 2019/2022
- **"Cannot find opencv_world*.dll"** → Add OpenCV bin directory to PATH
- **CMake can't find OpenCV** → Try specifying OpenCV path:
  ```powershell
  cmake -DOpenCV_DIR=C:\opencv\build\x64\vc16\lib ..
  ```