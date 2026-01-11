# Dublin 3D Reconstruction Project

This repository contains the course project for urban point cloud processing and 3D building reconstruction.  

The project focuses on a complete workflow from raw point cloud preprocessing to building-level 3D reconstruction.

---

## Project Structure

The project is organized into three independent submodules:

### 1. Dublin_3D_Reconstruction

Core C++ project managed with CMake and implemented using PCL (Point Cloud Library).  

This module corresponds to Chapters 4–7 of the project report and implements the full processing pipeline.

**Main functions include:**

- Point cloud preprocessing (downsampling, noise removal)

- Ground segmentation

- Non-ground object clustering

- Building surface reconstruction (Poisson)

**Source files:**

- `01\_preprocess.cpp`

- `02\_ground\_segmentation.cpp`

- `03\_segment\_euclidean.cpp`

- `04\_reconstruction\_poisson.cpp`

Build outputs are generated under:out/build/x64-Release/

---

### 2. Random\_Forest\_Script

An independent script-based project implementing random forest classification for point cloud semantic labeling.  

This module is used as an extended experiment for comparing traditional geometric segmentation with learning-based methods.

---

### 3. Buildings\_Mesh\_Script

Python scripts for building-level mesh reconstruction and visualization.  

This module performs post-processing on extracted building point clouds and generates 3D mesh models for visualization.

---

## Build and Run (Core Project)

### Environment

- C++17

- CMake

- PCL

- Visual Studio (CMake workflow)

- Windows x64

### Basic Workflow

1. Configure and build the project using CMake (Release / RelWithDebInfo).

2. Run executables in the following order:

&nbsp;  - `01\_preprocess`

&nbsp;  - `02\_ground\_segmentation`

&nbsp;  - `03\_segment\_euclidean`

&nbsp;  - `04\_reconstruction\_poisson`

1. Load intermediate and final results using CloudCompare for verification.

---

## Notes

- Debug build artifacts and IDE cache files (e.g. `.vs/`) are intentionally excluded.

- The repository keeps only source code and release build outputs for clarity and lightweight distribution.

---

## Author

Course Project – Urban 3D Modeling
