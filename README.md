# Analysis of a multi-layered map access on GPUs

This project provides a CUDA based volumetric scoring kernel with multiple memory layouts.

## Prerequisites

Ensure that the following software is installed:

* A C++ compiler with CUDA support (e.g., NVIDIA CUDA Toolkit).
* CMake version 3.10 or higher.

## Compilation

1. Create a build directory in the project root:

   ```bash
   mkdir build
   cd build
   ```
2. Configure the project with CMake:

   ```bash
   cmake ..
   ```
3. Build the project:

   ```bash
   make
   ```

After these steps, the compiled binaries will be located in the `build` directory.
