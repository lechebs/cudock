# Analysis of a multi-layered map access on GPUs

This repository contains CUDA based kernels and analysis code used to evaluate the performance of different memory layouts for volumetric scoring in a pocket-ligand docking scenario.

## Compilation

A C++ compiler with CUDA support and CMake (version 3.20 or higher) are required.

1. Create and enter a build directory:

   ```bash
   mkdir build && cd build
   ```
2. Configure the project:

   ```bash
   cmake ..
   ```
3. Build the binaries:

   ```bash
   make
   ```

## Usage

After building, the `main` executable will be available in the `build` directory.

```bash
./main
```
