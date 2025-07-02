This repository contains a template for a CUDA applications.
It is meant to be the as simple as possible and easy to customize.
To add/change the source files that compose the application it is enough to modify the two related lists in `CMakeLists.txt`.


## How to compile the application

We use CMake as building system, so the overall procedure is composed of three steps:
1. Configure the application
1. Compile the application
1. Optionally install the application

For example, Assuming that the current working directory of the terminal is in the repository root, and you want to compile everything in a dedicated folder `build`, you can issue the following three commands:
```bash
$ cmake -B ./build -S . -DCMAKE_INSTALL_PREFIX=.
$ cmake --build ./build
$ cmake --install ./build
```
After issuing all the commands, you will have the executable in the `bin` folder inside the repository.
The install command is actually optional.
If you omit the command, you will have the executable in `build/src` folder.

## TODO
 - custom software swizzling in global memory (done)
 - evaluate __ldg on global memory (it's already done)
 - channel packing (e.g. two float4 textures and global "float8")
 - software interpolation on global memory
 - half precision (possibly with tensor cores reduction?)
 - memory compression??

 - precompute interaction for all atom types, this effectively allows to reduce the fetch
 - sort atoms based on morton order to increase spatial locality within a warp
  (maybe more relevant for bigger ligands)
 - introduce divergence to conditionally fetch based on the atom type
   consider sorting atoms by type to reduce divergence
 - predicated execution may cause texture fetches even when guarded by conditionals,
   so perhaps the only way is to group same atom types in the same warp, sorting around
   stuff will then be needed
