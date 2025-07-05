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
 - software interpolation on global memory (done)
 - half precision (possibly with tensor cores reduction?)
 - memory compression (done)

 - collaborativey load the neighbours using tex3d and perform sw interpolation
   (should allow greater throughput, does not)

 - try to load the ligand bbox voxels in shared memory, so then when fetching neighbours
   you don't fetch them again

 - try to group different poses in the same block, perhaps even assign one warp per pose,  but make sure that threads in the same block are assigned to poses translated to the
   close spots, this should in theory allow for better L1 locality

 - precompute interaction for all atom types, this effectively allows to reduce the fetch
 - sort atoms based on morton order to increase spatial locality within a warp
  (maybe more relevant for bigger ligands)
 - introduce divergence to conditionally fetch based on the atom type
   consider sorting atoms by type to reduce divergence
 - predicated execution may cause texture fetches even when guarded by conditionals,
   so perhaps the only way is to group same atom types in the same warp, sorting around
   stuff will then be needed

 - spatial hashing to store only non empty voxels or store sorted octree leaves and perform binary searches to determine if they're empty or not
 consider compression on the indirection table


 Looks like my custom swizzle works better for bigger grids (by tweaking the tile size), and compression is also even better for larger grids (which i assume are thus sparser, so probably because of that). Nevertheless for a spacing ~0.375A they behave mostly the same, even with compression. I should try to collaboratively fetch the texture and interpolate in sw to see if that is better (not faster).


Packing is always faster than individual channels (when fetching all channels), and 4 channels seems to be the best option, 8 channel packing is slower than 4 on global memory. Global memory is on pair for 3 channel packing against native float4 tmem.
But going back to float channels, native tmem with conditional loads performs better than anything else, while conditional load in global memory does not, but try again there, cause the compute overhead doesn't seem to cause that much divergence. 

I guess that precomputed grids stored in single channel textures, conditionally loaded based on the atom type are going to be the best option out there.
But channel packing on swizzled global memory is a fair option if higher precision interpolation is needed.
