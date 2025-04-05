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
