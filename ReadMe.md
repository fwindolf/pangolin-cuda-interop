# Pangolin-Cuda-Interop

A minimal example showing how to modify the GLTextures provided by Pangolin with CUDA.

`main.cpp` contains all the OpenGL/Pangolin code.

It will first open an image, then create the Pangolin GlTexture and get its cudaArray, which will then be used in the cuda kernel (in form of a texture object). The output of the kernel is put to tmp, and copied back into the GlTexture's cudaArray.

`kernel.cu` contains the kernel and the logic to call the kernel

The kernel is templated, so the code should also work with float, and also in greyscale. Major changes would be in allocating the host array.

`util.h` contains some helper functions for cuda


## Install

With an existing (recent) installation of cuda toolkit, Eigen and OpenCV, proceed to install Pangolin in the `third_party` directory.

```
git submodule init

cd third_party/Pangolin
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=.. [-DCMAKE_BUILD_TYPE=Debug] ..
make
``` 

When having trouble to build Pangolin, take a look at the [Pangolin ReadMe](https://github.com/stevenlovegrove/Pangolin/#dependencies)

After installing Pangolin to `third_party`, go back to the source directory and build this Application.

```
mkdir build && cd build
cmake ..
make
```

## Run

Just execute the `./interop` executable in the build folder.
