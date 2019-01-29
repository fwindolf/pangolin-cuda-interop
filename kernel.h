#include <cuda.h>
#include <cuda_runtime.h>

#include "util.h"

template <typename T>
void transformImage(cudaArray_t input, T* output, int width, int height);