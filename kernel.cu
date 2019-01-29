
#include "kernel.h"

#include <stdio.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void kernel(cudaTextureObject_t tex, T* output, int width, int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;
    
    unsigned int x_new = (x + 3) % width;
    
    T data = tex2D<T>(tex, x_new, y);
    output[x + y * width] = data;
}

template __global__ void kernel(cudaTextureObject_t tex, uchar1* output, int width, int height);
template __global__ void kernel(cudaTextureObject_t tex, uchar4* output, int width, int height);
template __global__ void kernel(cudaTextureObject_t tex, float1* output, int width, int height);
template __global__ void kernel(cudaTextureObject_t tex, float4* output, int width, int height);

template<typename T>
void transformImage(cudaArray_t input, T* output, int width, int height)
{
    cudaTextureObject_t tex;

    cudaResourceDesc texRes;
    memset(&texRes,0, sizeof(cudaResourceDesc));
    texRes.resType            = cudaResourceTypeArray;
    texRes.res.array.array    = input;

    cudaTextureDesc texDescr;
    memset(&texDescr,0,sizeof(cudaTextureDesc));
    // texDescr.normalizedCoords = 1;
    // texDescr.filterMode       = cudaFilterModeLinear;
    texDescr.addressMode[0]   = cudaAddressModeClamp;
    texDescr.addressMode[1]   = cudaAddressModeClamp;
    texDescr.readMode         = cudaReadModeElementType;
    
    cudaSafeCall(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));
    
    // Invoke kernel
    dim3 dimBlock(32, 32);
    dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
                 (height + dimBlock.y - 1) / dimBlock.y);

    kernel <<<dimGrid, dimBlock>>> (tex, (T*)output, width, height);    
    cudaCheck(cudaGetLastError(), __FILE__, __LINE__);

    cudaSafeCall(cudaDestroyTextureObject(tex));
}


template void transformImage(cudaArray_t, uchar1*, int, int);
template void transformImage(cudaArray_t, uchar4*, int, int);
template void transformImage(cudaArray_t, float1*, int, int);
template void transformImage(cudaArray_t, float4*, int, int);
