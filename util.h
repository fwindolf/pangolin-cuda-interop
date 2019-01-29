#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <stdio.h>

/**
 * Cuda Initialization
 */

static inline void initCuda()
{
    // TODO: Use optimal device
    cudaSetDevice(0);
}

/**
 * Safe calls for cuda functions
 */

#define cudaSafeCall(call_status)  cudaCheck((call_status), __FILE__ ,__LINE__)

static inline void cudaCheck(cudaError call_status, const char* file, const int line)
{
    if(call_status != cudaSuccess)
    {
        std::cerr << file << "(" << line << ") : Cuda call returned with error " 
                  << cudaGetErrorName(call_status) << "(=" << cudaGetErrorString(call_status) << ")" 
                  << std::endl;
        exit(call_status);
    }
}

/**
 * Make device types from a single float
 */
template <typename T>
__host__ __device__ inline T make(float data);

template <>
__host__ __device__ inline uchar1 make(float data)
{
    return make_uchar1(data); 
}

template <>
__host__ __device__ inline uchar4 make(float data)
{
    return make_uchar4(data, data, data, 1); 
}

template <>
__host__ __device__ inline float1 make(float data)
{
    return make_float1(data); 
}

template <>
__host__ __device__ inline float4 make(float data)
{
    return make_float4(data, data, data, 1.f); 
}

/**
 * Print device types
 */

template <typename T>
__device__ inline void print(T& data);

template <>
__device__ inline void print(uchar1& data)
{
    printf("[ %d ]\n", data.x);
}

template <>
__device__ inline void print(uchar4& data)
{
    printf("[ %d, %d, %d, %d ]\n", data.x, data.y, data.z, data.w);
}

template <>
__device__ inline void print(float1& data)
{
    printf("[ %.2f ]\n", data.x);
}

template <>
__device__ inline void print(float4& data)
{
    printf("[ %.2f, %.2f, %.2f, %.2f ] \n", data.x, data.y, data.z, data.w);
}