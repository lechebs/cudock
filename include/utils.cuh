#pragma once

#include <iostream>
#include <cstring>

#include "vec3.hpp"

#define CUDA_CHECK_ERR(api_call)                            \
{                                                           \
    cudaError_t res = api_call;                             \
    if (res != cudaSuccess) {                               \
        std::cout << "[ERROR:"                              \
                  << strrchr(__FILE__, '/') + 1 << ":"      \
                  << __LINE__ << "] "                       \
                  << cudaGetErrorString(res)                \
                  << std::endl;                             \
    }                                                       \
}

enum GPUMemType { GPU_GMEM, GPU_TMEM };

template<typename T> void CUDA_TIME_EXEC(const std::string &tag,
                                         const T &launch_kernel)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    launch_kernel();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    CUDA_CHECK_ERR(cudaGetLastError());

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "[" << tag << "] elapsed: " << ms << "ms" << std::endl;
}

inline void aos_to_soa(vec3 *aos, float *dst[], int size)
{
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < 3; ++j) {
            dst[j][i] = aos[i][j];
        }
    }
}
