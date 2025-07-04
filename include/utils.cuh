#pragma once

#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

#include <cuda.h>

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

#define CUDADR_CHECK_ERR(api_call)                          \
{                                                           \
    CUresult res = api_call;                                \
    if (res != CUDA_SUCCESS) {                              \
        char err_string[128];                               \
        const char *ptr = err_string;                       \
        cuGetErrorString(res, &ptr);                        \
        std::cout << "[ERROR:"                              \
                  << strrchr(__FILE__, '/') + 1 << ":"      \
                  << __LINE__ << "] "                       \
                  << err_string                             \
                  << std::endl;                             \
    }                                                       \
}



enum GPUMemType { GPU_GMEM, GPU_GMEM_SWIZZLED, GPU_TMEM };
enum InterpolateType { NN_INTERPOLATE, LIN_INTERPOLATE };

template<typename T> void CUDA_TIME_EXEC(const std::string &tag,
                                         const T &launch_kernel,
                                         int num_launches = 5,
                                         int num_warmup = 1)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<float> elapsed;
    elapsed.reserve(num_launches);

    // Warm-up launches
    for (int i = 0; i < num_warmup; ++i) {
        launch_kernel();
    }

    for (int i = 0; i < num_launches; ++i) {

        cudaEventRecord(start);
        launch_kernel();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        elapsed.push_back(ms);
    }

    CUDA_CHECK_ERR(cudaGetLastError());

    float elapsed_mean = 0.0;
    for (float ms : elapsed) {
        elapsed_mean += ms;
    }
    elapsed_mean /= num_launches;

    float elapsed_std = 0.0;
    for (float ms : elapsed) {
        float diff = ms - elapsed_mean;
        elapsed_std += diff * diff;
    }
    elapsed_std /= num_launches - (num_launches > 1 ? 1 : 0);
    elapsed_std = std::sqrt(elapsed_std);

    printf("[%s] launches=%d, mean=%.6f, std=%.6f\n",
           tag.c_str(),
           num_launches,
           elapsed_mean,
           elapsed_std);
}
