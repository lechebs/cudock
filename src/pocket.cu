#include "pocket.hpp"

#include <iostream>

#include <cuda_runtime_api.h>

#include "utils.cuh"

namespace
{
    void _alloc_global(float *src[], float *dst[], int size, int num_buffers)
    {
        for (int i = 0; i < num_buffers; ++i) {
            CUDA_CHECK_ERR(cudaMalloc(&dst[i], sizeof(float) * size));
            CUDA_CHECK_ERR(cudaMemcpy(dst[i],
                                      src[i],
                                      sizeof(float) * size,
                                      cudaMemcpyHostToDevice));
        }
    }

    void _free_global(float *buffers[], int num_buffers)
    {
        for (int i = 0; i < num_buffers; ++i) {
            CUDA_CHECK_ERR(cudaFree(buffers[i]));
        }
    }

    void _alloc_textures(float *src[],
                         cudaArray_t dst_arrays[],
                         cudaTextureObject_t dst_textures[],
                         int num_textures,
                         int width,
                         int height,
                         int depth)
    {
        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;
        cudaChannelFormatDesc fmt_desc =
            cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaExtent extent = make_cudaExtent(width, height, depth);

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeClamp;
        tex_desc.addressMode[1] = cudaAddressModeClamp;
        tex_desc.addressMode[2] = cudaAddressModeClamp;
        //tex_desc.normalizedCoords = false;
        tex_desc.filterMode = cudaFilterModePoint;
        //tex_desc.disableTrilinearOptimization = true;

        cudaResourceViewDesc view_desc = {};
        view_desc.format = cudaResViewFormatFloat1;
        view_desc.width = width;
        view_desc.height = height;
        view_desc.depth = depth;

        for (int i = 0; i < num_textures; ++i) {
            CUDA_CHECK_ERR(cudaMalloc3DArray(&dst_arrays[i], &fmt_desc, extent));

            cudaMemcpy3DParms parms = {0};
            parms.srcPtr = make_cudaPitchedPtr(src[i],
                                               width * sizeof(float),
                                               width,
                                               height);
            parms.dstArray = dst_arrays[i];
            parms.extent = extent;
            parms.kind = cudaMemcpyHostToDevice;
            CUDA_CHECK_ERR(cudaMemcpy3D(&parms));

            res_desc.res.array.array = dst_arrays[i];
            CUDA_CHECK_ERR(cudaCreateTextureObject(&dst_textures[i],
                                                   &res_desc,
                                                   &tex_desc,
                                                   &view_desc));
        }
    }

    void _free_textures(cudaArray_t arrays[],
                        cudaTextureObject_t textures[],
                        int num_textures)
    {
        for (int i = 0; i < num_textures; ++i) {
            CUDA_CHECK_ERR(cudaFreeArray(arrays[i]));
            CUDA_CHECK_ERR(cudaDestroyTextureObject(textures[i]));
        }
    }
};

namespace cuDock
{
    bool Pocket::is_on_gpu() const
    {
        return _is_on_gpu[GPU_GMEM] || _is_on_gpu[GPU_TMEM];
    }

    bool Pocket::is_on_gpu(enum GPUMemType mem_type) const
    {
        return _is_on_gpu[mem_type];
    }

    void Pocket::to_gpu(enum GPUMemType mem_type)
    {
        if (!_is_on_gpu[mem_type]) {
            _is_on_gpu[mem_type] = true;

            if (mem_type == GPU_GMEM) {
                _alloc_global(_voxels.data(),
                              _gpu_global_voxels.data(),
                              get_size(),
                              NUM_CHANNELS);
            } else if (mem_type == GPU_TMEM) {
                _alloc_textures(_voxels.data(),
                                _gpu_array_voxels.data(),
                                _gpu_texture_voxels.data(),
                                NUM_CHANNELS,
                                get_shape(0),
                                get_shape(1),
                                get_shape(2));
            }
        }

    }

    void Pocket::off_gpu(enum GPUMemType mem_type)
    {
        if (_is_on_gpu[mem_type]) {
            _is_on_gpu[mem_type] = false;

            if (mem_type == GPU_GMEM) {
                _free_global(_gpu_global_voxels.data(), NUM_CHANNELS);
            } else if (mem_type == GPU_TMEM) {
                _free_textures(_gpu_array_voxels.data(),
                               _gpu_texture_voxels.data(),
                               NUM_CHANNELS);
            }
        }
    }

    const std::array<float *, Pocket::NUM_CHANNELS>
    &Pocket::get_gpu_gmem_voxels() const
    {
        return _gpu_global_voxels;
    }

    const std::array<cudaTextureObject_t, Pocket::NUM_CHANNELS>
    &Pocket::get_gpu_tmem_voxels() const
    {
        return _gpu_texture_voxels;
    }
};
