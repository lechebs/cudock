#include "pocket.hpp"

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "utils.cuh"
#include "swizzling.cuh"
#include "docker.hpp"

namespace
{
    void pack_channels(float *channels[],
                       float *dst,
                       int channel_offset,
                       int size)
    {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < 4; ++k) {
                dst[j * 4 + k] = channels[channel_offset + k][j];
            }
        }
    }

    void set_alloc_prop(CUmemAllocationProp *prop,
                        bool compressible)
    {
        CUdevice device;
        CUDADR_CHECK_ERR(cuCtxGetDevice(&device));

        std::memset(prop, 0, sizeof(CUmemAllocationProp));

        prop->type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop->location.id = device;
        prop->allocFlags.compressionType =
            compressible ? CU_MEM_ALLOCATION_COMP_GENERIC :
                           CU_MEM_ALLOCATION_COMP_NONE;
    }

    size_t get_alloc_global_size(CUmemAllocationProp *prop,
                                 int size,
                                 int num_buffers)
    {
        size_t granularity = 0;
        CUDADR_CHECK_ERR(cuMemGetAllocationGranularity(
            &granularity, prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

        size_t global_size = sizeof(float) * size * num_buffers;
        return ((global_size - 1) / granularity + 1) * granularity;
    }

    void _alloc_global(float *src[],
                       float *dst[],
                       int size,
                       int num_buffers,
                       bool compressible,
                       bool packed)
    {

        // Allocate a chunk of physical memory
        CUmemAllocationProp prop;
        set_alloc_prop(&prop, compressible);
        size_t global_size = get_alloc_global_size(&prop, size, num_buffers);
        CUmemGenericAllocationHandle alloc_handle;
        CUDADR_CHECK_ERR(cuMemCreate(&alloc_handle, global_size, &prop, 0));

        CUdeviceptr ptr;
        // Reserve a virtual address range and map it to physical memory
        CUDADR_CHECK_ERR(cuMemAddressReserve(&ptr, global_size, 0, 0, 0));
        CUDADR_CHECK_ERR(cuMemMap(ptr, global_size, 0, alloc_handle, 0));

        // Release physical allocation handle
        CUDADR_CHECK_ERR(cuMemRelease(alloc_handle));

        // Make the address accessible
        CUmemAccessDesc access_desc = {};
        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id = 0;
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CUDADR_CHECK_ERR(cuMemSetAccess(ptr, global_size, &access_desc, 1));

        int num_alloc = packed ? num_buffers / 4 : num_buffers;

        for (int i = 0; i < num_alloc; ++i) {
            dst[i] = (float *) ptr + size * i * (packed ? 4 : 1);

            float *curr_src = src[i];
            if (packed) {
                curr_src = new float[size * 4];
                pack_channels(src, curr_src, i * 4, size);
            }

            CUDA_CHECK_ERR(cudaMemcpy(dst[i],
                                      curr_src,
                                      sizeof(float) * size * (packed ? 4 : 1),
                                      cudaMemcpyHostToDevice));

            if (packed) {
                delete[] curr_src;
            }
        }
    }

    void _free_global(float *ptr,
                      int size,
                      int num_buffers,
                      bool compressible)
    {
        CUmemAllocationProp prop;
        set_alloc_prop(&prop, compressible);
        size_t global_size = get_alloc_global_size(&prop, size, num_buffers);

        CUDADR_CHECK_ERR(cuMemUnmap((CUdeviceptr) ptr, global_size));
        CUDADR_CHECK_ERR(cuMemAddressFree((CUdeviceptr) ptr, global_size));
    }

    void _alloc_textures(float *src[],
                         cudaArray_t dst_arrays[],
                         cudaTextureObject_t dst_textures[],
                         int num_channels,
                         int width,
                         int height,
                         int depth,
                         bool lerp,
                         bool packed)
    {
        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeArray;

        cudaChannelFormatDesc fmt_desc;
        if (packed) {
            fmt_desc = cudaCreateChannelDesc(
                32, 32, 32, 32, cudaChannelFormatKindFloat);
        } else {
            fmt_desc = cudaCreateChannelDesc(
                32, 0, 0, 0, cudaChannelFormatKindFloat);
        }

        cudaExtent extent = make_cudaExtent(width, height, depth);

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeClamp;
        tex_desc.addressMode[1] = cudaAddressModeClamp;
        tex_desc.addressMode[2] = cudaAddressModeClamp;
        //tex_desc.normalizedCoords = false;
        tex_desc.filterMode = lerp ? cudaFilterModeLinear :
                                     cudaFilterModePoint;
        tex_desc.disableTrilinearOptimization = true;

        cudaResourceViewDesc view_desc = {};
        view_desc.format = packed ? cudaResViewFormatFloat4 :
                                    cudaResViewFormatFloat1;
        view_desc.width = width;
        view_desc.height = height;
        view_desc.depth = depth;

        int num_textures = packed ? num_channels / 4 : num_channels;

        for (int i = 0; i < num_textures; ++i) {
            CUDA_CHECK_ERR(cudaMalloc3DArray(&dst_arrays[i],
                                             &fmt_desc,
                                             extent));

            float *src_buff = src[i];

            if (packed) {
                // Pack 4 channels
                int size = width * height * depth;
                float *packed_src = new float[size * 4];
                pack_channels(src, packed_src, i * 4, size);
                src_buff = packed_src;
            }

            cudaMemcpy3DParms parms = {0};
            parms.srcPtr = make_cudaPitchedPtr(src_buff,
                                               width * sizeof(float) *
                                                   (packed ? 4 : 1),
                                               width,
                                               height);
            parms.dstArray = dst_arrays[i];
            parms.extent = extent;
            parms.kind = cudaMemcpyHostToDevice;
            CUDA_CHECK_ERR(cudaMemcpy3D(&parms));

            if (packed) {
                delete[] src_buff;
            }

            res_desc.res.array.array = dst_arrays[i];
            CUDA_CHECK_ERR(cudaCreateTextureObject(&dst_textures[i],
                                                   &res_desc,
                                                   &tex_desc,
                                                   &view_desc));
        }
    }

    void _free_textures(cudaArray_t arrays[],
                        cudaTextureObject_t textures[],
                        int num_channels,
                        bool packed)
    {
        int num_textures = packed ? num_channels / 4 : num_channels;

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
        return _is_on_gpu[GPU_GMEM] ||
               _is_on_gpu[GPU_GMEM_SWIZZLED] ||
               _is_on_gpu[GPU_TMEM];
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
                              NUM_CHANNELS,
                              use_compressible_memory_,
                              is_packed());

            } else if (mem_type == GPU_GMEM_SWIZZLED) {

                int w = get_shape(0);
                int h = get_shape(1);
                int d = get_shape(2);

                int swizzled_size = Swizzling::get_swizzled_size(
                    w, h, d, get_swizzled_tile_size());

                std::array<float *, NUM_CHANNELS> voxels_swizzled;
                for (int c = 0; c < NUM_CHANNELS; ++c) {
                    voxels_swizzled[c] = new float[swizzled_size];
                    Swizzling::
                    to_swizzled_format(w, h, d,
                                       get_swizzled_tile_size(),
                                       _voxels[c],
                                       voxels_swizzled[c]);
                }

                _alloc_global(voxels_swizzled.data(),
                              _gpu_global_voxels.data(),
                              swizzled_size,
                              NUM_CHANNELS,
                              use_compressible_memory_,
                              is_packed());

                for (int c = 0; c < NUM_CHANNELS; ++c) {
                    delete[] voxels_swizzled[c];
                }

            } else if (mem_type == GPU_TMEM) {
                _alloc_textures(_voxels.data(),
                                _gpu_array_voxels.data(),
                                _gpu_texture_voxels.data(),
                                NUM_CHANNELS,
                                get_shape(0),
                                get_shape(1),
                                get_shape(2),
                                get_interpolate() == LIN_INTERPOLATE,
                                is_packed());
            }
        }

    }

    void Pocket::off_gpu(enum GPUMemType mem_type)
    {
        if (_is_on_gpu[mem_type]) {
            _is_on_gpu[mem_type] = false;

            if (mem_type == GPU_GMEM) {
                _free_global(_gpu_global_voxels[0],
                             get_size(),
                             NUM_CHANNELS,
                             use_compressible_memory_);
            } else if (mem_type == GPU_GMEM_SWIZZLED) {
                _free_global(_gpu_global_voxels[0],
                             Swizzling::get_swizzled_size(
                                get_shape(0),
                                get_shape(1),
                                get_shape(2),
                                get_swizzled_tile_size()),
                             NUM_CHANNELS,
                             use_compressible_memory_);
            } else if (mem_type == GPU_TMEM) {
                _free_textures(_gpu_array_voxels.data(),
                               _gpu_texture_voxels.data(),
                               NUM_CHANNELS,
                               is_packed());
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
