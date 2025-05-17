#include "docker.hpp"

#include <iostream>
#include <vector>

#include "ligand.hpp"
#include "utils.cuh"

#define WARP_SIZE 32
#define BLOCK_SIZE 128

namespace
{
    __global__ void _score_gmem(const float * const voxels[],
                                int num_channels,
                                int num_poses)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx > num_poses) {
            return;
        }
    }

    __global__ void _score_tmem(const cudaTextureObject_t textures[],
                                int num_textures,
                                int num_poses)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx > num_poses) {
            return;
        }
    }
}

namespace cuDock
{
    __constant__ struct Ligand::GPUData _gpu_ligand_data;

    void Docker::to_gpu()
    {
        int num_poses = _translations.size();
        if (num_poses == 0) {
            return;
        }

        int alloc_size = num_poses * sizeof(float);

        if (!_is_on_gpu) {
            _is_on_gpu = true;
            CUDA_CHECK_ERR(cudaMalloc(&_gpu_translations_x, alloc_size));
            CUDA_CHECK_ERR(cudaMalloc(&_gpu_translations_y, alloc_size));
            CUDA_CHECK_ERR(cudaMalloc(&_gpu_translations_z, alloc_size));
            CUDA_CHECK_ERR(cudaMalloc(&_gpu_rotations_x, alloc_size));
            CUDA_CHECK_ERR(cudaMalloc(&_gpu_rotations_y, alloc_size));
            CUDA_CHECK_ERR(cudaMalloc(&_gpu_rotations_z, alloc_size));
        }

        std::vector<float> buffer_x;
        std::vector<float> buffer_y;
        std::vector<float> buffer_z;
        buffer_x.reserve(num_poses);
        buffer_y.reserve(num_poses);
        buffer_z.reserve(num_poses);

        float *dst_buffers[3];
        float *src_buffers[] = { buffer_x.data(),
                                 buffer_y.data(),
                                 buffer_z.data() };

        // Copying poses to global memory in SoA buffer

        aos_to_soa(_translations.data(), src_buffers, num_poses);
        dst_buffers[0] = _gpu_translations_x;
        dst_buffers[1] = _gpu_translations_y;
        dst_buffers[2] = _gpu_translations_z;
        for (int i = 0; i < 3; ++i) {
            CUDA_CHECK_ERR(cudaMemcpy(dst_buffers[i],
                                      src_buffers[i],
                                      alloc_size,
                                      cudaMemcpyHostToDevice));
        }

        aos_to_soa(_rotations.data(), src_buffers, num_poses);
        dst_buffers[0] = _gpu_rotations_x;
        dst_buffers[1] = _gpu_rotations_y;
        dst_buffers[2] = _gpu_rotations_z;
        for (int i = 0; i < 3; ++i) {
            CUDA_CHECK_ERR(cudaMemcpy(dst_buffers[i],
                                      src_buffers[0],
                                      alloc_size,
                                      cudaMemcpyHostToDevice));
        }

        // Copying ligand data to SoA constant memory buffer

        const std::vector<Ligand::Atom> &atoms = _ligand.get_atoms();
        int num_atoms = atoms.size();

        struct Ligand::GPUData ligand_data;
        for (int i = 0; i < num_atoms; ++i) {
            const Ligand::Atom &atom = atoms[i];
            ligand_data.atoms_x[i] = atom.pos[0];
            ligand_data.atoms_y[i] = atom.pos[1];
            ligand_data.atoms_z[i] = atom.pos[2];
            ligand_data.atom_type[i] = atom.type;
        }

        CUDA_CHECK_ERR(cudaMemcpyToSymbol(_gpu_ligand_data,
                                          &ligand_data,
                                          sizeof(struct Ligand::GPUData)));
    }

    void Docker::off_gpu()
    {
        if (_is_on_gpu) {
            _is_on_gpu = false;
            CUDA_CHECK_ERR(cudaFree(_gpu_translations_x));
            CUDA_CHECK_ERR(cudaFree(_gpu_translations_y));
            CUDA_CHECK_ERR(cudaFree(_gpu_translations_z));
            CUDA_CHECK_ERR(cudaFree(_gpu_rotations_x));
            CUDA_CHECK_ERR(cudaFree(_gpu_rotations_y));
            CUDA_CHECK_ERR(cudaFree(_gpu_rotations_z));
        }
    }

    void Docker::_score_poses_gpu(int num_poses)
    {
        int num_blocks = (num_poses - 1) / BLOCK_SIZE + 1;

        if (_pocket.is_on_gpu(GPU_GMEM)) {
            CUDA_TIME_EXEC("_score_gmem", [&](){
                _score_gmem<<<
                    num_blocks,
                    BLOCK_SIZE>>>(_pocket.get_gpu_gmem_voxels().data(),
                                  Pocket::NUM_CHANNELS,
                                  num_poses);
            });
        } else {
            CUDA_TIME_EXEC("_score_tmem", [&](){
                _score_tmem<<<
                    num_blocks,
                    BLOCK_SIZE>>>(_pocket.get_gpu_tmem_voxels().data(),
                                  Pocket::NUM_CHANNELS,
                                  num_poses);
            });
        }
    }
}
