#include "docker.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <type_traits>

#include "utils.cuh"
#include "swizzling.cuh"
#include "ligand.hpp"

#define WARP_SIZE 32
#define BLOCK_SIZE 128

namespace
{
    __constant__
    struct cuDock::Ligand::GPUData _gpu_ligand_data;

    __constant__
    float *_gpu_gmem_voxels[cuDock::Pocket::NUM_CHANNELS];

    __constant__
    cudaTextureObject_t _gpu_tmem_voxels[cuDock::Pocket::NUM_CHANNELS];

    __device__ __inline__ float warp_reduce(float value)
    {
        #pragma unroll 5
        for (int i = 0, d = 16; i < 5; ++i, d >>= 1) {
            value += __shfl_down_sync(0xffffffff, value, d);
        }

        return value;
    }

    __device__ float block_reduce(float value,
                                  int num_elements,
                                  float *shmem)
    {
        int lane_idx = threadIdx.x % WARP_SIZE;
        int warp_idx = threadIdx.x / WARP_SIZE;

        #pragma unroll
        while (num_elements > 1) {

            shmem[threadIdx.x] = 0;

            __syncthreads();

            if (warp_idx <= ((num_elements - 1) >> 5)) {
                value = warp_reduce(value);
                if (lane_idx == 0) {
                    shmem[warp_idx] = value;
                }
            }

            num_elements = ((num_elements - 1) >> 5) + 1;

            __syncthreads();

            if (warp_idx <= ((num_elements - 1) >> 5)) {
                value = shmem[threadIdx.x];
            }
        }

        return value;
    }

    __device__ __inline__ float3 _get_atom_pos(int idx)
    {
        float3 pos = {
            _gpu_ligand_data.atoms_x[idx],
            _gpu_ligand_data.atoms_y[idx],
            _gpu_ligand_data.atoms_z[idx]
        };

        return pos;
    }

    __device__ __inline__ float score_pose_gmem(float3 pos,
                                                float grid_cell_size,
                                                int grid_width,
                                                int grid_height,
                                                unsigned int mask,
                                                int num_channels)
    {
        float score = 0;
        // Computing lookup coordinates
        int i = pos.x / grid_cell_size;
        int j = pos.y / grid_cell_size;
        int k = pos.z / grid_cell_size;

        int grid_idx = k * grid_width * grid_height + j * grid_width + i;

        for (int c = 0; c < num_channels; ++c) {
            score += _gpu_gmem_voxels[c][grid_idx] * (mask & 1);
            mask >>= 1;
        }

        return score;
    }

    __device__ __inline__
    float score_pose_gmem_swizzled(float3 pos,
                                   float grid_cell_size,
                                   int grid_height,
                                   int grid_width,
                                   unsigned int mask,
                                   int num_channels)
    {
        float score = 0;
        // Computing lookup coordinates
        int i = pos.x / grid_cell_size;
        int j = pos.y / grid_cell_size;
        int k = pos.z / grid_cell_size;

        int tile_size_in_bits = 5;
        int padded_width = grid_width + cuDock::Swizzling::
                                        get_padding_size(grid_width,
                                                         tile_size_in_bits);
        int padded_height = grid_height + cuDock::Swizzling::
                                          get_padding_size(grid_height,
                                                           tile_size_in_bits);
        int idx = cuDock::Swizzling::get_swizzled_idx(i,
                                                      j,
                                                      k,
                                                      padded_width,
                                                      padded_height,
                                                      tile_size_in_bits);

        for (int c = 0; c < num_channels; ++c) {
            score += _gpu_gmem_voxels[c][idx] * (mask & 1);
            mask >>= 1;
        }

        return score;
    }

    __device__ __inline__ float score_pose_tmem(float3 pos,
                                                float grid_cell_size,
                                                unsigned int mask,
                                                int num_channels)
    {
        float score = 0;

        int tx = pos.x / grid_cell_size;
        int ty = pos.y / grid_cell_size;
        int tz = pos.z / grid_cell_size;

        for (int c = 0; c < num_channels; ++c) {
            score += tex3D<float>(_gpu_tmem_voxels[c], tx, ty, tz) *
                     (mask & 1);
            mask >>= 1;
        }

        return score;
    }

    __global__ void score_poses(float grid_cell_size,
                                int grid_width,
                                int grid_height,
                                int num_channels,
                                const float3 *translations,
                                const float3 *rotations,
                                float *scores,
                                int num_atoms,
                                int block_size,
                                enum GPUMemType mem_type)
    {
        // Broadcasting to warp
        float3 t = translations[blockIdx.x];
        float3 r = rotations[blockIdx.x];

        float r_mat[9];
        cuDock::Docker::compute_rot_mat(r.x, r.y, r.z, r_mat);

        float score = 0.0;

        for (int idx = threadIdx.x; idx < num_atoms; idx += block_size) {
            float3 pos = _get_atom_pos(idx);
            cuDock::Docker::transform_atom_pos(pos.x,
                                               pos.y,
                                               pos.z,
                                               t.x,
                                               t.y,
                                               t.z,
                                               r_mat,
                                               pos.x,
                                               pos.y,
                                               pos.z);

            unsigned int mask = _gpu_ligand_data.atoms_channel_mask[idx];

            if (mem_type == GPU_GMEM) {
                score += score_pose_gmem(pos,
                                         grid_cell_size,
                                         grid_height,
                                         grid_width,
                                         mask,
                                         num_channels);

            } else if (mem_type == GPU_GMEM_SWIZZLED) {
                score += score_pose_gmem_swizzled(pos,
                                                  grid_cell_size,
                                                  grid_height,
                                                  grid_width,
                                                  mask,
                                                  num_channels);

            } else if (mem_type == GPU_TMEM) {
                score += score_pose_tmem(pos,
                                         grid_cell_size,
                                         mask,
                                         num_channels);
            }
        }

        __shared__ float shmem[BLOCK_SIZE];
        score = block_reduce(score, block_size, shmem);

        if (threadIdx.x == 0) {
            scores[blockIdx.x] = score;
        }
    }
}

namespace cuDock
{
     __device__ __host__ void Docker::compute_rot_mat(float rx,
                                                      float ry,
                                                      float rz,
                                                      float mat[])
    {
        float sinrx = sinf(rx);
        float cosrx = cosf(rx);
        float sinry = sinf(ry);
        float cosry = cosf(ry);
        float sinrz = sinf(rz);
        float cosrz = cosf(rz);

        mat[0] = cosry * cosrz;
        mat[1] = sinrx * sinry * cosrz - cosrx * sinrz;
        mat[2] = cosrx * sinry * cosrz + sinrx * sinrz;

        mat[3] = cosry * sinrz;
        mat[4] = sinrx * sinry * sinrz + cosrx * cosrz;
        mat[5] = cosrx * sinry * sinrz - sinrx * cosrz;

        mat[6] = -sinry;
        mat[7] = sinrx * cosry;
        mat[8] = cosrx * cosry;
    }

    __device__ __host__ void
    Docker::transform_atom_pos(float x,
                               float y,
                               float z,
                               float tx,
                               float ty,
                               float tz,
                               const float r_mat[],
                               float &x_dst,
                               float &y_dst,
                               float &z_dst)
    {
        x_dst = r_mat[0] * x + r_mat[1] * y + r_mat[2] * z + tx;
        y_dst = r_mat[3] * x + r_mat[4] * y + r_mat[5] * z + ty;
        z_dst = r_mat[6] * x + r_mat[7] * y + r_mat[8] * z + tz;
    }

    void Docker::to_gpu()
    {
        int num_poses = _translations.size();
        if (num_poses == 0) {
            return;
        }

        int alloc_size = num_poses * sizeof(float3);


        if (!_is_on_gpu) {
            _is_on_gpu = true;
            CUDA_CHECK_ERR(cudaMalloc(&_gpu_translations, alloc_size));
            CUDA_CHECK_ERR(cudaMalloc(&_gpu_rotations, alloc_size));
            CUDA_CHECK_ERR(cudaMalloc(&_gpu_scores,
                                      num_poses * sizeof(float)));
        }

        std::vector<float3> t_buffer;
        std::vector<float3> r_buffer;
        t_buffer.reserve(num_poses);
        r_buffer.reserve(num_poses);

        for (int i = 0; i < num_poses; ++i) {
            t_buffer[i].x = _translations[i][0];
            t_buffer[i].y = _translations[i][1];
            t_buffer[i].z = _translations[i][2];
            r_buffer[i].x = _rotations[i][0];
            r_buffer[i].y = _rotations[i][1];
            r_buffer[i].z = _rotations[i][2];
        }

        // Copying poses to global memory

        CUDA_CHECK_ERR(cudaMemcpy(_gpu_translations,
                                  t_buffer.data(),
                                  alloc_size,
                                  cudaMemcpyHostToDevice));

        CUDA_CHECK_ERR(cudaMemcpy(_gpu_rotations,
                                  r_buffer.data(),
                                  alloc_size,
                                  cudaMemcpyHostToDevice));

        // Copying ligand data to SoA constant memory buffer

        const std::vector<Ligand::Atom> &atoms = _ligand.get_atoms();
        int num_atoms = atoms.size();

        struct Ligand::GPUData ligand_data;
        for (int i = 0; i < num_atoms; ++i) {
            const Ligand::Atom &atom = atoms[i];
            ligand_data.atoms_x[i] = atom.pos[0];
            ligand_data.atoms_y[i] = atom.pos[1];
            ligand_data.atoms_z[i] = atom.pos[2];
            ligand_data.atoms_mass[i] = Ligand::get_atom_mass(atom.type);
            // ligand_data.atom_type[i] = atom.type;
            ligand_data.atoms_channel_mask[i] =
                Ligand::get_atom_channel_mask(atom.type);
        }

        CUDA_CHECK_ERR(cudaMemcpyToSymbol(_gpu_ligand_data,
                                          &ligand_data,
                                          sizeof(struct Ligand::GPUData))); 
    }

    void Docker::off_gpu()
    {
        if (_is_on_gpu) {
            _is_on_gpu = false;
            CUDA_CHECK_ERR(cudaFree(_gpu_translations));
            CUDA_CHECK_ERR(cudaFree(_gpu_rotations));
            CUDA_CHECK_ERR(cudaFree(_gpu_scores));
        }
    }

    void Docker::get_scores(std::vector<float> &dst) const
    {
        dst.resize(_translations.size());

        if (_is_on_gpu) {
            cudaMemcpy(dst.data(),
                       _gpu_scores,
                       dst.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
        } else {
            std::copy(_scores.begin(), _scores.end(), dst.begin());
        }
    }

    void Docker::_score_poses_gpu(int num_poses)
    {
        if (_pocket.is_on_gpu(GPU_GMEM) ||
            _pocket.is_on_gpu(GPU_GMEM_SWIZZLED)) {
            // Copy pocket voxels gmem pointers to constant memory
            CUDA_CHECK_ERR(cudaMemcpyToSymbol(
                _gpu_gmem_voxels,
                _pocket.get_gpu_gmem_voxels().data(),
                sizeof(float *) * Pocket::NUM_CHANNELS));
        } else {
            // Copy pocket voxels tmem pointers to constant memory
            CUDA_CHECK_ERR(cudaMemcpyToSymbol(
                _gpu_tmem_voxels,
                _pocket.get_gpu_tmem_voxels().data(),
                sizeof(cudaTextureObject_t) * Pocket::NUM_CHANNELS));
        }

        int num_atoms = _ligand.get_num_atoms();
        int block_size = std::ceil((float) num_atoms / WARP_SIZE) * WARP_SIZE;

        if (_pocket.is_on_gpu(GPU_GMEM)) {
            CUDA_TIME_EXEC("_score_gmem", [&](){
                score_poses<<<
                    num_poses,
                    block_size>>>(_pocket.get_cell_size(),
                                  _pocket.get_shape(0),
                                  _pocket.get_shape(1),
                                  Pocket::NUM_CHANNELS,
                                  _gpu_translations,
                                  _gpu_rotations,
                                  _gpu_scores,
                                  num_atoms,
                                  block_size,
                                  GPU_GMEM);
            });
        } else if (_pocket.is_on_gpu(GPU_GMEM_SWIZZLED)) {
            CUDA_TIME_EXEC("_score_gmem_swizzled", [&](){
                score_poses<<<
                    num_poses,
                    block_size>>>(_pocket.get_cell_size(),
                                  _pocket.get_shape(0),
                                  _pocket.get_shape(1),
                                  Pocket::NUM_CHANNELS,
                                  _gpu_translations,
                                  _gpu_rotations,
                                  _gpu_scores,
                                  num_atoms,
                                  block_size,
                                  GPU_GMEM_SWIZZLED);
            });
        } else {
            CUDA_TIME_EXEC("_score_tmem", [&](){
                score_poses<<<
                    num_poses,
                    block_size>>>(_pocket.get_cell_size(),
                                  _pocket.get_shape(0),
                                  _pocket.get_shape(1),
                                  Pocket::NUM_CHANNELS,
                                  _gpu_translations,
                                  _gpu_rotations,
                                  _gpu_scores,
                                  num_atoms,
                                  block_size,
                                  GPU_TMEM);
            });

        }
    }
}
