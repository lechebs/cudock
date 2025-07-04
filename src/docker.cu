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
    struct cuDock::Ligand::GPUData GPU_LIGAND_DATA;

    __constant__
    float * __restrict__ GPU_GMEM_VOXELS[cuDock::Pocket::NUM_CHANNELS];

    __constant__
    cudaTextureObject_t GPU_TMEM_VOXELS[cuDock::Pocket::NUM_CHANNELS];

    __constant__ int NUM_CHANNELS;

    __constant__ int GRID_WIDTH;
    __constant__ int GRID_HEIGHT;
    __constant__ int GRID_DEPTH;
    __constant__ float GRID_CELL_SIZE;

    __constant__ int SWIZZLED_X_OFFSET_MULT;
    __constant__ int SWIZZLED_Y_OFFSET_MULT;
    __constant__ int SWIZZLED_Z_OFFSET_MULT;
    __constant__ int SWIZZLED_TILE_SIZE;

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
            GPU_LIGAND_DATA.atoms_x[idx],
            GPU_LIGAND_DATA.atoms_y[idx],
            GPU_LIGAND_DATA.atoms_z[idx]
        };

        return pos;
    }

    __device__ __forceinline__ float lerp(float t, float v1, float v2)
    {
        return v1 * (1.0f - t) + v2 * t;
    }

    /*
    __device__ __forceinline__ float bilerp(float x, float y,
                                            float v1, float v2,
                                            float v3, float v4)
    {
        return lerp(y, lerp(x, v1, v2), lerp(x, v3, v4));
    }

    __device__ float trilerp(float x, float y, float z, float v[8])
    {
        float v1 = bilerp(x,
                          y,
                          v[0], v[1],
                          v[2], v[3]);

        float v2 = bilerp(x,
                          y,
                          v[4], v[5],
                          v[6], v[7]);

        return lerp(z, v1, v2);
    }
    */

    template<GPUMemType MEM_TYPE>
    __device__ __forceinline__
    float voxel_fetch(int c, int i, int j, int k)
    {
        if constexpr (MEM_TYPE == GPU_GMEM) {
            int idx = k * GRID_WIDTH * GRID_HEIGHT +
                      j * GRID_WIDTH + i;
            return GPU_GMEM_VOXELS[c][idx];

        } else if (MEM_TYPE == GPU_GMEM_SWIZZLED) {
            int idx = cuDock::Swizzling::
                      get_swizzled_idx(i, j, k,
                                       SWIZZLED_X_OFFSET_MULT,
                                       SWIZZLED_Y_OFFSET_MULT,
                                       SWIZZLED_Z_OFFSET_MULT,
                                       SWIZZLED_TILE_SIZE);
            return GPU_GMEM_VOXELS[c][idx];

        } else {
            return tex3D<float>(GPU_TMEM_VOXELS[c], i, j, k);
        }
    }

    template<GPUMemType MEM_TYPE>
    __device__ float score_pose_nn(float3 pos, unsigned int mask)
    {
        float score = 0;

        int i = pos.x / GRID_CELL_SIZE;
        int j = pos.y / GRID_CELL_SIZE;
        int k = pos.z / GRID_CELL_SIZE;

        for (int c = 0; c < NUM_CHANNELS; ++c) {
            score += voxel_fetch<MEM_TYPE>(c, i, j, k) * (mask & 1);
            mask >>= 1;
        }

        return score;
    }

    template<GPUMemType MEM_TYPE>
    __device__ float score_pose_lerp(int atom_idx,
                                     int num_atoms,
                                     int block_size,
                                     float3 pos,
                                     unsigned int mask)
    {
        float score = 0;

        int i = floorf(pos.x / GRID_CELL_SIZE - 0.5f);
        int j = floorf(pos.y / GRID_CELL_SIZE - 0.5f);
        int k = floorf(pos.z / GRID_CELL_SIZE - 0.5f);

        for (int atom = 0; atom < WARP_SIZE; atom += WARP_SIZE / 8) {

            int lane_idx = threadIdx.x % WARP_SIZE;
            int src_lane = atom + lane_idx / 8;

            int ii = __shfl_sync(0xffffffff, i, src_lane);
            int jj = __shfl_sync(0xffffffff, j, src_lane);
            int kk = __shfl_sync(0xffffffff, k, src_lane);
            unsigned int m = __shfl_sync(0xffffffff, mask, src_lane);

            int ni = 0;
            int nj = 0;
            int nk = 0;

            bool active = src_lane + threadIdx.x / WARP_SIZE *
                                     WARP_SIZE < num_atoms;

            if (active) {
                int dir_idx = threadIdx.x % 8;
                unsigned int dirs = 0b111110101100011010001000;
                unsigned dir = (dirs >> (3 * dir_idx)) & 7u;

                // Clamp to border
                ni = max(0, min(ii + (dir & 1u), GRID_WIDTH - 1));
                nj = max(0, min(jj + ((dir >> 1) & 1u), GRID_HEIGHT - 1));
                nk = max(0, min(kk + ((dir >> 2) & 1u), GRID_DEPTH - 1));
            }

            float x = __shfl_sync(0xffffffff, pos.x, src_lane);
            float y = __shfl_sync(0xffffffff, pos.y, src_lane);
            float z = __shfl_sync(0xffffffff, pos.z, src_lane);

            x = x / GRID_CELL_SIZE - 0.5f - ii;
            y = y / GRID_CELL_SIZE - 0.5f - jj;
            z = z / GRID_CELL_SIZE - 0.5f - kk;

            for (int c = 0; c < NUM_CHANNELS; ++c) {
                float v1 = active ? voxel_fetch<MEM_TYPE>(c, ni, nj, nk) : 0;
                float v2 = __shfl_down_sync(0xffffffff, v1, 1, 2);

                float v12 = lerp(x, v1, v2);
                float v34 = __shfl_down_sync(0xffffffff, v12, 2, 4);

                float v1234 = lerp(y, v12, v34);
                float v5678 = __shfl_down_sync(0xffffffff, v1234, 4, 8);

                score += lerp(z, v1234, v5678) * (m & 1);
                m >>= 1;
            }
        }

        if (threadIdx.x % 8 != 0) {
            score = 0;
        }

        return score;
    }

    __device__ float score_pose_tmem_native(float3 pos,
                                            unsigned int mask)
    {
        float score = 0;

        float tx = pos.x / GRID_CELL_SIZE;
        float ty = pos.y / GRID_CELL_SIZE;
        float tz = pos.z / GRID_CELL_SIZE;

        for (int c = 0; c < NUM_CHANNELS; ++c) {
            score += tex3D<float>(GPU_TMEM_VOXELS[c], tx, ty, tz) *
                     (mask & 1);
            mask >>= 1;
        }

        return score;
    }

    template<GPUMemType MEM_TYPE>
    __global__ void score_poses(const float3 *translations,
                                const float3 *rotations,
                                float *scores,
                                int num_atoms,
                                int block_size,
                                enum InterpolateType int_type)
    {
        // Broadcasting to warp
        float3 t = translations[blockIdx.x];
        float3 r = rotations[blockIdx.x];

        float r_mat[9];
        cuDock::Docker::compute_rot_mat(r.x, r.y, r.z, r_mat);

        float score = 0.0;

        float3 pos;
        unsigned int mask;

        int idx = threadIdx.x;

        if (idx < num_atoms) {
            pos = _get_atom_pos(idx);
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

            mask = GPU_LIGAND_DATA.atoms_channel_mask[idx];
        }

        __syncthreads();

        if (MEM_TYPE != GPU_TMEM) {
            score = score_pose_lerp<MEM_TYPE>(
                idx, num_atoms, block_size, pos, mask);
        } else {

            if (idx < num_atoms) {
                score = score_pose_tmem_native(pos, mask);
            }
        }

        __syncthreads();

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

        CUDA_CHECK_ERR(cudaMemcpyToSymbol(GPU_LIGAND_DATA,
                                          &ligand_data,
                                          sizeof(struct Ligand::GPUData)));

        // Copying various data to constant memory

        CUDA_CHECK_ERR(cudaMemcpyToSymbol(NUM_CHANNELS,
                                          &Pocket::NUM_CHANNELS,
                                          sizeof(int)));

        int shape = _pocket.get_shape(0);
        CUDA_CHECK_ERR(cudaMemcpyToSymbol(GRID_WIDTH, &shape, sizeof(int)));

        shape = _pocket.get_shape(1);
        CUDA_CHECK_ERR(cudaMemcpyToSymbol(GRID_HEIGHT, &shape, sizeof(int)));

        shape = _pocket.get_shape(2);
        CUDA_CHECK_ERR(cudaMemcpyToSymbol(GRID_DEPTH, &shape, sizeof(int)));

        float cell_size = _pocket.get_cell_size();
        CUDA_CHECK_ERR(cudaMemcpyToSymbol(GRID_CELL_SIZE,
                                          &cell_size,
                                          sizeof(float)));

        int tile_size = _pocket.get_swizzled_tile_size();
        CUDA_CHECK_ERR(cudaMemcpyToSymbol(SWIZZLED_TILE_SIZE,
                                          &tile_size,
                                          sizeof(int)));

        int padded_width = _pocket.get_shape(0) +
                           cuDock::Swizzling::
                           get_padding_size(_pocket.get_shape(0),
                                            tile_size);

        int padded_height = _pocket.get_shape(1) +
                            cuDock::Swizzling::
                            get_padding_size(_pocket.get_shape(1),
                                             tile_size);

        int width_in_tiles = padded_width / tile_size;
        int height_in_tiles = padded_height / tile_size;
        int brick_size = tile_size * tile_size * tile_size;

        int x_offset_mult = brick_size;
        int y_offset_mult = brick_size * width_in_tiles;
        int z_offset_mult = brick_size * width_in_tiles * height_in_tiles;

        CUDA_CHECK_ERR(cudaMemcpyToSymbol(SWIZZLED_X_OFFSET_MULT,
                                          &x_offset_mult,
                                          sizeof(int)));

        CUDA_CHECK_ERR(cudaMemcpyToSymbol(SWIZZLED_Y_OFFSET_MULT,
                                          &y_offset_mult,
                                          sizeof(int)));

        CUDA_CHECK_ERR(cudaMemcpyToSymbol(SWIZZLED_Z_OFFSET_MULT,
                                          &z_offset_mult,
                                          sizeof(int)));
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
                GPU_GMEM_VOXELS,
                _pocket.get_gpu_gmem_voxels().data(),
                sizeof(float *) * Pocket::NUM_CHANNELS));
        } else {
            // Copy pocket voxels tmem pointers to constant memory
            CUDA_CHECK_ERR(cudaMemcpyToSymbol(
                GPU_TMEM_VOXELS,
                _pocket.get_gpu_tmem_voxels().data(),
                sizeof(cudaTextureObject_t) * Pocket::NUM_CHANNELS));
        }

        int num_atoms = _ligand.get_num_atoms();
        int block_size = std::ceil((float) num_atoms / WARP_SIZE) * WARP_SIZE;

        if (_pocket.is_on_gpu(GPU_GMEM)) {
            CUDA_TIME_EXEC("_score_gmem", [&](){
                score_poses<GPU_GMEM><<<
                    num_poses,
                    block_size>>>(_gpu_translations,
                                  _gpu_rotations,
                                  _gpu_scores,
                                  num_atoms,
                                  block_size,
                                  _pocket.get_interpolate());
            });
        } else if (_pocket.is_on_gpu(GPU_GMEM_SWIZZLED)) {
            CUDA_TIME_EXEC("_score_gmem_swizzled", [&](){
                score_poses<GPU_GMEM_SWIZZLED><<<
                    num_poses,
                    block_size>>>(_gpu_translations,
                                  _gpu_rotations,
                                  _gpu_scores,
                                  num_atoms,
                                  block_size,
                                  _pocket.get_interpolate());
            });
        } else {
            CUDA_TIME_EXEC("_score_tmem", [&](){
                score_poses<GPU_TMEM><<<
                    num_poses,
                    block_size>>>(_gpu_translations,
                                  _gpu_rotations,
                                  _gpu_scores,
                                  num_atoms,
                                  block_size,
                                  _pocket.get_interpolate());
            });

        }
    }
}
