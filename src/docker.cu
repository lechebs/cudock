#include "docker.hpp"

#include <iostream>
#include <vector>

#include "ligand.hpp"
#include "utils.cuh"

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

    __device__ void _compute_rot_mat(float3 r, float mat[])
    {
        float sinrx = __sinf(r.x);
        float cosrx = __cosf(r.x);
        float sinry = __sinf(r.y);
        float cosry = __cosf(r.y);
        float sinrz = __sinf(r.z);
        float cosrz = __cosf(r.z);

        mat[0] = cosry * cosrz;
        mat[1] = sinrx * sinry * cosrz - cosrx * sinrz;
        mat[2] = cosrx * sinry * cosrz + sinrx * sinrz;

        mat[3] = cosry * sinrz;
        mat[4] = sinrx * sinry * sinrz + cosrx * sinrz;
        mat[5] = cosrx * sinry * sinrz - sinrx * cosrz;

        mat[6] = -sinry;
        mat[7] = sinrx * cosry;
        mat[8] = cosrx * cosry;
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

    __device__ __inline__ void
    _transform_atom_pos(float x,
                        float y,
                        float z,
                        float3 t,
                        const float r_mat[],
                        float3 &dst)
    {
        dst.x = r_mat[0] * x + r_mat[1] * y + r_mat[2] * z + t.x;
        dst.y = r_mat[3] * x + r_mat[4] * y + r_mat[5] * z + t.y;
        dst.z = r_mat[6] * x + r_mat[7] * y + r_mat[8] * z + t.z;
    }

    __global__ void _score_gmem(float grid_cell_size,
                                int grid_width,
                                int grid_height,
                                int num_channels,
                                const float3 *translations,
                                const float3 *rotations,
                                int num_atoms,
                                int block_size)
    {
        // Broadcasting to warp
        float3 t = translations[blockIdx.x];
        float3 r = rotations[blockIdx.x];

        float r_mat[9];
        _compute_rot_mat(r, r_mat);

        for (int idx = threadIdx.x; idx < num_atoms; idx += block_size) {
            float3 pos = _get_atom_pos(idx);
            _transform_atom_pos(pos.x, pos.y, pos.z, t, r_mat, pos);

            // Computing lookup coordinates
            int i = pos.x / grid_cell_size;
            int j = pos.y / grid_cell_size;
            int k = pos.z / grid_cell_size;

            int grid_idx = k * grid_width * grid_height + j * grid_width + i;

            float val = 0.0;
            for (int c = 0; c < num_channels; ++c) {
                val += _gpu_gmem_voxels[c][grid_idx];
            }

            if (threadIdx.x == 0) {
                printf("[%02d:%02d] x=%.2f y=%.2f z=%.2f idx=%d val=%.2f\n",
                       blockIdx.x,
                       threadIdx.x,
                       pos.x,
                       pos.y,
                       pos.z,
                       grid_idx,
                       val);
            }
        }
    }

    __global__ void _score_tmem(float3 domain_size,
                                int num_channels,
                                const float3 *translations,
                                const float3 *rotations,
                                int num_atoms,
                                int block_size)
    {
        float3 t = translations[blockIdx.x];
        float3 r = rotations[blockIdx.x];

        float r_mat[9];
        _compute_rot_mat(r, r_mat);

        for (int idx = threadIdx.x; idx < num_atoms; idx += block_size) {
            float3 pos = _get_atom_pos(idx);
            _transform_atom_pos(pos.x, pos.y, pos.z, t, r_mat, pos);

            // Texture coordinates are normalized
            float tx = pos.x / domain_size.x;
            float ty = pos.y / domain_size.y;
            float tz = pos.z / domain_size.z;

            float val = 0.0;
            for (int c = 0; c < num_channels; ++c) {
                val += tex3D<float>(_gpu_tmem_voxels[c], tx, ty, tz);
            }

            if (threadIdx.x == 0) {
                printf("[%02d:%02d] x=%.2f y=%.2f z=%.2f val=%.2f\n",
                       blockIdx.x,
                       threadIdx.x,
                       pos.x,
                       pos.y,
                       pos.z,
                       val);
            }

        }
    }
}

namespace cuDock
{
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

        // Copy pocket voxels gmem pointers to constant memory

        CUDA_CHECK_ERR(cudaMemcpyToSymbol(
            _gpu_gmem_voxels,
            _pocket.get_gpu_gmem_voxels().data(),
            sizeof(float *) * Pocket::NUM_CHANNELS));

        // Copy pocket voxels tmem pointers to constant memory

        CUDA_CHECK_ERR(cudaMemcpyToSymbol(
            _gpu_tmem_voxels,
            _pocket.get_gpu_tmem_voxels().data(),
            sizeof(cudaTextureObject_t) * Pocket::NUM_CHANNELS));

    }

    void Docker::off_gpu()
    {
        if (_is_on_gpu) {
            _is_on_gpu = false;
            CUDA_CHECK_ERR(cudaFree(_gpu_translations));
            CUDA_CHECK_ERR(cudaFree(_gpu_rotations));
        }
    }

    void Docker::_score_poses_gpu(int num_poses)
    {
        int num_atoms = _ligand.get_num_atoms();
        int block_size = num_atoms / WARP_SIZE * WARP_SIZE;

        if (_pocket.is_on_gpu(GPU_GMEM)) {

            CUDA_TIME_EXEC("_score_gmem", [&](){
                _score_gmem<<<
                    num_poses,
                    block_size>>>(_pocket.get_cell_size(),
                                  _pocket.get_shape(0),
                                  _pocket.get_shape(1),
                                  Pocket::NUM_CHANNELS,
                                  _gpu_translations,
                                  _gpu_rotations,
                                  num_atoms,
                                  block_size);
            });
        } else {
            CUDA_TIME_EXEC("_score_tmem", [&](){
                _score_tmem<<<
                    num_poses,
                    block_size>>>({ _pocket.get_domain_size(0),
                                    _pocket.get_domain_size(1),
                                    _pocket.get_domain_size(2) },
                                  Pocket::NUM_CHANNELS,
                                  _gpu_translations,
                                  _gpu_rotations,
                                  num_atoms,
                                  block_size);
            });
        }
    }
}
