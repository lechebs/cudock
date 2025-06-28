#pragma once

#include <array>
#include <vector>
#include <random>

#include "vec3.hpp"
#include "pocket.hpp"
#include "ligand.hpp"

namespace cuDock
{
    class Docker
    {
    public:
        Docker(const Pocket &pocket, const Ligand &ligand);

        void generate_random_poses(int num_poses);

        void to_gpu();
        void off_gpu();

        void run();

        void get_scores(std::vector<float> &dst) const;

        ~Docker();

        static __device__ __host__
        void compute_rot_mat(float rx,
                             float ry,
                             float rz,
                             float mat[]);

        static __device__ __host__
        void transform_atom_pos(float x,
                                float y,
                                float z,
                                float tx,
                                float ty,
                                float tz,
                                const float mat[],
                                float &x_dst,
                                float &y_dst,
                                float &z_dst);

    private:
        void _reinit_buffers(int num_poses);
        void _score_poses(int num_poses);

        void _score_poses_gpu(int num_poses);

        const Pocket &_pocket;
        const Ligand &_ligand;

        std::vector<float> _scores;
        std::vector<vec3> _translations;
        std::vector<vec3> _rotations;

        std::default_random_engine _rng;
        std::uniform_real_distribution<float> _dist;

        bool _is_on_gpu = 0;

        float3 *_gpu_translations;
        float3 *_gpu_rotations;
        float *_gpu_scores;
    };
}
