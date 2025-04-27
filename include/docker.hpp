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

        void run_random_poses(int num_poses);

        const std::vector<float> &get_scores();

    private:
        static void _rotate(vec3 pos, const vec3 &angles, vec3 &dst);
        static void _translate(vec3 pos, const vec3 &delta, vec3 &dst);

        void _reinit_buffers(int num_poses);
        void _score_poses(int num_poses);

        const Pocket &_pocket;
        const Ligand &_ligand;

        std::vector<float> _scores;
        std::vector<vec3> _translations;
        std::vector<vec3> _rotations;

        std::default_random_engine _rng;
        std::uniform_real_distribution<float> _dist;
    };
}
