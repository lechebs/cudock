#include "docker.hpp"

#include <vector>
#include <random>
#include <cmath>

#include "vec3.hpp"
#include "pocket.hpp"
#include "ligand.hpp"

namespace cuDock
{
    Docker::Docker(const Pocket &pocket, const Ligand &ligand) :
        _pocket(pocket),
        _ligand(ligand),
        _rng(42)
    {}

    void Docker::generate_random_poses(int num_poses)
    {
        _reinit_buffers(num_poses);

        for (int i = 0; i < num_poses; ++i) {
            vec3 delta, angles;
            for (int j = 0; j < 3; ++j) {
                float domain_size = _pocket.get_domain_size(j);
                // Random translation constrained within the pocket domain
                delta[j] = _ligand.get_radius() +
                    _dist(_rng) * (domain_size - 2 * _ligand.get_radius());
                // Random rotation around current axis
                angles[j] = _dist(_rng) * 2 * (float) M_PI;
            }
            _translations.push_back(delta);
            _rotations.push_back(angles);
        }
    }

    void Docker::run()
    {
        int num_poses = _translations.size();
        if (num_poses > 0) {
            if (_pocket.is_on_gpu() && _is_on_gpu) {
                _score_poses_gpu(num_poses);
            } else {
                _score_poses(num_poses);
            }
        }
    }

    void Docker::_reinit_buffers(int num_poses)
    {
        _scores.clear();
        _translations.clear();
        _rotations.clear();

        _scores.reserve(num_poses);
        _translations.reserve(num_poses);
        _rotations.reserve(num_poses);
    }

    void Docker::_score_poses(int num_poses)
    {
        for (int i = 0; i < num_poses; ++i) {

            float score = 0.0;

            float r_mat[9];
            compute_rot_mat(_rotations[i][0],
                            _rotations[i][1],
                            _rotations[i][2],
                            r_mat);

            for (const Ligand::Atom &atom : _ligand.get_atoms()) {

                vec3 pos;
                transform_atom_pos(atom.pos[0],
                                   atom.pos[1],
                                   atom.pos[2],
                                   _translations[i][0],
                                   _translations[i][1],
                                   _translations[i][2],
                                   r_mat,
                                   pos[0],
                                   pos[1],
                                   pos[2]);

                std::array<float, Pocket::NUM_CHANNELS> values;
                _pocket.lookup(pos, values);

                unsigned int mask = Ligand::get_atom_channel_mask(atom.type);
                // Accumulating dot product between the looked up values
                // and the maske
                for (int c = 0; c < Pocket::NUM_CHANNELS; ++c, mask >>= 1)
                {
                    score += values[c] * (mask & 1);
                }
            }
            _scores.push_back(score);
        }
    }

    Docker::~Docker()
    {
        off_gpu();
    }
}
