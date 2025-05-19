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
                delta[j] =
                    _ligand.get_radius() +
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

    const std::vector<float> &Docker::get_scores()
    {
        return _scores;
    }

    void Docker::_rotate(vec3 pos, const vec3 &angles, vec3 &dst)
    {
        float sina = std::sin(angles[0]);
        float cosa = std::cos(angles[0]);
        float sinb = std::sin(angles[1]);
        float cosb = std::cos(angles[1]);
        float sinc = std::sin(angles[2]);
        float cosc = std::cos(angles[2]);

        float x = pos[0];
        float y = pos[1];
        float z = pos[2];

        dst[0] = cosb * cosc * x +
                 (sina * sinb * cosc - cosa * sinc) * y +
                 (cosa * sinb * cosc + sina * sinc) * z;

        dst[1] = cosb * sinc * x +
                 (sina * sinb * sinc + cosa * sinc) * y +
                 (cosa * sinb * sinc - sina * cosc) * z;

        dst[2] = -sinb * x + sina * cosb * y + cosa * cosb * z;
    }

    void Docker::_translate(vec3 pos, const vec3 &delta, vec3 &dst)
    {
        for (int i = 0; i < 3; ++i) {
            dst[i] = pos[i] + delta[i];
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
            for (const Ligand::Atom &atom : _ligand.get_atoms()) {

                vec3 pos;
                _rotate(atom.pos, _rotations[i], pos);
                _translate(pos, _translations[i], pos);

                std::array<float, Pocket::NUM_CHANNELS> values;
                _pocket.lookup(pos, values);

                unsigned int mask = Ligand::get_atom_channel_mask(atom.type);
                // Accumulating dot product between the looked up values
                // and the mask
                for (int c = 0, b = 1; c < Pocket::NUM_CHANNELS; ++c, b <<= 1)
                {
                    score += values[c] * ((mask & b) > 0);
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
