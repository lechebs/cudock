#include "scoring.hpp"

#include <array>
#include <iostream>

#include "pocket.hpp"
#include "ligand.hpp"

namespace cuDock
{
    namespace Scoring
    {
        float evaluate_score(const Pocket &pocket, const Ligand &ligand)
        {
            float score = 0.0;

            for (const Ligand::Atom &atom : ligand.get_atoms()) {

                std::array<float, Pocket::NUM_CHANNELS> values;
                pocket.lookup(atom.pos, values);

                unsigned int mask = Ligand::get_atom_channel_mask(atom.type);
                // Accumulating dot product between the looked up values and the mask
                for (int c = 0, b = 1; c < Pocket::NUM_CHANNELS; ++c, b <<= 1) {
                    score += values[c] * ((mask & b) > 0);
                }
            }

            return score;
        }
    }
}
