#pragma once

#include "pocket.hpp"
#include "ligand.hpp"

namespace cuDock
{
    namespace Scoring
    {
        float evaluate_score(const Pocket &pocket, const Ligand &ligand);
    }
}
