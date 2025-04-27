#pragma once

#include <string>
#include <vector>

#include "ligand.hpp"
#include "pocket.hpp"

namespace cuDock
{
    namespace Parsing
    {
        void read_pocket_csv(const std::string &file_path,
                             std::vector<Pocket::Point> &points);

        void read_ligand_mol2(const std::string &file_path,
                              std::vector<Ligand::Atom> &atoms);
    }
}
