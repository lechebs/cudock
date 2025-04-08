#pragma once

#include <ostream>
#include <string>
#include <vector>
#include <map>

namespace cuDock
{
    class Ligand
    {
    public:
        struct Atom
        {
            unsigned int type;
            float pos[3];
        };

        static unsigned int get_atom_type_by_name(const std::string &name);

        Ligand(const std::string &mol2_file_path);
        Ligand(std::vector<Atom> atoms);

        const std::vector<Atom> &get_atoms();

    private:
        const static std::map<std::string, unsigned int> _atom_type_map;

        std::vector<Atom> _atoms;
    };

    std::ostream &operator<<(std::ostream &os, const Ligand::Atom &atom);
}

