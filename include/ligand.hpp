#pragma once

#include <ostream>
#include <string>
#include <vector>
#include <map>

#include "pocket.hpp"

namespace cuDock
{
    class Ligand
    {
    public:
        struct Atom
        {
            unsigned int type;
            vec3 pos;
        };

        static unsigned int get_atom_type_by_name(const std::string &name);
        static unsigned int get_atom_channel_mask(unsigned int atom_type);
        static float get_atom_mass(unsigned int atom_type);

        Ligand(const std::string &mol2_file_path);
        Ligand(std::vector<Atom> atoms);

        const std::vector<Atom> &get_atoms() const;
        float get_radius() const;

    private:
        // Translates the atoms so that the ligand
        // center of mass ends up in (0, 0, 0)
        void _translate_com_to_origin();

        const static std::map<std::string, unsigned int> _atom_type_map;
        const static std::vector<unsigned int> _channel_masks;
        const static std::vector<float> _atom_mass;

        std::vector<Atom> _atoms;
        float _radius;
    };

    std::ostream &operator<<(std::ostream &os, const Ligand::Atom &atom);
}

