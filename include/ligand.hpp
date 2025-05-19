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
        static const unsigned int MAX_NUM_ATOMS = 64;

        struct GPUData
        {
            float atoms_x[MAX_NUM_ATOMS];
            float atoms_y[MAX_NUM_ATOMS];
            float atoms_z[MAX_NUM_ATOMS];
            float atoms_mass[MAX_NUM_ATOMS];
            // unsigned int atom_type[MAX_NUM_ATOMS];
            unsigned int atoms_channel_mask[MAX_NUM_ATOMS];
        };

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
        int get_num_atoms() const;
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

        bool _is_on_gpu;
    };

    std::ostream &operator<<(std::ostream &os, const Ligand::Atom &atom);
}

