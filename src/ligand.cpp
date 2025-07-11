#include "ligand.hpp"

#include <string>
#include <vector>
#include <cmath>

#include "parsing.hpp"

namespace cuDock
{
    const std::map<std::string, unsigned int>
    Ligand::_atom_type_map = {
        { "N.4",   0 },
        { "C.3",   1 },
        { "C.2",   2 },
        { "O.co2", 3 },
        { "O.2",   4 },
        { "N.am",  5 },
        { "S.3",   6 },
        { "C.ar",  7 },
        { "H",     8 }
    };

    /*
     * Channels:
     * 1. Simmetric
     * 2. Gravitational
     * 3. Electrostatic
     * 4. Lipophilicity
     * 5. Hidrophilicity
     * 6. Polar
     * 7. HB Acceptor
     * 8. HB Donor
     */

    const std::vector<unsigned int>
    Ligand::_channel_masks = {
        0b01100101,
        0b10101000,
        0b00101101,
        0b11111100,
        0b01111101,
        0b01100101,
        0b10011011,
        0b01101101,
        0b10010010
    };

    const std::vector<float>
    Ligand::_atom_mass = {
        14.01,
        12.01,
        12.01,
        16.00,
        16.00,
        14.01,
        32.07,
        12.01,
        1.008
    };

    unsigned int Ligand::get_atom_type_by_name(const std::string &name)
    {
        return _atom_type_map.at(name);
    }

    unsigned int Ligand::get_atom_channel_mask(unsigned int atom_type)
    {
        return _channel_masks[atom_type];
    }

    float Ligand::get_atom_mass(unsigned int atom_type)
    {
        return _atom_mass[atom_type];
    }

    Ligand::Ligand(const std::string &mol2_file_path)
    {
        Parsing::read_ligand_mol2(mol2_file_path, _atoms);
        _translate_com_to_origin();
    }

    Ligand::Ligand(std::vector<Ligand::Atom> atoms) : _atoms(atoms)
    {
        _translate_com_to_origin();
    }

    const std::vector<Ligand::Atom> &Ligand::get_atoms() const
    {
        return _atoms;
    }

    int Ligand::get_num_atoms() const
    {
        return _atoms.size();
    }

    float Ligand::get_radius() const
    {
        return _radius;
    }

    void Ligand::_translate_com_to_origin()
    {
        float tot_mass = 0;
        float com[3] = { 0, 0, 0 };

        for (Ligand::Atom &atom : _atoms) {
            float mass = Ligand::get_atom_mass(atom.type);
            for (int i = 0; i < 3; ++i) {
                com[i] += atom.pos[i] * mass;
            }
            tot_mass += mass;
        }

        for (int i = 0; i < 3; ++i) {
            com[i] /= tot_mass;
        }

        float radius = 0;
        for (Ligand::Atom &atom : _atoms) {
            float r = 0;
            for (int i = 0; i < 3; ++i) {
                // Translating
                atom.pos[i] -= com[i];
                // Computing distance to com
                r += atom.pos[i] * atom.pos[i];
            }

            if (r > radius) {
                radius =  r;
            }
        }

        _radius = std::sqrt(radius);
    }

    std::ostream &operator<<(std::ostream &os, const Ligand::Atom &atom)
    {
        os << "x=" << atom.pos[0] << ", "
           << "y=" << atom.pos[1] << ", "
           << "z=" << atom.pos[2] << ", "
           << "type=" << atom.type;

        return os;
    }
}
