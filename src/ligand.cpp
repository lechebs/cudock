#include "ligand.hpp"

#include <string>
#include <vector>

#include "parser.hpp"

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

    unsigned int Ligand::get_atom_type_by_name(const std::string &name)
    {
        return _atom_type_map.at(name);
    }

    Ligand::Ligand(const std::string &mol2_file_path)
    {
        Parser::read_ligand_mol2(mol2_file_path, _atoms);
    }

    Ligand::Ligand(std::vector<Ligand::Atom> atoms) : _atoms(atoms) {}

    const std::vector<Ligand::Atom> &Ligand::get_atoms()
    {
        return _atoms;
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
