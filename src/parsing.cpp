#include "parsing.hpp"

#include <fstream>
#include <vector>
#include <limits>
#include <cstring>

#include <iostream>

#include "pocket.hpp"

namespace cuDock
{
    namespace Parsing
    {
        const static int MAX_LINE_SIZE = 256;

        void read_pocket_csv(const std::string &file_path,
                             std::vector<Pocket::Point> &points)
        {
            std::ifstream input(file_path);

            char buff[MAX_LINE_SIZE];

            // Read csv header
            input.ignore(MAX_LINE_SIZE, '\n');

            while (!input.getline(buff, MAX_LINE_SIZE).eof()) {
                char *ptr = std::strtok(buff, " ,");

                // Ignore n, pocket_n, pocket_score and pocket_overlap fields
                for (int i = 0; i < 4; ++i) {
                    ptr = std::strtok(nullptr, " ,");
                }

                Pocket::Point point;
                // Read x, y, z
                for (int i = 0; i < 3; ++i) {
                    point.pos[i] = std::strtof(ptr, nullptr);
                    ptr = std::strtok(nullptr, " ,");
                }
                // Ignore psi field
                ptr = std::strtok(nullptr, " ,");
                // Read psi1-8 fields
                for (int i = 0; i < Pocket::NUM_CHANNELS; ++i) {
                    point.channels[i] = std::strtof(ptr, nullptr);
                    ptr = std::strtok(nullptr, " ,");
                }

                points.push_back(point);
            }

            points.shrink_to_fit();
        }

        void read_ligand_mol2(const std::string &file_path,
                              std::vector<Ligand::Atom> &atoms)
        {
            std::ifstream input(file_path);

            char buff[MAX_LINE_SIZE];

            // Read until @<TRIPOS>MOLECULE
            input.ignore(MAX_LINE_SIZE, '@');
            input.ignore(MAX_LINE_SIZE, '\n');
            // Ignore mol_name
            input.ignore(MAX_LINE_SIZE, '\n');

            int num_atoms;
            input >> num_atoms;
            // Read until @<TRIPOS>ATOM
            input.ignore(MAX_LINE_SIZE, '@');
            input.ignore(MAX_LINE_SIZE, '\n');

            for (int i = 0; i < num_atoms; ++i) {
                input.getline(buff, MAX_LINE_SIZE);

                char *ptr = std::strtok(buff, " ");
                // Ignore atom_id and atom_name
                for (int j = 0; j < 2; ++j) {
                    ptr = std::strtok(nullptr, " ");
                }

                Ligand::Atom atom;
                // Read x, y, z
                for (int j = 0; j < 3; ++j) {
                    atom.pos[j] = std::strtof(ptr, nullptr);
                    ptr = std::strtok(nullptr, " ");
                }
                // Read atom_type
                atom.type = Ligand::get_atom_type_by_name(ptr);

                atoms.push_back(atom);
            }

            atoms.shrink_to_fit();
        }
    }
}
