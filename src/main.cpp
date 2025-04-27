#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

#include "parsing.hpp"
#include "pocket.hpp"
#include "docker.hpp"

using namespace cuDock;

int main([[maybe_unused]] int argc, [[maybe_unused]] const char *argv[]) {

    const std::string pocket_file_path = "../data/10gs_protein_pocket1.csv";
    const std::string ligand_file_path = "../data/10gs_ligand1.mol2";

    std::vector<Pocket::Point> points;
    Parsing::read_pocket_csv(pocket_file_path, points);

    std::vector<Ligand::Atom> atoms;
    Parsing::read_ligand_mol2(ligand_file_path, atoms);

    Pocket pocket(points, 2.0);
    Ligand ligand(atoms);

    Docker docker(pocket, ligand);
    docker.run_random_poses(10);

    for (float score : docker.get_scores()) {
        std::cout << score << std::endl;
    }

    // std::cout << Scoring::evaluate_score(pocket, ligand) << std::endl;

    /*
    int d = pocket.shape(1);
    int h = pocket.shape(2);
    int w = pocket.shape(3);

    std::cout << std::setprecision(3) << std::fixed;

    std::cout << std::endl;
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < h; ++j) {
            for (int k = 0; k < w; ++k) {
                std::cout << pocket(2, i, j, k) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    */

    return EXIT_SUCCESS;
}
