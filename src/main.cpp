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

    Pocket pocket(points, 1.0);
    Ligand ligand(atoms);

    const int num_poses = 2 << 17;

    std::cout << "num_poses=" << num_poses << std::endl;

    Docker docker(pocket, ligand);
    docker.generate_random_poses(num_poses);

    pocket.to_gpu(GPU_GMEM);
    docker.to_gpu();
    docker.run();

    pocket.off_gpu(GPU_GMEM);
    pocket.to_gpu(GPU_TMEM);
    docker.run();

    return EXIT_SUCCESS;
}
