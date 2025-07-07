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

    Pocket pocket(points, 1.0); // 0.25
    Ligand ligand(atoms);

    const int num_poses = 1 << 5; // 20

    std::cout << "num_poses=" << num_poses << std::endl;
    std::cout << "domain=(" << pocket.get_domain_size(0) 
              << ", " << pocket.get_domain_size(1)
              << ", " << pocket.get_domain_size(2)
              << ")" << std::endl;
    std::cout << "shape=(" << pocket.get_shape(0) 
              << ", " << pocket.get_shape(1)
              << ", " << pocket.get_shape(2)
              << ")" << std::endl;

    Docker docker(pocket, ligand);
    docker.generate_random_poses(num_poses);

    std::vector<float> ref_scores;
    std::vector<float> res_scores;

    pocket.set_swizzled_tile_size(32); // 32
    pocket.use_compressible_memory(false); // true
    pocket.set_interpolate(LIN_INTERPOLATE);
    pocket.set_packed(true); // false

    docker.to_gpu();

    pocket.to_gpu(GPU_GMEM);
    docker.run();
    pocket.off_gpu(GPU_GMEM);

    pocket.to_gpu(GPU_GMEM_SWIZZLED);
    docker.run();
    docker.get_scores(res_scores);
    pocket.off_gpu(GPU_GMEM_SWIZZLED);

    pocket.to_gpu(GPU_TMEM);
    docker.run();
    docker.get_scores(ref_scores);

    std::cout << std::fixed;

    // Validate results
    for (size_t i = 0; i < ref_scores.size(); ++i) {
        float ref_score = ref_scores[i];
        float res_score = res_scores[i];

        if (std::abs(ref_score - res_score) > 1e-1f) {
            std::cout << "WARNING: incorrect score at pose "
                      << i << " ("
                      << "ref: " << ref_score << ", res: "
                      << res_score << ")" << std::endl;
        }
    }

    return EXIT_SUCCESS;
}
