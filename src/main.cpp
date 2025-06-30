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

    Pocket pocket(points, 0.2);
    Ligand ligand(atoms);

    const int num_poses = 1 << 20;

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

    std::vector<float> cpu_scores;
    std::vector<float> gpu_gmem_scores;
    std::vector<float> gpu_tmem_scores;

    /*
    docker.run();
    docker.get_scores(cpu_scores);
    docker.to_gpu();
    */

    docker.to_gpu();

    pocket.to_gpu(GPU_TMEM);
    docker.run();

    /*
    pocket.off_gpu(GPU_GMEM);
    pocket.to_gpu(GPU_GMEM_SWIZZLED);
    docker.run();
    docker.get_scores(gpu_gmem_scores);

    pocket.off_gpu(GPU_GMEM_SWIZZLED);
    pocket.to_gpu(GPU_TMEM);
    docker.run();
    docker.get_scores(gpu_tmem_scores);
    */

    /*
    std::cout << std::fixed;

    // Validate results
    for (size_t i = 0; i < cpu_scores.size(); ++i) {
        float cpu_score = cpu_scores[i];
        float gpu_gmem_score = gpu_gmem_scores[i];
        float gpu_tmem_score = gpu_tmem_scores[i];

        if (std::abs(cpu_score - gpu_gmem_score) > 1e-4f) {
            std::cout << "[GMEM] WARNING: incorrect score at pose "
                      << i << " ("
                      << "CPU: " << cpu_score << ", GPU: "
                      << gpu_gmem_score << ")" << std::endl;
        }

        if (std::abs(cpu_score - gpu_tmem_score) > 1e-4f) {
            std::cout << "[TMEM] WARNING: incorrect score at pose "
                      << i << " ("
                      << "CPU: " << cpu_score << ", GPU: "
                      << gpu_tmem_score << ")" << std::endl;
        }
    }
    */

    return EXIT_SUCCESS;
}
