#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

#include "parser.hpp"
#include "pocket.hpp"

using namespace cuDock;

int main([[maybe_unused]] int argc, [[maybe_unused]] const char *argv[]) {

    const std::string csv_file_path = "../data/10gs_protein_pocket1.csv";

    std::vector<PocketPoint> points = Parser::readPocketCSV(csv_file_path);

    Pocket pocket(points, 2.0);

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

    return EXIT_SUCCESS;
}
