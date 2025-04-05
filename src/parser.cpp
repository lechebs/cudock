#include "parser.hpp"

#include <fstream>
#include <vector>
#include <limits>
#include <cstring>

#include <iostream>

#include "pocket.hpp"

namespace cuDock
{
    namespace Parser
    {
        const static int MAX_LINE_SIZE = 256;

        std::vector<PocketPoint> readPocketCSV(const std::string &file_path)
        {
            std::vector<PocketPoint> points;
            std::ifstream input(file_path);

            char buff[MAX_LINE_SIZE];

            // Read csv header
            input.getline(buff, MAX_LINE_SIZE);

            while (!input.getline(buff, MAX_LINE_SIZE).eof()) {
                char *ptr = strtok(buff, " ,");

                // Ignore n, pocket_n, pocket_score and pocket_overlap fields
                for (int i = 0; i < 4; ++i) {
                    ptr = strtok(nullptr, " ,");
                }

                PocketPoint point;
                // Read x, y, z
                for (int i = 0; i < 3; ++i) {
                    point.pos[i] = std::strtof(ptr, nullptr);
                    ptr = strtok(nullptr, " ,");
                }
                // Read psi channels
                for (int i = 0; i < 9; ++i) {
                    point.channels[i] = std::strtof(ptr, nullptr);
                    ptr = strtok(nullptr, " ,");
                }

                points.push_back(point);
            }

            return points;
        }
    }
}
