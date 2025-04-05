#pragma once

#include <string>
#include <vector>

#include "pocket.hpp"

namespace cuDock
{
    namespace Parser
    {
        std::vector<PocketPoint>
        readPocketCSV(const std::string &file_path);
    }
}
