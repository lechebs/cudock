#pragma once

#include <array>
#include <vector>
#include <iostream>

namespace cuDock
{
    constexpr int NUM_POCKET_CHANNELS = 9;

    struct PocketPoint
    {
        float pos[3]; // x, y, z
        float channels[NUM_POCKET_CHANNELS];
    };

    std::ostream &operator<<(std::ostream &os, const PocketPoint &point);

    class Pocket
    {
    public:
        Pocket(const std::string &csv_file_path, float cell_size);
        Pocket(const std::vector<PocketPoint> &pocket_points, float cell_size);

        unsigned int size() const;

        unsigned int shape(int axis) const;

        float operator()(unsigned int c,
                         unsigned int i,
                         unsigned int j,
                         unsigned int k) const;

        const float *operator()(unsigned int c,
                                unsigned int i = 0) const;

        ~Pocket();
    private:
        // Converts a 3d subscript into a one-dimensional index
        unsigned int _sub_to_idx(unsigned int i = 0,
                                 unsigned int j = 0,
                                 unsigned int k = 0) const;

        void _voxelize(const std::vector<PocketPoint> &points,
                       float cell_size);

        std::array<float *, NUM_POCKET_CHANNELS> _voxels;

        unsigned int _shape[3];
    };
}
