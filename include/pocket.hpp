#pragma once

#include <ostream>
#include <array>
#include <vector>
#include <array>
#include <iostream>

namespace cuDock
{
    class Pocket
    {
    public:
        constexpr static int NUM_CHANNELS = 8;

        struct Point
        {
            float pos[3]; // x, y, z
            float channels[NUM_CHANNELS];
        };

        Pocket(const std::string &csv_file_path, float cell_size);
        Pocket(const std::vector<Point> &pocket_points, float cell_size);

        unsigned int size() const;

        unsigned int shape(int axis) const;

        void domain(int axis, float &min, float &max) const;

        float voxel(unsigned int c,
                    unsigned int i,
                    unsigned int j,
                    unsigned int k) const;

        const float *voxels(unsigned int c,
                            unsigned int i = 0) const;

        void lookup(const float pos[3],
                    std::array<float, NUM_CHANNELS> &values) const;

        ~Pocket();
    private:
        // Converts a 3d subscript into a one-dimensional index
        unsigned int _sub_to_idx(unsigned int i = 0,
                                 unsigned int j = 0,
                                 unsigned int k = 0) const;
        // Converts a (x, y, z) position into a 3d subscript
        void _pos_to_sub(const float pos[3],
                         unsigned int sub[3]) const;

        void _voxelize(const std::vector<Point> &points);

        std::array<float *, NUM_CHANNELS> _voxels;

        float _cell_size;
        float _domain[6];
        unsigned int _shape[3];
    };

    std::ostream &operator<<(std::ostream &os, const Pocket::Point &point);
}
