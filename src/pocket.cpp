#include "pocket.hpp"

#include <ostream>
#include <string>
#include <vector>
#include <limits>
#include <cmath>
#include <cstring>
#include <iomanip>

#include "parser.hpp"

namespace cuDock
{
    std::ostream &operator<<(std::ostream &os, const PocketPoint &point)
    {
        os << "x=" << point.pos[0] << ", y="
                   << point.pos[1] << ", z="
                   << point.pos[2];

        for (int c = 0; c < NUM_POCKET_CHANNELS; ++c) {
            os << ", psi" << c << "=" << point.channels[c];
        }

        return os;
    }

    Pocket::Pocket(const std::string &csv_file_path, float cell_size)
    {
        std::vector<PocketPoint> points =
            Parser::readPocketCSV(csv_file_path);

        _voxelize(points, cell_size);
    }

    Pocket::Pocket(const std::vector<PocketPoint> &pocket_points,
                   float cell_size)
    {
        _voxelize(pocket_points, cell_size);
    }

    unsigned int Pocket::size() const
    {
        unsigned int size = NUM_POCKET_CHANNELS;

        for (int i = 0; i < 3; ++i) {
            size *= _shape[i];
        }

        return size;
    }

    unsigned int Pocket::shape(int axis) const
    {
        if (axis == 0) {
            return NUM_POCKET_CHANNELS;
        } else {
            return _shape[axis - 1];
        }
    }

    float Pocket::operator()(unsigned int c,
                             unsigned int i,
                             unsigned int j,
                             unsigned int k) const
    {
        return _voxels[c][_sub_to_idx(i, j, k)];
    }

    const float *Pocket::operator()(unsigned int c,
                                    unsigned int i) const
    {
        return &_voxels[c][_sub_to_idx(i)];
    }

    Pocket::~Pocket()
    {
        return;
        for (int c = 0; c < NUM_POCKET_CHANNELS; ++c) {
            delete[] _voxels[c];
        }
    }

    unsigned int Pocket::_sub_to_idx(unsigned int i,
                                     unsigned int j,
                                     unsigned int k) const
    {
        return i * _shape[1] * _shape[2] +
               j * _shape[2] + k;
    }

    void Pocket::_voxelize(const std::vector<PocketPoint> &points,
                           float cell_size)
    {
        float inf = std::numeric_limits<float>::infinity();
        float min_pos[3] = { inf, inf, inf };
        float max_pos[3] = { -inf, -inf, -inf };

        // Computing bounding box
        for (const PocketPoint &p : points) {
            for (int i = 0; i < 3; ++i) {
                float val = p.pos[i];
                if (val < min_pos[i]) {
                    min_pos[i] = val;
                } else if (val > max_pos[i]) {
                    max_pos[i] = val;
                }
            }
        }

        for (int i = 0; i < 3; ++i) {
            // PocketPoint coordinates are x, y, z
            // while voxels are stored as depth, height, width
            _shape[i] = static_cast<unsigned int>(
                std::ceil((max_pos[2 - i] - min_pos[2 - i]) / cell_size));
        }

        unsigned int size = _shape[0] * _shape[1] * _shape[2];
        for (int c = 0; c < NUM_POCKET_CHANNELS; ++c) {
            // Allocating and zero-initializing the channel
            _voxels[c] = new float[size];
            std::memset(_voxels[c], 0, size * sizeof(float));
        }

        // Keeps track of the number of points that fall within the same voxel
        int *points_count = new int[size];
        std::memset(points_count, 0, size * sizeof(int));

        // Assigning points to voxels
        for (const PocketPoint &p : points) {
            // Computing corresponding voxel coordinates
            unsigned int sub[3];
            for (int i = 0; i < 3; ++i) {
                sub[i] = static_cast<unsigned int>(
                    std::floor((p.pos[2 - i] - min_pos[2 - i]) / cell_size));
            }
            unsigned int idx = _sub_to_idx(sub[0], sub[1], sub[2]);
            points_count[idx]++;

            for (int c = 0; c < NUM_POCKET_CHANNELS; ++c) {
                _voxels[c][idx] += p.channels[c];
            }
        }

        // Averaging the values of the points in the same voxel
        for (int c = 0; c < NUM_POCKET_CHANNELS; ++c) {
            for (unsigned int i = 0; i < size; ++i) {
                _voxels[c][i] /= std::max(1, points_count[i]);
            }
        }

        delete[] points_count;
    }
}
