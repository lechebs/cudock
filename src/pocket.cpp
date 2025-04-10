#include "pocket.hpp"

#include <ostream>
#include <string>
#include <vector>
#include <array>
#include <limits>
#include <cmath>
#include <cstring>
#include <iomanip>

#include "parser.hpp"

namespace cuDock
{
    Pocket::Pocket(const std::string &csv_file_path, float cell_size) :
        _cell_size(cell_size)
    {
        std::vector<Pocket::Point> points;
        Parser::read_pocket_csv(csv_file_path, points);

        _voxelize(points);
    }

    Pocket::Pocket(const std::vector<Pocket::Point> &pocket_points,
                   float cell_size) :
        _cell_size(cell_size)
    {
        _voxelize(pocket_points);
    }

    unsigned int Pocket::size() const
    {
        unsigned int size = Pocket::NUM_CHANNELS;

        for (int i = 0; i < 3; ++i) {
            size *= _shape[i];
        }

        return size;
    }

    unsigned int Pocket::shape(int axis) const
    {
        return _shape[axis];
    }

    void Pocket::domain(int axis, float &min, float &max) const
    {
        min = _domain[axis];
        max = _domain[axis + 3];
    }

    float Pocket::voxel(unsigned int c,
                        unsigned int i,
                        unsigned int j,
                        unsigned int k) const
    {
        return _voxels[c][_sub_to_idx(i, j, k)];
    }

    const float *Pocket::voxels(unsigned int c,
                                unsigned int i) const
    {
        return &_voxels[c][_sub_to_idx(i)];
    }

    void Pocket::lookup(const float pos[3],
                        std::array<float, Pocket::NUM_CHANNELS> &values) const
    {
        unsigned int sub[3];
        _pos_to_sub(pos, sub);

        for (int c = 0; c < Pocket::NUM_CHANNELS; ++c) {
            values[c] = voxel(c, sub[0], sub[1], sub[2]);
        }
    }

    Pocket::~Pocket()
    {
        return;
        for (int c = 0; c < Pocket::NUM_CHANNELS; ++c) {
            delete[] _voxels[c];
        }
    }

    std::ostream &operator<<(std::ostream &os, const Pocket::Point &point)
    {
        os << "x=" << point.pos[0] << ", y="
                   << point.pos[1] << ", z="
                   << point.pos[2];

        for (int c = 0; c < Pocket::NUM_CHANNELS; ++c) {
            os << ", psi" << c << "=" << point.channels[c];
        }

        return os;
    }

    unsigned int Pocket::_sub_to_idx(unsigned int i,
                                     unsigned int j,
                                     unsigned int k) const
    {
        return i * _shape[1] * _shape[2] +
               j * _shape[2] + k;
    }

    // TODO: handle pos out of domain
    void Pocket::_pos_to_sub(const float pos[3],
                             unsigned int sub[3]) const
    {
        for (int i = 0; i < 3; ++i) {
            sub[i] = static_cast<unsigned int>(
                std::floor((pos[2 - i] - _domain[i]) / _cell_size));
        }
    }

    void Pocket::_voxelize(const std::vector<Pocket::Point> &points)
    {
        float inf = std::numeric_limits<float>::infinity();
        float min_pos[3] = { inf, inf, inf };
        float max_pos[3] = { -inf, -inf, -inf };

        // Computing bounding box
        for (const Pocket::Point &p : points) {
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
            _domain[i] = min_pos[2 - i];
            _domain[i + 3] = max_pos[2 - i];
            // PocketPoint coordinates are x, y, z
            // while voxels are stored as depth, height, width
            _shape[i] = static_cast<unsigned int>(
                std::ceil((_domain[i + 3] - _domain[i]) / _cell_size));
        }

        unsigned int size = _shape[0] * _shape[1] * _shape[2];
        for (int c = 0; c < Pocket::NUM_CHANNELS; ++c) {
            // Allocating and zero-initializing the channel
            _voxels[c] = new float[size];
            std::memset(_voxels[c], 0, size * sizeof(float));
        }

        // Keeps track of the number of points that fall within the same voxel
        int *points_count = new int[size];
        std::memset(points_count, 0, size * sizeof(int));

        // Assigning points to voxels
        for (const Pocket::Point &p : points) {
            // Computing corresponding voxel coordinates
            unsigned int sub[3];
            _pos_to_sub(p.pos, sub);

            unsigned int idx = _sub_to_idx(sub[0], sub[1], sub[2]);
            points_count[idx]++;

            for (int c = 0; c < Pocket::NUM_CHANNELS; ++c) {
                _voxels[c][idx] += p.channels[c];
            }
        }

        // Averaging the values of the points in the same voxel
        for (int c = 0; c < Pocket::NUM_CHANNELS; ++c) {
            for (unsigned int i = 0; i < size; ++i) {
                _voxels[c][i] /= std::max(1, points_count[i]);
            }
        }

        delete[] points_count;
    }
}
