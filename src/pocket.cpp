#include "pocket.hpp"

#include <ostream>
#include <string>
#include <vector>
#include <array>
#include <limits>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <cassert>
#include <cstdlib>

#include "parsing.hpp"
#include "vec3.hpp"

namespace cuDock
{
    Pocket::Pocket(const std::string &csv_file_path, float cell_size) :
        _cell_size(cell_size)
    {
        std::vector<Pocket::Point> points;
        Parsing::read_pocket_csv(csv_file_path, points);

        _voxelize(points);
    }

    Pocket::Pocket(const std::vector<Pocket::Point> &pocket_points,
                   float cell_size) :
        _cell_size(cell_size)
    {
        _voxelize(pocket_points);
    }

    void Pocket::set_interpolate(enum InterpolateType int_type)
    {
        int_type_ = int_type;
    }

    enum InterpolateType Pocket::get_interpolate() const
    {
        return int_type_;
    }

    void Pocket::set_swizzled_tile_size(int tile_size)
    {
        swizzled_tile_size_ = tile_size;
    }

    int Pocket::get_swizzled_tile_size() const
    {
        return swizzled_tile_size_;
    }

    void Pocket::use_compressible_memory(bool flag)
    {
        use_compressible_memory_ = flag;
    }

    void Pocket::set_packed(bool flag)
    {
        is_packed_ = flag;
    }

    bool Pocket::is_packed() const
    {
        return is_packed_;
    }

    float Pocket::get_cell_size() const
    {
        return _cell_size;
    }

    unsigned int Pocket::get_size() const
    {
        unsigned int size = 1;

        for (int i = 0; i < 3; ++i) {
            size *= _shape[i];
        }

        return size;
    }

    unsigned int Pocket::get_shape(int cartesian_axis) const
    {
        return _shape[2 - cartesian_axis];
    }

    float Pocket::get_domain_size(int cartesian_axis) const
    {
        return _domain_size[2 - cartesian_axis];
    }

    float Pocket::get_voxel(unsigned int c,
                            unsigned int i,
                            unsigned int j,
                            unsigned int k) const
    {
        return _voxels[c][_sub_to_idx(i, j, k)];
    }

    const float *Pocket::get_voxels(unsigned int c,
                                    unsigned int i) const
    {
        return &_voxels[c][_sub_to_idx(i)];
    }

    void Pocket::lookup(const vec3 &pos,
                        std::array<float, Pocket::NUM_CHANNELS> &values) const
    {
        vec3ui sub;
        _pos_to_sub(pos, sub);

        for (int c = 0; c < Pocket::NUM_CHANNELS; ++c) {
            values[c] = get_voxel(c, sub[0], sub[1], sub[2]);
        }
    }

    Pocket::~Pocket()
    {
        for (int c = 0; c < Pocket::NUM_CHANNELS; ++c) {
            delete[] _voxels[c];
        }

        off_gpu(GPU_GMEM);
        off_gpu(GPU_GMEM_SWIZZLED);
        off_gpu(GPU_TMEM);
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
        assert(i < _shape[0]);
        assert(j < _shape[1]);
        assert(k < _shape[2]);

        return i * _shape[1] * _shape[2] +
               j * _shape[2] + k;
    }

    void Pocket::_pos_to_sub(const vec3 &pos, vec3ui &sub) const
    {
        for (int i = 0; i < 3; ++i) {
            if (pos[2 - i] > _domain_size[i]) {
                std::cout << pos[2 - i] << ", "
                          << _domain_size[i] << std::endl;
            }

            assert(pos[2 - i] <= _domain_size[i]);

            sub[i] = pos[2 - i] / _cell_size;
            if (sub[i] == _shape[i]) {
                sub[i] -= 1;
            }
        }
    }

    float trilerp(float x, float y, float z, float values[8])
    {
        assert(x >= 0 && x <= 1 && y >= 0 && y <= 1 && z >= 0 && z <= 1);

        float v12 = values[0] * (1.0f - x) + values[1] * x;
        float v34 = values[2] * (1.0f - x) + values[3] * x;
        float v56 = values[4] * (1.0f - x) + values[5] * x;
        float v78 = values[6] * (1.0f - x) + values[7] * x;

        float v1234 = v12 * (1.0f - y) + v34 * y;
        float v5678 = v56 * (1.0f - y) + v78 * y;

        return v1234 * (1.0f - z) + v5678 * z;
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
            // PocketPoint coordinates are x, y, z
            // while voxels are stored as depth, height, width
            _domain_size[2 - i] = max_pos[i] - min_pos[i];
            _shape[2 - i] = static_cast<unsigned int>(
                std::ceil(_domain_size[2 - i] / _cell_size));
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
            vec3 pos;
            vec3ui sub;
            // Translating domain bottom-left corner into origin
            for (int i = 0; i < 3; ++i) {
                pos[i] = p.pos[i] - min_pos[i];
            }
            _pos_to_sub(pos, sub);

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
