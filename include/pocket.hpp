#pragma once

#include <ostream>
#include <array>
#include <vector>
#include <array>
#include <iostream>

#include <cuda_runtime_api.h>

#include "vec3.hpp"
#include "utils.cuh"

namespace cuDock
{
    class Pocket
    {
    public:
        constexpr static int NUM_CHANNELS = 8;
        constexpr static float BASE_CELL_SIZE = 2.0;

        struct Point
        {
            vec3 pos; // x, y, z
            std::array<float, NUM_CHANNELS> channels;
        };

        Pocket(const std::string &csv_file_path, float cell_size);
        Pocket(const std::vector<Point> &pocket_points, float cell_size);

        void to_gpu(enum GPUMemType mem_type);
        void off_gpu(enum GPUMemType mem_type);

        bool is_on_gpu() const;
        bool is_on_gpu(enum GPUMemType mem_type) const;

        void set_interpolate(enum InterpolateType int_type);
        enum InterpolateType get_interpolate() const;

        void set_swizzled_tile_size(int tile_size);
        int get_swizzled_tile_size() const;

        void use_compressible_memory(bool flag);

        const std::array<float *, NUM_CHANNELS> &get_gpu_gmem_voxels() const;

        const std::array<cudaTextureObject_t, NUM_CHANNELS>
        &get_gpu_tmem_voxels() const;

        float get_cell_size() const;

        unsigned int get_size() const;

        unsigned int get_shape(int cartesian_axis) const;

        // The pocket is enclosed in the box of space spanning from (0, 0, 0)
        // to (get_domain_size(0), get_domain_size(1), get_domain_size(2))
        float get_domain_size(int cartesian_axis) const;

        float get_voxel(unsigned int c,
                        unsigned int i,
                        unsigned int j,
                        unsigned int k) const;

        const float *get_voxels(unsigned int c,
                                unsigned int i = 0) const;

        void lookup(const vec3 &pos,
                    std::array<float, NUM_CHANNELS> &values) const;

        ~Pocket();
    private:
        // Converts a 3d subscript into a one-dimensional index
        unsigned int _sub_to_idx(unsigned int i = 0,
                                 unsigned int j = 0,
                                 unsigned int k = 0) const;
        // Converts a (x, y, z) position into a 3d subscript
        void _pos_to_sub(const vec3 &pos,
                         vec3ui &sub) const;

        void _voxelize(const std::vector<Point> &points);

        std::array<float *, NUM_CHANNELS> _voxels;

        float _cell_size;
        vec3 _domain_size;
        vec3ui _shape;

        std::array<float *, NUM_CHANNELS> _gpu_global_voxels;
        std::array<cudaArray_t, NUM_CHANNELS> _gpu_array_voxels;
        std::array<cudaTextureObject_t, NUM_CHANNELS> _gpu_texture_voxels;

        std::array<bool, 4> _is_on_gpu = { 0, 0, 0, 0 };

        enum InterpolateType int_type_ = NN_INTERPOLATE;

        int swizzled_tile_size_ = 32;
        bool use_compressible_memory_ = false;
    };

    std::ostream &operator<<(std::ostream &os, const Pocket::Point &point);
}
