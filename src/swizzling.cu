#include "swizzling.cuh"

#include <cassert>
#include <iostream>

namespace cuDock
{
    namespace Swizzling
    {
        __host__ __device__
        int get_padding_size(int size, int tile_size)
        {
            return tile_size - size % tile_size;
        }

        int get_swizzled_size(int width,
                              int height,
                              int depth,
                              int tile_size)
        {
            int w_pad = get_padding_size(width, tile_size);
            int h_pad = get_padding_size(height, tile_size);
            int d_pad = get_padding_size(depth, tile_size);

            return (width + w_pad) *
                   (height + h_pad) *
                   (depth + d_pad);
        }

        template<typename T>
        void to_swizzled_format(int width,
                                int height,
                                int depth,
                                int tile_size,
                                const T *src,
                                T *dst)
        {
            int padded_width = width + get_padding_size(width, tile_size);
            int padded_height = height + get_padding_size(height, tile_size);

            int width_in_tiles = padded_width / tile_size;
            int height_in_tiles = padded_height / tile_size;

            int brick_size = tile_size * tile_size * tile_size;

            int x_offset_mult = brick_size;
            int y_offset_mult = brick_size * width_in_tiles;
            int z_offset_mult = brick_size * width_in_tiles * height_in_tiles;

            for (int z = 0; z < depth; ++z) {
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {

                        int src_idx = z * height * width + y * width + x;
                        int dst_idx = get_swizzled_idx(x,
                                                       y,
                                                       z,
                                                       x_offset_mult,
                                                       y_offset_mult,
                                                       z_offset_mult,
                                                       tile_size);

                        assert(src_idx < width * height * depth);
                        assert(dst_idx < padded_width * padded_height *
                                         (depth +
                                          get_padding_size(depth,
                                                           tile_size)));

                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
        }

        template void to_swizzled_format<float>(int width,
                                                int height,
                                                int depth,
                                                int tile_size,
                                                const float *src,
                                                float *dst);
    }
}
