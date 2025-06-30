#include "swizzling.cuh"

#include <cassert>
#include <iostream>

namespace cuDock
{
    namespace Swizzling
    {
        __host__ __device__
        int get_padding_size(int size, int tile_size_in_bits)
        {
            int tile_size = 1 << tile_size_in_bits;
            return tile_size - size % tile_size;
        }

        int get_swizzled_size(int width,
                              int height,
                              int depth,
                              int tile_size_in_bits)
        {
            int w_pad = get_padding_size(width, tile_size_in_bits);
            int h_pad = get_padding_size(height, tile_size_in_bits);
            int d_pad = get_padding_size(depth, tile_size_in_bits);

            return (width + w_pad) *
                   (height + h_pad) *
                   (depth + d_pad);
        }

        __host__ __device__
        unsigned int expand_bits(unsigned int v, int num_bits_per_dim)
        {
            v &= (1 << num_bits_per_dim) - 1;

            v = (v * 0x00010001u) & 0xFF0000FFu;
            v = (v * 0x00000101u) & 0x0F00F00Fu;
            v = (v * 0x00000011u) & 0xC30C30C3u;
            v = (v * 0x00000005u) & 0x49249249u;

            return v;
        }

        __host__ __device__
        unsigned int morton_encode(unsigned int x,
                                   unsigned int y,
                                   unsigned int z,
                                   int num_bits_per_dim)
        {
            return expand_bits(x, num_bits_per_dim) * 4 +
                   expand_bits(y, num_bits_per_dim) * 2 +
                   expand_bits(z, num_bits_per_dim);
        }


        __host__ __device__
        int get_swizzled_idx(int x,
                             int y,
                             int z,
                             int padded_width,
                             int padded_height,
                             int tile_size_in_bits)
        {
            int tile_size = 1 << tile_size_in_bits;

            int brick_size = tile_size * tile_size * tile_size;

            int width_in_tiles = padded_width / tile_size;
            int height_in_tiles = padded_height / tile_size;

            int tile_x = x / tile_size;
            int tile_y = y / tile_size;
            int tile_z = z / tile_size;

            // Tile offset
            int dst_offset = tile_z * brick_size *
                                      width_in_tiles *
                                      height_in_tiles +
                             tile_y * brick_size *
                                      width_in_tiles +
                             tile_x * brick_size;

            return morton_encode(x, y, z, tile_size_in_bits) + dst_offset;
        }

        template<typename T>
        void to_swizzled_format(int width,
                                int height,
                                int depth,
                                int tile_size_in_bits,
                                const T *src,
                                T *dst)
        {
            int padded_width = width + get_padding_size(width,
                                                        tile_size_in_bits);
            int padded_height = height + get_padding_size(height,
                                                          tile_size_in_bits);

            for (int z = 0; z < depth; ++z) {
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {

                        int src_idx = z * height * width + y * width + x;
                        int dst_idx = get_swizzled_idx(x,
                                                       y,
                                                       z,
                                                       padded_width,
                                                       padded_height,
                                                       tile_size_in_bits);

                        assert(src_idx < width * height * depth);
                        assert(dst_idx < padded_width * padded_height *
                                         (depth +
                                          get_padding_size(depth,
                                                           tile_size_in_bits)));

                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
        }

        template void to_swizzled_format<float>(int width,
                                                int height,
                                                int depth,
                                                int tile_size_in_bits,
                                                const float *src,
                                                float *dst);
    }
}
