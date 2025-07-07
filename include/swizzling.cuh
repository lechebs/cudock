#pragma once

namespace cuDock
{
    namespace Swizzling
    {
        int get_swizzled_size(int width,
                              int height,
                              int depth,
                              int tile_size);

        template<typename T>
        void to_swizzled_format(int width,
                                int height,
                                int depth,
                                int tile_size,
                                const T *src,
                                T *dst);

        __host__ __device__
        int get_padding_size(int size, int tile_size);

        __host__ __device__ __forceinline__
        int get_swizzled_idx(int x,
                             int y,
                             int z,
                             int x_offset_mult,
                             int y_offset_mult,
                             int z_offset_mult,
                             int tile_size);

        __host__ __device__ __forceinline__
        unsigned int expand_bits(unsigned int v, unsigned int tile_size)
        {
            v &= tile_size - 1u;
            //v &= (1 << 4) - 1u;

            v = (v * 0x00010001u) & 0xFF0000FFu;
            v = (v * 0x00000101u) & 0x0F00F00Fu;
            v = (v * 0x00000011u) & 0xC30C30C3u;
            v = (v * 0x00000005u) & 0x49249249u;

            return v;
        }

        __host__ __device__ __forceinline__
        unsigned int morton_encode(unsigned int x,
                                   unsigned int y,
                                   unsigned int z,
                                   unsigned int tile_size)
        {
            return expand_bits(x, tile_size) * 4 +
                   expand_bits(y, tile_size) * 2 +
                   expand_bits(z, tile_size);
        }

#define TILE_SIZE 32

        __host__ __device__ __forceinline__
        int get_swizzled_idx(int x,
                             int y,
                             int z,
                             int x_offset_mult,
                             int y_offset_mult,
                             int z_offset_mult,
                             int tile_size)
        {
            int tile_x = x / TILE_SIZE;
            int tile_y = y / TILE_SIZE;
            int tile_z = z / TILE_SIZE;

            // Tile offset
            int dst_offset = tile_z * z_offset_mult +
                             tile_y * y_offset_mult +
                             tile_x * x_offset_mult;

            /*
            return dst_offset + (TILE_SIZE * TILE_SIZE * (z % TILE_SIZE) +
                                 TILE_SIZE * (y % TILE_SIZE) +
                                 (x % TILE_SIZE));
            */

            return morton_encode(x, y, z, tile_size) + dst_offset;
        }

    }
}
