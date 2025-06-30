#pragma once

namespace cuDock
{
    namespace Swizzling
    {
         int get_swizzled_size(int width,
                              int height,
                              int depth,
                              int tile_size_in_bits);

        template<typename T>
        void to_swizzled_format(int width,
                                int height,
                                int depth,
                                int tile_size_in_bits,
                                const T *src,
                                T *dst);

        __host__ __device__
        int get_padding_size(int size, int tile_size_in_bits);

        __host__ __device__
        int get_swizzled_idx(int x,
                             int y,
                             int z,
                             int padded_width,
                             int padded_height,
                             int tile_size_in_bits);
    }
}
