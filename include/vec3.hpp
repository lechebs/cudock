#pragma once

#include <cassert>

template<typename T> class _vec3
{
public:
    _vec3() : _data({ 0, 0, 0 }) {}
    _vec3(T x, T y, T z) : _data({ x, y, z }) {}

    T &operator[](unsigned int i)
    {
        assert(i < 3);
        return _data[i];
    }

    const T &operator[](unsigned int i) const
    {
        assert(i < 3);
        return _data[i];
    }

private:
    std::array<T, 3> _data;
};

using vec3 = _vec3<float>;
using vec3ui = _vec3<unsigned int>;
