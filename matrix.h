#pragma once
#include <intrin.h>

struct alignas(16) Matrix
{
    union
    {
        __m128 r[4];
        float m[4][4];
    };
};

#define MATRIX_ROW_COMP(r, c) (const float *)(&r) + c

inline Matrix multiply(const Matrix& m1, const Matrix& m2) noexcept
{
    Matrix m;
    __m128 x, y, z, w;
    x = _mm_broadcast_ss(MATRIX_ROW_COMP(m1.r[0], 0));
    y = _mm_broadcast_ss(MATRIX_ROW_COMP(m1.r[0], 1));
    z = _mm_broadcast_ss(MATRIX_ROW_COMP(m1.r[0], 2));
    w = _mm_broadcast_ss(MATRIX_ROW_COMP(m1.r[0], 3));
    x = _mm_mul_ps(x, m2.r[0]);
    y = _mm_mul_ps(y, m2.r[1]);
    z = _mm_mul_ps(z, m2.r[2]);
    w = _mm_mul_ps(w, m2.r[3]);
    x = _mm_add_ps(x, z);
    y = _mm_add_ps(y, w);
    x = _mm_add_ps(x, y);
    m.r[0] = x;
    x = _mm_broadcast_ss(MATRIX_ROW_COMP(m1.r[1], 0));
    y = _mm_broadcast_ss(MATRIX_ROW_COMP(m1.r[1], 1));
    z = _mm_broadcast_ss(MATRIX_ROW_COMP(m1.r[1], 2));
    w = _mm_broadcast_ss(MATRIX_ROW_COMP(m1.r[1], 3));
    x = _mm_mul_ps(x, m2.r[0]);
    y = _mm_mul_ps(y, m2.r[1]);
    z = _mm_mul_ps(z, m2.r[2]);
    w = _mm_mul_ps(w, m2.r[3]);
    x = _mm_add_ps(x, z);
    y = _mm_add_ps(y, w);
    x = _mm_add_ps(x, y);
    m.r[1] = x;
    x = _mm_broadcast_ss(MATRIX_ROW_COMP(m1.r[2], 0));
    y = _mm_broadcast_ss(MATRIX_ROW_COMP(m1.r[2], 1));
    z = _mm_broadcast_ss(MATRIX_ROW_COMP(m1.r[2], 2));
    w = _mm_broadcast_ss(MATRIX_ROW_COMP(m1.r[2], 3));
    x = _mm_mul_ps(x, m2.r[0]);
    y = _mm_mul_ps(y, m2.r[1]);
    z = _mm_mul_ps(z, m2.r[2]);
    w = _mm_mul_ps(w, m2.r[3]);
    x = _mm_add_ps(x, z);
    y = _mm_add_ps(y, w);
    x = _mm_add_ps(x, y);
    m.r[2] = x;
    x = _mm_broadcast_ss(MATRIX_ROW_COMP(m1.r[3], 0));
    y = _mm_broadcast_ss(MATRIX_ROW_COMP(m1.r[3], 1));
    z = _mm_broadcast_ss(MATRIX_ROW_COMP(m1.r[3], 2));
    w = _mm_broadcast_ss(MATRIX_ROW_COMP(m1.r[3], 3));
    x = _mm_mul_ps(x, m2.r[0]);
    y = _mm_mul_ps(y, m2.r[1]);
    z = _mm_mul_ps(z, m2.r[2]);
    w = _mm_mul_ps(w, m2.r[3]);
    x = _mm_add_ps(x, z);
    y = _mm_add_ps(y, w);
    x = _mm_add_ps(x, y);
    m.r[3] = x;
    return m;
}
