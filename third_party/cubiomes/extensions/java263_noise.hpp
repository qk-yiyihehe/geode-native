// SPDX-License-Identifier: GPL-3.0-only
// 移自MinecraftFinders；保留原许可。该可选C++扩展不改变上游MIT的旧版噪声接口。
// 26.3要求float中间值与原求值顺序；不能用旧版double Perlin或允许FMA替代。
#pragma once
#include <cmath>
#include <cstdint>

namespace cubiomes::java263 {
inline double wrap(double v) {
    return v - std::floor(v / 33554432.0 + 0.5) * 33554432.0;
}

inline float lerp(float t, float a, float b) {
    return a + t * (b - a);
}

inline float smooth(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

extern "C" float samplePerlinFloatRaw(const uint8_t *, double, double, double,
    double, double, double, double);

inline float perlin(const uint8_t *permutation, double ox, double oy, double oz, double x, double y, double z,
                    double smear = 0) {
    return samplePerlinFloatRaw(permutation, ox, oy, oz, x, y, z, smear);
}
}
