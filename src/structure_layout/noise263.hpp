#pragma once
#include <cmath>
#include <cstdint>

namespace geode_layout::noise263 {
inline double wrap(double v) {
    return v - std::floor(v / 33554432.0 + 0.5) * 33554432.0;
}

inline float lerp(float t, float a, float b) {
    return a + t * (b - a);
}

inline float smooth(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

inline float perlin(const uint8_t *permutation, double ox, double oy, double oz, double x, double y, double z,
                    double smear = 0) {
    double ax = wrap(x) + ox, ay = wrap(y) + oy, az = wrap(z) + oz;
    int ix = int(std::floor(ax)), iy = int(std::floor(ay)), iz = int(std::floor(az));
    float a = float(ax - ix), b = float(ay - iy), c = float(az - iz), original_b = b;
    if (smear != 0) {
        double relative = ay - iy, limit = y >= 0 && y < relative ? y : relative;
        b = float(relative - std::floor(limit / smear + double(1.0e-7f)) * smear);
    }
    auto p = [&](int i) {
        return permutation[i & 255];
    };
    static constexpr int g[16][3] = {{1, 1, 0},  {-1, 1, 0},  {1, -1, 0}, {-1, -1, 0}, {1, 0, 1},  {-1, 0, 1},
                                     {1, 0, -1}, {-1, 0, -1}, {0, 1, 1},  {0, -1, 1},  {0, 1, -1}, {0, -1, -1},
                                     {1, 1, 0},  {0, -1, 1},  {-1, 1, 0}, {0, -1, -1}};
    auto dot = [&](int dx, int dy, int dz) {
        auto &v = g[p(p(p(ix + dx) + iy + dy) + iz + dz) & 15];
        return (float(v[0]) * (a - dx) + float(v[1]) * (b - dy)) + float(v[2]) * (c - dz);
    };
    float tx = smooth(a), ty = smooth(original_b), tz = smooth(c);
    return lerp(tz, lerp(ty, lerp(tx, dot(0, 0, 0), dot(1, 0, 0)), lerp(tx, dot(0, 1, 0), dot(1, 1, 0))),
                lerp(ty, lerp(tx, dot(0, 0, 1), dot(1, 0, 1)), lerp(tx, dot(0, 1, 1), dot(1, 1, 1))));
}
}
