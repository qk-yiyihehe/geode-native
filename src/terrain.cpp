#include "terrain.hpp"
#include "geode.hpp"
#include "carver.hpp"
#include "structures.hpp"
#include "extensions/java263_noise.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include <vector>
#include <bit>
#include <unordered_map>
#include <math.h>

namespace terrain_noise {
#include "noise.h"
}

using terrain_noise::PerlinNoise;
using terrain_noise::Xoroshiro;
using terrain_noise::xSetSeed;
using terrain_noise::xNextLong;
using terrain_noise::xNextInt;
using terrain_noise::xPerlinInit;
using terrain_noise::samplePerlin;

namespace geode {
enum class Op {
    Constant,
    Add,
    Mul,
    Min,
    Max,
    Abs,
    Square,
    Cube,
    HalfNegative,
    QuarterNegative,
    Squeeze,
    Clamp,
    YGradient,
    Range,
    Noise,
    ShiftA,
    ShiftB,
    Shifted,
    Weird,
    Blended,
    Spline,
    Cache2D,
    FlatCache,
    Interpolated,
    Sub,
    Div,
    Negate,
    Lerp,
    Surface,
    Float,
    Ridge118,
    Slide118,
    Blended118
};

struct Node {
    Op op;
    int a, b, c, d;
    double p, q, r, s;
};

struct SplinePoint {
    float location, derivative;
    int value;
};

struct NoiseSpec {
    int first;
    std::vector<double> amplitudes;
    uint64_t lo, hi;
};
}

#include "terrain_data.hpp"
#include "terrain263_data.hpp"
#include "terrain118_data.hpp"

namespace geode {
namespace {
double wrap(double x) {
    return x - std::floor(x / 33554432.0 + 0.5) * 33554432.0;
}

double lerp(double t, double a, double b) {
    return a + t * (b - a);
}

float lerpf(float t, float a, float b) {
    return a + t * (b - a);
}

struct Octave {
    PerlinNoise noise;
    double frequency, weight;
};

struct Normal {
    std::vector<Octave> first, second;
    double factor;
    bool float_noise;

    explicit Normal(const NoiseSpec &spec, Xoroshiro seed, bool modern26 = false) : float_noise(modern26) {
        auto build = [&](std::vector<Octave> &out) {
            uint64_t lo = xNextLong(&seed), hi = xNextLong(&seed);
            int count = int(spec.amplitudes.size());
            double frequency = std::ldexp(1.0, spec.first),
                   weight = std::ldexp(1.0, count - 1) / (std::ldexp(1.0, count) - 1);
            for (int i = 0; i < count; ++i, frequency *= 2, weight *= 0.5)
                if (spec.amplitudes[i] != 0) {
                    auto *hash = terrain_data::octave_hashes[spec.first + i + 16];
                    Xoroshiro random{lo ^ hash[0], hi ^ hash[1]};
                    Octave o{};
                    xPerlinInit(&o.noise, &random);
                    o.frequency = frequency;
                    o.weight = modern26 ? spec.amplitudes[i] : weight * spec.amplitudes[i];
                    out.push_back(o);
                }
        };
        build(first);
        build(second);
        int lo = 0, hi = int(spec.amplitudes.size()) - 1;
        while (spec.amplitudes[lo] == 0)
            ++lo;
        while (spec.amplitudes[hi] == 0)
            --hi;
        int count = hi - lo + 1;
        factor = (1.0 / 6.0) / (0.1 * (1.0 + 1.0 / count));
    }

    double sample(double x, double y, double z) const {
        if (float_noise) {
            float value = 0;
            for (size_t i = 0; i < first.size(); ++i) {
                auto &a = first[i];
                auto &b = second[i];
                auto sample = [&](const Octave &o, double scale) {
                    double f = o.frequency * scale;
                    return cubiomes::java263::perlin(o.noise.d, o.noise.a, o.noise.b, o.noise.c, x * f, y * f, z * f);
                };
                value += float(a.weight) * sample(a, 1);
                value += float(b.weight) * sample(b, 1.0181268882175227);
            }
            return value;
        }
        auto sum = [](const std::vector<Octave> &octaves, double x, double y, double z) {
            double value = 0;
            for (auto &o : octaves)
                value += o.weight * samplePerlin(&o.noise, wrap(x * o.frequency), wrap(y * o.frequency),
                                                 wrap(z * o.frequency), 0, 0);
            return value;
        };
        return (sum(first, x, y, z) +
                sum(second, x * 1.0181268882175227, y * 1.0181268882175227, z * 1.0181268882175227)) *
               factor;
    }
};

struct Blended {
    std::array<PerlinNoise, 16> low, high;
    std::array<PerlinNoise, 8> main;
    std::array<double, 5> p;

    Blended(Xoroshiro random, std::array<double, 5> params) : p(params) {
        for (auto &n : low)
            xPerlinInit(&n, &random);
        for (auto &n : high)
            xPerlinInit(&n, &random);
        for (auto &n : main)
            xPerlinInit(&n, &random);
    }

    double sample(int x, int y, int z) const {
        double xx = x * 684.412 * p[0], yy = y * 684.412 * p[1], zz = z * 684.412 * p[0];
        double ax = xx / p[2], ay = yy / p[3], az = zz / p[2], smear = 684.412 * p[1] * p[4], main_smear = smear / p[3];
        double value = 0, frequency = 1;
        for (int i = 0; i < 8; ++i, frequency *= 0.5)
            value += samplePerlin(&main[i], wrap(ax * frequency), wrap(ay * frequency), wrap(az * frequency),
                                  main_smear * frequency, ay * frequency) /
                     frequency;
        double mix = (value / 10 + 1) / 2, lo = 0, hi = 0;
        frequency = 1;
        for (int i = 0; i < 16; ++i, frequency *= 0.5) {
            double dx = wrap(xx * frequency), dy = wrap(yy * frequency), dz = wrap(zz * frequency),
                   scale = smear * frequency;
            if (mix < 1)
                lo += samplePerlin(&low[i], dx, dy, dz, scale, yy * frequency) / frequency;
            if (mix > 0)
                hi += samplePerlin(&high[i], dx, dy, dz, scale, yy * frequency) / frequency;
        }
        return lerp(std::clamp(mix, 0.0, 1.0), lo / 512, hi / 512) / 128;
    }

    float sample263(int x, int y, int z) const {
        double xx = x * 684.412 * p[0], yy = y * 684.412 * p[1], zz = z * 684.412 * p[0], smear = 684.412 * p[1] * p[4];
        auto sum = [&](const auto &octaves, double x, double y, double z, double smear, float weight) {
            float value = 0;
            double frequency = 1;
            for (auto &n : octaves) {
                value += weight * cubiomes::java263::perlin(n.d, n.a, n.b, n.c, x * frequency, y * frequency, z * frequency,
                                                   smear * frequency);
                frequency *= 0.5;
                weight *= 2;
            }
            return value;
        };
        float mix = std::clamp(sum(main, xx / p[2], yy / p[3], zz / p[2], smear / p[3], 0.05f) + 0.5f, 0.0f, 1.0f);
        float a = mix < 1 ? sum(low, xx, yy, zz, smear, 1.0f / 65536) : 0,
              b = mix > 0 ? sum(high, xx, yy, zz, smear, 1.0f / 65536) : 0;
        return cubiomes::java263::lerp(mix, a, b);
    }
};

struct Key {
    int id, x, y, z;
    uint64_t context = 0;
    bool operator==(const Key &) const = default;
};

struct Entry {
    Key key{-1, 0, 0, 0};
    double value = 0;
};

struct Hash {
    size_t operator()(Key k) const {
        uint64_t v = uint64_t(uint32_t(k.x)) * 0x9E3779B97F4A7C15ULL + uint64_t(uint32_t(k.z)) * 0xC2B2AE3D27D4EB4FULL +
                     uint32_t(k.y) * 0x165667B1ULL + uint32_t(k.id) + k.context * 0xD6E8FEB86659FD93ULL;
        v ^= v >> 33;
        v *= 0xff51afd7ed558ccdULL;
        return v ^ (v >> 33);
    }
};

struct Fluid {
    int level = -32512, type = 1;

    int at(int y) const {
        return y < level ? type : 0;
    }
};

struct Cell {
    int x, y, z;
    Fluid fluid;
    bool ready = false;
};
}

struct Terrain::Impl {
    const std::vector<Node> &nodes;
    const std::vector<SplinePoint> &spline_points;
    std::vector<Normal> noises;
    std::vector<Blended> blends;
    std::vector<Entry> cache;
    bool interpolate = true, aquifer_eval = false;
    int context_x = 0, context_z = 0;
    uint64_t aquifer_lo = 0, aquifer_hi = 0, bedrock_lo = 0, bedrock_hi = 0;
    Carver carver;
    uint64_t seed;
    int generation;
    std::unique_ptr<Structures> structures;

    struct Neighbor {
        GeodeShape shape;
        int valid = -1;
    };

    std::unordered_map<uint64_t, Neighbor> neighbors;
    Simulator geodes;
    std::unordered_map<Key, Cell, Hash> cells;
    std::unordered_map<uint64_t, int> surfaces;

    struct SamplingBound {
        int threshold = -26;
        bool complete = false;
    };

    std::unordered_map<uint64_t, SamplingBound> sampling_bounds;

    explicit Impl(uint64_t seed, int generation)
        : nodes(uses_118_terrain(generation) ? terrain118_data::nodes : terrain_data::nodes),
          spline_points(uses_118_terrain(generation) ? terrain118_data::points : terrain_data::points),
          carver(seed), seed(seed), generation(generation), geodes(seed, generation) {
        Xoroshiro random;
        xSetSeed(&random, seed);
        uint64_t lo = xNextLong(&random), hi = xNextLong(&random);
        for (auto &spec : uses_118_terrain(generation) ? terrain118_data::noises :
                         generation >= 5 ? terrain263_data::noises : terrain_data::noises)
            noises.emplace_back(spec, Xoroshiro{lo ^ spec.lo, hi ^ spec.hi}, generation >= 5);
        for (auto params : uses_118_terrain(generation) ? terrain118_data::blends :
                           generation >= 5 ? terrain263_data::blends : terrain_data::blends)
            blends.emplace_back(Xoroshiro{lo ^ terrain_data::terrain_hash[0], hi ^ terrain_data::terrain_hash[1]},
                                params);
        Xoroshiro aq{lo ^ terrain_data::aquifer_hash[0], hi ^ terrain_data::aquifer_hash[1]};
        aquifer_lo = xNextLong(&aq);
        aquifer_hi = xNextLong(&aq);
        Xoroshiro br{lo ^ terrain_data::bedrock_hash[0], hi ^ terrain_data::bedrock_hash[1]};
        bedrock_lo = xNextLong(&br);
        bedrock_hi = xNextLong(&br);
        cache.resize(1 << 18);
    }

    double eval(int id, int x, int y, int z) {
        if (generation >= 5)
            return eval263(id, x, y, z);
        auto &n = nodes[id];
        if (n.op == Op::Constant)
            return n.p;
        if (n.op == Op::Cache2D && interpolate)
            y = 0;
        if (n.op == Op::FlatCache && interpolate &&
            (!aquifer_eval || (floor_div(x, 4) >= context_x * 4 && floor_div(x, 4) <= context_x * 4 + 4 &&
                               floor_div(z, 4) >= context_z * 4 && floor_div(z, 4) <= context_z * 4 + 4))) {
            x = floor_div(x, 4) * 4;
            y = 0;
            z = floor_div(z, 4) * 4;
        }
        Key key{id + int(nodes.size()) * (aquifer_eval ? 2 : interpolate ? 0 : 1),
                x, y, z, aquifer_eval ? geode::key(context_x, context_z) : 0};
        size_t slot = Hash{}(key) & (cache.size() - 1);
        if (cache[slot].key == key)
            return cache[slot].value;
        auto a = [&] {
            return eval(n.a, x, y, z);
        };
        auto b = [&] {
            return eval(n.b, x, y, z);
        };
        double result = 0;
        switch (n.op) {
            case Op::Add: {
                double va = a();
                result = va + b();
                break;
            }
            case Op::Mul: {
                double va = a();
                result = va == 0 ? 0 : va * b();
                break;
            }
            case Op::Min: {
                double va = a();
                result = std::min(va, b());
                break;
            }
            case Op::Max: {
                double va = a();
                result = std::max(va, b());
                break;
            }
            case Op::Abs:
                result = std::abs(a());
                break;
            case Op::Square: {
                double v = a();
                result = v * v;
                break;
            }
            case Op::Cube: {
                double v = a();
                result = v * v * v;
                break;
            }
            case Op::HalfNegative: {
                double v = a();
                result = v > 0 ? v : v * 0.5;
                break;
            }
            case Op::QuarterNegative: {
                double v = a();
                result = v > 0 ? v : v * 0.25;
                break;
            }
            case Op::Squeeze: {
                double v = std::clamp(a(), -1.0, 1.0);
                result = v / 2 - v * v * v / 24;
                break;
            }
            case Op::Clamp:
                result = std::clamp(a(), n.p, n.q);
                break;
            case Op::YGradient:
                result = lerp(std::clamp((y - n.p) / (n.q - n.p), 0.0, 1.0), n.r, n.s);
                break;
            case Op::Range: {
                double v = a();
                result = eval(v >= n.p && v < n.q ? n.b : n.c, x, y, z);
                break;
            }
            case Op::Noise:
                result = noises[n.a].sample(x * n.p, y * n.q, z * n.p);
                break;
            case Op::ShiftA:
                result = noises[n.a].sample(x * 0.25, 0, z * 0.25) * 4;
                break;
            case Op::ShiftB:
                result = noises[n.a].sample(z * 0.25, x * 0.25, 0) * 4;
                break;
            case Op::Shifted:
                result = noises[n.a].sample(x * n.p + eval(n.b, x, y, z), y * n.q + eval(n.c, x, y, z),
                                            z * n.p + eval(n.d, x, y, z));
                break;
            case Op::Weird: {
                double v = a(), scale;
                if (n.p == 0)
                    scale = v < -0.5 ? 0.75 : v < 0 ? 1 : v < 0.5 ? 1.5 : 2;
                else
                    scale = v < -0.75 ? 0.5 : v < -0.5 ? 0.75 : v < 0.5 ? 1 : v < 0.75 ? 2 : 3;
                result = scale * std::abs(noises[n.b].sample(x / scale, y / scale, z / scale));
                break;
            }
            case Op::Blended:
                result = blends[n.a].sample(x, y, z);
                break;
            case Op::Blended118:
                result = blends[n.a].sample(floor_div(x, 4) * 4, floor_div(y, 8) * 8, floor_div(z, 4) * 4);
                break;
            case Op::Float:
                result = float(a());
                break;
            case Op::Ridge118: {
                float value = float(a());
                result = -(std::abs(std::abs(value) - 0.6666667f) - 0.33333334f) * 3.0f;
                break;
            }
            case Op::Slide118: {
                auto slide = [](double target, double value, double alpha) {
                    return alpha <= 0 ? target : alpha >= 1 ? value : lerp(alpha, target, value);
                };
                int cell_y = y / 8 - n.b;
                result = slide(n.p, a(), (n.c - cell_y - n.r) / n.q);
                result = slide(n.s, result, double(cell_y) / n.d);
                break;
            }
            case Op::Cache2D:
            case Op::FlatCache:
                result = a();
                break;
            case Op::Interpolated: {
                if (!interpolate) {
                    result = a();
                    break;
                }
                int bx = floor_div(x, 4) * 4, by = floor_div(y, 8) * 8, bz = floor_div(z, 4) * 4;
                double tx = (x - bx) / 4.0, ty = (y - by) / 8.0, tz = (z - bz) / 4.0;
                double corners[2][2];
                for (int iz = 0; iz < 2; ++iz)
                    for (int ix = 0; ix < 2; ++ix) {
                        double lo = eval(n.a, bx + ix * 4, by, bz + iz * 4),
                               hi = eval(n.a, bx + ix * 4, by + 8, bz + iz * 4);
                        corners[iz][ix] = lerp(ty, lo, hi);
                    }
                result = lerp(tz, lerp(tx, corners[0][0], corners[0][1]), lerp(tx, corners[1][0], corners[1][1]));
                break;
            }
            case Op::Spline: {
                float input = float(a());
                auto begin = spline_points.data() + n.b;
                int count = n.c, index = -1;
                while (index + 1 < count && input >= begin[index + 1].location)
                    ++index;
                auto value = [&](int i) {
                    return float(eval(begin[i].value, x, y, z));
                };
                if (index < 0)
                    result = value(0) + begin[0].derivative * (input - begin[0].location);
                else if (index == count - 1)
                    result = value(index) + begin[index].derivative * (input - begin[index].location);
                else {
                    float span = begin[index + 1].location - begin[index].location,
                          t = (input - begin[index].location) / span, v0 = value(index), v1 = value(index + 1),
                          delta = v1 - v0;
                    float d0 = begin[index].derivative * span - delta, d1 = -begin[index + 1].derivative * span + delta;
                    result = lerpf(t, v0, v1) + t * (1 - t) * lerpf(t, d0, d1);
                }
                break;
            }
            default:
                throw std::logic_error("Unsupported native terrain operator");
        }
        cache[slot] = {key, result};
        return result;
    }

    float eval263(int id, int x, int y, int z) {
        auto &n = terrain263_data::nodes[id];
        if (n.op == Op::Constant)
            return float(n.p);
        Key key{id, x, y, z};
        size_t slot = Hash{}(key) & (cache.size() - 1);
        if (cache[slot].key == key)
            return float(cache[slot].value);
        auto a = [&] {
            return eval263(n.a, x, y, z);
        };
        auto b = [&] {
            return eval263(n.b, x, y, z);
        };
        float result = 0;
        switch (n.op) {
            case Op::Add:
                result = a() + b();
                break;
            case Op::Sub:
                result = a() - b();
                break;
            case Op::Mul:
                result = a() * b();
                break;
            case Op::Div:
                result = a() / b();
                break;
            case Op::Min:
                result = std::min(a(), b());
                break;
            case Op::Max:
                result = std::max(a(), b());
                break;
            case Op::Abs:
                result = std::abs(a());
                break;
            case Op::Negate:
                result = -a();
                break;
            case Op::Square: {
                float v = a();
                result = v * v;
                break;
            }
            case Op::Cube: {
                float v = a();
                result = v * v * v;
                break;
            }
            case Op::HalfNegative: {
                float v = a();
                result = v > 0 ? v : v * 0.5f;
                break;
            }
            case Op::QuarterNegative: {
                float v = a();
                result = v > 0 ? v : v * 0.25f;
                break;
            }
            case Op::Squeeze: {
                float v = std::clamp(a(), -1.0f, 1.0f);
                result = v / 2 - v * v * v / 24;
                break;
            }
            case Op::Clamp:
                result = std::clamp(a(), float(n.p), float(n.q));
                break;
            case Op::YGradient:
                result = float(n.r) + float(std::clamp(y, int(std::min(n.p, n.q)), int(std::max(n.p, n.q))) - n.p) *
                                          ((float(n.s) - float(n.r)) / float(n.q - n.p));
                break;
            case Op::Lerp: {
                float t = a();
                result = t == 0 ? b() : t == 1 ? eval263(n.c, x, y, z) : cubiomes::java263::lerp(t, b(), eval263(n.c, x, y, z));
                break;
            }
            case Op::Range: {
                float v = a();
                result = eval263(v >= n.p && v < n.q ? n.b : n.c, x, y, z);
                break;
            }
            case Op::ShiftA:
                result = float(noises[n.a].sample(x * 0.25, 0, z * 0.25)) * 4;
                break;
            case Op::ShiftB:
                result = float(noises[n.a].sample(z * 0.25, x * 0.25, 0)) * 4;
                break;
            case Op::Shifted:
                result = float(noises[n.a].sample(x * n.p + eval263(n.b, x, y, z), y * n.q + eval263(n.c, x, y, z),
                                                  z * n.p + eval263(n.d, x, y, z)));
                break;
            case Op::Blended:
                result = blends[n.a].sample263(x, y, z);
                break;
            case Op::Interpolated: {
                int sx = int(n.p), sy = int(n.q), bx = floor_div(x, sx) * sx, by = floor_div(y, sy) * sy,
                    bz = floor_div(z, sx) * sx;
                float tx = float(x - bx) / sx, ty = float(y - by) / sy, tz = float(z - bz) / sx, corners[2][2];
                for (int iz = 0; iz < 2; ++iz)
                    for (int iy = 0; iy < 2; ++iy)
                        corners[iz][iy] = cubiomes::java263::lerp(tx, eval263(n.a, bx, by + iy * sy, bz + iz * sx),
                                                         eval263(n.a, bx + sx, by + iy * sy, bz + iz * sx));
                result = cubiomes::java263::lerp(tz, cubiomes::java263::lerp(ty, corners[0][0], corners[0][1]),
                                        cubiomes::java263::lerp(ty, corners[1][0], corners[1][1]));
                break;
            }
            case Op::Spline: {
                float input = a();
                auto *begin = terrain263_data::points.data() + n.b;
                int count = n.c, index = -1;
                while (index + 1 < count && input >= begin[index + 1].location)
                    ++index;
                auto value = [&](int i) {
                    return eval263(begin[i].value, x, y, z);
                };
                if (index < 0)
                    result = value(0) + begin[0].derivative * (input - begin[0].location);
                else if (index == count - 1)
                    result = value(index) + begin[index].derivative * (input - begin[index].location);
                else {
                    float span = begin[index + 1].location - begin[index].location,
                          t = (input - begin[index].location) / span, v0 = value(index), v1 = value(index + 1),
                          delta = v1 - v0;
                    float d0 = begin[index].derivative * span - delta, d1 = -begin[index + 1].derivative * span + delta;
                    result = cubiomes::java263::lerp(t, v0, v1) + t * (1 - t) * cubiomes::java263::lerp(t, d0, d1);
                }
                break;
            }
            case Op::Surface: {
                int step = int(n.q), top = int(std::floor(b() / step)) * step;
                result = float(n.p);
                for (int py = top; py >= n.p; py -= step)
                    if (eval263(n.a, x, py, z) > 0) {
                        result = float(py);
                        break;
                    }
                break;
            }
            default:
                throw std::logic_error("Unsupported 26.3 terrain operation");
        }
        cache[slot] = {key, result};
        return result;
    }

    static Fluid global(int y) {
        return y < -54 ? Fluid{-54, 2} : Fluid{63, 1};
    }

    int surface(int x, int z) {
        if (generation >= 5) {
            auto k = key(x, z);
            auto found = surfaces.find(k);
            if (found != surfaces.end())
                return found->second;
            int height = int(eval263(terrain263_data::surface_level, x, 0, z));
            surfaces.emplace(k, height);
            return height;
        }
        x = floor_div(x, 4) * 4;
        z = floor_div(z, 4) * 4;
        auto k = key(x, z);
        auto found = surfaces.find(k);
        if (found != surfaces.end())
            return found->second;
        int height = 2147483647, top = 320;
        if (generation >= 4) {
            double upper = 128 + -128 * (0.2734375 * (1 / eval(terrain_data::surface_factor, x, 0, z)) +
                                         -1 * eval(terrain_data::surface_offset, x, 0, z));
            top = int(std::floor(std::clamp(upper, -40.0, 320.0) / 8)) * 8;
            height = -64;
        }
        for (int y = top; y >= -64; y -= 8)
            if (eval(uses_118_terrain(generation) ? terrain118_data::initial_density_without_jaggedness :
                     terrain_data::initial_density_without_jaggedness, x, y, z) > 0.390625) {
                height = y;
                break;
            }
        surfaces.emplace(k, height);
        return height;
    }

    Fluid compute_fluid(int x, int y, int z) {
        Fluid result = global(y);
        int lowest = 2147483647;
        bool under_water = false;
        static constexpr int offsets[][2] = {{0, 0},  {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {-3, 0}, {-2, 0},
                                             {-1, 0}, {1, 0},   {-2, 1},  {-1, 1}, {0, 1},  {1, 1}};
        for (auto &offset : offsets) {
            int px = x + offset[0] * 16, pz = z + offset[1] * 16, h = surface(px, pz), top = h + 8;
            bool center = offset[0] == 0 && offset[1] == 0;
            if (center && y - 12 > top)
                return result;
            bool reaches = y + 12 > top;
            if (reaches || center) {
                auto other = global(top);
                if (other.at(top) != 0) {
                    if (center)
                        under_water = true;
                    if (reaches)
                        return other;
                }
            }
            lowest = std::min(lowest, h);
        }
        double wet = -1, full = -1;
        bool excluded = generation >= 5 ? eval263(terrain263_data::exclusion, x, y, z) > 0
                                        : !uses_118_terrain(generation) &&
                                              eval(terrain_data::erosion, x, y, z) < -0.22499999403953552 &&
                                              eval(terrain_data::depth, x, y, z) > 0.8999999761581421;
        if (!excluded) {
            double water = under_water ? 1.0 - std::clamp((lowest + 8 - y) / 64.0, 0.0, 1.0) : 0.0;
            double noise = std::clamp(
                eval(uses_118_terrain(generation) ? terrain118_data::fluid_level_floodedness :
                     generation >= 5 ? terrain263_data::fluid_level_floodedness : terrain_data::fluid_level_floodedness,
                     x, y, z),
                -1.0, 1.0);
            wet = noise - lerp(1 - water, -0.8, 0.4);
            full = noise - lerp(1 - water, -0.3, 0.8);
        }
        int level = -32512, spread_level = -32512;
        if (full > 0)
            level = result.level;
        else if (wet > 0) {
            int gx = floor_div(x, 16), gy = floor_div(y, 40), gz = floor_div(z, 16);
            double spread =
                eval(uses_118_terrain(generation) ? terrain118_data::fluid_level_spread :
                     generation >= 5 ? terrain263_data::fluid_level_spread : terrain_data::fluid_level_spread, gx, gy,
                     gz) *
                10;
            spread_level = gy * 40 + 20 + int(std::floor(spread / 3)) * 3;
            level = std::min(lowest, spread_level);
        }
        // 1.18以截断到地表前的扩散高度选流体；新版使用最终高度。
        int lava_level = uses_118_terrain(generation) ? spread_level : level;
        if (lava_level <= -10 && lava_level != -32512 && result.type != 2 &&
            std::abs(eval(uses_118_terrain(generation) ? terrain118_data::lava :
                          generation >= 5 ? terrain263_data::lava : terrain_data::lava, floor_div(x, 64),
                          floor_div(y, 40), floor_div(z, 64))) > 0.3)
            result.type = 2;
        result.level = level;
        return result;
    }

    Cell &cell(int x, int y, int z) {
        Key k{0, x, y, z, key(context_x, context_z)};
        auto found = cells.find(k);
        if (found != cells.end())
            return found->second;
        uint64_t hash = uint64_t(int64_t(std::bit_cast<int32_t>(uint32_t(x) * 3129871U))) ^
                        (uint64_t(int64_t(z)) * 116129781ULL) ^ uint64_t(int64_t(y));
        hash = hash * hash * 42317861ULL + hash * 11;
        hash = uint64_t(std::bit_cast<int64_t>(hash) >> 16);
        Xoroshiro r{aquifer_lo ^ hash, aquifer_hi};
        int px = x * 16 + xNextInt(&r, 10), py = y * 12 + xNextInt(&r, 9), pz = z * 16 + xNextInt(&r, 10);
        return cells.emplace(k, Cell{px, py, pz, {}, false}).first->second;
    }

    Fluid status(const Key &k) {
        auto &c = cell(k.x, k.y, k.z);
        if (!c.ready) {
            c.fluid = compute_fluid(c.x, c.y, c.z);
            c.ready = true;
        }
        return c.fluid;
    }

    double pressure(int x, int y, int z, Fluid a, Fluid b, double &barrier) {
        int aa = a.at(y), bb = b.at(y);
        if ((aa == 1 && bb == 2) || (aa == 2 && bb == 1))
            return 2;
        int difference = std::abs(a.level - b.level);
        if (!difference)
            return 0;
        double middle = 0.5 * (a.level + b.level), dy = y + 0.5 - middle, overlap = difference / 2.0 - std::abs(dy), p;
        if (dy > 0)
            p = overlap / (overlap > 0 ? 1.5 : 2.5);
        else {
            overlap += 3;
            p = overlap / (overlap > 0 ? 3.0 : 10.0);
        }
        double noise = 0;
        if (p >= -2 && p <= 2) {
            if (std::isnan(barrier))
                barrier = eval(uses_118_terrain(generation) ? terrain118_data::barrier :
                               generation >= 5 ? terrain263_data::barrier : terrain_data::barrier, x, y, z);
            noise = barrier;
        }
        return 2 * (noise + p);
    }

    bool aquifer_solid(int x, int y, int z, double density) {
        // 旧采样器在基础密度触及-64时直接使用全局流体，不再施加含水层压力。
        if (generation == -2 && density <= -11.0 / 24 && eval(terrain118_data::base_density, x, y, z) <= -64)
            return false;
        if (density > 0)
            return true;
        if (global(y).at(y) == 2)
            return false;
        // 限制含水层缓存，避免大范围开发扫描累计所有历史查询；值可确定重算。
        if (cells.size() >= 65536)
            cells.clear();
        if (surfaces.size() >= 65536)
            surfaces.clear();
        context_x = floor_div(x, 16);
        context_z = floor_div(z, 16);
        aquifer_eval = true;
        if (generation == 4 && y > -26) {
            auto &bound = sampling_bounds[key(context_x, context_z)];
            if (y > bound.threshold && !bound.complete) {
                bool complete = true;
                for (int pz = (context_z - 1) * 16; pz <= (context_z + 1) * 16 + 9 && complete; pz += 4)
                    for (int px = (context_x - 1) * 16; px <= (context_x + 1) * 16 + 9; px += 4) {
                        int h = surface(px, pz);
                        bound.threshold = std::max(bound.threshold, (floor_div(h + 20, 12) + 1) * 12 + 10);
                        // 任一采样表面若能证明上界高于查询位置，即可省去剩余列扫描。
                        if (y <= bound.threshold) {
                            complete = false;
                            break;
                        }
                    }
                bound.complete = complete;
            }
            if (y > bound.threshold) {
                aquifer_eval = false;
                return false;
            }
        }
        int gx = floor_div(x - 5, 16), gy = floor_div(y + 1, 12), gz = floor_div(z - 5, 16);
        int distances[3] = {2147483647, 2147483647, 2147483647};
        Key nearest[3]{};
        for (int dx = 0; dx <= 1; ++dx)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dz = 0; dz <= 1; ++dz) {
                    auto &p = cell(gx + dx, gy + dy, gz + dz);
                    int xx = p.x - x, yy = p.y - y, zz = p.z - z, d = xx * xx + yy * yy + zz * zz;
                    Key k{0, gx + dx, gy + dy, gz + dz};
                    if (distances[0] >= d) {
                        distances[2] = distances[1];
                        nearest[2] = nearest[1];
                        distances[1] = distances[0];
                        nearest[1] = nearest[0];
                        distances[0] = d;
                        nearest[0] = k;
                    } else if (distances[1] >= d) {
                        distances[2] = distances[1];
                        nearest[2] = nearest[1];
                        distances[1] = d;
                        nearest[1] = k;
                    } else if (distances[2] >= d) {
                        distances[2] = d;
                        nearest[2] = k;
                    }
                }
        auto similarity = [](int a, int b) {
            return 1.0 - std::abs(b - a) / 25.0;
        };
        Fluid a = status(nearest[0]);
        double sim = similarity(distances[0], distances[1]);
        bool solid = false;
        if (generation == -2 && a.at(y) == 1 && global(y - 1).at(y - 1) == 2) {
            // 1.18/1.18.1在全局水/岩浆接缝加密度1，1.18.2起直接保留流体。
            solid = density + 1 > 0;
        } else if (sim > 0 && !(a.at(y) == 1 && global(y - 1).at(y - 1) == 2)) {
            double barrier = std::nan("");
            Fluid b = status(nearest[1]);
            solid = density + sim * pressure(x, y, z, a, b, barrier) > 0;
            if (!solid) {
                Fluid c = status(nearest[2]);
                double ac = similarity(distances[0], distances[2]), bc = similarity(distances[1], distances[2]);
                solid = (ac > 0 && density + sim * ac * pressure(x, y, z, a, c, barrier) > 0) ||
                        (bc > 0 && density + sim * bc * pressure(x, y, z, b, c, barrier) > 0);
            }
        }
        aquifer_eval = false;
        return solid;
    }
};

Terrain::Terrain(uint64_t seed, int generation) : impl_(std::make_unique<Impl>(seed, generation)) {}

Terrain::~Terrain() = default;

double Terrain::density(int x, int y, int z) {
    return impl_->eval(uses_118_terrain(impl_->generation) ? terrain118_data::root :
                       impl_->generation >= 5 ? terrain263_data::root : terrain_data::root, x, y, z);
}

double Terrain::raw_density(int x, int y, int z) {
    impl_->interpolate = false;
    double v = impl_->eval(uses_118_terrain(impl_->generation) ? terrain118_data::root :
                          impl_->generation >= 5 ? terrain263_data::root : terrain_data::root, x, y, z);
    impl_->interpolate = true;
    return v;
}

double Terrain::noise(int id, double x, double y, double z) {
    return impl_->noises.at(id).sample(x, y, z);
}

bool Terrain::base_solid(int x, int y, int z) {
    return y >= -64 && y < 320 && impl_->aquifer_solid(x, y, z, density(x, y, z));
}

bool Terrain::carved(int x, int y, int z) {
    return impl_->carver.contains(x, y, z);
}

bool Terrain::overwritten(int x, int y, int z) {
    if (bedrock(x, y, z))
        return true;
    if (impl_->generation <= 0)
        return false;
    if (!impl_->structures)
        impl_->structures = std::make_unique<Structures>(impl_->seed, impl_->generation);
    return impl_->structures->trial_written(x, y, z);
}

bool Terrain::bedrock(int x, int y, int z) {
    if (y <= -64)
        return true;
    if (y >= -59)
        return false;
    uint64_t hash = uint64_t(int64_t(std::bit_cast<int32_t>(uint32_t(x) * 3129871U))) ^
                    (uint64_t(int64_t(z)) * 116129781ULL) ^ uint64_t(int64_t(y));
    hash = hash * hash * 42317861ULL + hash * 11;
    hash = uint64_t(std::bit_cast<int64_t>(hash) >> 16);
    Xoroshiro r{impl_->bedrock_lo ^ hash, impl_->bedrock_hi};
    double sample = (xNextLong(&r) >> 40) * (1.0 / 16777216);
    return sample < 1 - (y + 64) / 5.0;
}

bool Terrain::overwritten(int x, int y, int z, int source_x, int source_z) {
    if (overwritten(x, y, z))
        return true;
    if (impl_->neighbors.size() > 32768)
        impl_->neighbors.clear();
    int cx = floor_div(x, 16), cz = floor_div(z, 16);
    // 使用规定的来源顺序；实际加载顺序变化由原版参照另外核对。
    for (int nx = cx - 2; nx <= cx + 2; ++nx)
        for (int nz = cz - 2; nz <= cz + 2; ++nz) {
            if (!in_world(nx, nz) || nx < source_x || (nx == source_x && nz <= source_z))
                continue;
            auto k = key(nx, nz);
            auto found = impl_->neighbors.find(k);
            if (found == impl_->neighbors.end())
                found = impl_->neighbors.emplace(k, Impl::Neighbor{impl_->geodes.describe(nx, nz)}).first;
            auto &n = found->second;
            if (!n.shape.exists || impl_->geodes.material(n.shape, {x, y, z}) == Material::Unchanged)
                continue;
            if (n.valid < 0) {
                int invalid = 0;
                for (int i = 0; i < n.shape.count; ++i) {
                    auto p = n.shape.points[i];
                    if (!solid(n.shape.x + p.x, n.shape.y + p.y, n.shape.z + p.z) && ++invalid > 1)
                        break;
                }
                n.valid = invalid <= 1;
            }
            if (n.valid)
                return true;
        }
    return false;
}

void Terrain::filter_budding(Simulation &result, int source_x, int source_z) {
    if (result.budding.empty())
        return;
    if (impl_->neighbors.size() > 32768)
        impl_->neighbors.clear();
    Block lo = result.budding.front(), hi = lo;
    for (auto p : result.budding) {
        lo.x = std::min(lo.x, p.x);
        lo.y = std::min(lo.y, p.y);
        lo.z = std::min(lo.z, p.z);
        hi.x = std::max(hi.x, p.x);
        hi.y = std::max(hi.y, p.y);
        hi.z = std::max(hi.z, p.z);
    }
    std::vector<Impl::Neighbor *> neighbors;
    for (int nx = floor_div(lo.x, 16) - 2; nx <= floor_div(hi.x, 16) + 2; ++nx)
        for (int nz = floor_div(lo.z, 16) - 2; nz <= floor_div(hi.z, 16) + 2; ++nz) {
            if (!in_world(nx, nz) || nx < source_x || (nx == source_x && nz <= source_z))
                continue;
            auto k = key(nx, nz);
            auto found = impl_->neighbors.find(k);
            if (found == impl_->neighbors.end())
                found = impl_->neighbors.emplace(k, Impl::Neighbor{impl_->geodes.describe(nx, nz)}).first;
            auto &n = found->second;
            const auto &s = n.shape;
            if (s.exists && s.x + 16 >= lo.x && s.x - 16 <= hi.x && s.y + 16 >= lo.y && s.y - 16 <= hi.y &&
                s.z + 16 >= lo.z && s.z - 16 <= hi.z)
                neighbors.push_back(&n);
        }
    // 每洞只收集一次相关邻洞；延迟有效性检查保持逐点覆盖判定语义。
    std::erase_if(result.budding, [&](Block p) {
        if (overwritten(p.x, p.y, p.z))
            return true;
        for (auto *neighbor : neighbors) {
            auto &n = *neighbor;
            if (impl_->geodes.material(n.shape, p) == Material::Unchanged)
                continue;
            if (n.valid < 0) {
                int invalid = 0;
                for (int i = 0; i < n.shape.count; ++i) {
                    auto q = n.shape.points[i];
                    if (!solid(n.shape.x + q.x, n.shape.y + q.y, n.shape.z + q.z) && ++invalid > 1)
                        break;
                }
                n.valid = invalid <= 1;
            }
            if (n.valid)
                return true;
        }
        return false;
    });
}

bool Terrain::solid(int x, int y, int z) {
    if (y < -64 || y >= 320)
        return false;
    double d = density(x, y, z) + beard(x, y, z);
    if (d <= 0)
        return impl_->aquifer_solid(x, y, z, d);
    return !carved(x, y, z) || impl_->aquifer_solid(x, y, z, 0);
}

double Terrain::beard(int x, int y, int z) {
    if (!impl_->structures)
        impl_->structures = std::make_unique<Structures>(impl_->seed, impl_->generation);
    return impl_->structures->beard(x, y, z);
}
}
