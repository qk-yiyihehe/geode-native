#include "climate263.hpp"
#include "noise263.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <math.h>

namespace climate_noise {
#include "noise.h"
}

using climate_noise::PerlinNoise;
using climate_noise::Xoroshiro;
using climate_noise::xNextLong;
using climate_noise::xSetSeed;
using climate_noise::xPerlinInit;

namespace geode_layout::climate263_model {
enum class Op { Constant, Add, Mul, Abs, Lerp, ShiftA, ShiftB, Shifted, Spline, YGradient };

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

#include "climate263_data.hpp"

namespace geode_layout {
using namespace climate263_model;

struct Climate263::Impl {
    struct Octave {
        PerlinNoise noise;
        double frequency;
        float weight;
    };

    struct Normal {
        std::vector<Octave> first, second;

        Normal(const NoiseSpec &spec, Xoroshiro seed) {
            for (auto *out : {&first, &second}) {
                uint64_t lo = xNextLong(&seed), hi = xNextLong(&seed);
                double frequency = std::ldexp(1.0, spec.first);
                for (int i = 0; i < int(spec.amplitudes.size()); ++i, frequency *= 2)
                    if (spec.amplitudes[i] != 0) {
                        auto *hash = octave_hashes[spec.first + i + 16];
                        Xoroshiro random{lo ^ hash[0], hi ^ hash[1]};
                        Octave o{};
                        xPerlinInit(&o.noise, &random);
                        o.frequency = frequency;
                        o.weight = float(spec.amplitudes[i]);
                        out->push_back(o);
                    }
            }
        }

        float sample(double x, double y, double z) const {
            float result = 0;
            for (size_t i = 0; i < first.size(); ++i) {
                auto sample = [&](const Octave &o, double scale) {
                    double f = o.frequency * scale;
                    return noise263::perlin(o.noise.d, o.noise.a, o.noise.b, o.noise.c, x * f, y * f, z * f);
                };
                result += first[i].weight * sample(first[i], 1);
                result += second[i].weight * sample(second[i], 1.0181268882175227);
            }
            return result;
        }
    };

    struct Entry {
        int x = 0, y = 0, z = 0;
        float value = 0;
        bool valid = false;
    };

    std::vector<Normal> fields;
    std::vector<Entry> cache;

    explicit Impl(uint64_t seed) : cache(nodes.size()) {
        Xoroshiro random;
        xSetSeed(&random, seed);
        uint64_t lo = xNextLong(&random), hi = xNextLong(&random);
        for (auto &spec : noises)
            fields.emplace_back(spec, Xoroshiro{lo ^ spec.lo, hi ^ spec.hi});
    }

    float eval(int id, int x, int y, int z) {
        auto &n = nodes[id];
        if (n.op == Op::Constant)
            return float(n.p);
        auto &e = cache[id];
        if (e.valid && e.x == x && e.y == y && e.z == z)
            return e.value;
        auto a = [&] {
            return eval(n.a, x, y, z);
        };
        auto b = [&] {
            return eval(n.b, x, y, z);
        };
        float result = 0;
        switch (n.op) {
            case Op::Constant:
                result = float(n.p);
                break;
            case Op::Add:
                result = a() + b();
                break;
            case Op::Mul:
                result = a() * b();
                break;
            case Op::Abs:
                result = std::abs(a());
                break;
            case Op::Lerp: {
                float t = a();
                result = t == 0 ? b() : t == 1 ? eval(n.c, x, y, z) : noise263::lerp(t, b(), eval(n.c, x, y, z));
                break;
            }
            case Op::YGradient:
                result = float(n.r) + float(std::clamp(y, int(std::min(n.p, n.q)), int(std::max(n.p, n.q))) - n.p) *
                                          ((float(n.s) - float(n.r)) / float(n.q - n.p));
                break;
            case Op::ShiftA:
                result = fields[n.a].sample(x * 0.25, 0, z * 0.25) * 4;
                break;
            case Op::ShiftB:
                result = fields[n.a].sample(z * 0.25, x * 0.25, 0) * 4;
                break;
            case Op::Shifted:
                result = fields[n.a].sample(x * n.p + eval(n.b, x, y, z), y * n.q + eval(n.c, x, y, z),
                                            z * n.p + eval(n.d, x, y, z));
                break;
            case Op::Spline: {
                float input = a();
                auto *begin = points.data() + n.b;
                int count = n.c, index = -1;
                while (index + 1 < count && input >= begin[index + 1].location)
                    ++index;
                auto value = [&](int i) {
                    return eval(begin[i].value, x, y, z);
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
                    result = noise263::lerp(t, v0, v1) + t * (1 - t) * noise263::lerp(t, d0, d1);
                }
                break;
            }
        }
        e = {x, y, z, result, true};
        return result;
    }
};

Climate263::Climate263(uint64_t seed) : impl_(std::make_unique<Impl>(seed)) {}

Climate263::~Climate263() = default;

std::pair<float, float> Climate263::sample(int x, int y, int z) {
    return {impl_->eval(erosion, x, y, z), impl_->eval(depth, x, y, z)};
}

bool Climate263::allowed(int x, int y, int z) {
    return impl_->eval(erosion, x, y, z) >= -0.225f || impl_->eval(depth, x, y, z) <= 0.9f;
}
}
