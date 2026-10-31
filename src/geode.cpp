#include "geode.hpp"
#include "terrain.hpp"
#include "extensions/java263_noise.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <utility>

namespace geode {
int floor_div(int x, int d) {
    int q = x / d;
    return q - (x % d < 0);
}

uint64_t key(int x, int z) {
    return (uint64_t(uint32_t(x)) << 32) | uint32_t(z);
}

int key_x(uint64_t v) {
    return std::bit_cast<int32_t>(uint32_t(v >> 32));
}

int key_z(uint64_t v) {
    return std::bit_cast<int32_t>(uint32_t(v));
}

bool in_world(int x, int z) {
    return x >= world_min_chunk && x <= world_max_chunk && z >= world_min_chunk && z <= world_max_chunk;
}

static uint64_t mix(uint64_t v) {
    v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ULL;
    v = (v ^ (v >> 27)) * 0x94D049BB133111EBULL;
    return v ^ (v >> 31);
}

void Random::seed_with(uint64_t seed) {
    uint64_t v = seed ^ 0x6A09E667F3BCC909ULL;
    low_ = mix(v);
    high_ = mix(v + 0x9E3779B97F4A7C15ULL);
}

uint32_t Random::bits(int n) {
    uint64_t a = low_, b = high_, v = a ^ b;
    low_ = std::rotl(a, 49) ^ v ^ (v << 21);
    high_ = std::rotl(v, 28);
    return uint32_t((std::rotl(a + b, 17) + a) >> (64 - n));
}

int Random::bounded(int bound) {
    if ((bound & -bound) == bound)
        return int((uint64_t(bound) * bits(31)) >> 31);
    uint32_t sample, value;
    do {
        sample = bits(31);
        value = sample % uint32_t(bound);
    } while (sample - value + uint32_t(bound - 1) >= 0x80000000U);
    return int(value);
}

uint64_t Random::java_long() {
    uint64_t hi = uint64_t(bits(32)) << 32;
    return hi + uint64_t(int64_t(std::bit_cast<int32_t>(bits(32))));
}

float Random::next_float() {
    return bits(24) * 0x1.0p-24f;
}

double Random::next_double() {
    uint64_t hi = bits(26), lo = bits(27);
    return ((hi << 27) + lo) * 0x1.0p-53;
}

namespace {
struct Legacy {
    uint64_t state;

    explicit Legacy(uint64_t seed) : state((seed ^ 0x5DEECE66DULL) & ((1ULL << 48) - 1)) {}

    uint32_t bits(int n) {
        state = (state * 0x5DEECE66DULL + 11) & ((1ULL << 48) - 1);
        return uint32_t(state >> (48 - n));
    }

    int bounded(int n) {
        if ((n & -n) == n)
            return int((uint64_t(n) * bits(31)) >> 31);
        uint32_t s, r;
        do {
            s = bits(31);
            r = s % uint32_t(n);
        } while (s - r + uint32_t(n - 1) >= 0x80000000U);
        return int(r);
    }

    uint64_t next_long() {
        uint64_t hi = uint64_t(bits(32)) << 32;
        return hi + uint64_t(int64_t(std::bit_cast<int32_t>(bits(32))));
    }

    double next_double() {
        uint64_t hi = bits(26), lo = bits(27);
        return ((hi << 27) + lo) * 0x1.0p-53;
    }

    Legacy fork() {
        return Legacy(440898198ULL ^ next_long());
    }
};

double lerp(double t, double a, double b) {
    return a + t * (b - a);
}

struct Axis {
    int cell;
    double local, smooth;
};

struct Interval {
    double lo, hi;
};

Interval lerp_bound(double lo, double hi, Interval a, Interval b) {
    return {std::min(lerp(lo, a.lo, b.lo), lerp(hi, a.lo, b.lo)) - 1e-14,
            std::max(lerp(lo, a.hi, b.hi), lerp(hi, a.hi, b.hi)) + 1e-14};
}

Axis axis(double v) {
    int i = int(std::floor(v));
    double t = v - i;
    return {i, t, t * t * t * (t * (t * 6 - 15) + 10)};
}

struct Noise {
    std::array<uint8_t, 256> permutation{};
    double dx, dy, dz;

    explicit Noise(Legacy random) {
        dx = random.next_double() * 256;
        dy = random.next_double() * 256;
        dz = random.next_double() * 256;
        for (int i = 0; i < 256; ++i)
            permutation[i] = uint8_t(i);
        for (int i = 0; i < 256; ++i)
            std::swap(permutation[i], permutation[i + random.bounded(256 - i)]);
    }

    int p(int i) const {
        return permutation[i & 255];
    }

    static double gradient(int hash, double x, double y, double z) {
        static constexpr int g[16][3] = {{1, 1, 0},  {-1, 1, 0},  {1, -1, 0}, {-1, -1, 0}, {1, 0, 1},  {-1, 0, 1},
                                         {1, 0, -1}, {-1, 0, -1}, {0, 1, 1},  {0, -1, 1},  {0, 1, -1}, {0, -1, -1},
                                         {1, 1, 0},  {0, -1, 1},  {-1, 1, 0}, {0, -1, -1}};
        auto &a = g[hash & 15];
        return a[0] * x + a[1] * y + a[2] * z;
    }

    double sample(Axis x, Axis y, Axis z) const {
        int px = p(x.cell), pX = p(x.cell + 1), xy = p(px + y.cell), Xy = p(pX + y.cell), xY = p(px + y.cell + 1),
            XY = p(pX + y.cell + 1);
        double a = x.local, b = y.local, c = z.local;
        return lerp(
            z.smooth,
            lerp(y.smooth, lerp(x.smooth, gradient(p(xy + z.cell), a, b, c), gradient(p(Xy + z.cell), a - 1, b, c)),
                 lerp(x.smooth, gradient(p(xY + z.cell), a, b - 1, c), gradient(p(XY + z.cell), a - 1, b - 1, c))),
            lerp(y.smooth,
                 lerp(x.smooth, gradient(p(xy + z.cell + 1), a, b, c - 1),
                      gradient(p(Xy + z.cell + 1), a - 1, b, c - 1)),
                 lerp(x.smooth, gradient(p(xY + z.cell + 1), a, b - 1, c - 1),
                      gradient(p(XY + z.cell + 1), a - 1, b - 1, c - 1))));
    }

    Interval bound(Axis x0, Axis x1, Axis y0, Axis y1, Axis z0, Axis z1) const {
        auto join = [](Interval a, Interval b) {
            return Interval{std::min(a.lo, b.lo), std::max(a.hi, b.hi)};
        };
        // 三格小块可能跨越噪声格点平面，须在平面处分段以保留有效上界。
        if (x0.cell != x1.cell)
            return join(bound(x0, {x0.cell, 1, 1}, y0, y1, z0, z1), bound({x1.cell, 0, 0}, x1, y0, y1, z0, z1));
        if (y0.cell != y1.cell)
            return join(bound(x0, x1, y0, {y0.cell, 1, 1}, z0, z1), bound(x0, x1, {y1.cell, 0, 0}, y1, z0, z1));
        if (z0.cell != z1.cell)
            return join(bound(x0, x1, y0, y1, z0, {z0.cell, 1, 1}), bound(x0, x1, y0, y1, {z1.cell, 0, 0}, z1));
        static constexpr int g[16][3] = {{1, 1, 0},  {-1, 1, 0},  {1, -1, 0}, {-1, -1, 0}, {1, 0, 1},  {-1, 0, 1},
                                         {1, 0, -1}, {-1, 0, -1}, {0, 1, 1},  {0, -1, 1},  {0, 1, -1}, {0, -1, -1},
                                         {1, 1, 0},  {0, -1, 1},  {-1, 1, 0}, {0, -1, -1}};
        Interval corners[2][2][2];
        for (int z = 0; z < 2; ++z)
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 2; ++x) {
                    auto &gradient = g[p(p(p(x0.cell + x) + y0.cell + y) + z0.cell + z) & 15];
                    double lo[3] = {x0.local - x, y0.local - y, z0.local - z},
                           hi[3] = {x1.local - x, y1.local - y, z1.local - z};
                    Interval v{0, 0};
                    for (int i = 0; i < 3; ++i) {
                        if (gradient[i] > 0) {
                            v.lo += lo[i];
                            v.hi += hi[i];
                        } else if (gradient[i] < 0) {
                            v.lo -= hi[i];
                            v.hi -= lo[i];
                        }
                    }
                    corners[z][y][x] = v;
                }
        Interval yz[2][2];
        for (int z = 0; z < 2; ++z)
            for (int y = 0; y < 2; ++y)
                yz[z][y] = lerp_bound(x0.smooth, x1.smooth, corners[z][y][0], corners[z][y][1]);
        return lerp_bound(z0.smooth, z1.smooth, lerp_bound(y0.smooth, y1.smooth, yz[0][0], yz[0][1]),
                          lerp_bound(y0.smooth, y1.smooth, yz[1][0], yz[1][1]));
    }
};

struct Field {
    Noise first, second;

    explicit Field(uint64_t seed) : Field(Legacy(seed)) {}

    explicit Field(Legacy random) : first(random.fork()), second(random.fork()) {}

    double sample263(int x, int y, int z) const {
        auto sample = [&](const Noise &n, double scale) {
            return cubiomes::java263::perlin(n.permutation.data(), n.dx, n.dy, n.dz, x * scale / 16.0, y * scale / 16.0,
                                    z * scale / 16.0);
        };
        float a = float(5.0 / 6.0) * sample(first, 1), b = float(5.0 / 6.0) * sample(second, 1.0181268882175227);
        return double(a + b) * 0.05;
    }

    struct Grid {
        std::array<Axis, 33> x, y, z;
    };

    static Grid grid(const Noise &noise, int x, int y, int z, double factor) {
        auto wrap = [](double v) {
            return v - std::floor(v / 67108864.0 + 0.5) * 67108864.0;
        };
        Grid result;
        for (int i = 0; i < 33; ++i) {
            result.x[i] = axis(wrap((x - 16 + i) * factor / 16.0) + noise.dx);
            result.y[i] = axis(wrap((y - 16 + i) * factor / 16.0) + noise.dy);
            result.z[i] = axis(wrap((z - 16 + i) * factor / 16.0) + noise.dz);
        }
        return result;
    }
};

const std::array<double, 4096> &inverse_sqrt(bool legacy, bool half_offset) {
    auto build = [](bool old, bool half) {
        std::array<double, 4096> a{};
        for (int i = 1; i < 4096; ++i) {
            // 1.18/1.18.1从方块中心计算距离：(dx+0.5)^2三项的常数和为0.75。
            double v = double(i) + (half ? 0.75 : 0);
            if (old) {
                // 1.19.3及以前使用一次牛顿迭代；改成精确平方根会改变壳层和后续RNG。
                double estimate =
                    std::bit_cast<double>(6910469410427058090ULL - (std::bit_cast<uint64_t>(v) >> 1));
                a[i] = estimate * (1.5 - 0.5 * v * estimate * estimate);
            } else {
                a[i] = 1.0 / std::sqrt(v);
            }
        }
        return a;
    };
    if (half_offset) {
        static const auto table = build(true, true);
        return table;
    }
    if (legacy) {
        static const auto table = build(true, false);
        return table;
    }
    static const auto table = build(false, false);
    return table;
}

const std::vector<double> &radial_fields(bool legacy, bool half_offset) {
    auto build = [](bool old, bool half) {
        std::vector<double> a(54 * 33 * 33 * 33);
        auto &inv = inverse_sqrt(old, half);
        for (int px = 4; px <= 6; ++px)
            for (int py = 4; py <= 6; ++py)
                for (int pz = 4; pz <= 6; ++pz)
                    for (int offset = 1; offset <= 2; ++offset) {
                        int field = (((px - 4) * 3 + py - 4) * 3 + pz - 4) * 2 + offset - 1;
                        size_t i = size_t(field) * 33 * 33 * 33;
                        for (int z = -16; z <= 16; ++z)
                            for (int y = -16; y <= 16; ++y)
                                for (int x = -16; x <= 16; ++x) {
                                    int dx = x - px, dy = y - py, dz = z - pz;
                                    a[i++] = inv[dx * dx + dy * dy + dz * dz + offset + (half ? dx + dy + dz : 0)];
                                }
                    }
        return a;
    };
    if (half_offset) {
        static const auto table = build(true, true);
        return table;
    }
    if (legacy) {
        static const auto table = build(true, false);
        return table;
    }
    static const auto table = build(false, false);
    return table;
}
}

struct Simulator::Impl {
    uint64_t seed, x_scale, z_scale;
    bool float_noise, legacy_shape, half_offset;
    Random random;
    Field field;

    explicit Impl(uint64_t s, int generation)
        : seed(s), float_noise(generation >= 5), legacy_shape(generation < 0), half_offset(generation == -2),
          random(s), field(s) {
        x_scale = (random.java_long() | 1) * 16;
        z_scale = (random.java_long() | 1) * 16;
    }

    void reset(int x, int z) {
        uint64_t v = (uint64_t(int64_t(x)) * x_scale + uint64_t(int64_t(z)) * z_scale) ^ seed;
        random.seed_with(v + 20002);
    }
};

Simulator::Simulator(uint64_t seed, int generation) : impl_(std::make_unique<Impl>(seed, generation)) {}

Simulator::~Simulator() = default;
Simulator::Simulator(Simulator &&) noexcept = default;
Simulator &Simulator::operator=(Simulator &&) noexcept = default;

bool Simulator::check(int x, int z, bool old) {
    impl_->reset(x, z);
    return impl_->random.bits(24) < (old ? 699050U : 699051U);
}

uint32_t Simulator::placement_bits(int x, int z) {
    impl_->reset(x, z);
    return impl_->random.bits(24);
}

void Simulator::placement_row(int x, int z, int count, uint32_t *output) const {
    std::fill(output, output + count, 0xffffffffU);
    if (z < world_min_chunk || z > world_max_chunk)
        return;
    int lo = std::clamp(world_min_chunk - x, 0, count), hi = std::clamp(world_max_chunk + 1 - x, 0, count);
    uint64_t position = uint64_t(int64_t(x + lo)) * impl_->x_scale + uint64_t(int64_t(z)) * impl_->z_scale;
    // 只需第一次随机抽取；递推区块种子的仿射变换，避免重建完整RNG状态。
    for (int i = lo; i < hi; ++i, position += impl_->x_scale) {
        uint64_t v = ((position ^ impl_->seed) + 20002) ^ 0x6A09E667F3BCC909ULL;
        uint64_t a = mix(v), b = mix(v + 0x9E3779B97F4A7C15ULL);
        output[i] = uint32_t((std::rotl(a + b, 17) + a) >> 40);
    }
}

GeodeShape Simulator::describe(int cx, int cz) {
    impl_->reset(cx, cz);
    auto &r = impl_->random;
    GeodeShape result;
    if (r.next_float() >= 1.0f / 24.0f)
        return result;
    result.exists = true;
    result.x = r.bounded(16) + cx * 16;
    result.z = r.bounded(16) + cz * 16;
    result.y = r.bounded(89) - 58;
    int n = result.count = r.bounded(2) + 3;
    double adjustment = n / 6.0;
    result.crack_limit = 1 / std::sqrt(2 + r.next_double() / 2 + (n > 3 ? adjustment : 0));
    result.cracked = r.next_float() < 0.95f;
    for (int i = 0; i < n; ++i) {
        auto &p = result.points[i];
        p.x = r.bounded(3) + 4;
        p.y = r.bounded(3) + 4;
        p.z = r.bounded(3) + 4;
        p.offset = r.bounded(2) + 1;
    }
    if (result.cracked) {
        int k = r.bounded(4), o = n * 2 + 1;
        result.crack_x = (k == 0 || k == 2) ? o : 0;
        result.crack_z = (k == 1 || k == 2) ? o : 0;
    }
    return result;
}

Material Simulator::material(const GeodeShape &s, Block point) const {
    if (!s.exists)
        return Material::Unchanged;
    int x = point.x - s.x, y = point.y - s.y, z = point.z - s.z;
    if (x < -16 || x > 16 || y < -16 || y > 16 || z < -16 || z > 16)
        return Material::Unchanged;
    auto sample = [&](const Noise &noise, double factor) {
        auto coord = [&](int p, double offset) {
            double v = p * factor / 16.0;
            v -= std::floor(v / 67108864.0 + 0.5) * 67108864.0;
            return axis(v + offset);
        };
        return noise.sample(coord(point.x, noise.dx), coord(point.y, noise.dy), coord(point.z, noise.dz));
    };
    double noise =
        impl_->float_noise
            ? impl_->field.sample263(point.x, point.y, point.z)
            : (sample(impl_->field.first, 1) + sample(impl_->field.second, 1.0181268882175227)) * (5.0 / 6.0) * 0.05;
    auto &inv = inverse_sqrt(impl_->legacy_shape, impl_->half_offset);
    double shell = 0;
    for (int i = 0; i < s.count; ++i) {
        auto p = s.points[i];
        int dx = x - p.x, dy = y - p.y, dz = z - p.z;
        shell += inv[dx * dx + dy * dy + dz * dz + p.offset + (impl_->half_offset ? dx + dy + dz : 0)];
    }
    if (impl_->float_noise || impl_->legacy_shape) {
        shell = 0;
        for (int i = 0; i < s.count; ++i) {
            auto p = s.points[i];
            int dx = x - p.x, dy = y - p.y, dz = z - p.z;
            shell += inv[dx * dx + dy * dy + dz * dz + p.offset + (impl_->half_offset ? dx + dy + dz : 0)] + noise;
        }
    } else
        shell += noise * s.count;
    if (shell < 1 / std::sqrt(4.2 + s.count / 6.0))
        return Material::Unchanged;
    if (shell >= 1 / std::sqrt(1.7))
        return Material::Air;
    if (s.cracked) {
        double crack = 0;
        int dx = x - s.crack_x, dz = z - s.crack_z;
        for (int height : {7, 5, 1}) {
            int dy = y - height;
            crack += inv[dx * dx + dy * dy + dz * dz + 2 + (impl_->half_offset ? dx + dy + dz : 0)];
        }
        crack += noise * 3;
        if (crack >= s.crack_limit)
            return Material::Air;
    }
    return Material::Solid;
}

Simulation Simulator::generate(int cx, int cz, Terrain *terrain) {
    auto shape = describe(cx, cz);
    Simulation result;
    if (!shape.exists)
        return result;
    result.exists = true;
    result.x = shape.x;
    result.y = shape.y;
    result.z = shape.z;
    auto &r = impl_->random;
    int n = shape.count;
    double adjustment = n / 6.0;
    double amethyst = 1 / std::sqrt(2.2 + adjustment), basalt = 1 / std::sqrt(4.2 + adjustment),
           air = 1 / std::sqrt(1.7);
    auto &points = shape.points;
    bool cracked = shape.cracked;
    double crack_limit = shape.crack_limit;
    int invalid = 0;
    for (int i = 0; i < n; ++i) {
        auto &p = points[i];
        // 原版遇到第二个无效控制点时会取消整座晶洞。
        if (terrain && !terrain->solid(result.x + p.x, result.y + p.y, result.z + p.z) && ++invalid > 1) {
            result.exists = false;
            result.rejected = true;
            return result;
        }
    }
    int crack_x = shape.crack_x, crack_z = shape.crack_z;
    auto a = Field::grid(impl_->field.first, result.x, result.y, result.z, 1);
    auto b = Field::grid(impl_->field.second, result.x, result.y, result.z, 1.0181268882175227);
    // 按格点平面分段后，五格小块可平衡区间计算成本与排除效果。
    constexpr int noise_tile = 5, tiles = (33 + noise_tile - 1) / noise_tile;
    Interval noise_bounds[tiles][tiles][tiles];
    for (int z = 0; z < tiles; ++z)
        for (int y = 0; y < tiles; ++y)
            for (int x = 0; x < tiles; ++x) {
                int x0 = x * noise_tile, y0 = y * noise_tile, z0 = z * noise_tile,
                    x1 = std::min(32, x0 + noise_tile - 1), y1 = std::min(32, y0 + noise_tile - 1),
                    z1 = std::min(32, z0 + noise_tile - 1);
                auto first = impl_->field.first.bound(a.x[x0], a.x[x1], a.y[y0], a.y[y1], a.z[z0], a.z[z1]);
                auto second = impl_->field.second.bound(b.x[x0], b.x[x1], b.y[y0], b.y[y1], b.z[z0], b.z[z1]);
                // 26.3的float插值需要比double路径更宽的舍入边界。
                double epsilon = impl_->float_noise ? 1e-5 : 1e-12;
                noise_bounds[z][y][x] = {(first.lo + second.lo) * (5.0 / 6.0) * 0.05 - epsilon,
                                         (first.hi + second.hi) * (5.0 / 6.0) * 0.05 + epsilon};
            }
    double row_noise_hi[tiles][tiles];
    for (int z = 0; z < tiles; ++z)
        for (int y = 0; y < tiles; ++y) {
            double hi = -2;
            for (auto bound : noise_bounds[z][y])
                hi = std::max(hi, bound.hi);
            row_noise_hi[z][y] = hi;
        }
    auto &inv = inverse_sqrt(impl_->legacy_shape, impl_->half_offset);
    result.budding.reserve(96);
    const double *radial[4]{};
    auto &fields = radial_fields(impl_->legacy_shape, impl_->half_offset);
    for (int i = 0; i < n; ++i) {
        auto p = points[i];
        int field = (((p.x - 4) * 3 + p.y - 4) * 3 + p.z - 4) * 2 + p.offset - 1;
        radial[i] = fields.data() + size_t(field) * 33 * 33 * 33;
    }
    for (int z = -16; z <= 16; ++z)
        for (int y = -16; y <= 16; ++y) {
            const size_t row = size_t((z + 16) * 33 + y + 16) * 33;
            double shells[33];
            double noise_hi = row_noise_hi[(z + 16) / noise_tile][(y + 16) / noise_tile];
            // 各径向项在自己的控制点X处取峰值，因此其和可约束整行。
            double maximum = 0;
            for (int i = 0; i < n; ++i)
                maximum += radial[i][row + points[i].x + 16];
            if (maximum + noise_hi * n < amethyst)
                continue;
            if (n == 3)
                for (int x = 0; x < 33; ++x)
                    shells[x] = (radial[0][row + x] + radial[1][row + x]) + radial[2][row + x];
            else
                for (int x = 0; x < 33; ++x)
                    shells[x] = ((radial[0][row + x] + radial[1][row + x]) + radial[2][row + x]) + radial[3][row + x];
            for (int x = -16; x <= 16; ++x) {
                double shell = shells[x + 16];
                // 区间上界只跳过不可能触发母岩随机抽取的位置。
                auto bound = noise_bounds[(z + 16) / noise_tile][(y + 16) / noise_tile][(x + 16) / noise_tile];
                if (shell + bound.hi * n < amethyst || shell + bound.lo * n >= air)
                    continue;
                double noise = impl_->float_noise
                                   ? impl_->field.sample263(result.x + x, result.y + y, result.z + z)
                                   : (impl_->field.first.sample(a.x[x + 16], a.y[y + 16], a.z[z + 16]) +
                                      impl_->field.second.sample(b.x[x + 16], b.y[y + 16], b.z[z + 16])) *
                                         (5.0 / 6.0) * 0.05;
                if (impl_->float_noise || impl_->legacy_shape) {
                    shell = 0;
                    for (int i = 0; i < n; ++i)
                        shell += radial[i][row + x + 16] + noise;
                } else
                    shell += noise * n;
                if (shell < basalt || shell >= air)
                    continue;
                double crack = 0;
                if (cracked) {
                    int dx = x - crack_x, dz = z - crack_z;
                    for (int height : {7, 5, 1}) {
                        int dy = y - height;
                        crack += inv[dx * dx + dy * dy + dz * dz + 2 + (impl_->half_offset ? dx + dy + dz : 0)];
                    }
                    crack += noise * 3;
                }
                if (cracked && crack >= crack_limit)
                    continue;
                if (shell >= amethyst && r.next_float() < 0.083) {
                    result.budding.push_back({result.x + x, result.y + y, result.z + z});
                    r.bits(32);
                }
            }
        }
    if (terrain)
        terrain->filter_budding(result, cx, cz);
    return result;
}
}
