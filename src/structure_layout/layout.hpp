#pragma once
#include "climate263.hpp"
#include <array>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <vector>
struct Generator;
extern "C" Generator *geode_biomes_create(uint64_t seed);
extern "C" void geode_biomes_free(Generator *generator);
extern "C" int geode_biomes_allowed(Generator *generator, int x, int y, int z);

namespace geode_layout {
struct Vec3 {
    int x{}, y{}, z{};
    auto operator<=>(const Vec3 &) const = default;

    Vec3 operator+(Vec3 b) const {
        return {x + b.x, y + b.y, z + b.z};
    }

    Vec3 operator-(Vec3 b) const {
        return {x - b.x, y - b.y, z - b.z};
    }
};

struct Point {
    int x{}, z{};
    auto operator<=>(const Point &) const = default;
};

struct Box {
    Vec3 lo, hi;

    Box moved(Vec3 p) const {
        return {lo + p, hi + p};
    }

    bool contains(Vec3 p) const {
        return p.x >= lo.x && p.x <= hi.x && p.y >= lo.y && p.y <= hi.y && p.z >= lo.z && p.z <= hi.z;
    }

    bool contains(Box b) const {
        return contains(b.lo) && contains(b.hi);
    }

    bool intersects(Box b) const {
        return hi.x >= b.lo.x && lo.x <= b.hi.x && hi.y >= b.lo.y && lo.y <= b.hi.y && hi.z >= b.lo.z && lo.z <= b.hi.z;
    }
};

struct Connector {
    Vec3 pos;
    int facing, top, name, target, pool, rollable, selection, placement;
    int match_id{};
};

struct Template {
    Vec3 size;
    std::vector<Connector> connectors;
    std::vector<Vec3> spawners;
};

struct Pool {
    int fallback;
    std::vector<int> elements;
};

struct AliasChoice {
    int weight;
    std::vector<std::array<int, 2>> mappings;
};

struct AliasGroup {
    std::vector<AliasChoice> choices;
};

struct LayoutData {
    std::vector<Template> templates;
    std::vector<Pool> pools;
    std::vector<AliasGroup> aliases;
    int start_pool{}, depth{}, distance{}, padding_bottom{}, padding_top{}, min_y{}, max_y{};
    int start_anchor = -1;
    bool constant_y = false, deep_dark = false;
};

class Random {
    uint64_t state_{};
    inline static constexpr auto bounds_ = [] {
        std::array<std::array<uint32_t, 2>, 1025> values{};
        for (uint64_t n = 1; n < values.size(); ++n)
            values[n] = {uint32_t((1ULL << 32) / n), uint32_t((1ULL << 31) / n * n)};
        return values;
    }();
    inline static constexpr auto four_steps_ = [] {
        std::array<uint64_t, 2> affine{1, 0};
        for (int i = 0; i < 4; ++i) {
            affine[0] *= 0x5deece66dULL;
            affine[1] = affine[1] * 0x5deece66dULL + 11;
        }
        return affine;
    }();

  public:
    explicit Random(uint64_t seed = 0) {
        seed_with(seed);
    }

    void seed_with(uint64_t seed) {
        state_ = (seed ^ 0x5deece66dULL) & ((1ULL << 48) - 1);
    }

    uint32_t bits(int n) {
        state_ = (state_ * 0x5deece66dULL + 11) & ((1ULL << 48) - 1);
        return uint32_t(state_ >> (48 - n));
    }

    int integer(int bound) {
        if ((bound & -bound) == bound)
            return int((uint64_t(bound) * bits(31)) >> 31);
        uint32_t b, r;
        if (bound < int(bounds_.size())) {
            const auto &info = bounds_[bound];
            do {
                b = bits(31);
            } while (b >= info[1]);
            // 对31位输入，向下取整倒数对商的低估至多为一。
            r = b - uint32_t((uint64_t(b) * info[0]) >> 32) * uint32_t(bound);
            return int(r >= uint32_t(bound) ? r - bound : r);
        }
        do {
            b = bits(31);
            r = b % uint32_t(bound);
        } while (b - r + uint32_t(bound - 1) >= 0x80000000U);
        return int(r);
    }

    uint64_t long_value() {
        auto hi = bits(32);
        auto lo = int32_t(bits(32));
        return (uint64_t(hi) << 32) + uint64_t(int64_t(lo));
    }

    template <class Container> void shuffle(Container &values) {
        for (int n = int(values.size()); n > 1; --n)
            std::swap(values[n - 1], values[integer(n)]);
    }

    void consume_shuffle(int size) {
        for (int n = size; n > 1; --n) {
            if (n < int(bounds_.size())) {
                while (bits(31) >= bounds_[n][1]) {
                }
            } else
                integer(n);
        }
    }

    void consume_four_shuffles(int size) {
        if (size <= 1)
            return;
        // 二元素洗牌每次只抽一次且无拒绝路径；四次LCG可合成一次仿射更新。
        if (size == 2) {
            state_ = (state_ * four_steps_[0] + four_steps_[1]) & ((1ULL << 48) - 1);
            return;
        }
        for (int i = 0; i < 4; ++i)
            consume_shuffle(size);
    }
};

int floor_div(int x, int denominator);
Point placement(uint64_t seed, int region_x, int region_z);
Point city_placement(uint64_t seed, int region_x, int region_z);

struct Variant {
    Box box;
    std::vector<Connector> connectors;
    std::vector<Vec3> spawners;
    // -1表示无匹配，-2表示多匹配，否则为唯一匹配的连接器下标。
    std::vector<int> matches;
    bool uniform_priority{};
    void connector_order(Random &random, std::vector<int> &order) const;
};

struct Tables {
    LayoutData data;
    std::vector<std::array<Variant, 4>> variants;
    explicit Tables(LayoutData source);
};

const Tables &tables(int generation);

struct Placed {
    int id, rotation;
    Vec3 pos;
    Box box;
    int ground_delta = 1;
    std::vector<Vec3> junctions;
};

struct Chamber {
    bool exists{};
    Vec3 biome_position{};
    std::vector<Vec3> spawners;
    int pieces{};
    std::vector<Placed> placed;
};

class LayoutEngine {
    struct Space {
        Box bounds;
        bool grid;
        int nx{}, ny{}, nz{}, query{};

        struct Blocked {
            Box box;
            int last_query{};
        };

        std::vector<Blocked> blocked;
        std::vector<std::vector<int>> cells;
        Space(Box box, bool indexed);
        void reset(Box box, bool indexed);
        void occupy(Box box);
        bool fits(Box box);

        int index(int x, int y, int z) const {
            return (x * ny + y) * nz + z;
        }

        std::array<int, 3> cell(Vec3 p) const;
    };

    const Tables &tables_;
    uint64_t seed_;
    std::unique_ptr<Generator, void (*)(Generator *)> biomes_;
    std::unique_ptr<Climate263> climate263_;
    std::deque<Space> spaces_;
    std::vector<int> aliases_, parents_, connectors_, candidates_, fallbacks_;

    struct Rejected {
        uint64_t attempt{};
        unsigned rotations{};
    };

    std::vector<Rejected> rejected_;
    uint64_t attempt_{};

    struct Start {
        int id = -1, rotation = 0;
        Vec3 pos, biome;
        Box box;
    };

    Start start(int chunk_x, int chunk_z, Random &random);

  public:
    LayoutEngine(int generation, uint64_t seed);
    Box possible_bounds(int chunk_x, int chunk_z);
    Chamber generate(int chunk_x, int chunk_z);
};
}
