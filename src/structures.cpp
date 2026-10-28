#include "structures.hpp"
#include "geode.hpp"
#include "structure_layout/layout.hpp"
#include <algorithm>
#include <unordered_map>
#include <array>
#include <bit>
#include <cmath>
#include <optional>

namespace geode {
namespace {
struct Mask {
    geode_layout::Vec3 size;
    std::vector<std::array<int, 2>> holes;
};

#include "structure_masks.hpp"

geode_layout::Vec3 rotate(geode_layout::Vec3 p, int r) {
    switch (r) {
        case 1:
            return {-p.z, p.y, p.x};
        case 2:
            return {-p.x, p.y, -p.z};
        case 3:
            return {p.z, p.y, -p.x};
        default:
            return p;
    }
}

double contribution(int x, int y, int z, int signed_y) {
    if (x < -12 || x >= 12 || y < -12 || y >= 12 || z < -12 || z >= 12)
        return 0;
    static const auto kernel = [] {
        std::array<float, 24 * 24 * 24> values{};
        for (int z = -12; z < 12; ++z)
            for (int x = -12; x < 12; ++x)
                for (int y = -12; y < 12; ++y)
                    values[(z + 12) * 576 + (x + 12) * 24 + y + 12] =
                        float(std::pow(std::exp(1.0), -(x * x + (y + 0.5) * (y + 0.5) + z * z) / 16));
        return values;
    }();
    double d = signed_y + 0.5, e = (x * x + d * d + z * z) / 2, half = e * 0.5;
    double inverse = std::bit_cast<double>(6910469410427058090ULL - (std::bit_cast<uint64_t>(e) >> 1));
    inverse *= 1.5 - half * inverse * inverse;
    return (-d * inverse / 2) * kernel[(z + 12) * 576 + (x + 12) * 24 + y + 12];
}
}

struct Structures::Impl {
    uint64_t seed;
    int generation;
    geode_layout::LayoutEngine engine;
    const std::vector<Mask> &masks;
    geode_layout::LayoutEngine city;

    struct Start {
        geode_layout::Point position;
        geode_layout::Box bounds;
        std::optional<geode_layout::Chamber> layout;
    };

    std::unordered_map<uint64_t, Start> regions;
    std::unordered_map<uint64_t, std::vector<geode_layout::Placed>> chunks;
    std::unordered_map<uint64_t, Start> cities;
    std::unordered_map<uint64_t, std::vector<geode_layout::Placed>> beard_chunks, layout_beard_chunks;

    Impl(uint64_t s, int g)
        : seed(s), generation(g), engine(g, s), masks(g == 1 ? legacy_masks : modern_masks), city(g >= 5 ? 6 : 3, s) {}

    const std::vector<geode_layout::Placed> &beard_pieces(int cx, int cz, bool ancient) {
        auto &cache = ancient ? beard_chunks : layout_beard_chunks;
        auto &starts = ancient ? cities : regions;
        auto &layout = ancient ? city : engine;
        int spacing = ancient ? 24 : 34;
        auto k = key(cx, cz);
        auto old = cache.find(k);
        if (old != cache.end())
            return old->second;
        // 这些是记忆缓存容量，不是搜索截断；淘汰的布局可相同地重算。
        if (cache.size() >= 8192)
            cache.clear();
        if (starts.size() >= 1024)
            starts.clear();
        std::vector<geode_layout::Placed> found;
        int rx = floor_div(cx, spacing), rz = floor_div(cz, spacing);
        geode_layout::Box padded{{cx * 16 - 12, -64, cz * 16 - 12}, {cx * 16 + 27, 319, cz * 16 + 27}};
        for (int x = rx - 1; x <= rx + 1; ++x)
            for (int z = rz - 1; z <= rz + 1; ++z) {
                auto region = key(x, z);
                auto existing = starts.find(region);
                if (existing == starts.end()) {
                    auto p = ancient ? geode_layout::city_placement(seed, x, z) : geode_layout::placement(seed, x, z);
                    existing =
                        starts
                            .emplace(region,
                                     Start{p,
                                           layout.possible_bounds(geode_layout::floor_div(p.x, 16), geode_layout::floor_div(p.z, 16)),
                                           {}})
                            .first;
                }
                auto &s = existing->second;
                if (!s.bounds.intersects(padded))
                    continue;
                if (!s.layout)
                    s.layout = layout.generate(geode_layout::floor_div(s.position.x, 16), geode_layout::floor_div(s.position.z, 16));
                for (const auto &source : s.layout->placed) {
                    if (source.box.hi.x < padded.lo.x || source.box.lo.x > padded.hi.x ||
                        source.box.hi.z < padded.lo.z || source.box.lo.z > padded.hi.z)
                        continue;
                    auto p = source;
                    std::erase_if(p.junctions, [&](auto j) {
                        return j.x <= padded.lo.x || j.x >= padded.hi.x || j.z <= padded.lo.z || j.z >= padded.hi.z;
                    });
                    found.push_back(std::move(p));
                }
            }
        return cache.emplace(k, std::move(found)).first->second;
    }

    const std::vector<geode_layout::Placed> &pieces(int cx, int cz) {
        auto k = key(cx, cz);
        auto old = chunks.find(k);
        if (old != chunks.end())
            return old->second;
        std::vector<geode_layout::Placed> found;
        if (chunks.size() >= 8192)
            chunks.clear();
        if (regions.size() >= 1024)
            regions.clear();
        int rx = floor_div(cx, 34), rz = floor_div(cz, 34);
        geode_layout::Box chunk{{cx * 16, -64, cz * 16}, {cx * 16 + 15, 319, cz * 16 + 15}};
        for (int x = rx - 1; x <= rx + 1; ++x)
            for (int z = rz - 1; z <= rz + 1; ++z) {
                auto region = key(x, z);
                auto existing = regions.find(region);
                if (existing == regions.end()) {
                    auto p = geode_layout::placement(seed, x, z);
                    existing =
                        regions
                            .emplace(region,
                                     Start{p,
                                           engine.possible_bounds(geode_layout::floor_div(p.x, 16), geode_layout::floor_div(p.z, 16)),
                                           {}})
                            .first;
                }
                auto &s = existing->second;
                if (!s.bounds.intersects(chunk))
                    continue;
                if (!s.layout)
                    s.layout = engine.generate(geode_layout::floor_div(s.position.x, 16), geode_layout::floor_div(s.position.z, 16));
                for (const auto &p : s.layout->placed)
                    if (p.box.intersects(chunk))
                        found.push_back(p);
            }
        return chunks.emplace(k, std::move(found)).first->second;
    }
};

Structures::Structures(uint64_t seed, int generation) : impl_(std::make_unique<Impl>(seed, generation)) {}

Structures::~Structures() = default;

bool Structures::trial_written(int x, int y, int z) {
    geode_layout::Vec3 world{x, y, z};
    for (const auto &p : impl_->pieces(floor_div(x, 16), floor_div(z, 16))) {
        if (!p.box.contains(world))
            continue;
        auto local = rotate(world - p.pos, (4 - p.rotation) & 3);
        auto &mask = impl_->masks[p.id];
        int index = local.x + mask.size.x * (local.z + mask.size.z * local.y);
        auto found = std::lower_bound(mask.holes.begin(), mask.holes.end(), index, [](const auto &hole, int point) {
            return hole[1] < point;
        });
        if (found == mask.holes.end() || (*found)[0] > index)
            return true;
    }
    return false;
}

double Structures::beard(int x, int y, int z) {
    // 1.18没有远古城市或试炼密室，不套用后续版本的结构地形适配。
    if (uses_118_terrain(impl_->generation))
        return 0;
    const auto &pieces = impl_->beard_pieces(floor_div(x, 16), floor_div(z, 16), true);
    double value = 0;
    for (const auto &p : pieces) {
        int dx = std::max({0, p.box.lo.x - x, x - p.box.hi.x}), dz = std::max({0, p.box.lo.z - z, z - p.box.hi.z}),
            ground = p.box.lo.y + p.ground_delta;
        int dy = std::max({0, ground - y, y - p.box.hi.y});
        value += contribution(dx, dy, dz, y - ground) * 0.8;
    }
    for (const auto &p : pieces)
        for (auto j : p.junctions)
            value += contribution(x - j.x, y - j.y, z - j.z, y - j.y) * 0.4;
    if (impl_->generation > 0) {
        const auto &trials = impl_->beard_pieces(floor_div(x, 16), floor_div(z, 16), false);
        for (const auto &p : trials) {
            double dx = std::max({0, p.box.lo.x - x, x - p.box.hi.x}) / 2.0,
                   dz = std::max({0, p.box.lo.z - z, z - p.box.hi.z}) / 2.0;
            double dy = std::max({0, p.box.lo.y - y, y - p.box.hi.y}) / 2.0;
            value += std::clamp(1 - std::sqrt(dx * dx + dy * dy + dz * dz) / 6, 0.0, 1.0) * 0.8;
        }
        for (const auto &p : trials)
            for (auto j : p.junctions)
                value += contribution(x - j.x, y - j.y, z - j.z, y - j.y) * 0.4;
    }
    return value;
}
}
