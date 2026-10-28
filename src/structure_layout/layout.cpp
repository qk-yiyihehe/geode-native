#include "layout.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace geode_layout {
#include "trial_data.hpp"
#include "city_data.hpp"

int floor_div(int x, int d) {
    int q = x / d;
    return q - (x % d < 0);
}

Point placement(uint64_t seed, int rx, int rz) {
    Random random(uint64_t(int64_t(rx)) * 341873128712ULL + uint64_t(int64_t(rz)) * 132897987541ULL + seed +
                  94251327ULL);
    int x = rx * 34 + random.integer(22), z = rz * 34 + random.integer(22);
    return {x * 16 + 8, z * 16 + 8};
}

Point city_placement(uint64_t seed, int rx, int rz) {
    Random random(uint64_t(int64_t(rx)) * 341873128712ULL + uint64_t(int64_t(rz)) * 132897987541ULL + seed +
                  20083232ULL);
    int x = rx * 24 + random.integer(16), z = rz * 24 + random.integer(16);
    return {x * 16 + 8, z * 16 + 8};
}

static Vec3 rotate(Vec3 p, int r) {
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

static constexpr Vec3 directions[] = {{0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}, {-1, 0, 0}, {1, 0, 0}};

static int rotate_direction(int d, int r) {
    auto p = rotate(directions[d], r);
    for (int i = 0; i < 6; ++i)
        if (p == directions[i])
            return i;
    return d;
}

Tables::Tables(LayoutData source) : data(std::move(source)) {
    std::map<std::tuple<int, int, int>, int> signatures;
    variants.resize(data.templates.size());
    for (size_t i = 0; i < data.templates.size(); ++i)
        for (int r = 0; r < 4; ++r) {
            const auto &t = data.templates[i];
            auto &v = variants[i][r];
            auto b = rotate(t.size - Vec3{1, 1, 1}, r);
            v.box = {{std::min(0, b.x), 0, std::min(0, b.z)}, {std::max(0, b.x), b.y, std::max(0, b.z)}};
            v.connectors = t.connectors;
            for (auto &c : v.connectors) {
                c.pos = rotate(c.pos, r);
                c.facing = rotate_direction(c.facing, r);
                c.top = rotate_direction(c.top, r);
                c.match_id =
                    signatures.try_emplace({c.target, c.facing, c.rollable ? -1 : c.top}, int(signatures.size()))
                        .first->second;
            }
            v.uniform_priority = std::all_of(v.connectors.begin(), v.connectors.end(), [&](const auto &c) {
                return c.selection == v.connectors.front().selection;
            });
            for (auto p : t.spawners)
                v.spawners.push_back(rotate(p, r));
        }
    for (auto &rotations : variants)
        for (auto &variant : rotations) {
            variant.matches.assign(signatures.size(), -1);
            for (auto [signature, id] : signatures) {
                auto [target, facing, top] = signature;
                for (int i = 0; i < int(variant.connectors.size()); ++i) {
                    const auto &c = variant.connectors[i];
                    if (c.name == target && c.facing == (facing ^ 1) && (top < 0 || top == c.top)) {
                        if (variant.matches[id] >= 0) {
                            variant.matches[id] = -2;
                            break;
                        }
                        variant.matches[id] = i;
                    }
                }
            }
        }
}

const Tables &tables(int generation) {
    if (generation == 3 || generation == 6) {
        static const Tables city(city_data());
        return city;
    }
    static const Tables legacy(legacy_data()), modern(modern_data());
    return generation == 1 ? legacy : modern;
}

LayoutEngine::Space::Space(Box box, bool indexed) {
    reset(box, indexed);
    blocked.reserve(indexed ? 256 : 8);
}

void LayoutEngine::Space::reset(Box box, bool indexed) {
    bounds = box;
    grid = indexed;
    query = 0;
    blocked.clear();
    if (grid) {
        nx = (box.hi.x - box.lo.x) / 16 + 1;
        ny = (box.hi.y - box.lo.y) / 16 + 1;
        nz = (box.hi.z - box.lo.z) / 16 + 1;
        cells.resize(nx * ny * nz);
        for (auto &c : cells)
            c.clear();
    }
}

std::array<int, 3> LayoutEngine::Space::cell(Vec3 p) const {
    return {std::clamp(floor_div(p.x - bounds.lo.x, 16), 0, nx - 1),
            std::clamp(floor_div(p.y - bounds.lo.y, 16), 0, ny - 1),
            std::clamp(floor_div(p.z - bounds.lo.z, 16), 0, nz - 1)};
}

void LayoutEngine::Space::occupy(Box box) {
    int id = int(blocked.size());
    blocked.push_back({box, 0});
    if (!grid)
        return;
    auto lo = cell(box.lo), hi = cell(box.hi);
    for (int x = lo[0]; x <= hi[0]; ++x)
        for (int y = lo[1]; y <= hi[1]; ++y)
            for (int z = lo[2]; z <= hi[2]; ++z)
                cells[index(x, y, z)].push_back(id);
}

bool LayoutEngine::Space::fits(Box box) {
    if (!bounds.contains(box))
        return false;
    if (!grid) {
        for (auto &b : blocked)
            if (b.box.intersects(box))
                return false;
        return true;
    }
    ++query;
    auto lo = cell(box.lo), hi = cell(box.hi);
    for (int x = lo[0]; x <= hi[0]; ++x)
        for (int y = lo[1]; y <= hi[1]; ++y)
            for (int z = lo[2]; z <= hi[2]; ++z)
                for (int id : cells[index(x, y, z)]) {
                    auto &b = blocked[id];
                    if (b.last_query == query)
                        continue;
                    b.last_query = query;
                    if (b.box.intersects(box))
                        return false;
                }
    return true;
}

LayoutEngine::LayoutEngine(int generation, uint64_t seed)
    : tables_(tables(generation)), seed_(seed), biomes_(geode_biomes_create(seed), geode_biomes_free),
      aliases_(tables_.data.pools.size()), rejected_(tables_.data.templates.size()) {
    if (generation >= 5)
        climate263_ = std::make_unique<Climate263>(seed);
    parents_.reserve(64);
    connectors_.reserve(64);
    candidates_.reserve(128);
    fallbacks_.reserve(32);
}

void Variant::connector_order(Random &random, std::vector<int> &order) const {
    order.resize(connectors.size());
    std::iota(order.begin(), order.end(), 0);
    random.shuffle(order);
    if (uniform_priority)
        return;
    for (size_t i = 1; i < order.size(); ++i) {
        int id = order[i];
        size_t j = i;
        while (j && connectors[order[j - 1]].selection < connectors[id].selection) {
            order[j] = order[j - 1];
            --j;
        }
        order[j] = id;
    }
}

LayoutEngine::Start LayoutEngine::start(int cx, int cz, Random &random) {
    const auto &data = tables_.data;
    auto a = random.long_value(), b = random.long_value();
    random.seed_with(uint64_t(int64_t(cx)) * a ^ uint64_t(int64_t(cz)) * b ^ seed_);
    int y = data.min_y + (data.constant_y ? 0 : random.integer(data.max_y - data.min_y + 1));
    Vec3 origin{cx * 16, y, cz * 16};
    Random positional(seed_);
    auto positional_seed = positional.long_value();
    // Java先按int乘X再扩宽，必须保留相同的溢出回绕行为。
    uint64_t hash = uint64_t(int64_t(int32_t(uint32_t(origin.x) * 3129871U))) ^
                    uint64_t(int64_t(origin.z)) * 116129781ULL ^ uint64_t(int64_t(origin.y));
    hash = hash * hash * 42317861ULL + hash * 11ULL;
    Random alias_random((hash >> 16) ^ positional_seed);
    auto &aliases = aliases_;
    std::iota(aliases.begin(), aliases.end(), 0);
    for (const auto &group : data.aliases) {
        int sum = 0;
        for (const auto &choice : group.choices)
            sum += choice.weight;
        int sample = alias_random.integer(sum);
        for (const auto &choice : group.choices) {
            sample -= choice.weight;
            if (sample < 0) {
                for (auto pair : choice.mappings)
                    aliases[pair[0]] = pair[1];
                break;
            }
        }
    }
    int rotation = random.integer(4);
    const auto &root_pool = data.pools[aliases[data.start_pool]];
    int root_id = root_pool.elements[random.integer(int(root_pool.elements.size()))];
    if (root_id < 0)
        return {};
    const auto &root = tables_.variants[root_id][rotation];
    Vec3 anchor{};
    if (data.start_anchor >= 0) {
        root.connector_order(random, parents_);
        for (int index : parents_)
            if (root.connectors[index].name == data.start_anchor) {
                anchor = root.connectors[index].pos;
                break;
            }
    }
    Vec3 root_pos = origin - anchor - Vec3{0, 1, 0};
    Box root_box = root.box.moved(root_pos);
    Vec3 biome_pos{(root_box.lo.x + root_box.hi.x) / 2, origin.y, (root_box.lo.z + root_box.hi.z) / 2};
    return {root_id, rotation, root_pos, biome_pos, root_box};
}

Box LayoutEngine::possible_bounds(int cx, int cz) {
    Random random(seed_);
    auto root = start(cx, cz, random);
    if (root.id < 0)
        return {{1, 1, 1}, {0, 0, 0}};
    int d = tables_.data.distance;
    auto p = root.biome;
    auto b = root.box;
    // 后代部件位于根空闲空间或父内部包围盒中，根部件本身可能超出空闲空间。
    return {{std::min(b.lo.x, p.x - d), std::min(b.lo.y, p.y - d), std::min(b.lo.z, p.z - d)},
            {std::max(b.hi.x, p.x + d), std::max(b.hi.y, p.y + d), std::max(b.hi.z, p.z + d)}};
}

Chamber LayoutEngine::generate(int cx, int cz) {
    const auto &data = tables_.data;
    Random random(seed_);
    auto root = start(cx, cz, random);
    if (root.id < 0)
        return {};
    const auto &aliases = aliases_;
    int root_id = root.id, rotation = root.rotation;
    auto root_pos = root.pos, biome_pos = root.biome;
    auto root_box = root.box;
    Chamber result;
    result.biome_position = biome_pos;
    bool outside_deep_dark = climate263_
                                 ? climate263_->allowed(floor_div(biome_pos.x, 4) * 4, floor_div(biome_pos.y, 4) * 4,
                                                        floor_div(biome_pos.z, 4) * 4)
                                 : geode_biomes_allowed(biomes_.get(), floor_div(biome_pos.x, 4),
                                                        floor_div(biome_pos.y, 4), floor_div(biome_pos.z, 4));
    if (outside_deep_dark == data.deep_dark)
        return result;
    result.exists = true;
    result.placed.push_back({root_id, rotation, root_pos, root_box, 1, {}});

    struct Piece {
        int id, rotation;
        Vec3 pos;
        Box box;
        int index, ground;
    };

    struct Task {
        Piece piece;
        Space *space;
        int depth;
    };

    size_t spaces_used = 0;
    auto space = [&](Box box, bool grid) {
        if (spaces_used == spaces_.size())
            spaces_.emplace_back(box, grid);
        else
            spaces_[spaces_used].reset(box, grid);
        return &spaces_[spaces_used++];
    };
    int distance = data.distance;
    Space *root_space = space(
        Box{{biome_pos.x - distance, std::max(biome_pos.y - distance, -64 + data.padding_bottom),
             biome_pos.z - distance},
            {biome_pos.x + distance, std::min(biome_pos.y + distance, 319 - data.padding_top), biome_pos.z + distance}},
        true);
    root_space->occupy(root_box);
    std::map<int, std::deque<Task>, std::greater<int>> pending;
    Task task{{root_id, rotation, root_pos, root_box, 0, 1}, root_space, 0};
    auto &parents = parents_;
    auto &connectors = connectors_;
    auto &candidates = candidates_;
    auto &fallbacks = fallbacks_;
    for (;;) {
        auto piece = task.piece;
        const auto &variant = tables_.variants[piece.id][piece.rotation];
        ++result.pieces;
        for (auto p : variant.spawners)
            result.spawners.push_back(piece.pos + p);
        variant.connector_order(random, parents);
        Space *internal = nullptr;
        for (int parent_index : parents) {
            const auto &parent = variant.connectors[parent_index];
            uint64_t attempt = ++attempt_;
            Vec3 connector_pos = piece.pos + parent.pos;
            Vec3 attachment = connector_pos + directions[parent.facing];
            const auto &pool = data.pools[aliases[parent.pool]];
            const auto &fallback = data.pools[pool.fallback];
            Space *available = task.space;
            if (piece.box.contains(attachment)) {
                if (!internal)
                    internal = space(piece.box, false);
                available = internal;
            }
            candidates.clear();
            if (task.depth != data.depth) {
                candidates = pool.elements;
                random.shuffle(candidates);
            }
            fallbacks = fallback.elements;
            random.shuffle(fallbacks);
            candidates.insert(candidates.end(), fallbacks.begin(), fallbacks.end());
            bool attached = false;
            for (int child_id : candidates) {
                if (child_id < 0)
                    break;
                auto &rejected = rejected_[child_id];
                if (rejected.attempt != attempt)
                    rejected = {attempt, 0};
                if (rejected.rotations == 15) {
                    // 权重池会重复模板；当前连接口成功连接前，碰撞几何保持不变。
                    // 旋转不改变连接器数量，因此可用任意旋转顺序消耗被丢弃的洗牌。
                    random.consume_shuffle(4);
                    random.consume_four_shuffles(int(tables_.variants[child_id][0].connectors.size()));
                    continue;
                }
                std::array<int, 4> rotations{0, 1, 2, 3};
                random.shuffle(rotations);
                for (int child_rotation : rotations) {
                    const auto &child = tables_.variants[child_id][child_rotation];
                    if (rejected.rotations & (1U << child_rotation)) {
                        random.consume_shuffle(int(child.connectors.size()));
                        continue;
                    }
                    int match = child.matches[parent.match_id];
                    if (match == -2)
                        child.connector_order(random, connectors);
                    else {
                        // 即使拒绝候选或只有一个连接器，也须消耗原版洗牌及拒绝采样的随机数。
                        random.consume_shuffle(int(child.connectors.size()));
                        if (match < 0) {
                            rejected.rotations |= 1U << child_rotation;
                            continue;
                        }
                        connectors.clear();
                        connectors.push_back(match);
                    }
                    for (int connector_index : connectors) {
                        const auto &target = child.connectors[connector_index];
                        if (parent.facing != (target.facing ^ 1) || (!parent.rollable && parent.top != target.top) ||
                            parent.target != target.name)
                            continue;
                        Vec3 pos = attachment - target.pos;
                        Box box = child.box.moved(pos);
                        if (!available->fits(box))
                            continue;
                        available->occupy(box);
                        int q = parent.pos.y - target.pos.y + directions[parent.facing].y;
                        int ground = piece.ground - q, index = int(result.placed.size());
                        result.placed[piece.index].junctions.push_back(
                            {attachment.x, piece.box.lo.y + piece.ground, attachment.z});
                        int child_ground_y = connector_pos.y - target.pos.y + ground;
                        result.placed.push_back({child_id,
                                                 child_rotation,
                                                 pos,
                                                 box,
                                                 ground,
                                                 {{connector_pos.x, child_ground_y, connector_pos.z}}});
                        Piece placed{child_id, child_rotation, pos, box, index, ground};
                        if (task.depth + 1 <= data.depth)
                            pending[parent.placement].push_back({placed, available, task.depth + 1});
                        else {
                            ++result.pieces;
                            for (auto p : child.spawners)
                                result.spawners.push_back(pos + p);
                        }
                        attached = true;
                        break;
                    }
                    if (attached)
                        break;
                    rejected.rotations |= 1U << child_rotation;
                }
                if (attached)
                    break;
            }
        }
        if (pending.empty())
            break;
        auto bucket = pending.begin();
        task = bucket->second.front();
        bucket->second.pop_front();
        if (bucket->second.empty())
            pending.erase(bucket);
    }
    std::sort(result.spawners.begin(), result.spawners.end());
    result.spawners.erase(std::unique(result.spawners.begin(), result.spawners.end()), result.spawners.end());
    return result;
}
}
