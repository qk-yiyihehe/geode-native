#pragma once
#include "geode.hpp"
#include <functional>
#include <string>
#include <unordered_map>

namespace geode {
// 比较器、距离参照或Top K截断时机变化时提升版本。
constexpr int kRankingRuleVersion = 3;
constexpr char kRankingRule[] = "primary-desc-distance-asc-x-asc-z-asc";

struct Options {
    uint64_t seed = 0;
    std::string seed_text = "0", version = "1.21.1", shape = "circle", scan_backend = "auto";
    std::string coverage = "game";
    int inner_radius = 128;
    int center_x = 0, center_z = 0, radius = 10000, tick_radius = 128, simulation_distance = 10;
    int min_geodes = 1, min_budding = 1, prefilter = 0, limit = 100, threads = 8;
    bool modern = false, terrain = true, full_world = false;
    int generation = 1;
    std::function<void()> checkpoint;
    std::string checkpoint_directory;
    std::function<double()> paused_ms;
    std::function<void(double, uint64_t, uint64_t)> restored;
    bool contains(int x, int z) const;
    bool intersects_chunk(int cx, int cz) const;
    int activity_radius() const;
    bool chunk_constant() const { return coverage == "game" && modern; }
    bool custom() const { return coverage != "game"; }
};

const std::vector<std::string> &versions();
void set_version(Options &o, const std::string &version);
// 全图简写必须先展开，再计算排名距离，
// 确保实际榜单中心为原点，而不是用户输入但未生效的中心。
void expand_full_world(Options &o);

struct Weight {
    int x, z, budding;
};

struct Center {
    int x = 0, z = 0, budding = -1, geodes = 0;
};

using Weights = std::unordered_map<uint64_t, int>;
bool active(const Options &o, int player_x, int player_z, int cx, int cz);
bool active_point(const Options &o, int player_x, int player_z, int x, int z);
std::pair<int, int> coarse_span(const Options &o, int dz);
Center optimize(const Options &o, int cx, int cz, const std::vector<Weight> &weights,
                const std::vector<Weight> &origins = {});
bool same_hotspot(const Options &o, Center a, Center b);
Center evaluate(const Options &o, int x, int z);
// 最终榜单使用水平距离；世界边界内的坐标保证
// 距离平方小于2^53，JSON可精确携带该整数。
int64_t center_distance_sq(int center_x, int center_z, int x, int z);
// 从精确距离平方计算十分之一格显示值：round(10*sqrt(distanceSq))。
int64_t distance_tenths(int center_x, int center_z, int x, int z);
// 正式榜单规则v3：母岩数降序，距离平方、X、Z依次升序。
// 与粗筛、候选池和组内几何比较器分开，后者保留原有语义；
// 只有此规则决定最终名次、Top K截断和正式导出顺序。
bool final_better(int center_x, int center_z, Center a, Center b);

struct Stats {
    uint64_t scanned = 0, candidates = 0, simulated = 0, weights = 0, pruned = 0, rejected = 0, hotspots = 0,
             optimized = 0, resumed_scan = 0, resumed_simulation = 0;
    // 两段式裁剪：石模型（无地形）先算母岩数上界，只有上界仍可能进榜的候选才跑地形。
    // stone_simulated 是上界那一遍的模拟次数，simulated 只统计地形那一遍。
    uint64_t stone_simulated = 0, two_pass_kept = 0;
    int two_pass_excluded_bound = 0, two_pass_floor_bound = 0;
    bool two_pass = false, two_pass_retried = false;
    double scan_ms = 0, simulate_ms = 0, rank_ms = 0, total_ms = 0, resumed_ms = 0;
    std::string scan_backend = "cpu";
};

struct Results {
    std::vector<Center> centers;
    Stats stats;
    std::string checkpoint_path;
};

// 进度回调：阶段名、阶段内完成量、阶段内总量、已扫描区块数。
// 第四个参数让界面在运行中显示真实扫描量，而不是等结果才知道。
using ProgressFn = std::function<void(const std::string &, uint64_t, uint64_t, uint64_t)>;

Results search(const Options &o, const ProgressFn &progress = {});
}
