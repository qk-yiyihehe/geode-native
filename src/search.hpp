#pragma once
#include "geode.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

namespace geode {
// 比较器、距离参照或Top K截断时机变化时提升版本。
constexpr int kRankingRuleVersion = 3;
constexpr char kRankingRule[] = "primary-desc-distance-asc-x-asc-z-asc";

struct Options {
    uint64_t seed = 0;
    std::string seed_text = "0", version = "1.21.1", shape = "circle", scan_backend = "auto";
    int center_x = 0, center_z = 0, radius = 10000, tick_radius = 128, simulation_distance = 10;
    int min_geodes = 1, min_budding = 1, prefilter = 0, limit = 100, threads = 8;
    // search_rect = true 时使用精确四至（闭区间），忽略 center/radius/shape。
    // 四至的排名中心取 (min+max)/2，与协议 bounds 的 center 一致。
    bool search_rect = false;
    int min_x = 0, max_x = 0, min_z = 0, max_z = 0;
    // 0 = 与 threads 相同；大于 threads 时只提高粗筛并行度，模拟仍用 threads。
    int scan_threads = 0;
    // 0 = 逐 512 区块瓦片记录断点（旧行为）；>0 = 按该区块边长的区域聚合断点。
    // 全图搜索应设为 8192（131,072 方块），避免 5,000 万瓦片的断点爆炸。
    int checkpoint_region = 0;
    bool modern = false, terrain = true, full_world = false;
    int generation = 1;
    std::function<void()> checkpoint;
    std::string checkpoint_directory;
    std::function<double()> paused_ms;
    std::function<void(double, uint64_t, uint64_t)> restored;
    bool contains(int x, int z) const;
    bool intersects_chunk(int cx, int cz) const;
    int activity_radius() const;
};

void set_version(Options &o, const std::string &version);
// 全图简写必须先展开，再计算排名距离，
// 确保实际榜单中心为原点，而不是用户输入但未生效的中心。
void expand_full_world(Options &o);

// 协作式暂停/取消：工作线程在检查点阻塞或抛出 Cancelled。
// 与旧的 Options::checkpoint 相比，暂停延迟为一个工作单位，且取消可正常收敛。
struct Cancelled {};

class Control {
    std::mutex mutex_;
    std::condition_variable changed_;
    std::atomic<bool> paused_{};
    std::atomic<bool> cancelled_{};
    std::atomic<int> waiting_{};
    std::chrono::steady_clock::time_point paused_at_{};
    std::chrono::steady_clock::duration paused_time_{};

  public:
    void checkpoint();
    void pause();
    void resume();
    void cancel();
    bool paused() const {
        return paused_.load(std::memory_order_relaxed);
    }
    bool cancelled() const {
        return cancelled_.load(std::memory_order_relaxed);
    }
    // 当前阻塞在检查点等待恢复的工作线程数；宿主可用它确认暂停已生效。
    int waiting() const {
        return waiting_.load(std::memory_order_relaxed);
    }
    // 累计暂停毫秒，含当前尚未恢复的暂停段。
    double paused_ms();
};

struct Weight {
    int x, z, budding;
};

struct Center {
    int x = 0, z = 0, budding = -1, geodes = 0;
};

using Weights = std::unordered_map<uint64_t, int>;
bool active(const Options &o, int player_x, int player_z, int cx, int cz);
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
    double scan_ms = 0, simulate_ms = 0, rank_ms = 0, total_ms = 0, resumed_ms = 0;
    std::string scan_backend = "cpu";
};

struct Results {
    std::vector<Center> centers;
    Stats stats;
    std::string checkpoint_path;
    // G-UP1：Control 取消时正常返回，不抛异常。
    bool cancelled = false;
    // G-UP5：引擎直接给出结果语义，宿主不再推断。
    bool approximate = false;
    int ranking_rule_version = kRankingRuleVersion;
    std::string scan_backend = "cpu";
};

using Progress = std::function<void(const std::string &, uint64_t, uint64_t)>;
using Backend = std::function<void(const std::string &, const std::string &)>;
Results search(const Options &o, const Progress &progress = {}, const Backend &backend = {});
// 可暂停/可取消的重载；旧 Options::checkpoint 仍会在此路径生效。
Results search(const Options &o, Control &control, const Progress &progress = {}, const Backend &backend = {});
}
