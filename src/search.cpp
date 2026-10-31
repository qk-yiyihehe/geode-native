#include "search.hpp"
#include "terrain.hpp"
#include "coarse_gpu.hpp"
#include "checkpoint.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstdio>
#include <exception>
#include <mutex>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>

namespace geode {
namespace {
using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

template <class F> void parallel(size_t size, int threads, F work) {
    std::atomic<size_t> next{};
    std::exception_ptr failure;
    std::mutex mutex;
    auto worker = [&](int id) {
        try {
            for (;;) {
                size_t i = next.fetch_add(1);
                if (i >= size)
                    break;
                work(i, id);
            }
        } catch (...) {
            std::lock_guard lock(mutex);
            if (!failure)
                failure = std::current_exception();
            next = size;
        }
    };
    std::vector<std::thread> pool;
    for (int i = 0; i < threads; ++i)
        pool.emplace_back(worker, i);
    for (auto &t : pool)
        t.join();
    if (failure)
        std::rethrow_exception(failure);
}

int isqrt(int64_t n) {
    int s = int(std::sqrt(double(n)));
    while (int64_t(s + 1) * (s + 1) <= n)
        ++s;
    while (int64_t(s) * s > n)
        --s;
    return s;
}

struct Candidate {
    int x, z, count;
};

bool better(Candidate a, Candidate b) {
    if (a.count != b.count)
        return a.count > b.count;
    if (a.x != b.x)
        return a.x < b.x;
    return a.z < b.z;
}

struct Better {
    bool operator()(Candidate a, Candidate b) const {
        return better(a, b);
    }
};

using Heap = std::priority_queue<Candidate, std::vector<Candidate>, Better>;

bool offer(Heap &q, Candidate c, int limit) {
    if (!limit || q.size() < size_t(limit)) {
        q.push(c);
        return true;
    }
    if (better(c, q.top())) {
        q.pop();
        q.push(c);
        return true;
    }
    return false;
}

// 日志只保存已完成单位；线程中尚未提交的候选堆和地形缓存
// 可以丢弃，回放后的全局Top K池为各线程提供安全起始阈值。
class SearchCheckpoint {
    const Options &o;
    Clock::time_point start;
    double pause_start = 0, history = 0, scan_history = 0, simulation_history = 0;
    std::unique_ptr<CheckpointJournal> journal;
    std::mutex mutex;
    std::exception_ptr commit_failure;
    std::vector<bool> scans;
    std::unordered_set<uint64_t> simulations;
    size_t scan_tiles_x = 0, scan_tiles_z = 0;
    int scan_width = 0, scan_height = 0;
    int region_chunks = 0;
    size_t region_cols = 0, region_rows = 0;
    std::vector<uint8_t> regions;
    std::vector<uint32_t> region_remaining;
    std::vector<uint64_t> region_covered;
    std::vector<std::vector<Candidate>> region_correct, region_legacy;
    std::vector<uint8_t> region_gpu;
    int unit_size = 0, unit_cols = 0, unit_rows = 0, region_units_x = 1, region_units_z = 1;

    void times(CheckpointRecord &record) {
        record.put(total_ms());
        record.put(stats.scan_ms);
        record.put(stats.simulate_ms);
    }

    void read_times(CheckpointRecord &record) {
        stats.total_ms = record.get<double>();
        stats.scan_ms = record.get<double>();
        stats.simulate_ms = record.get<double>();
    }

    static Candidate read_candidate(CheckpointRecord &r) {
        return {r.get<int>(), r.get<int>(), r.get<int>()};
    }

    static void put_candidate(CheckpointRecord &r, Candidate c) {
        r.put(c.x);
        r.put(c.z);
        r.put(c.count);
    }

    size_t region_of(size_t unit) const {
        int col = int(unit % size_t(unit_cols)), row = int(unit / size_t(unit_cols));
        return size_t(row / region_units_z) * region_cols + size_t(col / region_units_x);
    }

    void write_region(size_t r) {
        if (commit_failure)
            std::rethrow_exception(commit_failure);
        try {
            CheckpointRecord record;
            record.put(uint8_t(4));
            stats.scan_ms = scan_ms();
            times(record);
            record.put(uint64_t(r));
            record.put(uint8_t(region_gpu[r]));
            record.put(region_covered[r]);
            auto add = [&](const std::vector<Candidate> &pending) {
                record.put(uint32_t(pending.size()));
                for (auto c : pending)
                    put_candidate(record, c);
            };
            add(region_correct[r]);
            add(region_legacy[r]);
            journal->append(record);
            // 区域记录只在完成后写入；同时把该区域的候选并入全局 Top-K，
            // 保证不启用断点时也能从 saved.correct/legacy 得到最终候选。
            for (auto c : region_correct[r])
                offer(correct, c, o.prefilter);
            for (auto c : region_legacy[r])
                offer(legacy, c, o.prefilter);
            regions[r] = 1;
            stats.scan_backend = region_gpu[r] ? "gpu" : "cpu";
            region_correct[r].clear();
            region_legacy[r].clear();
            region_covered[r] = 0;
        } catch (...) {
            commit_failure = std::current_exception();
            throw;
        }
    }

  public:
    Stats stats;
    Heap correct, legacy;
    Weights weights, origins;
    std::vector<Candidate> candidates;
    bool scan_complete = false;

    SearchCheckpoint(const Options &options, Clock::time_point begin, size_t scan_count, int minx, int maxx, int minz,
                     int maxz)
        : o(options), start(begin), scans(o.checkpoint_directory.empty() || o.checkpoint_region > 0 ? 0 : scan_count) {
        int width = std::max(0, maxx - minx + 1), height = std::max(0, maxz - minz + 1);
        scan_width = width;
        scan_height = height;
        scan_tiles_x = size_t((width + 511) / 512);
        scan_tiles_z = size_t((height + 511) / 512);
        pause_start = o.paused_ms ? o.paused_ms() : 0;
        if (!enabled())
            return;
        if (o.checkpoint_region > 0) {
            // 区域断点对齐到搜索范围左上角，与粗筛/模拟的瓦片划分保持一致。
            region_chunks = std::max(512, o.checkpoint_region);
            region_cols = (size_t(width) + size_t(region_chunks) - 1) / size_t(region_chunks);
            region_rows = (size_t(height) + size_t(region_chunks) - 1) / size_t(region_chunks);
            regions.assign(region_cols * region_rows, 0);
        }
        // 生成逻辑、归组或任务归属变化时，必须更新模型/工作划分修订。
        std::ostringstream signature;
        signature << "geode-native-v2;model=20261027-range;work=512-4096-64-twopass;int=" << sizeof(int) << ";seed=" << o.seed << ';'
                  << o.version << ';' << o.generation << ';' << o.terrain << ';' << o.modern << ';' << o.center_x << ';'
                  << o.center_z << ';' << o.radius << ';' << o.shape << ';' << o.full_world << ';' << o.tick_radius
                  << ';' << o.simulation_distance << ';' << o.min_geodes << ';' << o.min_budding << ';' << o.prefilter
                  << ';' << o.limit << ';' << o.threads << ';' << o.scan_backend << ';' << kRankingRuleVersion
                  << ';' << o.coverage << ';' << o.inner_radius << ';' << CUBIOMES_REVISION
                  << ';' << o.search_rect << ';' << o.min_x << ';' << o.max_x << ';' << o.min_z << ';' << o.max_z
                  << ';' << o.checkpoint_region;
        journal = std::make_unique<CheckpointJournal>(o.checkpoint_directory, signature.str());        journal->replay([&](CheckpointRecord &record) {
            if (o.checkpoint)
                o.checkpoint();
            auto kind = record.get<uint8_t>();
            read_times(record);
            if (kind == 1) {
                auto id = record.get<uint64_t>();
                if (id >= scans.size() || scans[size_t(id)] || scan_complete)
                    throw std::runtime_error("Invalid geode coarse checkpoint unit");
                scans[size_t(id)] = true;
                stats.scan_backend = record.get<uint8_t>() ? "gpu" : "cpu";
                stats.scanned += record.get<uint64_t>();
                for (auto *pool : {&correct, &legacy}) {
                    auto n = record.get<uint32_t>();
                    for (uint32_t i = 0; i < n; ++i)
                        offer(*pool, read_candidate(record), o.prefilter);
                }
            } else if (kind == 2) {
                if (scan_complete)
                    throw std::runtime_error("Duplicate geode coarse checkpoint phase");
                scan_complete = true;
                stats.scanned = record.get<uint64_t>();
                stats.scan_backend = record.get<uint8_t>() ? "gpu" : "cpu";
                auto n = record.get<uint64_t>();
                if (n > (record.bytes.size() - record.offset) / 12)
                    throw std::runtime_error("Invalid geode candidate checkpoint size");
                candidates.reserve(size_t(n));
                for (uint64_t i = 0; i < n; ++i)
                    candidates.push_back(read_candidate(record));
                stats.candidates = n;
            } else if (kind == 3) {
                auto id = record.get<uint64_t>();
                if (!scan_complete || !simulations.insert(id).second)
                    throw std::runtime_error("Invalid geode simulation checkpoint unit");
                stats.simulated += record.get<uint64_t>();
                stats.rejected += record.get<uint64_t>();
                auto n = record.get<uint32_t>();
                for (uint32_t i = 0; i < n; ++i) {
                    auto k = record.get<uint64_t>();
                    weights[k] += record.get<int>();
                }
                n = record.get<uint32_t>();
                for (uint32_t i = 0; i < n; ++i)
                    origins.emplace(record.get<uint64_t>(), 1);
            } else if (kind == 4) {
                auto id = record.get<uint64_t>();
                if (o.checkpoint_region <= 0 || id >= regions.size() || regions[size_t(id)] || scan_complete)
                    throw std::runtime_error("Invalid geode region checkpoint unit");
                regions[size_t(id)] = 1;
                stats.scan_backend = record.get<uint8_t>() ? "gpu" : "cpu";
                stats.scanned += record.get<uint64_t>();
                for (auto *pool : {&correct, &legacy}) {
                    auto n = record.get<uint32_t>();
                    for (uint32_t i = 0; i < n; ++i)
                        offer(*pool, read_candidate(record), o.prefilter);
                }
            } else
                throw std::runtime_error("Unknown geode checkpoint record");
            record.end();
        });
        history = stats.total_ms;
        scan_history = stats.scan_ms;
        simulation_history = stats.simulate_ms;
        stats.resumed_ms = history;
        if (o.checkpoint_region <= 0)
            stats.resumed_scan = std::count(scans.begin(), scans.end(), true);
        stats.resumed_simulation = simulations.size();
        if (o.restored)
            o.restored(history, stats.resumed_scan, stats.resumed_simulation);
    }

    bool enabled() const {
        return !o.checkpoint_directory.empty();
    }

    double pause_ms() const {
        return (o.paused_ms ? o.paused_ms() : 0) - pause_start;
    }

    double total_ms() const {
        return history + elapsed(start) - pause_ms();
    }

    double scan_ms() const {
        return scan_history + elapsed(start) - pause_ms();
    }

    double simulation_ms(Clock::time_point begin, double paused_at) const {
        return simulation_history + elapsed(begin) - (pause_ms() - paused_at);
    }

    // 区域断点必须按实际的并行单位（CPU 瓦片 512 / GPU 批次 4096）建立
    // 每区域剩余单位计数；只把完整区域写入日志，未完成区域重扫。
    void prepare_regions(int unit) {
        if (!enabled() || o.checkpoint_region <= 0 || unit_size == unit)
            return;
        std::lock_guard lock(mutex);
        unit_size = unit;
        unit_cols = (scan_width + unit - 1) / unit;
        unit_rows = (scan_height + unit - 1) / unit;
        region_units_x = std::max(1, region_chunks / unit);
        region_units_z = region_units_x;
        region_cols = (size_t(unit_cols) + size_t(region_units_x) - 1) / size_t(region_units_x);
        region_rows = (size_t(unit_rows) + size_t(region_units_z) - 1) / size_t(region_units_z);
        size_t count = region_cols * region_rows;
        region_remaining.assign(count, 0);
        region_covered.assign(count, 0);
        region_correct.assign(count, {});
        region_legacy.assign(count, {});
        region_gpu.assign(count, 0);
        regions.resize(count, 0);
        uint64_t completed = 0;
        for (size_t rz = 0; rz < region_rows; ++rz)
            for (size_t rx = 0; rx < region_cols; ++rx) {
                int ux = std::min(region_units_x, unit_cols - int(rx) * region_units_x);
                int uz = std::min(region_units_z, unit_rows - int(rz) * region_units_z);
                uint32_t units = uint32_t(std::max(0, ux) * std::max(0, uz));
                region_remaining[rz * region_cols + rx] = units;
                if (regions[rz * region_cols + rx])
                    completed += units;
            }
        stats.resumed_scan = completed;
    }

    bool scan_done(size_t id) {
        if (!enabled())
            return false;
        std::lock_guard lock(mutex);
        if (o.checkpoint_region > 0) {
            if (!unit_size)
                return false;
            return regions[region_of(id)] != 0;
        }
        return scans.at(id);
    }

    bool simulation_done(uint64_t id) {
        if (!enabled())
            return false;
        std::lock_guard lock(mutex);
        return simulations.contains(id);
    }

    std::string path() const {
        return journal ? journal->path() : std::string{};
    }

    void scan_commit(size_t id, uint64_t covered, bool gpu, const std::vector<Candidate> &a,
                     const std::vector<Candidate> &b) {
        if (!enabled())
            return;
        std::lock_guard lock(mutex);
        if (commit_failure)
            std::rethrow_exception(commit_failure);
        if (o.checkpoint_region > 0) {
            if (!unit_size)
                throw std::runtime_error("Geode region checkpoint is not prepared");
            size_t r = region_of(id);
            if (regions[r])
                return;
            region_covered[r] += covered;
            if (gpu)
                region_gpu[r] = 1;
            region_correct[r].insert(region_correct[r].end(), a.begin(), a.end());
            region_legacy[r].insert(region_legacy[r].end(), b.begin(), b.end());
            stats.scanned += covered;
            if (--region_remaining[r] == 0)
                write_region(r);
            return;
        }
        try {
            CheckpointRecord record;
            record.put(uint8_t(1));
            stats.scan_ms = scan_ms();
            times(record);
            record.put(uint64_t(id));
            record.put(uint8_t(gpu));
            record.put(covered);
            auto add = [&](Heap &pool, const std::vector<Candidate> &pending) {
                std::vector<Candidate> accepted;
                for (auto c : pending)
                    if (offer(pool, c, o.prefilter))
                        accepted.push_back(c);
                record.put(uint32_t(accepted.size()));
                for (auto c : accepted)
                    put_candidate(record, c);
            };
            add(correct, a);
            add(legacy, b);
            journal->append(record);
            scans.at(id) = true;
            stats.scanned += covered;
            stats.scan_backend = gpu ? "gpu" : "cpu";
        } catch (...) {
            commit_failure = std::current_exception();
            throw;
        }
    }

    void scan_finish(const std::vector<Candidate> &values, const Stats &measured) {
        if (!enabled() || scan_complete)
            return;
        std::lock_guard lock(mutex);
        stats.scan_ms = measured.scan_ms;
        if (commit_failure)
            std::rethrow_exception(commit_failure);
        try {
            CheckpointRecord record;
            record.put(uint8_t(2));
            times(record);
            record.put(measured.scanned);
            record.put(uint8_t(measured.scan_backend == "gpu"));
            record.put(uint64_t(values.size()));
            for (auto c : values)
                put_candidate(record, c);
            journal->append(record);
            scan_complete = true;
            stats.scanned = measured.scanned;
            stats.candidates = values.size();
        } catch (...) {
            commit_failure = std::current_exception();
            throw;
        }
    }

    void simulation_commit(uint64_t id, const Weights &delta, const std::vector<uint64_t> &source, uint64_t count,
                           uint64_t rejected, Clock::time_point begin, double paused_at) {
        if (!enabled())
            return;
        std::lock_guard lock(mutex);
        stats.simulate_ms = simulation_ms(begin, paused_at);
        if (commit_failure)
            std::rethrow_exception(commit_failure);
        try {
            CheckpointRecord record;
            record.put(uint8_t(3));
            times(record);
            record.put(id);
            record.put(count);
            record.put(rejected);
            record.put(uint32_t(delta.size()));
            for (auto [k, v] : delta) {
                record.put(k);
                record.put(v);
            }
            record.put(uint32_t(source.size()));
            for (auto k : source)
                record.put(k);
            journal->append(record);
            simulations.insert(id);
            stats.simulated += count;
            stats.rejected += rejected;
            for (auto [k, v] : delta)
                weights[k] += v;
            for (auto k : source)
                origins.emplace(k, 1);
        } catch (...) {
            commit_failure = std::current_exception();
            throw;
        }
    }
};

// 候选池和组内几何仍以晶洞数打破同分，决定哪些位置参与计算；
// 最终榜单不能沿用该规则，因为晶洞数在最终结果中只展示。
bool center_better(Center a, Center b) {
    if (a.budding != b.budding)
        return a.budding > b.budding;
    if (a.geodes != b.geodes)
        return a.geodes > b.geodes;
    if (a.x != b.x)
        return a.x < b.x;
    return a.z < b.z;
}

std::vector<Weight> gather(const Weights &all, int cx, int cz, int radius) {
    std::vector<Weight> result;
    result.reserve(256);
    for (int z = cz - radius; z <= cz + radius; ++z)
        for (int x = cx - radius; x <= cx + radius; ++x) {
            auto it = all.find(key(x, z));
            if (it != all.end() && it->second)
                result.push_back({x, z, it->second});
        }
    return result;
}

// 区块空间的稠密权重网格。游戏覆盖规则的权重键就是区块坐标，搜索范围通常远小于
// 世界边界，因此可以铺成二维数组，把逐候选的哈希探测换成数组读取；再建一张二维
// 前缀和，供"活动形状为按区块方形"的版本把窗口求和降到常数次读取。
// 这只是查找方式的替换，不改变任何数值判定与遍历顺序。
struct DenseGrid {
    int min_x = 0, min_z = 0, width = 0, height = 0;
    std::vector<int> cell;
    std::vector<int64_t> prefix;

    void build(const Weights &all, int lo_x, int hi_x, int lo_z, int hi_z) {
        min_x = lo_x;
        min_z = lo_z;
        width = hi_x - lo_x + 1;
        height = hi_z - lo_z + 1;
        cell.assign(size_t(width) * size_t(height), 0);
        for (const auto &[k, v] : all) {
            if (!v)
                continue;
            int x = key_x(k), z = key_z(k);
            if (x < lo_x || x > hi_x || z < lo_z || z > hi_z)
                continue;
            cell[size_t(z - lo_z) * size_t(width) + size_t(x - lo_x)] += v;
        }
        size_t stride = size_t(width) + 1;
        prefix.assign(stride * (size_t(height) + 1), 0);
        for (int z = 0; z < height; ++z) {
            int64_t row = 0;
            for (int x = 0; x < width; ++x) {
                row += cell[size_t(z) * size_t(width) + size_t(x)];
                prefix[size_t(z + 1) * stride + size_t(x + 1)] =
                    prefix[size_t(z) * stride + size_t(x + 1)] + row;
            }
        }
    }

    bool covers(int x, int z) const {
        return x >= min_x && x < min_x + width && z >= min_z && z < min_z + height;
    }

    int at(int x, int z) const {
        if (!covers(x, z))
            return 0;
        return cell[size_t(z - min_z) * size_t(width) + size_t(x - min_x)];
    }

    // 半开区间矩形和；区间先夹到网格边界，越界部分按 0 计。
    int64_t rect(int x0, int z0, int x1, int z1) const {
        int cx0 = std::clamp(x0 - min_x, 0, width), cx1 = std::clamp(x1 - min_x, 0, width);
        int cz0 = std::clamp(z0 - min_z, 0, height), cz1 = std::clamp(z1 - min_z, 0, height);
        if (cx0 >= cx1 || cz0 >= cz1)
            return 0;
        size_t stride = size_t(width) + 1;
        return prefix[size_t(cz1) * stride + size_t(cx1)] - prefix[size_t(cz0) * stride + size_t(cx1)] -
               prefix[size_t(cz1) * stride + size_t(cx0)] + prefix[size_t(cz0) * stride + size_t(cx0)];
    }
};
}

void Control::checkpoint() {
    if (cancelled_.load(std::memory_order_relaxed))
        throw Cancelled{};
    if (!paused_.load(std::memory_order_relaxed))
        return;
    std::unique_lock lock(mutex_);
    ++waiting_;
    changed_.wait(lock, [this] {
        return !paused_.load(std::memory_order_relaxed) || cancelled_.load(std::memory_order_relaxed);
    });
    --waiting_;
    if (cancelled_.load(std::memory_order_relaxed))
        throw Cancelled{};
}

void Control::pause() {
    std::lock_guard lock(mutex_);
    if (!paused_.load(std::memory_order_relaxed)) {
        paused_.store(true, std::memory_order_relaxed);
        paused_at_ = std::chrono::steady_clock::now();
    }
}

void Control::resume() {
    std::lock_guard lock(mutex_);
    if (paused_.load(std::memory_order_relaxed)) {
        paused_time_ += std::chrono::steady_clock::now() - paused_at_;
        paused_.store(false, std::memory_order_relaxed);
    }
    changed_.notify_all();
}

void Control::cancel() {
    cancelled_.store(true, std::memory_order_relaxed);
    changed_.notify_all();
}

double Control::paused_ms() {
    std::lock_guard lock(mutex_);
    return std::chrono::duration<double, std::milli>(
               paused_time_ + (paused_.load(std::memory_order_relaxed) ? std::chrono::steady_clock::now() - paused_at_
                                                                      : std::chrono::steady_clock::duration{}))
        .count();
}


bool Options::contains(int x, int z) const {
    if (x < -30000000 || x >= 30000000 || z < -30000000 || z >= 30000000)
        return false;
    // 矩形四至是闭区间；排名中心与边界裁切都沿用同一判定。
    if (search_rect)
        return x >= min_x && x <= max_x && z >= min_z && z <= max_z;
    int64_t dx = int64_t(x) - center_x, dz = int64_t(z) - center_z;
    if (std::abs(dx) > radius || std::abs(dz) > radius)
        return false;
    return shape == "square" || dx * dx + dz * dz <= int64_t(radius) * radius;
}

bool Options::intersects_chunk(int x, int z) const {
    if (search_rect) {
        // 矩形与区块包围盒求交：哪怕只覆盖区块一角也要保留，
        // 之后再由 optimize() 把玩家位置夹到矩形内。
        int64_t left = int64_t(x) * 16, right = left + 15;
        int64_t top = int64_t(z) * 16, bottom = top + 15;
        if (right < -30000000 || left > 29999999 || bottom < -30000000 || top > 29999999)
            return false;
        return right >= min_x && left <= max_x && bottom >= min_z && top <= max_z;
    }
    int bx = std::clamp(center_x, x * 16, x * 16 + 15), bz = std::clamp(center_z, z * 16, z * 16 + 15);
    return contains(bx, bz);
}

int Options::activity_radius() const {
    if (custom())
        return (inner_radius + 15) / 16 + 1;
    return modern ? std::min(simulation_distance, 31) : (tick_radius + 15) / 16;
}

void expand_full_world(Options &o) {
    o.center_x = 0;
    o.center_z = 0;
    o.radius = 30000000;
    o.shape = "square";
    o.full_world = true;
}

const std::vector<std::string> &versions() {
    static const std::vector<std::string> values = {
        "1.18",
        "1.18.1",
        "1.18.2",
        "1.19",
        "1.19.1",
        "1.19.2",
        "1.19.3",
        "1.19.4",
        "1.20",
        "1.20.1",
        "1.20.2",
        "1.20.3",
        "1.20.4",
        "1.20.5",
        "1.20.6",
        "1.21",
        "1.21.1",
        "1.21.2",
        "1.21.3",
        "1.21.4",
        "1.21.5",
        "1.21.6",
        "1.21.7",
        "1.21.8",
        "1.21.9",
        "1.21.10",
        "1.21.11",
        "26.1",
        "26.1.1",
        "26.1.2",
        "26.2",
        "26.3",
    };
    return values;
}

void set_version(Options &o, const std::string &text) {
    int a = -1, b = -1, c = 0;
    char dot = 0;
    std::istringstream in(text);
    if (!(in >> a >> dot >> b) || dot != '.')
        throw std::invalid_argument("Invalid Minecraft version");
    if (in.peek() == '.') {
        in.get();
        if (!(in >> c))
            throw std::invalid_argument("Invalid Minecraft patch");
    }
    bool supported =
        a == 1 && b >= 18 && b <= 21 && c >= 0 && !(b == 18 && c > 2) && !(b == 19 && c > 4) &&
        !(b == 20 && c > 6) && !(b == 21 && c > 11);
    supported = supported || (a == 26 && ((b == 1 && c >= 0 && c <= 2) || ((b == 2 || b == 3) && c == 0)));
    if (!in.eof() || !supported)
        throw std::invalid_argument("Generation profile supports 1.18..1.21.11 and released 26.1..26.3 only");
    o.version = text;
    o.modern = a == 26 || (b == 21 && c >= 5);
    // 1.18/1.18.1保留半格距离和旧含水层接缝；1.19.4起晶洞改用精确倒平方根。
    o.generation = a == 26 ? (b >= 3 ? 5 : 4) : b == 18 ? (c < 2 ? -2 : -1) :
                   b == 19 && c < 4 ? -3 : b < 21 ? 0 : c <= 1 ? 1 : c >= 9 ? 4 : 2;
}

bool active(const Options &o, int px, int pz, int cx, int cz) {
    if (o.chunk_constant())
        return std::abs(floor_div(px, 16) - cx) <= o.activity_radius() &&
               std::abs(floor_div(pz, 16) - cz) <= o.activity_radius();
    if (o.custom())
        return active_point(o, px, pz, cx * 16 + 8, cz * 16 + 8);
    int64_t dx = int64_t(cx) * 16 + 8 - px, dz = int64_t(cz) * 16 + 8 - pz;
    return dx * dx + dz * dz < int64_t(o.tick_radius) * o.tick_radius;
}

bool active_point(const Options &o, int px, int pz, int x, int z) {
    if (!o.custom())
        return active(o, px, pz, floor_div(x, 16), floor_div(z, 16));
    int64_t dx = int64_t(x) - px, dz = int64_t(z) - pz;
    if (o.coverage == "square")
        return std::abs(dx) <= o.inner_radius && std::abs(dz) <= o.inner_radius;
    return dx * dx + dz * dz <= int64_t(o.inner_radius) * o.inner_radius;
}

std::pair<int, int> coarse_span(const Options &o, int dz) {
    if (o.chunk_constant()) {
        int q = o.activity_radius();
        return std::abs(dz) <= q ? std::pair{-q, q} : std::pair{0, -1};
    }
    int distance = dz > 0 ? dz * 16 - 7 : dz < 0 ? -dz * 16 - 8 : 0;
    int radius = o.custom() ? o.inner_radius : o.tick_radius;
    int64_t remaining = int64_t(radius) * radius - (o.custom() ? 0 : 1) - int64_t(distance) * distance;
    if (o.coverage == "square")
        remaining = distance <= radius ? int64_t(radius) * radius : -1;
    if (remaining < 0)
        return {0, -1};
    int reach = isqrt(remaining);
    return {-((reach + 8) / 16), (reach + 7) / 16};
}

bool same_hotspot(const Options &o, Center a, Center b) {
    if (o.custom())
        return active_point(o, a.x, a.z, b.x, b.z);
    return active(o, a.x, a.z, floor_div(b.x, 16), floor_div(b.z, 16)) &&
           active(o, b.x, b.z, floor_div(a.x, 16), floor_div(a.z, 16));
}

int64_t center_distance_sq(int center_x, int center_z, int x, int z) {
    int64_t dx = int64_t(x) - center_x, dz = int64_t(z) - center_z;
    return dx * dx + dz * dz;
}

int64_t distance_tenths(int center_x, int center_z, int x, int z) {
    // 显示值与排名使用同一整数距离平方；先乘100再开方，
    // 按四舍五入取到十分之一格，并用修正循环吸收浮点初值误差。
    int64_t squared = center_distance_sq(center_x, center_z, x, z) * 100, root = int64_t(std::sqrt(double(squared)));
    while ((root + 1) * (root + 1) <= squared)
        ++root;
    while (root * root > squared)
        --root;
    if (squared - root * root > root)
        ++root;
    return root;
}

bool final_better(int center_x, int center_z, Center a, Center b) {
    if (a.budding != b.budding)
        return a.budding > b.budding;
    // 横纵差各不超过6000万，64位距离平方和小于2^63，也小于JSON的2^53精确整数界限。
    int64_t da = center_distance_sq(center_x, center_z, a.x, a.z),
            db = center_distance_sq(center_x, center_z, b.x, b.z);
    if (da != db)
        return da < db;
    if (a.x != b.x)
        return a.x < b.x;
    return a.z < b.z;
}

Center optimize(const Options &o, int cx, int cz, const std::vector<Weight> &weights,
                const std::vector<Weight> &origins) {
    Center best;
    int bx = cx * 16, bz = cz * 16;
    if (o.chunk_constant()) {
        int sum = 0, geodes = 0;
        for (auto w : weights)
            if (active(o, bx, bz, w.x, w.z))
                sum += w.budding;
        for (auto w : origins)
            if (active(o, bx, bz, w.x, w.z))
                geodes += w.budding;
        if (!origins.empty() && geodes < o.min_geodes)
            return best;
        // 同一玩家区块内的方形活动范围不变，只需处理搜索边界裁切。
        for (int x = bx; x < bx + 16; ++x)
            for (int z = bz; z < bz + 16; ++z)
                if (o.contains(x, z))
                    return {x, z, sum, geodes};
        return best;
    }
    int diff[16][17]{}, geode_diff[16][17]{};
    auto accumulate = [&](const std::vector<Weight> &source, int (&layer)[16][17], bool blocks) {
        for (auto w : source) {
            int tx = blocks ? w.x : w.x * 16 + 8, tz = blocks ? w.z : w.z * 16 + 8;
            for (int z = 0; z < 16; ++z) {
                int radius = o.custom() ? o.inner_radius : o.tick_radius;
                int64_t dz = int64_t(tz) - bz - z;
                int64_t remaining = int64_t(radius) * radius - (o.custom() ? 0 : 1) - dz * dz;
                if (o.coverage == "square")
                    remaining = std::abs(dz) <= radius ? int64_t(radius) * radius : -1;
                if (remaining < 0)
                    continue;
                int reach = isqrt(remaining), lo = std::max(0, tx - reach - bx), hi = std::min(15, tx + reach - bx);
                if (lo <= hi) {
                    layer[z][lo] += w.budding;
                    layer[z][hi + 1] -= w.budding;
                }
            }
        }
    };
    accumulate(weights, diff, o.custom());
    accumulate(origins, geode_diff, false);
    for (int z = 0; z < 16; ++z) {
        int sum = 0, geodes = 0;
        for (int x = 0; x < 16; ++x) {
            sum += diff[z][x];
            geodes += geode_diff[z][x];
            Center c{bx + x, bz + z, sum, geodes};
            if ((origins.empty() || geodes >= o.min_geodes) && o.contains(c.x, c.z) && center_better(c, best))
                best = c;
        }
    }
    return best;
}

Center evaluate(const Options &o, int px, int pz) {
    if (px < -30000000 || px >= 30000000 || pz < -30000000 || pz >= 30000000)
        throw std::invalid_argument("Player position outside world");
    Simulator simulator(o.seed, o.generation);
    auto terrain = o.terrain ? std::make_unique<Terrain>(o.seed, o.generation) : nullptr;
    Center out{px, pz, 0, 0};
    int pcx = floor_div(px, 16), pcz = floor_div(pz, 16), r = o.activity_radius() + 1;
    for (int cz = pcz - r; cz <= pcz + r; ++cz)
        for (int cx = pcx - r; cx <= pcx + r; ++cx)
            if (in_world(cx, cz)) {
                auto s = simulator.generate(cx, cz, terrain.get());
                if (!s.exists)
                    continue;
                if (active(o, px, pz, cx, cz))
                    ++out.geodes;
                for (auto b : s.budding)
                    if (in_world(floor_div(b.x, 16), floor_div(b.z, 16)) &&
                        active_point(o, px, pz, b.x, b.z))
                        ++out.budding;
            }
    return out;
}

static Results search_impl(const Options &o, const ProgressFn &progress, const Backend &backend, bool allow_two_pass,
                           bool &insufficient) {
    auto checkpoint = [&] {        if (o.checkpoint)
            o.checkpoint();
    };
    if (o.search_rect && (o.min_x > o.max_x || o.min_z > o.max_z || o.min_x < -30000000 ||
                          o.max_x > 29999999 || o.min_z < -30000000 || o.max_z > 29999999))
        throw std::invalid_argument("Invalid rectangular search bounds");
    checkpoint();
    auto start = Clock::now();
    Results out;
    int nthreads = std::clamp(o.threads, 1, 64), r = o.activity_radius() + 1;
    int scan_threads = std::clamp(o.scan_threads > 0 ? o.scan_threads : nthreads, 1, 256);
    int minx, maxx, minz, maxz;
    if (o.search_rect) {
        minx = std::max(world_min_chunk, floor_div(std::max(-30000000, o.min_x), 16));
        maxx = std::min(world_max_chunk, floor_div(std::min(29999999, o.max_x), 16));
        minz = std::max(world_min_chunk, floor_div(std::max(-30000000, o.min_z), 16));
        maxz = std::min(world_max_chunk, floor_div(std::min(29999999, o.max_z), 16));
    } else {
        minx = std::max(world_min_chunk,
                        floor_div(int(std::max<int64_t>(-30000000, int64_t(o.center_x) - o.radius)), 16));
        maxx = std::min(world_max_chunk,
                        floor_div(int(std::min<int64_t>(29999999, int64_t(o.center_x) + o.radius)), 16));
        minz = std::max(world_min_chunk,
                        floor_div(int(std::max<int64_t>(-30000000, int64_t(o.center_z) - o.radius)), 16));
        maxz = std::min(world_max_chunk,
                        floor_div(int(std::min<int64_t>(29999999, int64_t(o.center_z) + o.radius)), 16));
    }
    // 完整候选模式会枚举并保存每个区块；超大范围必须先给候选预算，
    // 否则会在模拟前耗尽内存。
    if (!o.prefilter && int64_t(maxx - minx + 1) * int64_t(maxz - minz + 1) > 16'000'000)
        throw std::invalid_argument("Complete candidate mode requires a candidate budget for this range");
    constexpr int tile_size = 512;
    int tiles_x = (maxx - minx + tile_size) / tile_size, tiles_z = (maxz - minz + tile_size) / tile_size;
    SearchCheckpoint saved(o, start, size_t(tiles_x) * tiles_z, minx, maxx, minz, maxz);
    out.stats = saved.stats;
    out.checkpoint_path = saved.path();
    std::vector<Heap> correct(scan_threads), legacy(scan_threads);
    std::vector<Simulator> scan_engines;
    for (int i = 0; i < scan_threads; ++i)
        scan_engines.emplace_back(o.seed, o.generation);
    if (!saved.scan_complete)
        for (int i = 0; i < scan_threads; ++i) {
            correct[i] = saved.correct;
            legacy[i] = saved.legacy;
        }
    std::atomic<uint64_t> scanned{saved.stats.scanned}, done{saved.stats.resumed_scan};
    std::mutex report;
    std::vector<Candidate> candidates;
    if (saved.scan_complete) {
        candidates = std::move(saved.candidates);
        if (progress) {
            uint64_t units = o.prefilter ? saved.stats.resumed_scan : uint64_t(scanned);
            progress("scan", units, units, uint64_t(scanned));
        }
    } else if (!o.prefilter) {
        for (int z = minz; z <= maxz; ++z) {
            checkpoint();
            for (int x = minx; x <= maxx; ++x)
                if (o.intersects_chunk(x, z))
                    candidates.push_back({x, z, 0});
        }
        scanned = uint64_t(maxx - minx + 1) * (maxz - minz + 1);
        if (progress)
            progress("scan", scanned, scanned, uint64_t(scanned));
    } else {
        bool gpu = false;
        std::string gpu_failure;
        // 约十亿区块以下，GPU初始化和首次回传会抵消较多计算收益。
        std::vector<Candidate> gpu_correct, gpu_legacy;
        const uint64_t chunks = uint64_t(maxx - minx + 1) * (maxz - minz + 1);
        bool gpu_requested = o.scan_backend == "gpu" || (o.scan_backend == "auto" && chunks >= 1000000000ULL);
        saved.prepare_regions(gpu_requested ? 4096 : 512);
        done.store(saved.stats.resumed_scan);
        bool partial_gpu = saved.stats.resumed_scan && saved.stats.scan_backend == "gpu";
        if (partial_gpu || (!saved.stats.resumed_scan && gpu_requested)) {
            gpu = gpu_scan(
                o, minx, maxx, minz, maxz,
                [&] {
                    int cut = o.min_geodes;
                    if (correct[0].size() >= size_t(o.prefilter) && legacy[0].size() >= size_t(o.prefilter))
                        cut = std::max(cut, std::min(correct[0].top().count, legacy[0].top().count));
                    return cut;
                },
                [&](int x, int z, int a, int b) {
                    if (a >= o.min_geodes && o.intersects_chunk(x, z) &&
                        offer(correct[0], {x, z, a}, o.prefilter) && saved.enabled())
                        gpu_correct.push_back({x, z, a});
                    if (b >= o.min_geodes && o.contains(x * 16 + 8, z * 16 + 8) &&
                        offer(legacy[0], {x, z, b}, o.prefilter) && saved.enabled())
                        gpu_legacy.push_back({x, z, b});
                },
                [&](uint64_t completed, uint64_t total) {
                    if (progress)
                        progress("scan", completed, total, uint64_t(scanned));
                },
                [&](uint64_t id) {
                    return saved.scan_done(size_t(id));
                },
                [&](uint64_t id, uint64_t covered) {
                    saved.scan_commit(size_t(id), covered, true, gpu_correct, gpu_legacy);
                    gpu_correct.clear();
                    gpu_legacy.clear();
                },
                &gpu_failure, [&](const std::string &device) { if (backend) backend("gpu", device); });
            if (backend && !gpu) backend("cpu", gpu_failure);
            if (!gpu && (o.scan_backend == "gpu" || partial_gpu))
                throw std::runtime_error(
                    "No compatible OpenCL GPU for this footprint/checkpoint; use a separate CPU checkpoint");
            if (gpu) {
                out.stats.scan_backend = "gpu";
                scanned = chunks;
            }
        }
        if (!gpu) {
            saved.prepare_regions(512);
            done.store(saved.stats.resumed_scan);
            struct Span {
                int z, lo, hi;
            };

            std::vector<Span> footprint;
            for (int dz = -r; dz <= r; ++dz) {
                auto [lo, hi] = coarse_span(o, dz);
                if (lo <= hi)
                    footprint.push_back({dz, lo, hi});
            }
            parallel(size_t(tiles_x) * tiles_z, scan_threads, [&](size_t tile, int id) {
                checkpoint();
                if (saved.scan_done(tile))
                    return;
                std::vector<Candidate> pending_correct, pending_legacy;
                auto accept = [&](bool old, Candidate c) {
                    if (offer(old ? legacy[id] : correct[id], c, o.prefilter) && saved.enabled())
                        (old ? pending_legacy : pending_correct).push_back(c);
                };
                int ax = minx + int(tile % tiles_x) * tile_size, bx = std::min(maxx, ax + tile_size - 1);
                int az = minz + int(tile / tiles_x) * tile_size, bz = std::min(maxz, az + tile_size - 1);
                auto complete = [&] {
                    auto covered = uint64_t(bx - ax + 1) * (bz - az + 1);
                    saved.scan_commit(tile, covered, false, pending_correct, pending_legacy);
                    scanned += covered;
                    uint64_t completed = ++done;
                    if (progress) {
                        std::lock_guard lock(report);
                        progress("scan", completed, uint64_t(tiles_x) * tiles_z, uint64_t(scanned));
                    }
                };
                if (!o.contains(std::clamp(o.center_x, ax * 16, bx * 16 + 15),
                                std::clamp(o.center_z, az * 16, bz * 16 + 15))) {
                    complete();
                    return;
                }
                int width = bx - ax + 1 + 2 * r, height = bz - az + 1 + 2 * r, stride = width + 1;
                if (o.chunk_constant()) {
                    // 方形滑窗只维护进入/离开行与各列总数。
                    // 保留双阈值；历史Java候选池相差一个24位随机数取值。
                    int q = o.activity_radius(), diameter = 2 * q + 1;
                    std::vector<uint8_t> history(size_t(width) * diameter);
                    std::vector<int> columns(width), old_columns(width);
                    std::vector<uint32_t> placements(width);
                    for (int z = 0; z < height; ++z) {
                        checkpoint();
                        scan_engines[id].placement_row(ax - r, az - r + z, width, placements.data());
                        auto *previous = history.data() + size_t(z % diameter) * width;
                        for (int x = 0; x < width; ++x) {
                            int a = placements[x] < 699051, b = placements[x] < 699050;
                            columns[x] += a - (previous[x] & 1);
                            old_columns[x] += b - (previous[x] >> 1);
                            previous[x] = uint8_t(a | (b << 1));
                        }
                        int cz = az - r + z - q;
                        if (cz < az || cz > bz)
                            continue;
                        int a = 0, b = 0, left = r - q;
                        for (int x = left; x < left + diameter; ++x) {
                            a += columns[x];
                            b += old_columns[x];
                        }
                        for (int cx = ax; cx <= bx; ++cx, ++left) {
                            int threshold = o.min_geodes;
                            if (correct[id].size() >= size_t(o.prefilter) && legacy[id].size() >= size_t(o.prefilter))
                                threshold =
                                    std::max(threshold, std::min(correct[id].top().count, legacy[id].top().count));
                            if ((a >= threshold || b >= threshold) && o.intersects_chunk(cx, cz)) {
                                if (a >= o.min_geodes)
                                    accept(false, {cx, cz, a});
                                if (b >= o.min_geodes && o.contains(cx * 16 + 8, cz * 16 + 8))
                                    accept(true, {cx, cz, b});
                            }
                            a += columns[left + diameter] - columns[left];
                            b += old_columns[left + diameter] - old_columns[left];
                        }
                    }
                    complete();
                    return;
                }
                std::vector<int> prefix(size_t(width + 1) * (height + 1)), old(prefix.size()), area(prefix.size());
                std::vector<uint32_t> placements(width);
                for (int z = 0; z < height; ++z) {
                    checkpoint();
                    scan_engines[id].placement_row(ax - r, az - r + z, width, placements.data());
                    int a = 0, b = 0;
                    for (int x = 0; x < width; ++x) {
                        uint32_t bits = placements[x];
                        a += bits < 699051;
                        b += bits < 699050;
                        size_t p = size_t(z + 1) * stride + x + 1;
                        area[p] = area[p - stride] + a;
                        prefix[p] = o.chunk_constant() ? area[p] : a;
                        old[p] = o.chunk_constant() ? old[p - stride] + b : b;
                    }
                }
                auto rectangle = [&](const std::vector<int> &p, int x0, int z0, int x1, int z1) {
                    return p[size_t(z1) * stride + x1] - p[size_t(z1) * stride + x0] - p[size_t(z0) * stride + x1] +
                           p[size_t(z0) * stride + x0];
                };
                int q = o.activity_radius();
                auto sum = [&](const std::vector<int> &p, int x, int z) {
                    int cx = x - ax + r, cz = z - az + r;
                    if (o.chunk_constant())
                        return rectangle(p, cx - q, cz - q, cx + q + 1, cz + q + 1);
                    int count = 0;
                    for (auto span : footprint) {
                        size_t row = size_t(cz + span.z + 1) * stride;
                        count += p[row + cx + span.hi + 1] - p[row + cx + span.lo];
                    }
                    return count;
                };
                constexpr int block = 4;
                for (int z = az; z <= bz; z += block)
                    for (int x = ax; x <= bx; x += block) {
                        int x1 = std::min(bx, x + block - 1), z1 = std::min(bz, z + block - 1),
                            threshold = o.min_geodes;
                        if (x == ax)
                            checkpoint();
                        if (correct[id].size() >= size_t(o.prefilter) && legacy[id].size() >= size_t(o.prefilter))
                            threshold = std::max(threshold, std::min(correct[id].top().count, legacy[id].top().count));
                        // 块内所有活动范围都包含在该来源矩形中；严格比较以保留同分候选。
                        if (rectangle(area, x - ax + r - q, z - az + r - q, x1 - ax + r + q + 1, z1 - az + r + q + 1) <
                            threshold)
                            continue;
                        for (int cz = z; cz <= z1; ++cz)
                            for (int cx = x; cx <= x1; ++cx) {
                                if (!o.intersects_chunk(cx, cz))
                                    continue;
                                int a = sum(prefix, cx, cz);
                                if (a >= o.min_geodes)
                                    accept(false, {cx, cz, a});
                                if (o.contains(cx * 16 + 8, cz * 16 + 8)) {
                                    int b = sum(old, cx, cz);
                                    if (b >= o.min_geodes)
                                        accept(true, {cx, cz, b});
                                }
                            }
                    }
                complete();
            });
        }
        Heap a, b;
        if (saved.enabled()) {
            a = std::move(saved.correct);
            b = std::move(saved.legacy);
        } else
            for (int i = 0; i < scan_threads; ++i) {
                while (!correct[i].empty()) {
                    if (correct[i].size() % 256 == 0)
                        checkpoint();
                    offer(a, correct[i].top(), o.prefilter);
                    correct[i].pop();
                }
                while (!legacy[i].empty()) {
                    if (legacy[i].size() % 256 == 0)
                        checkpoint();
                    offer(b, legacy[i].top(), o.prefilter);
                    legacy[i].pop();
                }
            }
        while (!a.empty()) {
            if (a.size() % 256 == 0)
                checkpoint();
            candidates.push_back(a.top());
            a.pop();
        }
        while (!b.empty()) {
            if (b.size() % 256 == 0)
                checkpoint();
            candidates.push_back(b.top());
            b.pop();
        }
        std::sort(candidates.begin(), candidates.end(), [](auto a, auto b) {
            return std::pair(a.x, a.z) < std::pair(b.x, b.z);
        });
        candidates.erase(std::unique(candidates.begin(), candidates.end(),
                                     [](auto a, auto b) {
                                         return a.x == b.x && a.z == b.z;
                                     }),
                         candidates.end());
    }
    out.stats.scanned = scanned;
    out.stats.candidates = candidates.size();
    out.stats.scan_ms = saved.scan_complete ? saved.stats.scan_ms : saved.scan_ms();
    saved.scan_finish(candidates, out.stats);
    // 模拟阶段的空间分块：分块越大，线程内地形/雕刻器缓存复用越好；
    // 但分块过大会让并行粒度不足。实测最优分块随搜索范围缩放，
    // 取"区域边长约 1/10"，并夹在 [128, 256]。
    const int simulation_tile = std::clamp((maxx - minx + 1) / 10, 128, 256);
    // ── 两段式精确裁剪 ──
    // 地形模拟只会删除母岩：控制点失效会取消整座晶洞，覆盖判定只会过滤已生成的母岩，
    // 不会新增。因此"纯几何模型（无地形）"算出的母岩数是任意挂机点上的严格上界。
    // 先用它给每个候选窗口定上界，只让上界仍可能进入最终榜单的候选参与昂贵的地形模拟。
    // 由于上界严格成立，这一步不改变最终榜单，只是少算了注定进不了榜的区域。
    std::vector<Candidate> excluded;
    int excluded_max_bound = 0;
    int two_pass_floor_bound = 0;
    uint64_t two_pass_kept_count = 0;
    bool two_pass = false;
    if (allow_two_pass && o.terrain && o.prefilter > 0 && candidates.size() > 4000) {
        struct StoneSpan {
            int z, lo, hi;
        };
        std::map<int, std::vector<std::pair<int, int>>> rows;
        for (auto c : candidates)
            for (int z = c.z - r; z <= c.z + r; ++z)
                if (z >= world_min_chunk && z <= world_max_chunk)
                    rows[z].push_back({std::max(world_min_chunk, c.x - r), std::min(world_max_chunk, c.x + r)});
        std::map<uint64_t, std::vector<StoneSpan>> tiles;
        for (auto &[z, line] : rows) {
            std::sort(line.begin(), line.end());
            std::vector<std::pair<int, int>> merged_rows;
            for (auto p : line) {
                if (merged_rows.empty() || p.first > merged_rows.back().second + 1)
                    merged_rows.push_back(p);
                else
                    merged_rows.back().second = std::max(merged_rows.back().second, p.second);
            }
            for (auto [lo, hi] : merged_rows)
                for (int x = lo; x <= hi;) {
                    int tx = floor_div(x, simulation_tile), end = std::min(hi, (tx + 1) * simulation_tile - 1);
                    tiles[key(tx, floor_div(z, simulation_tile))].push_back({z, x, end});
                    x = end + 1;
                }
        }
        rows.clear();
        std::vector<std::vector<StoneSpan>> stone_work;
        std::vector<uint64_t> stone_keys;
        stone_work.reserve(tiles.size());
        for (auto &[k, list] : tiles) {
            stone_keys.push_back(k);
            stone_work.push_back(std::move(list));
        }
        tiles.clear();
        // 上界那一遍：关闭地形，只算纯几何母岩数。权重键与正式精排保持一致，
        // 自定义覆盖按母岩实际 XZ 计分，其余按区块计。
        // 这一遍在后段任务里可能比地形那一遍还久，必须单独上报进度：
        // 否则界面会一直停在粗筛的最后那一帧，看起来像卡死。
        std::vector<Weights> stone_local(nthreads);
        std::vector<uint64_t> stone_counts(nthreads, 0);
        uint64_t stone_total = stone_work.size();
        std::atomic<uint64_t> stone_done{0};
        auto stone_reported = Clock::now();
        parallel(stone_total, nthreads, [&](size_t i, int id) {
            checkpoint();
            auto &simulator = scan_engines[id];
            auto &weights = stone_local[id];
            for (auto row : stone_work[i])
                for (int x = row.lo; x <= row.hi; ++x) {
                    if ((x - row.lo) % 32 == 0)
                        checkpoint();
                    if (!simulator.check(x, row.z))
                        continue;
                    ++stone_counts[size_t(id)];
                    auto s = simulator.generate(x, row.z, nullptr);
                    for (auto p : s.budding) {
                        int cx = floor_div(p.x, 16), cz = floor_div(p.z, 16);
                        if (in_world(cx, cz))
                            ++weights[o.custom() ? key(p.x, p.z) : key(cx, cz)];
                    }
                }
            uint64_t completed = ++stone_done;
            if (progress) {
                std::lock_guard lock(report);
                auto now = Clock::now();
                if (completed == stone_total || now - stone_reported >= std::chrono::milliseconds(60)) {
                    stone_reported = now;
                    progress("stone", completed, stone_total, uint64_t(scanned));
                }
            }
        });
        stone_work.clear();
        // 统一折算到区块粒度：上界只需要"每个区块最多能贡献多少母岩"，
        // 上界只需要"每个位置最多能贡献多少母岩"，与活动形状无关：
        // 非自定义按区块汇总，自定义按实际母岩 XZ 建区块索引后逐点判定。
        Weights chunk_totals;
        std::unordered_map<uint64_t, std::vector<Weight>> stone_buckets;
        for (auto i = 0; i < nthreads; ++i)
            out.stats.stone_simulated += stone_counts[size_t(i)];
        for (auto &m : stone_local) {
            for (auto [k, v] : m) {
                if (o.custom()) {
                    int x = key_x(k), z = key_z(k);
                    stone_buckets[key(floor_div(x, 16), floor_div(z, 16))].push_back({x, z, v});
                } else {
                    chunk_totals[k] += v;
                }
            }
            m.clear();
        }
        // 每个候选的活动窗口上界。coarse_span() 给出的行跨度已经把
        // "玩家在候选区块内任意位置"的偏移算进去了，因此按它求和覆盖了
        // 候选区块内所有挂机点，是对组内上界的严格上界。方形、欧式圆盘、
        // 旧版区块中心圆和新版按区块方形都由该函数统一给出。
        int q = o.activity_radius() + 1;
        std::vector<std::pair<int, size_t>> ranked_bounds(candidates.size());
        parallel(candidates.size(), nthreads, [&](size_t i, int) {
            checkpoint();
            auto c = candidates[i];
            int64_t sum = 0;
            if (o.chunk_constant()) {
                // 区块常量规则：候选区块内任意位置的活动区块集合相同，
                // 直接用按区块汇总的方形窗口和，等价于组内上界。
                for (int dz = -q; dz <= q; ++dz) {
                    auto span = coarse_span(o, dz);
                    if (span.first > span.second)
                        continue;
                    for (int dx = span.first; dx <= span.second; ++dx) {
                        auto it = chunk_totals.find(key(c.x + dx, c.z + dz));
                        if (it != chunk_totals.end())
                            sum += it->second;
                    }
                }
            } else {
                // 旧版圆形与自定义范围：直接复用正式精排的组内上界算法
                // （把权重点夹到候选区块内再判断活动形状），只是权重换成石模型。
                // 这样上界与真实评分同形同尺度，没有"外扩"带来的虚高。
                int wradius = o.activity_radius() + 1;
                for (int wz = c.z - wradius; wz <= c.z + wradius; ++wz)
                    for (int wx = c.x - wradius; wx <= c.x + wradius; ++wx) {
                        auto ck = key(wx, wz);
                        if (!o.custom()) {
                            auto found = chunk_totals.find(ck);
                            if (found == chunk_totals.end())
                                continue;
                            int bx = wx * 16 + 8, bz = wz * 16 + 8;
                            int px = std::clamp(bx, c.x * 16, c.x * 16 + 15);
                            int pz = std::clamp(bz, c.z * 16, c.z * 16 + 15);
                            if (active_point(o, px, pz, bx, bz))
                                sum += found->second;
                        } else {
                            auto bucket = stone_buckets.find(ck);
                            if (bucket == stone_buckets.end())
                                continue;
                            for (auto w : bucket->second) {
                                int px = std::clamp(w.x, c.x * 16, c.x * 16 + 15);
                                int pz = std::clamp(w.z, c.z * 16, c.z * 16 + 15);
                                if (active_point(o, px, pz, w.x, w.z))
                                    sum += w.budding;
                            }
                        }
                    }
            }
            ranked_bounds[i] = {int(std::min<int64_t>(sum, INT32_MAX)), i};
            if ((i % 4096) == 0)
                checkpoint();
        });
        chunk_totals.clear();
        std::sort(ranked_bounds.begin(), ranked_bounds.end(),
                  [](const auto &a, const auto &b) { return a.first > b.first; });
        // 保留规则按"上界阈值"而不是固定条数：高分候选往往挤在少数热点里，
        // 只按条数截断会漏掉其他区域本可进榜的热点。这里以第 limit 高的上界为基准
        // 再留 25% 余量，随着搜索范围自动伸缩，同时保证空间上覆盖足够多的热点。
        int cutIndex = int(std::min<size_t>(size_t(std::max(0, o.limit - 1)), ranked_bounds.size() - 1));
        int topBound = ranked_bounds[cutIndex].first;
        bool circleShape = !o.chunk_constant() && o.coverage != "square";
        // 圆形与自定义范围的石模型上界比地形实分高约 15%（地形删掉的母岩），
        // 方形规则只有约 10%，因此圆形留更多余量；不足仍由自检回退兜底。
        int floorBound = std::max(o.min_budding, circleShape ? topBound * 13 / 20 : topBound * 3 / 4);
        two_pass_floor_bound = floorBound;
        size_t keep = 0;
        while (keep < ranked_bounds.size() && ranked_bounds[keep].first >= floorBound)
            ++keep;
        keep = std::max<size_t>(keep, 4000);
        keep = std::min(keep, candidates.size());
        std::vector<Candidate> kept;
        kept.reserve(keep);
        for (size_t i = 0; i < keep; ++i)
            kept.push_back(candidates[ranked_bounds[i].second]);
        for (size_t i = keep; i < ranked_bounds.size(); ++i)
            excluded.push_back(candidates[ranked_bounds[i].second]);
        excluded_max_bound = excluded.empty() ? 0 : ranked_bounds[keep].first;
        std::sort(kept.begin(), kept.end(), [](auto a, auto b) { return std::pair(a.x, a.z) < std::pair(b.x, b.z); });
        candidates.swap(kept);
        two_pass_kept_count = candidates.size();
        two_pass = true;
    }

    out.approximate = o.prefilter != 0;
    out.scan_backend = out.stats.scan_backend;
#ifdef GEODE_SCAN_PROBE
    for (auto c : candidates)
        out.centers.push_back({c.x, c.z, c.count, 0});
    out.stats.total_ms = saved.total_ms();
    return out;
#endif
    if (candidates.empty()) {
        out.stats.total_ms = saved.total_ms();
        return out;
    }

    // 先合并重叠的来源矩形，再测试放置，避免逐候选窗口重复检查。
    struct Row {
        int z;
        std::vector<std::pair<int, int>> intervals;
    };

    std::vector<Row> intervals;
    if (!o.prefilter) {
        for (int z = std::max(world_min_chunk, minz - r); z <= std::min(world_max_chunk, maxz + r); ++z) {
            checkpoint();
            int lo = minx, hi = maxx;
            if (o.shape == "circle") {
                int64_t dz = int64_t(std::clamp(o.center_z, (z - r) * 16, (z + r) * 16 + 15)) - o.center_z;
                int reach = isqrt(int64_t(o.radius) * o.radius - dz * dz);
                lo = std::max(lo, floor_div(o.center_x - reach, 16));
                hi = std::min(hi, floor_div(o.center_x + reach, 16));
            }
            intervals.push_back({z, {{std::max(world_min_chunk, lo - r), std::min(world_max_chunk, hi + r)}}});
        }
    } else {
        std::map<int, std::vector<std::pair<int, int>>> rows;
        for (auto c : candidates)
            for (int z = c.z - r; z <= c.z + r; ++z)
                if (z >= world_min_chunk && z <= world_max_chunk) {
                    checkpoint();
                    rows[z].push_back({std::max(world_min_chunk, c.x - r), std::min(world_max_chunk, c.x + r)});
                }
        for (auto &[z, line] : rows) {
            checkpoint();
            std::sort(line.begin(), line.end());
            Row row{z, {}};
            for (auto p : line) {
                if (row.intervals.empty() || p.first > row.intervals.back().second + 1)
                    row.intervals.push_back(p);
                else
                    row.intervals.back().second = std::max(row.intervals.back().second, p.second);
            }
            intervals.push_back(std::move(row));
        }
    }
    std::vector<Weights> local_weights(nthreads);
    std::vector<std::vector<uint64_t>> local_origins(nthreads);
    std::vector<uint64_t> local_count(nthreads), local_rejected(nthreads);
    done = saved.stats.resumed_simulation;
    auto simulation_start = Clock::now();
    double simulation_paused = saved.pause_ms();
    std::vector<std::unique_ptr<Terrain>> terrains(nthreads);
    if (o.terrain)
        for (auto &t : terrains)
            t = std::make_unique<Terrain>(o.seed, o.generation);

    struct Span {
        int z, lo, hi;
    };

    std::map<uint64_t, std::vector<Span>> tile_rows;
    for (const auto &row : intervals)
        for (auto [lo, hi] : row.intervals)
            for (int x = lo; x <= hi;) {
                int tx = floor_div(x, simulation_tile), end = std::min(hi, (tx + 1) * simulation_tile - 1);
                tile_rows[key(tx, floor_div(row.z, simulation_tile))].push_back({row.z, x, end});
                x = end + 1;
            }
    std::vector<std::vector<Span>> work;
    std::vector<uint64_t> work_keys;
    work.reserve(tile_rows.size());
    for (auto &[k, rows] : tile_rows) {
        work_keys.push_back(k);
        work.push_back(std::move(rows));
    }
    tile_rows.clear();
    auto simulate_reported = Clock::now();
    if (progress && saved.stats.resumed_simulation)
        progress("simulate", done, work.size(), uint64_t(scanned));
    parallel(work.size(), nthreads, [&](size_t i, int id) {
        checkpoint();
        if (saved.simulation_done(work_keys[i]))
            return;
        auto &simulator = scan_engines[id];
        Weights delta;
        std::vector<uint64_t> source;
        uint64_t count = 0, rejected = 0;
        auto &weights = saved.enabled() ? delta : local_weights[id];
        auto &origin = saved.enabled() ? source : local_origins[id];
        for (auto row : work[i])
            for (int x = row.lo; x <= row.hi; ++x) {
                if ((x - row.lo) % 32 == 0)
                    checkpoint();
                if (!simulator.check(x, row.z))
                    continue;
                auto s = simulator.generate(x, row.z, terrains[id].get());
                ++count;
                rejected += s.rejected;
                if (s.exists)
                    origin.push_back(key(x, row.z));
                for (auto p : s.budding) {
                    int cx = floor_div(p.x, 16), cz = floor_div(p.z, 16);
                    if (in_world(cx, cz))
                        ++weights[o.custom() ? key(p.x, p.z) : key(cx, cz)];
                }
            }
        if (saved.enabled())
            saved.simulation_commit(work_keys[i], delta, source, count, rejected, simulation_start, simulation_paused);
        else {
            local_count[id] += count;
            local_rejected[id] += rejected;
        }
        uint64_t completed = ++done;
        // 按时间而不是固定条数节流：固定 32 条会让最后不足 32 个分区永远不上报，
        // 进度条卡在 99.x% 不动。
        if (progress) {
            std::lock_guard lock(report);
            auto now = Clock::now();
            if (completed == work.size() || now - simulate_reported >= std::chrono::milliseconds(60)) {
                simulate_reported = now;
                progress("simulate", completed, work.size(), uint64_t(scanned));
            }
        }
    });
    Weights weights = std::move(saved.weights), origins = std::move(saved.origins);
    if (saved.enabled()) {
        out.stats.simulated = saved.stats.simulated;
        out.stats.rejected = saved.stats.rejected;
    }
    for (int id = 0; id < nthreads; ++id) {
        out.stats.simulated += local_count[id];
        out.stats.rejected += local_rejected[id];
        for (auto [k, v] : local_weights[id])
            weights[k] += v;
        for (auto k : local_origins[id])
            origins.emplace(k, 1);
    }
    // 自定义范围按母岩实际XZ统计；按区块建立索引，避免每个候选遍历全部母岩。
    std::unordered_map<uint64_t, std::vector<Weight>> point_buckets;
    if (o.custom())
        for (auto [k, count] : weights) {
            int x = int(int32_t(k >> 32)), z = int(int32_t(k));
            point_buckets[key(floor_div(x, 16), floor_div(z, 16))].push_back({x, z, count});
        }
    // 稠密网格只在权重键为区块坐标且包围盒有限时启用；大范围或自定义范围
    // 继续走原哈希路径，行为与内存占用都不变。
    DenseGrid weight_grid, origin_grid;
    bool dense = false;
    if (!o.custom()) {
        int pad = o.activity_radius() + 2;
        int lo_x = std::max(world_min_chunk, minx - pad), hi_x = std::min(world_max_chunk, maxx + pad);
        int lo_z = std::max(world_min_chunk, minz - pad), hi_z = std::min(world_max_chunk, maxz + pad);
        int64_t cells = int64_t(hi_x - lo_x + 1) * int64_t(hi_z - lo_z + 1);
        if (cells > 0 && cells <= 40000000) {
            weight_grid.build(weights, lo_x, hi_x, lo_z, hi_z);
            origin_grid.build(origins, lo_x, hi_x, lo_z, hi_z);
            dense = true;
        }
    }
    auto gather_weights = [&](int cx, int cz, int radius) {
        if (!o.custom() && dense) {
            // 与 gather() 相同的遍历顺序与"跳过零值"规则，只把哈希探测换成数组读取。
            std::vector<Weight> result;
            result.reserve(256);
            for (int z = cz - radius; z <= cz + radius; ++z)
                for (int x = cx - radius; x <= cx + radius; ++x) {
                    int v = weight_grid.at(x, z);
                    if (v)
                        result.push_back({x, z, v});
                }
            return result;
        }
        if (!o.custom())
            return gather(weights, cx, cz, radius);
        std::vector<Weight> result;
        for (int z = cz - radius; z <= cz + radius; ++z)
            for (int x = cx - radius; x <= cx + radius; ++x) {
                auto it = point_buckets.find(key(x, z));
                if (it != point_buckets.end())
                    result.insert(result.end(), it->second.begin(), it->second.end());
            }
        return result;
    };
    auto gather_origins = [&](int cx, int cz, int radius) {
        if (!o.custom() && dense) {
            std::vector<Weight> result;
            result.reserve(64);
            for (int z = cz - radius; z <= cz + radius; ++z)
                for (int x = cx - radius; x <= cx + radius; ++x) {
                    int v = origin_grid.at(x, z);
                    if (v)
                        result.push_back({x, z, v});
                }
            return result;
        }
        return gather(origins, cx, cz, radius);
    };
    out.stats.weights = weights.size();
    out.stats.simulate_ms = saved.simulation_ms(simulation_start, simulation_paused);
    auto rank_start = Clock::now();
    double rank_paused = saved.pause_ms();
    if (progress)
        progress("rank", 0, out.stats.candidates, uint64_t(scanned));

    struct Prepared {
        Candidate chunk;
        Center center;
        int bound;
    };

    std::atomic<uint64_t> rank_done{0};
    auto rank_reported = Clock::now();
    std::vector<Prepared> prepared(candidates.size());
    parallel(candidates.size(), nthreads, [&](size_t i, int) {
        checkpoint();
        auto c = candidates[i];
        int x = c.x * 16 + 8, z = c.z * 16 + 8;
        if (!o.contains(x, z)) {
            x = std::clamp(o.center_x, c.x * 16, c.x * 16 + 15);
            z = std::clamp(o.center_z, c.z * 16, c.z * 16 + 15);
        }
        int bound = 0;
        Center score{x, z, 0, 0};
        if (dense && o.chunk_constant()) {
            // 区块常量规则下，候选区块内任意整数位置对应的活动区块集合都等于
            // 以该区块为中心的方形邻域（夹取到区块内不会改变所属区块），
            // 因此"组内上界"与"中心评分"是同一个窗口和，可用前缀和一次求出。
            int q = o.activity_radius();
            int64_t sum = weight_grid.rect(c.x - q, c.z - q, c.x + q + 1, c.z + q + 1);
            bound = int(sum);
            score.budding = int(sum);
            score.geodes = int(origin_grid.rect(c.x - q, c.z - q, c.x + q + 1, c.z + q + 1));
        } else {
            auto nearby = gather_weights(c.x, c.z, o.activity_radius() + 1);
            for (auto w : nearby) {
                int wx = o.custom() ? w.x : w.x * 16 + 8, wz = o.custom() ? w.z : w.z * 16 + 8;
                int px = std::clamp(wx, c.x * 16, c.x * 16 + 15),
                    pz = std::clamp(wz, c.z * 16, c.z * 16 + 15);
                if (active_point(o, px, pz, wx, wz))
                    bound += w.budding;
                if (active_point(o, x, z, wx, wz))
                    score.budding += w.budding;
            }
            for (auto w : gather_origins(c.x, c.z, o.activity_radius()))
                if (active(o, x, z, w.x, w.z))
                    score.geodes += w.budding;
        }
        prepared[i] = {c, score, bound};
        uint64_t done_count = ++rank_done;
        if (progress) {
            std::lock_guard lock(report);
            auto now = Clock::now();
            if (done_count == prepared.size() || now - rank_reported >= std::chrono::milliseconds(60)) {
                rank_reported = now;
                progress("rank", done_count, prepared.size(), uint64_t(scanned));
            }
        }
    });
    std::sort(prepared.begin(), prepared.end(), [](const auto &a, const auto &b) {
        return center_better(a.center, b.center);
    });

    struct Hotspot {
        Center anchor;
        std::vector<Prepared> members;
    };

    std::vector<Hotspot> hotspots;
    std::unordered_map<uint64_t, std::vector<size_t>> buckets;
    int bucket_width = (o.activity_radius() + 1) * 16;
    std::atomic<uint64_t> pruned{}, optimized{};
    for (auto p : prepared) {
        checkpoint();
        if (p.bound < o.min_budding) {
            ++pruned;
            continue;
        }
        int bx = floor_div(p.center.x, bucket_width), bz = floor_div(p.center.z, bucket_width);
        size_t selected = hotspots.size();
        for (int z = bz - 1; z <= bz + 1; ++z)
            for (int x = bx - 1; x <= bx + 1; ++x) {
                auto found = buckets.find(key(x, z));
                if (found == buckets.end())
                    continue;
                for (size_t id : found->second)
                    if (id < selected && same_hotspot(o, hotspots[id].anchor, p.center))
                        selected = id;
            }
        if (selected == hotspots.size()) {
            buckets[key(bx, bz)].push_back(selected);
            hotspots.push_back({p.center, {}});
        }
        hotspots[selected].members.push_back(p);
    }
    prepared.clear();
    candidates.clear();
    std::vector<Center> ranked(hotspots.size());
    parallel(hotspots.size(), nthreads, [&](size_t i, int) {
        checkpoint();
        auto &members = hotspots[i].members;
        std::sort(members.begin(), members.end(), [](const auto &a, const auto &b) {
            return a.bound > b.bound;
        });
        Center best;
        // 使用可靠几何上界排除成员，不固定截取每组前五个。
        for (size_t j = 0; j < members.size(); ++j) {
            checkpoint();
            auto p = members[j];
            if (p.bound < best.budding) {
                pruned += members.size() - j;
                break;
            }
            auto source = gather_origins(p.chunk.x, p.chunk.z, o.activity_radius());
            if (source.empty())
                continue;
            auto c = optimize(o, p.chunk.x, p.chunk.z, gather_weights(p.chunk.x, p.chunk.z, o.activity_radius() + 1),
                              source);
            ++optimized;
            if (center_better(c, best))
                best = c;
        }
        ranked[i] = best;
    });
    // 最终榜单以距实际搜索中心的距离打破同分，晶洞数不参与。
    // 上述热点归组保留自己的顺序，因此代表位置集合不变；
    // 这里只决定相互重叠的代表点保留谁，以及最终先后顺序。
    std::sort(ranked.begin(), ranked.end(), [&](const Center &a, const Center &b) {
        return final_better(o.center_x, o.center_z, a, b);
    });
    for (auto c : ranked) {
        if (c.budding < o.min_budding)
            continue;
        bool close = false;
        for (auto kept : out.centers)
            if (same_hotspot(o, c, kept)) {
                close = true;
                break;
            }
        if (!close) {
            if (c.geodes >= o.min_geodes)
                out.centers.push_back(c);
            if (out.centers.size() >= size_t(o.limit))
                break;
        }
    }
    // 裁剪自检：被排除候选的上界必须严格低于最终榜单的最低分，否则说明保留的
    // 候选不够，可能漏掉本应进榜的热点。这里只判定并记录，不静默改动输出。
    if (two_pass) {
        int threshold = out.centers.empty() ? 0 : out.centers.back().budding;
        insufficient = !out.centers.empty() && excluded_max_bound >= threshold;
        std::fprintf(stderr, "TWOPASS keptBound=%d excluded=%zu excludedMaxBound=%d threshold=%d sufficient=%d\n",
                     two_pass_floor_bound, excluded.size(), excluded_max_bound, threshold, int(!insufficient));
    }
    out.stats.two_pass = two_pass;
    out.stats.two_pass_kept = two_pass_kept_count;
    out.stats.two_pass_excluded_bound = excluded_max_bound;
    out.stats.two_pass_floor_bound = two_pass_floor_bound;
    out.stats.hotspots = hotspots.size();
    out.stats.optimized = optimized;
    out.stats.pruned = pruned;
    out.stats.rank_ms = elapsed(rank_start) - (saved.pause_ms() - rank_paused);
    out.stats.total_ms = saved.total_ms();
    return out;
}
Results search(const Options &o, const ProgressFn &progress, const Backend &backend) {
    bool insufficient = false;
    Results first = search_impl(o, progress, backend, true, insufficient);
    // 两段式裁剪依赖"石模型母岩数是严格上界"。自检若发现保留的候选不足，
    // 就按未裁剪的完整地形模拟重跑一次，保证不会因为裁剪漏掉热点。
    if (insufficient) {
        std::fprintf(stderr, "TWOPASS insufficient, retrying without pruning\n");
        Results retried = search_impl(o, progress, backend, false, insufficient);
        retried.stats.two_pass_retried = true;
        return retried;
    }
    return first;
}

Results search(const Options &options, Control &control, const ProgressFn &progress, const Backend &backend) {
    Options o = options;
    auto legacy = o.checkpoint;
    o.checkpoint = [&] {
        if (legacy)
            legacy();
        control.checkpoint();
    };
    o.paused_ms = [&control] {
        return control.paused_ms();
    };
    auto start = Clock::now();
    try {
        return search(o, progress, backend);
    } catch (const Cancelled &) {
        // 取消是正常终止：已完成的断点保留，内存结果由宿主丢弃。
        Results out;
        out.cancelled = true;
        out.approximate = o.prefilter != 0;
        out.stats.total_ms = elapsed(start) - control.paused_ms();
        out.stats.scan_ms = out.stats.total_ms;
        return out;
    }
}
}