#include "search.hpp"
#include "terrain.hpp"
#include "coarse_gpu.hpp"
#include "checkpoint.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
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

  public:
    Stats stats;
    Heap correct, legacy;
    Weights weights, origins;
    std::vector<Candidate> candidates;
    bool scan_complete = false;

    SearchCheckpoint(const Options &options, Clock::time_point begin, size_t scan_count)
        : o(options), start(begin), scans(o.checkpoint_directory.empty() ? 0 : scan_count) {
        pause_start = o.paused_ms ? o.paused_ms() : 0;
        if (!enabled())
            return;
        // 生成逻辑、归组或任务归属变化时，必须更新模型/工作划分修订。
        std::ostringstream signature;
        signature << "geode-native-v2;model=20261027-range;work=512-4096-64;int=" << sizeof(int) << ";seed=" << o.seed << ';'
                  << o.version << ';' << o.generation << ';' << o.terrain << ';' << o.modern << ';' << o.center_x << ';'
                  << o.center_z << ';' << o.radius << ';' << o.shape << ';' << o.full_world << ';' << o.tick_radius
                  << ';' << o.simulation_distance << ';' << o.min_geodes << ';' << o.min_budding << ';' << o.prefilter
                  << ';' << o.limit << ';' << o.threads << ';' << o.scan_backend << ';' << kRankingRuleVersion
                  << ';' << o.coverage << ';' << o.inner_radius << ';' << CUBIOMES_REVISION;
        journal = std::make_unique<CheckpointJournal>(o.checkpoint_directory, signature.str());
        journal->replay([&](CheckpointRecord &record) {
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
            } else
                throw std::runtime_error("Unknown geode checkpoint record");
            record.end();
        });
        history = stats.total_ms;
        scan_history = stats.scan_ms;
        simulation_history = stats.simulate_ms;
        stats.resumed_ms = history;
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

    bool scan_done(size_t id) {
        if (!enabled())
            return false;
        std::lock_guard lock(mutex);
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
}

bool Options::contains(int x, int z) const {
    if (x < -30000000 || x >= 30000000 || z < -30000000 || z >= 30000000)
        return false;
    int64_t dx = int64_t(x) - center_x, dz = int64_t(z) - center_z;
    if (std::abs(dx) > radius || std::abs(dz) > radius)
        return false;
    return shape == "square" || dx * dx + dz * dz <= int64_t(radius) * radius;
}

bool Options::intersects_chunk(int x, int z) const {
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

Results search(const Options &o, const std::function<void(const std::string &, uint64_t, uint64_t)> &progress) {
    auto checkpoint = [&] {
        if (o.checkpoint)
            o.checkpoint();
    };
    checkpoint();
    auto start = Clock::now();
    Results out;
    int nthreads = o.threads, r = o.activity_radius() + 1;
    int minx =
        std::max(world_min_chunk, floor_div(int(std::max<int64_t>(-30000000, int64_t(o.center_x) - o.radius)), 16));
    int maxx =
        std::min(world_max_chunk, floor_div(int(std::min<int64_t>(29999999, int64_t(o.center_x) + o.radius)), 16));
    int minz =
        std::max(world_min_chunk, floor_div(int(std::max<int64_t>(-30000000, int64_t(o.center_z) - o.radius)), 16));
    int maxz =
        std::min(world_max_chunk, floor_div(int(std::min<int64_t>(29999999, int64_t(o.center_z) + o.radius)), 16));
    constexpr int tile_size = 512;
    int tiles_x = (maxx - minx + tile_size) / tile_size, tiles_z = (maxz - minz + tile_size) / tile_size;
    SearchCheckpoint saved(o, start, size_t(tiles_x) * tiles_z);
    out.stats = saved.stats;
    out.checkpoint_path = saved.path();
    std::vector<Heap> correct(nthreads), legacy(nthreads);
    std::vector<Simulator> scan_engines;
    for (int i = 0; i < nthreads; ++i)
        scan_engines.emplace_back(o.seed, o.generation);
    if (!saved.scan_complete)
        for (int i = 0; i < nthreads; ++i) {
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
            progress("scan", units, units);
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
            progress("scan", scanned, scanned);
    } else {
        bool gpu = false;
        // 约十亿区块以下，GPU初始化和首次回传会抵消较多计算收益。
        std::vector<Candidate> gpu_correct, gpu_legacy;
        bool partial_gpu = saved.stats.resumed_scan && saved.stats.scan_backend == "gpu";
        if (partial_gpu ||
            (!saved.stats.resumed_scan &&
             (o.scan_backend == "gpu" ||
              (o.scan_backend == "auto" && uint64_t(maxx - minx + 1) * (maxz - minz + 1) >= 1000000000ULL)))) {
            gpu = gpu_scan(
                o, minx, maxx, minz, maxz,
                [&] {
                    int cut = o.min_geodes;
                    if (correct[0].size() >= size_t(o.prefilter) && legacy[0].size() >= size_t(o.prefilter))
                        cut = std::max(cut, std::min(correct[0].top().count, legacy[0].top().count));
                    return cut;
                },
                [&](int x, int z, int a, int b) {
                    if (a >= o.min_geodes && offer(correct[0], {x, z, a}, o.prefilter) && saved.enabled())
                        gpu_correct.push_back({x, z, a});
                    if (b >= o.min_geodes && o.contains(x * 16 + 8, z * 16 + 8) &&
                        offer(legacy[0], {x, z, b}, o.prefilter) && saved.enabled())
                        gpu_legacy.push_back({x, z, b});
                },
                [&](uint64_t completed, uint64_t total) {
                    if (progress)
                        progress("scan", completed, total);
                },
                [&](uint64_t id) {
                    return saved.scan_done(size_t(id));
                },
                [&](uint64_t id, uint64_t covered) {
                    saved.scan_commit(size_t(id), covered, true, gpu_correct, gpu_legacy);
                    gpu_correct.clear();
                    gpu_legacy.clear();
                });
            if (!gpu && (o.scan_backend == "gpu" || partial_gpu))
                throw std::runtime_error(
                    "No compatible OpenCL GPU for this footprint/checkpoint; use a separate CPU checkpoint");
            if (gpu) {
                out.stats.scan_backend = "gpu";
                scanned = uint64_t(maxx - minx + 1) * (maxz - minz + 1);
            }
        }
        if (!gpu) {
            struct Span {
                int z, lo, hi;
            };

            std::vector<Span> footprint;
            for (int dz = -r; dz <= r; ++dz) {
                auto [lo, hi] = coarse_span(o, dz);
                if (lo <= hi)
                    footprint.push_back({dz, lo, hi});
            }
            parallel(size_t(tiles_x) * tiles_z, nthreads, [&](size_t tile, int id) {
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
                        progress("scan", completed, uint64_t(tiles_x) * tiles_z);
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
            for (int i = 0; i < nthreads; ++i) {
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
    // 64区块瓦片加雕刻器八区块边缘共80×80，可放入8192来源缓存。
    constexpr int simulation_tile = 64;
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
    if (progress && saved.stats.resumed_simulation)
        progress("simulate", done, work.size());
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
        if (progress && completed % 32 == 0) {
            std::lock_guard lock(report);
            progress("simulate", completed, work.size());
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
    auto gather_weights = [&](int cx, int cz, int radius) {
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
    out.stats.weights = weights.size();
    out.stats.simulate_ms = saved.simulation_ms(simulation_start, simulation_paused);
    auto rank_start = Clock::now();
    double rank_paused = saved.pause_ms();
    if (progress)
        progress("rank", 0, out.stats.candidates);

    struct Prepared {
        Candidate chunk;
        Center center;
        int bound;
    };

    std::vector<Prepared> prepared(candidates.size());
    parallel(candidates.size(), nthreads, [&](size_t i, int) {
        checkpoint();
        auto c = candidates[i];
        auto nearby = gather_weights(c.x, c.z, o.activity_radius() + 1);
        int x = c.x * 16 + 8, z = c.z * 16 + 8;
        if (!o.contains(x, z)) {
            x = std::clamp(o.center_x, c.x * 16, c.x * 16 + 15);
            z = std::clamp(o.center_z, c.z * 16, c.z * 16 + 15);
        }
        int bound = 0;
        Center score{x, z, 0, 0};
        for (auto w : nearby) {
            int wx = o.custom() ? w.x : w.x * 16 + 8, wz = o.custom() ? w.z : w.z * 16 + 8;
            int px = std::clamp(wx, c.x * 16, c.x * 16 + 15),
                pz = std::clamp(wz, c.z * 16, c.z * 16 + 15);
            if (active_point(o, px, pz, wx, wz))
                bound += w.budding;
            if (active_point(o, x, z, wx, wz))
                score.budding += w.budding;
        }
        for (auto w : gather(origins, c.x, c.z, o.activity_radius()))
            if (active(o, x, z, w.x, w.z))
                score.geodes += w.budding;
        prepared[i] = {c, score, bound};
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
            auto source = gather(origins, p.chunk.x, p.chunk.z, o.activity_radius());
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
    out.stats.hotspots = hotspots.size();
    out.stats.optimized = optimized;
    out.stats.pruned = pruned;
    out.stats.rank_ms = elapsed(rank_start) - (saved.pause_ms() - rank_paused);
    out.stats.total_ms = saved.total_ms();
    return out;
}
}
