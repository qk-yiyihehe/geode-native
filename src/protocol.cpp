#include "protocol.hpp"
#include "search.hpp"
#include "checkpoint.hpp"
#include "json.hpp"
#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

namespace geode {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

namespace {
constexpr int kProtocolVersion = 3, kRankingVersion = kRankingRuleVersion;
constexpr char kEngineVersion[] = "3.0.0";

struct ProtocolCancelled {};

double ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// 阶段百分比按契约保留一位小数，后续节流会丢弃中间帧。
double percent(uint64_t done, uint64_t total) {
    if (!total)
        return 0;
    return std::round(1000.0 * double(done) / double(total)) / 10;
}

// 暂停计时在协议层累计；搜索循环只负责协作等待。
struct ProtocolControl {
    std::atomic<int> state{};
    std::mutex mutex;
    std::condition_variable changed;
    double paused_ms = 0;
    bool pause_open = false;
    Clock::time_point pause_begin;

    void set(int value) {
        std::lock_guard lock(mutex);
        if (value == 1 && !pause_open) {
            pause_begin = Clock::now();
            pause_open = true;
        }
        if (value == 0 && pause_open) {
            paused_ms += ms(pause_begin);
            pause_open = false;
        }
        state = value;
        changed.notify_all();
    }

    // 尚未结束的暂停也计时，确保取消任务仍报告实际等待时间。
    double paused() {
        std::lock_guard lock(mutex);
        return paused_ms + (pause_open ? ms(pause_begin) : 0);
    }

    void check() {
        if (state.load(std::memory_order_relaxed) == 0)
            return;
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] {
            return state != 1;
        });
        if (state == 2)
            throw Cancelled{};
    }
};

fs::path utf8_path(const std::string &text) {
    return fs::path(std::u8string(reinterpret_cast<const char8_t *>(text.data()), text.size()));
}

std::string utf8_text(const fs::path &path) {
    auto value = path.u8string();
    return std::string(reinterpret_cast<const char *>(value.data()), value.size());
}

// 显示距离由整数距离平方得到，保留一位小数以避免比较漂移。
std::string distance_text(int center_x, int center_z, int x, int z) {
    int64_t tenths = distance_tenths(center_x, center_z, x, z);
    return std::to_string(tenths / 10) + "." + std::to_string(tenths % 10);
}

uint64_t seed_value(const std::string &text) {
    if (text.empty())
        throw std::invalid_argument("种子不能为空");
    int64_t number;
    auto begin = text.data();
    if (text.front() == '+')
        ++begin;
    auto [end, error] = std::from_chars(begin, text.data() + text.size(), number);
    if (error == std::errc{} && end == text.data() + text.size())
        return uint64_t(number);
    if (error == std::errc::result_out_of_range)
        throw std::invalid_argument("种子超出64位整数范围");
    uint32_t hash = 0;
    for (size_t i = 0; i < text.size();) {
        uint32_t c = uint8_t(text[i++]);
        if (c >= 0xc0) {
            int n = c < 0xe0 ? 1 : c < 0xf0 ? 2 : 3;
            c &= (1U << (6 - n)) - 1;
            while (n-- && i < text.size())
                c = (c << 6) | (uint8_t(text[i++]) & 63);
        }
        if (c > 0xffff) {
            c -= 0x10000;
            hash = hash * 31 + 0xd800 + (c >> 10);
            c = 0xdc00 + (c & 1023);
        }
        hash = hash * 31 + c;
    }
    return uint64_t(int64_t(int32_t(hash)));
}

Options options(const Json &config) {
    if (config.at("protocolVersion") != kProtocolVersion || config.at("finder") != "geode")
        throw std::invalid_argument("需要桌面协议v3的母岩配置");
    Options o;
    o.seed_text = config.at("seed").get<std::string>();
    o.seed = seed_value(o.seed_text);
    o.checkpoint_directory = "config/checkpoints/geode-native";
    auto b = config.at("bounds");
    auto p = config.at("params");
    o.center_x = b.at("centerX");
    o.center_z = b.at("centerZ");
    o.radius = b.at("radiusBlocks");
    o.shape = b.at("shape");
    if (b.contains("minX") && b.contains("maxX") && b.contains("minZ") && b.contains("maxZ")) {
        o.search_rect = true;
        o.min_x = b.at("minX");
        o.max_x = b.at("maxX");
        o.min_z = b.at("minZ");
        o.max_z = b.at("maxZ");
        o.center_x = int((int64_t(o.min_x) + o.max_x) / 2);
        o.center_z = int((int64_t(o.min_z) + o.max_z) / 2);
        o.radius = std::max(o.max_x - o.min_x, o.max_z - o.min_z) / 2 + 1;
        o.shape = "rect";
    }
    // fullWorld只是简写，结果必须回显展开后的实际方形范围。
    if (b.value("fullWorld", false))
        expand_full_world(o);
    o.threads = config.at("threads");
    o.limit = config.at("resultLimit");
    set_version(o, p.at("mcVersion"));
    o.min_geodes = p.value("minGeodes", 1);
    o.min_budding = p.value("minBudding", 1);
    o.simulation_distance = p.value("simulationDistanceChunks", 10);
    o.prefilter = p.value("prefilterLimit", 200000);
    o.scan_backend = p.value("scanBackend", std::string("auto"));
    if (o.radius < 1 || o.radius > 60000000 || std::abs(int64_t(o.center_x)) > 29999999 ||
        std::abs(int64_t(o.center_z)) > 29999999 || o.threads < 1 || o.threads > 32 || o.limit < 1 || o.limit > 200 ||
        o.min_geodes < 1 || o.min_budding < 0 || o.prefilter < 0 || o.simulation_distance < 2 ||
        o.simulation_distance > 32 || (o.shape != "circle" && o.shape != "square" && o.shape != "rect") ||
        (o.scan_backend != "auto" && o.scan_backend != "cpu" && o.scan_backend != "gpu") ||
        (o.search_rect && (o.min_x > o.max_x || o.min_z > o.max_z || o.min_x < -30000000 || o.max_x > 29999999 ||
                           o.min_z < -30000000 || o.max_z > 29999999)))
        throw std::invalid_argument("搜索参数不合法");
    if (o.scan_backend == "gpu" && !o.prefilter)
        throw std::invalid_argument("GPU粗筛需要正数候选预算；全量模式请选择自动或CPU");
    if (!o.prefilter && o.radius > 100000)
        throw std::invalid_argument("大范围搜索请设置粗筛预算，例如200000");
    return o;
}

Json candidate(const Options &o, int rank, Center c) {
    return {{"id", "geode-" + std::to_string(c.x) + "-" + std::to_string(c.z)},
            {"rank", rank},
            {"x", c.x},
            {"z", c.z},
            {"score", c.budding},
            {"scoreUnit", "预计活跃母岩"},
            {"distance", double(distance_tenths(o.center_x, o.center_z, c.x, c.z)) / 10},
            {"distanceSq", center_distance_sq(o.center_x, o.center_z, c.x, c.z)},
            {"secondary", c.geodes},
            {"secondaryUnit", "活跃晶洞"},
            {"bestPlayer", {{"x", c.x}, {"z", c.z}}},
            {"tpCommand", "/tp @s " + std::to_string(c.x) + " ~ " + std::to_string(c.z)},
            {"verified", false}};
}

Json result_json(const Options &o, const Results &output, const std::string &job, double search_ms, double paused_ms) {
    Json candidates = Json::array();
    int rank = 0;
    for (auto c : output.centers)
        candidates.push_back(candidate(o, ++rank, c));
    auto &s = output.stats;
    Json bounds = {{"centerX", o.center_x},
                   {"centerZ", o.center_z},
                   {"radiusBlocks", o.radius},
                   {"shape", o.shape},
                   {"searchRect", o.search_rect},
                   {"minX", o.search_rect ? o.min_x : o.center_x - o.radius},
                   {"maxX", o.search_rect ? o.max_x : o.center_x + o.radius},
                   {"minZ", o.search_rect ? o.min_z : o.center_z - o.radius},
                   {"maxZ", o.search_rect ? o.max_z : o.center_z + o.radius},
                   {"fullWorld", o.full_world}};
    Json params = {{"minGeodes", o.min_geodes},     {"simulationDistanceChunks", o.simulation_distance},
                   {"prefilterLimit", o.prefilter}, {"minBudding", o.min_budding},
                   {"scanBackend", o.scan_backend}, {"mcVersion", o.version}};
    // 精排始终用CPU；GPU粗筛对应cpu+gpu，原始设备保留在stats.scanBackend。
    Json result = {
        {"jobId", job},
        {"protocolVersion", kProtocolVersion},
        {"rankingRuleVersion", kRankingVersion},
        {"rankingRule", kRankingRule},
        {"finder", "geode"},
        {"seed", o.seed_text},
        {"finderGeneration", o.version},
        {"engine", "geode-native"},
        {"engineVersion", kEngineVersion},
        {"bounds", bounds},
        {"params", params},
        {"threads", o.threads},
        {"device", s.scan_backend == "gpu" ? "cpu+gpu" : "cpu"},
        {"requestedLimit", o.limit},
        {"returnedCount", output.centers.size()},
        {"scannedChunks", s.scanned},
        {"elapsedMs", search_ms},
        {"timing", {{"searchMs", search_ms}, {"pausedMs", paused_ms}, {"resumedMs", s.resumed_ms}, {"totalMs", 0.0}}},
        {"stats",
         {{"candidatePositions", s.candidates},
          {"simulated", s.simulated},
          {"results", output.centers.size()},
          {"scanBackend", s.scan_backend},
          {"candidateChunks", s.candidates},
          {"simulatedGeodes", s.simulated},
          {"terrainRejectedGeodes", s.rejected},
          {"prunedCandidates", s.pruned},
          {"hotspotGroups", s.hotspots},
          {"geometricallyOptimizedChunks", s.optimized},
          {"scanMs", s.scan_ms},
          {"simulateMs", s.simulate_ms},
          {"rankMs", s.rank_ms},
          {"resumedCoarseUnits", s.resumed_scan},
          {"resumedSimulationUnits", s.resumed_simulation}}},
        {"candidates", candidates},
        {"model", o.terrain ? "terrain-control-estimate" : "stone-estimate"},
        {"verified", false},
        {"searchApproximate", o.prefilter != 0},
        {"mcVersion", o.version}};
    return result;
}
}

Exported export_results(const Options &o, const std::vector<Center> &centers, const std::string &directory,
                        const std::string &prefix, Json &body, Json *exports) {
    fs::path base = fs::absolute(utf8_path(directory));
    if (!exports)
        fs::create_directories(base);
    auto stamp =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
    base /= prefix + std::to_string(stamp);
    Exported paths;
    paths.csv = utf8_text(base) + ".csv";
    paths.txt = utf8_text(base) + ".txt";
    paths.json = utf8_text(base) + ".json";
    std::ostringstream table, text;
    table << "\xef\xbb\xbf排名;中心X;中心Z;中心区块X;中心区块Z;预计活跃母岩;活跃晶洞;距搜索中心距离;验证状态\r\n";
    text
        << "原生局部地形模型估计，未经游戏验证；传送指令保留当前Y坐标。\n排名\t中心X\t中心Z\t预计活跃母岩\t活跃晶洞\t距搜索中心距离\n";
    int rank = 0;
    for (auto c : centers) {
        auto tenths = distance_text(o.center_x, o.center_z, c.x, c.z);
        ++rank;
        table << rank << ';' << c.x << ';' << c.z << ';' << floor_div(c.x, 16) << ';' << floor_div(c.z, 16) << ';'
              << c.budding << ';' << c.geodes << ';' << tenths
              << (o.terrain ? ";未验证（局部地形估计）\r\n" : ";未验证（纯石头估计）\r\n");
        text << rank << '\t' << c.x << '\t' << c.z << '\t' << c.budding << '\t' << c.geodes << '\t' << tenths << '\n';
    }
    if (exports) {
        *exports = {{"csv", table.str()}, {"txt", text.str()}};
        body["output"] = Json::object();
        return paths;
    }
    std::ofstream csv_file(utf8_path(paths.csv), std::ios::binary), txt_file(utf8_path(paths.txt), std::ios::binary);
    csv_file << table.str();
    txt_file << text.str();
    csv_file.close();
    txt_file.close();
    if (!csv_file || !txt_file)
        throw std::runtime_error("结果文件写入失败");
    // 待调用方填齐字段后才由write_json写入JSON，
    // 保证文件与done事件使用同一个对象，避免另拼一份字段结构。
    body["output"] = {{"csvPath", paths.csv}, {"txtPath", paths.txt}, {"jsonPath", paths.json}};
    return paths;
}

void write_json(const std::string &path, const Json &body) {
    std::ofstream record(utf8_path(path), std::ios::binary);
    record << body.dump(2) << '\n';
    record.close();
    if (!record)
        throw std::runtime_error("结果文件写入失败");
}

int desktop_protocol() {
    std::mutex output_mutex, stats_mutex;
    std::thread worker;
    std::shared_ptr<ProtocolControl> control;
    std::atomic<bool> active{};
    std::string job;
    uint64_t sequence = 0;
    Json current;
    auto emit = [&](const Json &event) {
        std::lock_guard lock(output_mutex);
        std::cout << event.dump() << std::endl;
    };
    for (std::string line; std::getline(std::cin, line);) {
        try {
            auto command = Json::parse(line);
            std::string type = command.at("type");
            if (type == "start") {
                if (worker.joinable()) {
                    control->set(2);
                    worker.join();
                }
                job = "geode-" + std::to_string(++sequence);
                active = false;
                auto config = command.at("config");
                auto o = options(config);
                control = std::make_shared<ProtocolControl>();
                auto token = control;
                auto id = job;
                current = {{"phase", "prepare"},   {"current", 0},        {"total", 1},     {"unit", "文件"},
                           {"percentage", 0},      {"speedPerSecond", 0}, {"elapsedMs", 0}, {"etaMs", 0},
                           {"foundCandidates", 0}, {"state", "running"}};
                o.checkpoint = [token] {
                    token->check();
                };
                emit({{"type", "accepted"}, {"requestId", command.at("requestId")}, {"jobId", job}});
                active = true;
                worker = std::thread([&, o, config, token, id] {
                    try {
                        auto task_start = Clock::now(), last = task_start - std::chrono::seconds(1);
                        double restored_ms = 0;
                        Options running = o;
                        running.paused_ms = [token] {
                            return token->paused();
                        };
                        running.restored = [&](double history, uint64_t, uint64_t) {
                            restored_ms = history;
                        };
                        auto progress = [&](const std::string &phase, uint64_t done, uint64_t total) {
                            token->check();
                            auto now = Clock::now();
                            double elapsed_ms = restored_ms + ms(task_start) - token->paused();
                            Json p = {{"phase", phase == "scan"       ? "coarse"
                                                : phase == "simulate" ? "verify"
                                                                      : phase},
                                      {"current", done},
                                      {"total", total},
                                      {"unit", phase == "scan"       ? "分片"
                                               : phase == "simulate" ? "模拟分区"
                                               : phase == "rank"     ? "候选"
                                               : phase == "prepare"  ? "阶段"
                                                                     : "文件"},
                                      {"percentage", percent(done, total)},
                                      {"speedPerSecond", 0},
                                      {"elapsedMs", elapsed_ms},
                                      {"etaMs", 0},
                                      {"foundCandidates", 0},
                                      {"state", "running"},
                                      {"message", phase == "scan"       ? "晶洞分布粗筛"
                                                  : phase == "simulate" ? "局部地形与母岩估计"
                                                  : phase == "rank"     ? "热点精排"
                                                                        : "整理搜索结果"}};
                            // 当前阶段总量不能估计后续地形模拟和排名的剩余工作。
                            {
                                std::lock_guard lock(stats_mutex);
                                current = p;
                            }
                            std::lock_guard lock(output_mutex);
                            if (token->state == 0 && (now - last >= std::chrono::milliseconds(70) || done == total)) {
                                std::cout << Json({{"type", "progress"}, {"jobId", id}, {"progress", p}}).dump()
                                          << std::endl;
                                last = now;
                            }
                        };
                        progress("prepare", 0, 1);
                        auto found = search(running, progress);
                        double paused = token->paused();
                        double search_ms = found.stats.total_ms;
                        token->check();
                        progress("report", 0, 1);
                        auto result = result_json(o, found, id, search_ms, paused);
                        bool defer_export = config.value("deferExport", false);
                        Json exports;
                        auto paths = export_results(o, found.centers, "results", "results-geode-", result,
                                                    defer_export ? &exports : nullptr);
                        result["timing"]["totalMs"] = found.stats.resumed_ms + ms(task_start);
                        if (defer_export)
                            exports["json"] = result.dump(2) + "\n";
                        else
                            write_json(paths.json, result);
                        auto &values = result["candidates"];
                        for (size_t first = 0; first < values.size(); first += 100) {
                            token->check();
                            if (first)
                                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                            Json batch = Json::array();
                            for (size_t i = first; i < std::min(first + 100, values.size()); ++i)
                                batch.push_back(values[i]);
                            emit({{"type", "candidates"},
                                  {"jobId", id},
                                  {"batch", batch},
                                  {"totalFound", values.size()}});
                        }
                        token->check();
                        std::unique_lock lock(output_mutex);
                        while (token->state == 1) {
                            lock.unlock();
                            token->check();
                            lock.lock();
                        }
                        if (token->state == 2)
                            throw ProtocolCancelled{};
                        active = false;
                        remove_checkpoint(found.checkpoint_path);
                        Json p;
                        {
                            std::lock_guard stats_lock(stats_mutex);
                            p = current;
                        }
                        p["phase"] = "done";
                        p["state"] = "done";
                        p["percentage"] = 100;
                        p["overallPercentage"] = 100;
                        std::cout
                            << Json({{"type", "progress"}, {"jobId", id}, {"progress", p}}).dump() << '\n'
                            << Json({{"type", "done"}, {"jobId", id}, {"result", result}, {"exports", exports}}).dump()
                            << std::endl;
                    } catch (const ProtocolCancelled &) {
                        active = false;
                        emit({{"type", "failed"}, {"jobId", id}, {"code", "cancelled"}, {"message", "搜索已中止"}});
                    } catch (const std::bad_alloc &) {
                        active = false;
                        emit({{"type", "failed"},
                              {"jobId", id},
                              {"code", "out-of-memory"},
                              {"message", "内存不足，请减少线程或候选预算"}});
                    } catch (const std::exception &e) {
                        active = false;
                        emit({{"type", "failed"}, {"jobId", id}, {"code", "engine-crashed"}, {"message", e.what()}});
                    }
                });
            } else if (control && active && (type == "pause" || type == "resume" || type == "cancel")) {
                if (command.value("jobId", std::string{}) != job)
                    continue;
                std::lock_guard lock(output_mutex);
                control->set(type == "pause" ? 1 : type == "resume" ? 0 : 2);
                Json p;
                {
                    std::lock_guard stats_lock(stats_mutex);
                    p = current;
                }
                if (p.is_null())
                    p = Json::object();
                p["state"] = type == "pause" ? "paused" : type == "resume" ? "running" : "cancelling";
                std::cout << Json({{"type", "progress"}, {"jobId", job}, {"progress", p}}).dump() << std::endl;
            } else if (type == "export")
                std::cerr << "结果已自动保存至results目录\n";
        } catch (const std::exception &e) {
            emit({{"type", "failed"}, {"jobId", job}, {"code", "invalid-config"}, {"message", e.what()}});
        }
    }
    if (worker.joinable()) {
        control->set(2);
        worker.join();
    }
    return 0;
}
}
