#include "search.hpp"
#include "protocol.hpp"
#include "checkpoint.hpp"
#include "terrain.hpp"
#include "json.hpp"
#include "range_units.hpp"
#include "cli_config.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
using Json = nlohmann::json;
namespace fs = std::filesystem;
using namespace geode;

namespace {
fs::path path_utf8(const std::string &s) {
    return fs::path(std::u8string(reinterpret_cast<const char8_t *>(s.data()), s.size()));
}

int integer(const std::string &s) {
    size_t n = 0;
    long long v = std::stoll(s, &n);
    if (n != s.size() || v < INT32_MIN || v > INT32_MAX)
        throw std::invalid_argument("Invalid integer: " + s);
    return int(v);
}

uint64_t seed_value(const std::string &s) {
    size_t n = 0;
    auto v = std::stoll(s, &n);
    if (n != s.size())
        throw std::invalid_argument("Seed must be a signed 64-bit integer string");
    return uint64_t(v);
}

Json center_json(Center c) {
    return {{"x", c.x},
            {"z", c.z},
            {"chunkX", floor_div(c.x, 16)},
            {"chunkZ", floor_div(c.z, 16)},
            {"estimatedBudding", c.budding},
            {"geodeOrigins", c.geodes},
            {"verified", false}};
}

Json sample_json(const Json &query, Options base) {
    if (query.contains("seed")) {
        base.seed_text = query.at("seed").get<std::string>();
        base.seed = seed_value(base.seed_text);
    }
    if (query.contains("version"))
        set_version(base, query.at("version").get<std::string>());
    if (query.contains("simulationDistance"))
        base.simulation_distance = range_units::read(query.at("simulationDistance"), true);
    base.coverage = query.value("coverage", base.coverage);
    if (query.contains("innerRadius"))
        base.inner_radius = range_units::read(query.at("innerRadius"));
    if ((base.coverage != "game" && base.coverage != "circle" && base.coverage != "square") ||
        base.inner_radius < 0 || base.inner_radius > 60000000)
        throw std::invalid_argument("覆盖模式必须为game/circle/square，半径为0–60000000格");
    if (base.simulation_distance < 2 || base.simulation_distance > 32)
        throw std::invalid_argument("Simulation distance must be 2..32");
    if (query.contains("playerX")) {
        return center_json(evaluate(base, query.at("playerX").get<int>(), query.at("playerZ").get<int>()));
    }
    int x = query.at("chunkX").get<int>(), z = query.at("chunkZ").get<int>();
    if (!in_world(x, z))
        throw std::invalid_argument("Geode chunk outside world");
    Simulator simulator(base.seed, base.generation);
    auto terrain = base.terrain ? std::make_unique<Terrain>(base.seed, base.generation) : nullptr;
    auto result = simulator.generate(x, z, terrain.get());
    Json blocks = Json::array();
    for (auto p : result.budding)
        blocks.push_back({p.x, p.y, p.z});
    return {{"exists", result.exists}, {"origin", {result.x, result.y, result.z}}, {"budding", blocks}};
}
}

int run(int argc, char **argv) {
    try {
        for (int i = 1; i < argc; ++i)
            if (std::string(argv[i]) == "--desktop")
                return desktop_protocol();
        auto command = cli_config::arguments(std::vector<std::string>(argv + 1, argv + argc));
        std::map<std::string, std::string> args;
        for (size_t i = 0; i < command.size(); ++i) {
            const auto &k = command[i];
            if (k == "--versions" || k == "--help" || k == "--jsonl" || k == "--no-checkpoint" || k == "--full-world")
                args[k] = "true";
            else {
                if (!k.starts_with("--") || i + 1 >= command.size())
                    throw std::invalid_argument("Expected --option value");
                args[k] = command[++i];
            }
        }
        if (args.contains("--versions")) {
            std::cout << Json{{"versions", versions()}}.dump() << '\n';
            return 0;
        }
        if (args.contains("--help")) {
            std::cout << "紫水晶母岩独立引擎\n"
                         "用法: geode-finder --seed 种子 [参数]\n"
                         "通用: --config finder.json --version 26.3 --radius 10000 --shape circle|square\n"
                         "      --center-x 0 --center-z 0 --threads 8 --limit 100 --output results\n"
                         "      --radius-unit blocks|chunks --full-world --no-checkpoint\n"
                         "统计: --coverage game|circle|square --inner-radius 128 --inner-radius-unit blocks|chunks\n"
                         "      --simulation-distance 10 --simulation-distance-unit chunks|blocks\n"
                         "      --min-geodes 1 --min-budding 1 --prefilter-limit 200000 --scan-backend auto|cpu|gpu\n"
                         "      --min-x X --max-x X --min-z Z --max-z Z: 精确四至矩形(闭区间；排名中心取中点)，--rect 用默认矩形\n"
                         "      --scan-threads N: 粗筛线程数(1..256，模拟仍用--threads)；--checkpoint-region 区块: 断点聚合区域\n"
                         "桌面: --desktop (协议v3，stdin/stdout JSONL，不读取CLI配置)\n"
                         "开发: --model terrain|stone --samples 文件.json；--help只显示入口用法\n"
                         "配置键为CLI参数名去掉--，命令行覆盖配置；输出目录由--output指定。\n";
            return 0;
        }
        auto text = [&](const std::string &flag, const std::string &fallback) {
            auto it = args.find(flag);
            return it == args.end() ? fallback : it->second;
        };
        auto number = [&](const std::string &flag, int fallback) {
            return integer(text(flag, std::to_string(fallback)));
        };
        Options o;
        o.seed_text = text("--seed", "0");
        o.seed = seed_value(o.seed_text);
        set_version(o, text("--version", "1.21.1"));
        if (args.contains("--finder") && args.at("--finder") != "geode")
            throw std::invalid_argument("This executable only supports geode");
        o.radius = range_units::text(text("--radius", "10000"),
                                    text("--radius-unit", "blocks"));
        o.center_x = number("--center-x", 0);
        o.center_z = number("--center-z", 0);
        o.shape = text("--shape", "circle");
        // 精确四至（闭区间）：给了任一四至就要求给全，--rect 使用默认矩形。
        const bool any_bound = args.contains("--min-x") || args.contains("--max-x") || args.contains("--min-z") ||
                               args.contains("--max-z");
        const bool all_bound = args.contains("--min-x") && args.contains("--max-x") && args.contains("--min-z") &&
                               args.contains("--max-z");
        if (any_bound && !all_bound)
            throw std::invalid_argument("--min-x/--max-x/--min-z/--max-z must be given together");
        if (any_bound || args.contains("--rect")) {
            o.search_rect = true;
            o.min_x = number("--min-x", -16384);
            o.max_x = number("--max-x", 16383);
            o.min_z = number("--min-z", -16384);
            o.max_z = number("--max-z", 16383);
            // 矩形模式的排名中心取四至中点（向零截断）。
            o.center_x = int((int64_t(o.min_x) + o.max_x) / 2);
            o.center_z = int((int64_t(o.min_z) + o.max_z) / 2);
            o.radius = std::max(o.max_x - o.min_x, o.max_z - o.min_z) / 2 + 1;
            o.shape = "rect";
        }
        o.tick_radius = number("--tick-radius", 128);        o.coverage = text("--coverage", "game");
        o.inner_radius = range_units::text(text("--inner-radius", "128"),
                                          text("--inner-radius-unit", "blocks"));
        o.simulation_distance = range_units::text(text("--simulation-distance", "10"),
                                                 text("--simulation-distance-unit", "chunks"), true);
        o.threads = number("--threads", 8);
        o.scan_threads = number("--scan-threads", 0);
        o.checkpoint_region = number("--checkpoint-region", 0);
        o.min_geodes = number("--min-geodes", 1);
        o.min_budding = number("--min-budding", 1);
        o.prefilter = number("--prefilter-limit", 0);
        o.limit = number("--limit", 100);
        o.scan_backend = text("--scan-backend", "auto");
        if (o.scan_backend != "auto" && o.scan_backend != "cpu" && o.scan_backend != "gpu")
            throw std::invalid_argument("Scan backend must be auto, cpu or gpu");
        if (o.scan_backend == "gpu" && !o.prefilter)
            throw std::invalid_argument("GPU coarse scan requires a positive --prefilter-limit");
        auto model = text("--model", "terrain");
        if (model != "stone" && model != "terrain")
            throw std::invalid_argument("Model must be stone or terrain");
        o.terrain = model == "terrain";
        if ((o.coverage != "game" && o.coverage != "circle" && o.coverage != "square") ||
            o.inner_radius > 60000000 || o.radius < 1 || o.radius > 60000000 || std::abs(int64_t(o.center_x)) > 30000000 ||
            std::abs(int64_t(o.center_z)) > 30000000 || o.threads < 1 || o.threads > 64 || o.tick_radius < 1 ||
            o.tick_radius > 1024 || o.simulation_distance < 2 || o.simulation_distance > 32 || o.prefilter < 0 ||
            o.min_geodes < 1 || o.min_budding < 0 || o.limit < 1 ||
            (o.shape != "circle" && o.shape != "square" && o.shape != "rect") || o.scan_threads < 0 ||
            o.scan_threads > 256 || o.checkpoint_region < 0 || o.checkpoint_region > 1000000 ||
            (o.search_rect && (o.min_x > o.max_x || o.min_z > o.max_z || o.min_x < -30000000 || o.max_x > 29999999 ||
                               o.min_z < -30000000 || o.max_z > 29999999)))
            throw std::invalid_argument("Invalid search/radius/thread/threshold setting");
        if (args.contains("--full-world"))
            expand_full_world(o);
        if (!args.contains("--no-checkpoint"))
            o.checkpoint_directory = text("--checkpoint-dir",
                                          text("--output", "results") + "/checkpoints");
        const std::set<std::string> supported = {"--coverage", "--inner-radius", "--inner-radius-unit",
                                                 "--radius-unit", "--simulation-distance-unit",
                                                 "--model",
                                                 "--config", "--full-world",
                                                 "--seed",
                                                 "--finder",
                                                 "--version",
                                                 "--radius",
                                                 "--center-x",
                                                 "--center-z",
                                                 "--shape",
                                                 "--tick-radius",
                                                 "--simulation-distance",
                                                 "--threads",
                                                 "--min-geodes",
                                                 "--min-budding",
                                                 "--prefilter-limit",
                                                 "--limit",
                                                 "--output",
                                                 "--jsonl",
                                                 "--player-x",
                                                 "--player-z",
                                                 "--geode-chunk-x",
                                                 "--geode-chunk-z",
                                                 "--samples",
                                                 "--scan-backend",
                                                 "--checkpoint-dir",
                                                 "--checkpoint-region",
                                                 "--scan-threads",
                                                 "--min-x",
                                                 "--max-x",
                                                 "--min-z",
                                                 "--max-z",
                                                 "--rect",
                                                 "--no-checkpoint"};
        for (auto &[k, v] : args)
            if (!supported.contains(k))
                throw std::invalid_argument("Unknown option: " + k);
        if (args.contains("--samples")) {
            std::ifstream file(path_utf8(args.at("--samples")));
            if (!file)
                throw std::runtime_error("Cannot read samples");
            Json queries;
            file >> queries;
            Json results = Json::array();
            for (auto &q : queries)
                results.push_back(sample_json(q, o));
            std::cout << results.dump() << '\n';
            return 0;
        }
        if (args.contains("--player-x") || args.contains("--player-z")) {
            auto c = evaluate(o, integer(args.at("--player-x")), integer(args.at("--player-z")));
            std::cout << center_json(c).dump() << '\n';
            return 0;
        }
        if (args.contains("--geode-chunk-x") || args.contains("--geode-chunk-z")) {
            Json q = {{"chunkX", integer(args.at("--geode-chunk-x"))}, {"chunkZ", integer(args.at("--geode-chunk-z"))}};
            std::cout << sample_json(q, o).dump() << '\n';
            return 0;
        }
        auto task_start = std::chrono::steady_clock::now(), last = task_start;
        bool jsonl = args.contains("--jsonl");
        auto output = search(o, [&](const std::string &phase, uint64_t done, uint64_t total, uint64_t) {
            auto now = std::chrono::steady_clock::now();
            if (done != total && now - last < std::chrono::milliseconds(200))
                return;
            last = now;
            if (jsonl)
                std::cout << Json({{"type", "progress"},
                                   {"finder", "geode"},
                                   {"phase", phase},
                                   {"completed", done},
                                   {"total", total}})
                                 .dump()
                          << std::endl;
            else
                std::cerr << phase << ": " << done << '/' << total << '\n';
        });
        // CLI保持results条目的字段结构，与samples和单点查询
        // 使用相同格式；只有顺序改为v3榜单规则。
        Json list = Json::array();
        for (auto c : output.centers)
            list.push_back(center_json(c));
        auto &s = output.stats;
        Json result = {
            {"finder", "geode"},
            {"cubiomesCommit", CUBIOMES_REVISION},
            {"seed", o.seed_text},
            {"version", o.version},
            {"rankingRuleVersion", kRankingRuleVersion},
            {"rankingRule", kRankingRule},
            {"generationProfile", o.terrain ? "native-density-geode-20002" : "java-modern-stone-20002"},
            {"activityRule", o.custom() ? "custom-horizontal-" + o.coverage :
                o.modern ? "simulation-distance-square-java-compatible" : "legacy-chunk-center-circle"},
            {"model", o.terrain ? "terrain-control-estimate" : "stone-estimate"},
            {"verified", false},
            {"searchApproximate", output.approximate},
            {"cancelled", output.cancelled},
            {"candidateChunkOptimization", "exact-integer"},
            {"rankingUnit", "distinct-hotspot"},
            {"parameters",
             {{"centerX", o.center_x},
              {"centerZ", o.center_z},
              {"radius", o.radius},
              {"shape", o.shape},
              {"searchRect", o.search_rect},
              {"minX", o.search_rect ? o.min_x : o.center_x - o.radius},
              {"maxX", o.search_rect ? o.max_x : o.center_x + o.radius},
              {"minZ", o.search_rect ? o.min_z : o.center_z - o.radius},
              {"maxZ", o.search_rect ? o.max_z : o.center_z + o.radius},
              {"tickRadius", o.tick_radius},
              {"simulationDistance", o.simulation_distance},
              {"coverage", o.coverage},
              {"innerRadiusBlocks", o.inner_radius},
              {"effectiveSimulationDistanceChunks", std::min(o.simulation_distance, 31)},
              {"minGeodes", o.min_geodes},
              {"minBudding", o.min_budding},
              {"prefilterLimit", o.prefilter},
              {"threads", o.threads},
              {"scanThreads", o.scan_threads > 0 ? o.scan_threads : o.threads},
              {"checkpointRegion", o.checkpoint_region},
              {"limit", o.limit}}},
            {"stats",
             {{"scanBackend", s.scan_backend},
              {"scannedChunks", s.scanned},
              {"candidateChunks", s.candidates},
              {"simulatedGeodes", s.simulated},
              {"terrainRejectedGeodes", s.rejected},
              {"weightedChunks", s.weights},
              {"prunedCandidates", s.pruned},
              {"hotspotGroups", s.hotspots},
              {"geometricallyOptimizedChunks", s.optimized},
              {"scanMs", s.scan_ms},
              {"simulateMs", s.simulate_ms},
              {"rankMs", s.rank_ms},
              {"elapsedMs", s.total_ms}}},
            {"results", list}};
        result["timing"] = {
            {"searchMs", s.total_ms}, {"pausedMs", 0}, {"resumedMs", s.resumed_ms}, {"totalMs", s.total_ms}};
        result["stats"]["resumedCoarseUnits"] = s.resumed_scan;
        result["stats"]["resumedSimulationUnits"] = s.resumed_simulation;
        result["stats"]["stoneGeodes"] = s.stone_simulated;
        result["stats"]["twoPass"] = s.two_pass;
        result["stats"]["twoPassKeptCandidates"] = s.two_pass_kept;
        result["stats"]["twoPassExcludedBound"] = s.two_pass_excluded_bound;
        result["stats"]["twoPassRetried"] = s.two_pass_retried;
        // 与桌面入口共用最终中心列表和导出函数，生成三份文件；随后
        // 直接输出同一个JSON对象，并保留csv/json兼容键。jsonl模式
        // 在写文件前标记done事件，使文件与标准输出保持一致。
        if (jsonl)
            result["type"] = "done";
        auto paths = export_results(o, output.centers, text("--output", "results"),
                                    "results-geode-", result);
        result["output"]["csv"] = paths.csv;
        result["output"]["json"] = paths.json;
        result["timing"]["totalMs"] =
            s.resumed_ms +
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - task_start).count();
        write_json(paths.json, result);
        remove_checkpoint(output.checkpoint_path);
        if (jsonl)
            std::cout << Json({{"type", "candidates"}, {"finder", "geode"}, {"candidates", list}}).dump() << '\n';
        std::cout << result.dump() << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
#ifdef _WIN32
int wmain(int argc, wchar_t **argv) {
    std::vector<std::string> args;
    for (int i = 0; i < argc; ++i) {
        int n = WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, nullptr, 0, nullptr, nullptr);
        std::string s(n, '\0');
        WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, s.data(), n, nullptr, nullptr);
        s.pop_back();
        args.push_back(std::move(s));
    }
    std::vector<char *> raw;
    for (auto &s : args)
        raw.push_back(s.data());
    return run(argc, raw.data());
}
#else
int main(int argc, char **argv) {
    return run(argc, argv);
}
#endif