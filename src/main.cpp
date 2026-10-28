#include "search.hpp"
#include "protocol.hpp"
#include "checkpoint.hpp"
#include "terrain.hpp"
#include "json.hpp"
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

std::string trim(std::string v) {
    auto a = v.find_first_not_of(" \t\r\n\xef\xbb\xbf");
    if (a == std::string::npos)
        return {};
    return v.substr(a, v.find_last_not_of(" \t\r\n") - a + 1);
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
        base.simulation_distance = query.at("simulationDistance").get<int>();
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
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; ++i) {
            std::string k = argv[i];
            if (k == "--help" || k == "--jsonl" || k == "--no-checkpoint")
                args[k] = "true";
            else {
                if (!k.starts_with("--") || i + 1 >= argc)
                    throw std::invalid_argument("Expected --option value");
                args[k] = argv[++i];
            }
        }
        if (args.contains("--help")) {
            std::cout
                << "geode-finder --seed <int64> --version <1.18..1.21.11|26.1..26.3> --radius <blocks> [--shape circle|square] [--threads 8] [--prefilter-limit 50000] [--output results]\n"
                << "  --tick-radius 128 --simulation-distance 10 --min-geodes 1 --min-budding 1 --limit 100 --model terrain|stone\n"
                << "  --prefilter-limit 0: exhaustive player-chunk candidates in the selected model\n"
                << "  --scan-backend auto|cpu|gpu: optional OpenCL coarse scan for large finite-budget searches\n"
                << "  --checkpoint-dir <directory>: auto-resume matching unfinished searches (default: output/checkpoints)\n"
                << "  --checkpoint-region <chunks>: aggregate checkpoint units into square regions (8192 = 131072 blocks)\n"
                << "  --scan-threads N: coarse-scan worker count (1..256; simulation still uses --threads)\n"
                << "  --no-checkpoint: disable persistent checkpoints; completed exports remove the checkpoint\n"
                << "  --min-x X --max-x X --min-z Z --max-z Z: exact inclusive rectangle (implies --rect); ranking center = midpoint\n"
                << "  --rect: use the default rectangle [-16384,16383]^2 instead of --center-x/--radius\n"
                << "  --player-x X --player-z Z: evaluate one position; --geode-chunk-x X --geode-chunk-z Z: geode block coordinates\n"
                << "  --config finder.properties; --samples file.json: batch model checks; --jsonl: streamed CLI events\n"
                << "  --desktop: desktop protocol v3 (ranking rule primary-desc-distance-asc-x-asc-z-asc)\n"
                << "  Every search writes one leaderboard to CSV, TXT and JSON and reports all three paths in output\n";
            return 0;
        }
        std::map<std::string, std::string> config;
        if (args.contains("--config")) {
            std::ifstream in(path_utf8(args.at("--config")));
            if (!in)
                throw std::runtime_error("Cannot read configuration");
            std::string line;
            while (std::getline(in, line)) {
                line = trim(line);
                if (line.empty() || line[0] == '#' || line[0] == '!')
                    continue;
                auto p = line.find('=');
                if (p != std::string::npos)
                    config[trim(line.substr(0, p))] = trim(line.substr(p + 1));
            }
        }
        auto text = [&](const std::string &flag, const std::string &prop, const std::string &fallback) {
            if (args.contains(flag))
                return args.at(flag);
            if (config.contains(prop))
                return config.at(prop);
            return fallback;
        };
        auto number = [&](const std::string &flag, const std::string &prop, int fallback) {
            return integer(text(flag, prop, std::to_string(fallback)));
        };
        Options o;
        o.seed_text = text("--seed", "seed", "0");
        o.seed = seed_value(o.seed_text);
        set_version(o, text("--version", "minecraft-version", "1.21.1"));
        if (args.contains("--finder") && args.at("--finder") != "geode")
            throw std::invalid_argument("This executable only supports geode");
        o.radius = number("--radius", "search-radius-blocks", 10000);
        o.center_x = number("--center-x", "search-center-x", 0);
        o.center_z = number("--center-z", "search-center-z", 0);
        o.shape = text("--shape", "search-area-shape", "circle");
        const bool anyBound = args.contains("--min-x") || args.contains("--max-x") || args.contains("--min-z") ||
                              args.contains("--max-z");
        const bool allBound = args.contains("--min-x") && args.contains("--max-x") && args.contains("--min-z") &&
                              args.contains("--max-z");
        if (anyBound && !allBound)
            throw std::invalid_argument("--min-x/--max-x/--min-z/--max-z must be given together");
        if (anyBound || args.contains("--rect")) {
            o.search_rect = true;
            o.min_x = number("--min-x", "search-min-x", -16384);
            o.max_x = number("--max-x", "search-max-x", 16383);
            o.min_z = number("--min-z", "search-min-z", -16384);
            o.max_z = number("--max-z", "search-max-z", 16383);
            // 矩形模式的排名中心取四至中点（向零截断）。
            o.center_x = int((int64_t(o.min_x) + o.max_x) / 2);
            o.center_z = int((int64_t(o.min_z) + o.max_z) / 2);
            o.radius = std::max(o.max_x - o.min_x, o.max_z - o.min_z) / 2 + 1;
            o.shape = "rect";
        }
        o.tick_radius = number("--tick-radius", "geode-legacy-random-tick-radius-blocks", 128);
        o.simulation_distance = number("--simulation-distance", "geode-simulation-distance-chunks", 10);
        o.threads = number("--threads", "scan-threads", 8);
        o.scan_threads = number("--scan-threads", "scan-thread-count", 0);
        o.checkpoint_region = number("--checkpoint-region", "geode-checkpoint-region-chunks", 0);
        o.min_geodes = number("--min-geodes", "geode-min-geodes", 1);
        o.min_budding = number("--min-budding", "geode-min-budding-amethyst", 1);
        o.prefilter = number("--prefilter-limit", "geode-prefilter-limit", 0);
        o.limit = number("--limit", "geode-result-limit", 100);
        o.scan_backend = text("--scan-backend", "geode-scan-backend", "auto");
        if (o.scan_backend != "auto" && o.scan_backend != "cpu" && o.scan_backend != "gpu")
            throw std::invalid_argument("Scan backend must be auto, cpu or gpu");
        if (o.scan_backend == "gpu" && !o.prefilter)
            throw std::invalid_argument("GPU coarse scan requires a positive --prefilter-limit");
        auto model = text("--model", "geode-model", "terrain");
        if (model != "stone" && model != "terrain")
            throw std::invalid_argument("Model must be stone or terrain");
        o.terrain = model == "terrain";
        if (o.radius < 1 || o.radius > 60000000 || std::abs(int64_t(o.center_x)) > 30000000 ||
            std::abs(int64_t(o.center_z)) > 30000000 || o.threads < 1 || o.threads > 64 || o.tick_radius < 1 ||
            o.tick_radius > 1024 || o.simulation_distance < 2 || o.simulation_distance > 32 || o.prefilter < 0 ||
            o.min_geodes < 1 || o.min_budding < 0 || o.limit < 1 ||
            (o.shape != "circle" && o.shape != "square" && o.shape != "rect") || o.scan_threads < 0 ||
            o.scan_threads > 256 || o.checkpoint_region < 0 || o.checkpoint_region > 1000000 ||
            (o.search_rect && (o.min_x > o.max_x || o.min_z > o.max_z || o.min_x < -30000000 || o.max_x > 29999999 ||
                               o.min_z < -30000000 || o.max_z > 29999999)))
            throw std::invalid_argument("Invalid search/radius/thread/threshold setting");
        if (config.contains("full-world") && config.at("full-world") != "false")
            throw std::invalid_argument("Use an explicit radius; full-world mode is not implemented");
        if (!args.contains("--no-checkpoint"))
            o.checkpoint_directory = text("--checkpoint-dir", "geode-checkpoint-directory",
                                          text("--output", "output-directory", "results") + "/checkpoints");
        const std::set<std::string> supported = {"--model",
                                                 "--config",
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
        auto output = search(o, [&](const std::string &phase, uint64_t done, uint64_t total) {
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
            {"seed", o.seed_text},
            {"version", o.version},
            {"rankingRuleVersion", kRankingRuleVersion},
            {"rankingRule", kRankingRule},
            {"generationProfile", o.terrain ? "native-density-geode-20002" : "java-modern-stone-20002"},
            {"activityRule", o.modern ? "simulation-distance-square-java-compatible" : "legacy-chunk-center-circle"},
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
        // 与桌面入口共用最终中心列表和导出函数，生成三份文件；随后
        // 直接输出同一个JSON对象，并保留csv/json兼容键。jsonl模式
        // 在写文件前标记done事件，使文件与标准输出保持一致。
        if (jsonl)
            result["type"] = "done";
        auto paths = export_results(o, output.centers, text("--output", "output-directory", "results"),
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
