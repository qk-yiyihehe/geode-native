#pragma once

#include "../third_party/json.hpp"
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace cli_config {
// 配置键与命令行同名（去掉--）；先展开配置，再追加命令行，后者覆盖前者。
inline std::vector<std::string> arguments(const std::vector<std::string> &input) {
    std::string file;
    std::vector<std::string> command;
    for (size_t i = 0; i < input.size(); ++i) {
        if (input[i] != "--config") {
            command.push_back(input[i]);
            continue;
        }
        if (++i == input.size())
            throw std::invalid_argument("--config需要JSON配置路径");
        file = input[i];
    }
    if (file.empty())
        return command;
    auto path = std::filesystem::path(std::u8string(file.begin(), file.end()));
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::invalid_argument("无法读取JSON配置: " + file);
    auto config = nlohmann::json::parse(stream);
    if (!config.is_object())
        throw std::invalid_argument("配置必须是JSON对象，键名使用不带--的命令行参数名");
    // 先去掉被命令行覆盖的配置项，再校验/换算，避免旧单位下的值提前报错。
    std::set<std::string> supplied;
    for (size_t i = 0; i < command.size(); ++i) {
        const auto &name = command[i];
        supplied.insert(name);
        if (name != "--full-world" && name != "--no-checkpoint" && name != "--resume" &&
            name != "--safe-erosion" && name != "--help" && name != "--versions" &&
            name != "--desktop" && name != "--jsonl" && name != "--serve" && name != "--dev-no-biomes")
            ++i;
    }
    std::vector<std::string> expanded;
    for (const auto &[key, value] : config.items()) {
        if (key.empty() || key.starts_with("-") || key == "config" || key == "desktop" ||
            key == "jsonl" || key == "serve" || key == "help" || key == "dev-no-biomes")
            throw std::invalid_argument("配置中不允许设置入口选项: " + key);
        if (supplied.contains("--" + key))
            continue;
        if (key == "seed" && !value.is_string())
            throw std::invalid_argument("JSON中的seed必须是字符串，避免64位整数精度丢失");
        if (value.is_boolean()) {
            if (key != "full-world" && key != "no-checkpoint" && key != "resume")
                throw std::invalid_argument("此参数不接受布尔值: " + key);
            if (value.get<bool>())
                expanded.push_back("--" + key);
        } else if (value.is_string() || value.is_number()) {
            expanded.push_back("--" + key);
            expanded.push_back(value.is_string() ? value.get<std::string>() : value.dump());
        } else {
            throw std::invalid_argument("配置参数必须为字符串、数字或开关: " + key);
        }
    }
    expanded.insert(expanded.end(), command.begin(), command.end());
    return expanded;
}
}
