#pragma once

#include "../third_party/json.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace range_units {
using Json = nlohmann::json;

inline int convert(double value, const std::string &unit, bool chunks = false) {
    if (unit != "blocks" && unit != "chunks")
        throw std::invalid_argument("距离单位必须为blocks或chunks");
    double result = value * (unit == "chunks" ? 16.0 : 1.0) / (chunks ? 16.0 : 1.0);
    // 区块半径可用1/16区块表达；模拟距离必须精确落在整数区块，禁止静默取整。
    if (!std::isfinite(result) || result != std::floor(result) ||
        result < 0 || result > std::numeric_limits<int>::max())
        throw std::invalid_argument(chunks ? "模拟距离必须为整数区块，格数须为16的整数倍"
                                           : "距离必须精确换算为非负整数格");
    return int(result);
}

inline int read(const Json &value, bool chunks = false) {
    if (value.is_number()) {
        if (!value.is_number_integer())
            throw std::invalid_argument("标准距离字段必须为整数；带单位的小数请使用value/unit对象");
        return convert(value.get<double>(), chunks ? "chunks" : "blocks", chunks);
    }
    if (!value.is_object() || value.size() != 2 || !value.contains("value") ||
        !value.contains("unit") || !value.at("value").is_number() || !value.at("unit").is_string())
        throw std::invalid_argument("距离应为整数或{value:数值,unit:blocks/chunks}");
    return convert(value.at("value").get<double>(), value.at("unit").get<std::string>(), chunks);
}

inline int field(const Json &owner, const char *key, const char *legacy, int fallback, bool chunks = false) {
    if (owner.contains(key) && owner.contains(legacy))
        throw std::invalid_argument(std::string(key) + "与" + legacy + "不能同时指定");
    if (owner.contains(key))
        return read(owner.at(key), chunks);
    return owner.contains(legacy) ? read(owner.at(legacy), chunks) : fallback;
}

inline int text(const std::string &value, const std::string &unit, bool chunks = false) {
    size_t end = 0;
    double number = std::stod(value, &end);
    if (end != value.size())
        throw std::invalid_argument("距离包含非数字字符");
    return convert(number, unit, chunks);
}
}
