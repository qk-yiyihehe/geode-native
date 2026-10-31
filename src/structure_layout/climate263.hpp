#pragma once
#include <cstdint>
#include <memory>
#include <utility>

namespace geode_layout {
class Climate263 {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    explicit Climate263(uint64_t seed);
    ~Climate263();
    std::pair<float, float> sample(int x, int y, int z);
    bool allowed(int x, int y, int z);
};
}
