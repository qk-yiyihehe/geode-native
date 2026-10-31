#include "climate263.hpp"
#include <cmath>
// 仅改名C辅助函数，避免它与C++20的std::lerp发生全局声明冲突。
#define lerp cubiomes_lerp
#include "rng.h"
#undef lerp
#include "generator.h"

namespace trial {
struct Climate263::Impl {
    Generator generator;

    explicit Impl(uint64_t seed) {
        setupGenerator(&generator, MC_26_3, 0);
        applySeed(&generator, DIM_OVERWORLD, seed);
    }
};

Climate263::Climate263(uint64_t seed) : impl_(std::make_unique<Impl>(seed)) {}
Climate263::~Climate263() = default;

std::pair<float, float> Climate263::sample(int x, int y, int z) {
    float values[NP_MAX];
    sampleBiomeClimate(&impl_->generator.bn, values, x, y, z, SAMPLE_NO_TEMPERATURE | SAMPLE_NO_HUMIDITY);
    return {values[NP_EROSION], values[NP_DEPTH]};
}

bool Climate263::allowed(int x, int y, int z) {
    auto [erosion, depth] = sample(x, y, z);
    return erosion >= -0.225f || depth <= 0.9f;
}
}
