#include "generator.h"
#include <stdlib.h>

Generator *geode_biomes_create(uint64_t seed) {
    Generator *generator = malloc(sizeof(Generator));
    setupGenerator(generator, MC_1_21_1, 0);
    applySeed(generator, DIM_OVERWORLD, seed);
    return generator;
}

void geode_biomes_free(Generator *generator) {
    free(generator);
}

int geode_biomes_allowed(Generator *generator, int x, int y, int z) {
    // 此生成器固定为现代主世界噪声，输入已经是四分之一方块坐标。
    return sampleBiomeNoise(&generator->bn, NULL, x, y, z, NULL, 0) != deep_dark;
}
