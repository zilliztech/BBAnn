#include "util/random.h"

void rand_perm (uint64_t *perm, uint64_t n, uint64_t k, int32_t seed) {
    std::mt19937 generator(seed);

    for (int32_t i = 0; i < n; i++) {
        perm[i] = i;
    }

    for (int32_t i = 0; i < k; i++) {
        int i2 = i + generator() % (n - i);
        std::swap(perm[i], perm[i2]);
    }
}

float rand_float() {
    static std::mt19937 generator(1234);
    return generator() / (float)generator.max();
}

int rand_int() {
    static std::mt19937 generator(3456);
    return generator() & 0x7fffffff;
}
