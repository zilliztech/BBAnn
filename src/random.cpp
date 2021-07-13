#include "util/random.h"

void rand_perm (int64_t *perm, int64_t n, int64_t k, int64_t seed) {
    std::mt19937 generator(seed);

    for (int64_t i = 0; i < n; i++) {
        perm[i] = i;
    }

    for (int64_t i = 0; i < k; i++) {
        int64_t i2 = i + generator() % (n - i);
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
