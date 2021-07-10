#pragma once

#include <cstdint>
#include <algorithm>
#include <random>

// perm[0...k-1] is the results
// but assume `perm` has allocated n spaces

void rand_perm (int64_t *perm, int64_t n, int64_t k, int64_t seed);
float rand_float();
int rand_int();

