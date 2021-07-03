#pragma once

#include <cstdint>
#include <algorithm>
#include <random>

// perm[0...k-1] is the results
// but assume `perm` has allocated n spaces

void rand_perm (uint32_t *perm, int32_t n, int k, int32_t seed);
float rand_float();
int rand_int();

