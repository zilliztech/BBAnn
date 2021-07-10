#pragma once

#include <algorithm>
#include <math.h>
#include "util/distance.h"

template<typename T>
void stat_length(T *x, int n, int dim, double &max_len, double &min_len, double &avg_len) {
    double sum_len = 0;
    max_len = 0;
    min_len = std::numeric_limits<double>::max();

#pragma omp parallel for reduction(max:max_len) reduction(min:min_len) reduction(+:sum_len) 
    for (size_t i = 0; i < n; i++) {
        T* p = x + i * dim;
        double len = sqrt(IP<T, T, double>(p, p, dim));
        if (len > max_len) max_len = len;
        if (len < min_len) min_len = len;
        sum_len += len;
    }

    avg_len = sum_len / n;
}