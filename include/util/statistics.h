#pragma once

#include <algorithm>
#include <math.h>
#include "util/distance.h"

struct refine_stat {
    int64_t vector_load_cnt;
    int64_t id_load_cnt;
    int64_t vector_page_hit_cnt;
    int64_t id_page_hit_cnt;
    int64_t different_offset_cnt;
    refine_stat():vector_page_hit_cnt(0), vector_load_cnt(0), id_page_hit_cnt(0), id_load_cnt(0), different_offset_cnt(0) {}
};

template<typename T>
void stat_length(T *x, int64_t n, int64_t dim, double &max_len, double &min_len, double &avg_len) {
    double sum_len = 0;
    max_len = 0;
    min_len = std::numeric_limits<double>::max();

#pragma omp parallel for reduction(max:max_len) reduction(min:min_len) reduction(+:sum_len) 
    for (int64_t i = 0; i < n; i++) {
        T* p = x + i * dim;
        double len = sqrt(IP<T, T, double>(p, p, dim));
        if (len > max_len) max_len = len;
        if (len < min_len) min_len = len;
        sum_len += len;
    }

    avg_len = sum_len / n;
}
