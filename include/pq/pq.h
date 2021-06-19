#pragma once

#include "ivf/clustering.h"
#include "utils/heap.h"
#include "utils/distance.h"

#include <stdint.h>
#include <string.h>
#include <assert.h>

#include <omp.h>

#include <iostream>
#include <memory>
#include <vector>
#include <limits>
#include <functional>

template<typename T>
using PQ_Computer = std::function<float(const T*, const float*, int n)>;

// T: type of the database vector
// U: type of codes ID (PQ representation)
template<class C, typename T, typename U>
class PQ {
private:
    int32_t d, dsub, K, ntotal, npos;
    uint8_t m, nbits;

    float* centroids = nullptr;
    U* codes = nullptr;

    void compute_code(const T* x, U* c);

    void compute_dis_tab(const T* q, float* dis_tab, PQ_Computer<T> computer);
public:
    PQ(int32_t _ntotal, int32_t _d, uint8_t _m, uint8_t _nbits)
    : ntotal(_ntotal), d(_d), m(_m), nbits(_nbits), npos(0) {
        assert(d % m == 0);

        dsub = d / m;
        K = 1 << nbits;

        centroids = new float[m * K * dsub];
        codes = new U[ntotal * m];
    };

    ~PQ() {
        if (centroids != nullptr) {
            delete[] centroids;
        }

        if (codes != nullptr) {
            delete[] codes;
        }
    }

    void train(int32_t n, const T* x);

    void encode_vectors(int32_t n, const T* x);

    void search(int32_t nq, const T* q, int32_t topk,
                   typename C::T* values, typename C::TI* labels,
                   PQ_Computer<T> computer);
};

template<class C, typename T, typename U>
void PQ<C, T, U>::compute_code(const T* x, U* c) {
    for (uint8_t i = 0; i < m; ++i, x += dsub) {
        float min_dist = std::numeric_limits<float>::max();
        int32_t best_id = 0;

        const float* cen = centroids + i * K * dsub;

        // find the best centroid
        for (int32_t j = 0; j < K; ++j, cen += dsub) {
            float dis = L2sqr<const T, const float, float>(x, cen, dsub);
            if (dis < min_dist) {
                min_dist = dis;
                best_id = j;
            }
        }

        *c++ = best_id;
    }
}

template<class C, typename T, typename U>
void PQ<C, T, U>::compute_dis_tab(const T* q, float* dis_tab,
                                     PQ_Computer<T> computer)
{
    const float* c = centroids;
    for (uint8_t i = 0; i < m; ++i, q += dsub) {
        for (int32_t j = 0; j < K; ++j, c += dsub) {
            *dis_tab++ = computer(q, c, dsub);
        }
    }
}

template<class C, typename T, typename U>
void PQ<C, T, U>::train(int32_t n, const T* x) {
    T* xs = new T[n * dsub];

    for (uint8_t i = 0; i < m; ++i) {
        const int32_t tmp_d = i * dsub;

        // get slice of x in subspace m_i
        for (int32_t j = 0; j < n; ++j) {
            memcpy(xs + j * dsub, x + j * d + tmp_d, dsub * sizeof(T));
        }

        // compute centroids
        kmeans(n, xs, dsub, K, centroids + i * K * dsub);
    }

    delete[] xs;
};

template<class C, typename T, typename U>
void PQ<C, T, U>::encode_vectors(int32_t n, const T *x) {
    assert(npos + n <= ntotal);

    U* c = codes + npos * m;

#pragma omp parallel for
    for (int32_t i = 0; i < n; ++i) {
        compute_code(x + i * d, c + i * m);
    }

    npos += n;
}

template<class C, typename T, typename U>
void PQ<C, T, U>::search(int32_t nq, const T* q, int32_t topk,
            typename C::T* values, typename C::TI* labels,
            PQ_Computer<T> computer) {
    
    // precompute distance tables
    float* dis_tabs = new float[nq * m * K];

#pragma omp parallel for
    for (int32_t i = 0; i < nq; ++i) {
        compute_dis_tab(q + i * d, dis_tabs + i * m * K, computer);
    }

    // search
#pragma omp parallel for
    for (int32_t i = 0; i < nq; ++i) {
        const float* dis_tab = dis_tabs + i * m * K;

        auto* __restrict val_ = values  + i * topk;
        auto* __restrict ids_ = labels  + i * topk;

        heap_heapify<C>(topk, val_, ids_);
        const U* c = codes;

        for (int j = 0; j < ntotal; ++j) {
            float dis = 0;
            const float* __restrict dt = dis_tab;
            for (int mm = 0; mm < m; ++mm) {
                dis += dt[*c++];
                dt += K;
            }

            if (C::cmp(val_[0], dis)) {
                heap_swap_top<C>(topk, val_, ids_, dis, j);
            }
        }
        heap_reorder<C>(topk, val_, ids_);
    }
}