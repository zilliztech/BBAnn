#pragma once

#include "clusting.h"

#include <cstdint>
#include <cstring>

#include <omp.h>

#include <iostream>
#include <memory>
#include <vector>
#include <limits>

template <typename T1, typename T2, typename R>
using Computer = std::function<R(const T1*, const T2*, int n)>;

// T: type of the database vector
// R: type of the distance
template<class C, typename T, typename R>
class PQ {
private:
    int32_t d, dsub, K;
    uint8_t m, nbits;

    float* centroids = nullptr;
    
    uint8_t* codes = nullptr;

    void compute_code(const T* x, uint8_t* c);

    void compute_dis_tab(const T* q, float* dis_tab, Computer<T, T, typename C::T> computer);
public:
    PQ(int32_t _d, uint8_t _m, uint8_t _nbits)
    : d(_d), m(_m), nbits(_nbits) {
        if (d % m != 0) {
            std::cerr << "Dimension d = " << d
                      << "is not divisible by m = " << m
                      << std::endl;
        }

        dsub = d / m;
        K = 1 << nbits;

        centroids = new float[m * K * dsub];
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
                   Computer<T, T, typename C::T> computer);
};

template<class C, typename T, typename R>
void PQ<C, T, R>::compute_code(const T* x, uint8_t* c) {
    for (uint8_t i = 0; i < m; ++i) {
        float min_dist = std::numeric_limits<int32_t>::max();
        int32_t best_id = 0;

        // find the best centroid
        for (int32_t j = 0; j < K; ++j) {
            float dis = L2sqr(x + j * dsub, centroids + i * j * dsub, dsub);
            if (dis < min_dist) {
                dis = min_dist;
                best_id = j;
            }
        }

        (*c)++ = best_id;
    }
}

template<class C, typename T, typename R>
void PQ<C,T,R>::compute_dis_tab(const T* q, float* dis_tab,
                                 Computer<T, T, typename C::T> computer) {
    for (uint8_t i = 0; i < m; ++i) {
        for (int32_t j = 0; j < K; ++j) {
            dis_tab++ = computer(q, centroids);
            centroids += dsub;
        }
        q += dsub;
    }
}

template<class C, typename T, typename R>
void PQ<C, T, R>::train(int32_t n, const T* x) {
    T *xs = new T[n * dsub];

    for (uint8_t i = 0; i < m; ++i) {

        // get slice of x in subspace m_i
        for (int32_t j = 0; j < n; ++j) {
            memcpy(xs + j * dsub, x + j * d + i * dsub, sizeof(T));
        }

        // compute centroids
        kmeans(n, xs, dsub, K, centroids + i * K * dsub);
    }

    delete[] xs;
};

template<class C, typename T, typename R>
void PQ<C, T, R>::encode_vectors(int32_t n, const T* x) {
    if (codes != nullptr) {
        std::cout << "Re-encoding vectors using PQ" << std::endl;
        delete [] codes;
    }

    codes = new uint8_t[m*n];
    
    for (int32_t i = 0; i < n; ++i) {
        compute_code(x + i * d, codes + i * m);
    }
}

template<class C, typename T, typename R>
void PQ<C, T, R>::search(int32_t nq, const T* q, int32_t topk,
            typename C::T* values, typename C::TI* labels,
            Computer<T, T, typename C::T> computer) {
    
    // precompute distance tables
    float* dis_tabs = new float[nq * m * K];

#pragma omp parallel for
    for (int32_t i = 0; i < nq; ++i) {
        compute_dis_tab(q + i * d, dis_tabs + i * m * K, computer);
    }

    // search
#pragma omp parallel for
    for (int32_t i = 0; i < nq; ++i) {
        const float* dis_t = dis_tabs + i * m * K;

        int32_t* __restrict val_ = value  + i * topk;
        float* __restrict ids_ = labels  + i * topk;

        heap_heapify<C>(topk, val_, ids_);

        for (int j = 0; j < n; ++j) {
            float dis = 0;
            const float *__restrict dt = dis_tab;
            for (int mm = 0; mm < m; ++mm) {
                dis += dt[*codes++];
                dt += ksub;
            }

            if (C::cmp(val_[0], disij)) {
                heap_swap_top<C>(topk, val_, ids_, disij, id[k]);
            }
        }
        heap_reorder<C>(topk, val_, ids_);
    }
}