#pragma once

#include "util/heap.h"
#include "util/distance.h"

#include <stdint.h>
#include <string.h>
#include <assert.h>

#include <omp.h>

#include <iostream>
#include <memory>
#include <vector>
#include <limits>
#include <functional>
#include "util/utils.h"

template<typename T>
using PQ_Computer = std::function<float(const T*, const float*, int n)>;

// T: type of the database vector
// U: type of codes ID (PQ representation)
template<class C, typename T, typename U>
class ProductQuantizer {
private:
    int32_t d, dsub, K, ntotal, npos;
    uint8_t m, nbits;

    float* centroids = nullptr;
    U* codes = nullptr;
    float*  precompute_table = nullptr;

    void compute_code(const T* x, U* c);

    void compute_dis_tab(const T* q, float* dis_tab, PQ_Computer<T> computer);
public:
    // in-memory
    ProductQuantizer(int32_t _ntotal, int32_t _d, uint8_t _m, uint8_t _nbits)
    : ntotal(_ntotal), d(_d), m(_m), nbits(_nbits), npos(0) {
        assert(d % m == 0);

        dsub = d / m;
        K = 1 << nbits;

        centroids = new float[m * K * dsub];
        codes = new U[ntotal * m];
        precompute_table = nullptr;
    };

    ProductQuantizer(int32_t _d, uint8_t _m, uint8_t _nbits)
    : ntotal(0), d(_d), m(_m), nbits(_nbits), npos(0) {
        assert(d % m == 0);

        dsub = d / m;
        K = 1 << nbits;

        centroids = new float[m * K * dsub];
        codes = nullptr;
        precompute_table = nullptr;
    };

    ProductQuantizer(const ProductQuantizer& pq) {
        d = pq.d;
        dsub = pq.dsub;
        K = pq.K;
        m = pq.m;
        nbits = pq.nbits;
        centroids = pq.centroids;
        codes = nullptr;
        // precompute distance tables
        precompute_table = new float[m * K];
//        compute_dis_tab(q, precompute_table, computer);
    }

    ~ProductQuantizer() {
        if (centroids != nullptr) {
            delete[] centroids;
        }

        if (codes != nullptr) {
            delete[] codes;
        }

        if (precompute_table != nullptr) {
            delete[] precompute_table;
        }
    }

    void reset() {
        centroids = nullptr;
    }

    void cal_precompute_table(const T* q, PQ_Computer<T> computer) {
        const float* c = centroids;
        for (uint8_t i = 0; i < m; ++i, q += dsub) {
            for (int32_t j = 0; j < K; ++j, c += dsub) {
                *precompute_table++ = computer(q, c, dsub);
            }
        }
    }

    void train(int32_t n, const T* x);

    void encode_vectors(int32_t n, const T* x);

    void encode_vectors_and_save(int32_t n, const T* x, const std::string& save_file);

    void search(int32_t nq, const T* q, int32_t topk,
                   typename C::T* values, typename C::TI* labels,
                   PQ_Computer<T> computer);

    void search(const T* q, const U* pcodes, int32_t len, int32_t topk,
                typename C::T* values, typename C::TI* labels,
                PQ_Computer<T> computer, bool reorder, bool heapify,
                const uint32_t& cid, const uint32_t& off, const uint32_t& qid);

    void save_centroids(const std::string& save_file) {
        uint32_t num_centroids = (uint32_t)m * K;
        uint32_t dim_centroids = dsub;
        std::ofstream centroids_writer(save_file, std::ios::binary);
        centroids_writer.write((char*)&num_centroids, sizeof(uint32_t));
        centroids_writer.write((char*)&dim_centroids, sizeof(uint32_t));
        centroids_writer.write((char*)centroids, sizeof(float) * K * d);
        centroids_writer.close();
    }

    void load_centroids(const std::string& load_file) {
        uint32_t num_centroids ,dim_centroids;
        std::ifstream centroids_reader(load_file, std::ios::binary);
        centroids_reader.read((char*)&num_centroids, sizeof(uint32_t));
        centroids_reader.read((char*)&dim_centroids, sizeof(uint32_t));
        centroids_reader.read((char*)centroids, sizeof(float) * K * d);
        centroids_reader.close();
    }
};

template<class C, typename T, typename U>
void ProductQuantizer<C, T, U>::compute_code(const T* x, U* c) {
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
void ProductQuantizer<C, T, U>::compute_dis_tab(const T* q, float* dis_tab,
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
void ProductQuantizer<C, T, U>::train(int32_t n, const T* x) {
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
void ProductQuantizer<C, T, U>::encode_vectors(int32_t n, const T *x) {
    assert(npos + n <= ntotal);

    U* c = codes + npos * m;

#pragma omp parallel for
    for (int32_t i = 0; i < n; ++i) {
        compute_code(x + i * d, c + i * m);
    }

    npos += n;
}

template<class C, typename T, typename U>
void ProductQuantizer<C, T, U>::encode_vectors_and_save(int32_t n, const T *x, const std::string& save_file) {
    U* c = new U[n * m];

#pragma omp parallel for
    for (int32_t i = 0; i < n; ++i) {
        compute_code(x + i * d, c + i * m);
    }

    std::ofstream code_writer(save_file, std::ios::binary);
    code_writer.write((char*)&n, sizeof(uint32_t));
    code_writer.write((char*)&m, sizeof(uint32_t));
    code_writer.write((char*)c, sizeof(U) * n * m);
    code_writer.close();
    delete[] c;
    c = nullptr;
}

template<class C, typename T, typename U>
void ProductQuantizer<C, T, U>::search(int32_t nq, const T* q, int32_t topk,
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

template<class C, typename T, typename U>
void ProductQuantizer<C, T, U>::search(const T* q, const U* pcodes, 
                         int32_t codebook_len, int32_t topk,
                         typename C::T* values, typename C::TI* labels,
                         PQ_Computer<T> computer, bool reorder, 
                         bool heapify, const uint32_t& cid, 
                         const uint32_t& off, const uint32_t& qid) {
    

    // search
    {
        assert(precompute_table != nullptr);
        const float* dis_tab = precompute_table;

        auto* __restrict val_ = values;
        auto* __restrict ids_ = labels;

        if (heapify)
            heap_heapify<C>(topk, val_, ids_);
        const U* c = pcodes;

        for (int j = 0; j < codebook_len; ++j) {
            float dis = 0;
            const float* __restrict dt = dis_tab;
            for (int mm = 0; mm < m; ++mm) {
                dis += dt[*c++];
                dt += K;
            }

            if (C::cmp(val_[0], dis)) {
                heap_swap_top<C>(topk, val_, ids_, dis, gen_refine_id(cid, off + j, qid));
            }
        }
        if (reorder)
            heap_reorder<C>(topk, val_, ids_);
    }
}

