#pragma once

#include "util/heap.h"
#include "util/distance.h"

#include <stdint.h>
#include <string.h>
#include <assert.h>

#include <omp.h>

#include <iostream>
#include <memory>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <limits>
#include <functional>
#include "util/utils.h"
#include "ivf/clustering.h"

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
    };

    ProductQuantizer(int32_t _d, uint8_t _m, uint8_t _nbits)
    : ntotal(0), d(_d), m(_m), nbits(_nbits), npos(0) {
        assert(d % m == 0);

        dsub = d / m;
        K = 1 << nbits;

        centroids = new float[m * K * dsub];
        codes = nullptr;
    };

    ~ProductQuantizer() {
        if (centroids != nullptr) {
            delete[] centroids;
        }

        if (codes != nullptr) {
            delete[] codes;
        }
    }

    float* get_centroids() {
        return centroids;
    }

    U* get_codes() {
        return codes;
    }

    void show_centroids() {
        std::cout << "show pq.centroids:" << std::endl;
        auto pc = centroids;
        for (auto i = 0; i < m; i ++) {
            std::cout << "m = " << i << std::endl;
            for (auto j = 0; j < K; j ++) {
                std::cout << j << ": (";
                for (auto k = 0; k < dsub; k ++)
                    std::cout << *pc ++ << " ";
                std::cout << ")" << std::endl;
            }
            std::cout << std::endl;
        }
    }

    void show_pretab(const float* precompute_table) {
        assert(precompute_table != nullptr);
        std::cout << "show pq.precompute_table:" << std::endl;
        auto pp = precompute_table;
        for (auto i = 0; i < m; i ++) {
            std::cout << "m = " << std::endl;
            for (auto j = 0; j < K; j ++)
                std::cout << "(" << j << ", " << *pp++ << ") ";
            std::cout << std::endl;
        }
    }

    void calc_precompute_table(float*& precompute_table, const T* q, PQ_Computer<T> computer) {
        if (precompute_table == nullptr) {
            precompute_table = new float[K * m];
        }

        const float* c = centroids;
        float* dis_tab = precompute_table;
        for (uint8_t i = 0; i < m; ++i, q += dsub) {
            for (int32_t j = 0; j < K; ++j, c += dsub) {
                *dis_tab++ = computer(q, c, dsub);
            }
        }
    }

    void train(int32_t n, const T* x);

    void encode_vectors(float*& precomputer_table,
                        int32_t n, const T* x,
                        bool append = false);

    void encode_vectors_and_save(float*& precomputer_table,
                                 int32_t n, const T *x,
                                 const std::string& save_file);

    void search(int32_t nq, const T* q, int32_t topk,
                   typename C::T* values, typename C::TI* labels,
                   PQ_Computer<T> computer);

    void search(float* precompute_table, const T* q, const U* pcodes, int32_t len, int32_t topk,
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
    for (int i = 0; i < m; ++i, x += dsub) {
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
    const size_t sub_code_size = dsub * sizeof(T);
    T* xs = new T[n * dsub];

    bool remove_dup = false;
    if (sub_code_size <= 4) {
        printf("Remove duplicates dsub %d * sizeof(Type) %d\n", dsub, sizeof(T));
        remove_dup = true;
    }

    for (uint8_t i = 0; i < m; ++i) {
        if (remove_dup) {
            uint32_t idx = 0;
            std::unordered_set<uint32_t> st;

            if (sub_code_size == 4) {
                auto u_xd = reinterpret_cast<const uint32_t*>(x) + i;
                auto u_xs = reinterpret_cast<uint32_t*>(xs);
                for (int32_t j = 0; j < n; j++) {
                    if (st.find(*u_xd) == st.end()) {
                        st.insert(*u_xd);
                        u_xs[idx++] = *u_xd;
                    }
                    u_xd += m;
                }
            } else {
                uint32_t val = 0;
                auto xd = x + i * dsub;
                for (int32_t j = 0; j < n; j++){
                    memcpy(&val, xd, sub_code_size);
                    if (st.find(val) == st.end()) {
                        st.insert(val);
                        memcpy(xs + idx * dsub, xd, sub_code_size);
                        idx++;
                    }
                    xd += d;
                }
            }

            if (idx < K) {
                printf("Unable to find %d points from %d training data, found %d.\n", K, n, idx);
                // todo: add some random data into xs
            } else {
                printf("Duplicate points removed, n from %d to %d\n", n , idx);
            }

            kmeans<T>(idx, xs, dsub, K, centroids + i * K * dsub);

        } else {
            auto xd = x + i * dsub;
            for (int32_t j = 0; j < n; ++j) {
                memcpy(xs + j * dsub, xd, sub_code_size);
                xd += d;
            }

            kmeans<T>(n, xs, dsub, K, centroids + i * K * dsub);
        }
    }

    delete[] xs;
};

template<class C, typename T, typename U>
void ProductQuantizer<C, T, U>::encode_vectors(float*& precomputer_table,
                                               int32_t n, const T *x,
                                               bool append) {
    if (!append) {
        npos = 0;
    }

    if (npos + n < ntotal) {
        ntotal = npos + n;
        U* new_codes = new U[ntotal * m];
        if (npos != 0) {
            memcpy(new_codes, codes, npos * m * sizeof(U));
        }
        if (codes != nullptr) {
            delete[] codes;
        }
        codes = new_codes;
    }

    U* c = codes + npos * m;

    bool new_precomputer_table = false;
    if (precomputer_table == nullptr) {
        precomputer_table = new float[m * K * (K - 1) / 2];
        new_precomputer_table = true;
    }

    for (int32_t loop = 0; loop < m; loop++) {
        float *data = precomputer_table + loop * K * (K - 1) / 2;
        float* cen = centroids + loop * K * dsub;

        auto Y = [&](int32_t i, int32_t j) -> float& {
            assert(i != j);
            return (i > j) ? data[j + i * (i - 1) / 2] : data[i + j * (j - 1) / 2];
        };

        if (new_precomputer_table) {
#pragma omp parallel
            {
                int nt = omp_get_num_threads();
                int rank = omp_get_thread_num();
                for (int32_t i = 1 + rank; i < K; i += nt) {
                    float* y_i = cen + i * dsub;
                    for (int32_t j = 0; j < i; j++) {
                        float* y_j = cen + j * dsub;
                        Y(i, j) = L2sqr<float,float,float>(y_i, y_j, dsub);
                    }
                }
            }
        }

#pragma omp parallel for
        for (int32_t i = 0; i < n; i++) {
            const T* x_i = x + i * d + loop * dsub;

            int32_t ids_i = 0;
            float val_i = L2sqr<const T,const float,float>(x_i, cen, dsub);
            float val_i_time_4 = val_i * 4;
            for (int32_t j = 1; j < K; j++) {
                if (val_i_time_4 <= Y(ids_i, j)) {
                    continue;
                }
                const float *y_j = cen + j * dsub;
                float disij = L2sqr<const T,const float,float>(x_i, y_j, dsub);
                if (disij < val_i) {
                    ids_i = j;
                    val_i = disij;
                    val_i_time_4 = val_i * 4;
                }
            }

            c[i * m + loop] = ids_i;
        }
    }

    npos += n;
}

template<class C, typename T, typename U>
void ProductQuantizer<C, T, U>::encode_vectors_and_save(float*& precomputer_table,
                                                        int32_t n, const T *x,
                                                        const std::string& save_file) {
    encode_vectors(precomputer_table, n, x, false);

    uint32_t wm = m;
    std::ofstream code_writer(save_file, std::ios::binary | std::ios::out);
    code_writer.write((char*)&n, sizeof(uint32_t));
    code_writer.write((char*)&wm, sizeof(uint32_t));
    code_writer.write((char*)codes, n * m * sizeof(U));
    code_writer.close();
    std::cout << "ProductQuantizer encode " << n << " vectors with m = " << wm << " into file "
              << save_file << std::endl;
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
void ProductQuantizer<C, T, U>::search(float* precompute_table, const T* q, const U* pcodes,
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

