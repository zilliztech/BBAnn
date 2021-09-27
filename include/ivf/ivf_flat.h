#pragma once

#include "ivf/clustering.h"
#include "flat/flat.h"

#include <vector>
#include <algorithm>
#include <functional>
#include <omp.h>

template<typename T>
void ivf_flat_insert(int64_t nx, const T* x_in, int64_t dim,
                     int64_t nlist, const float* centroids,
                     std::vector<std::vector<T>> &codes,
                     std::vector<std::vector<uint32_t>> &ids) {

    std::unique_ptr<int64_t []> assign(new int64_t[nx]);
    std::unique_ptr<float []> dis(new float[nx]);
    elkan_L2_assign(x_in, centroids, dim, nx, nlist, assign.get(), dis.get());
    dis = nullptr;

    codes.resize(nlist);
    ids.resize(nlist);

#pragma omp parallel
    {
        int64_t nt = omp_get_num_threads();
        int64_t rank = omp_get_thread_num();

        // this thread is taking care of centroids c0:c1
        int64_t c0 = (nlist * rank) / nt;
        int64_t c1 = (nlist * (rank + 1)) / nt;

        for (int64_t i = 0; i < nx; i++) {
            int64_t ci = assign[i];
            if (ci >= c0 && ci < c1) {
                codes[ci].resize(codes[ci].size() + dim);
                memcpy(codes[ci].data() + ids[ci].size() * dim,
                       x_in + i * dim, dim * sizeof(T));
                ids[ci].push_back(i);
            }
        }
    }
}

template<class C, typename T, typename R>
void ivf_flat_search(int64_t nq, const T* q_in, int64_t dim,
                     int64_t nlist, const float* centroids,
                     std::vector<std::vector<T>> &codes,
                     std::vector<std::vector<uint32_t>> &ids,
                     int64_t nprobe, int64_t topk,
                     typename C::T * value,
                     typename C::TI * labels,
                     Computer<T, T, typename C::T> comptuer) {

    std::unique_ptr<uint32_t[]> idx(new uint32_t[nq * nprobe]);
    std::unique_ptr<float[]> coarse_dis(new float[nq * nprobe]);
    knn_1<CMax<float, uint32_t>, T, float> (
        q_in, centroids, nq, nlist, dim, nprobe, coarse_dis.get(), idx.get(), L2sqr<const T, const float, float>);
    coarse_dis = nullptr;

#pragma omp parallel for
    for (int64_t i = 0; i < nq; i++) {
        auto *q_i = q_in + i * dim;

        auto * __restrict val_ = value  + i * topk;
        auto * __restrict ids_ = labels  + i * topk;

        // init heap
        heap_heapify<C>(topk, val_, ids_);

        for (int64_t j = 0; j < nprobe; j++) {
            auto idx_j = idx[i * nprobe + j];
            auto code = codes[idx_j].data();
            auto id = ids[idx_j].data();
            int64_t count = ids[idx_j].size();

            for (int64_t k = 0; k < count; k++) {
                auto disij = comptuer (q_i, code, dim);
                if (C::cmp(val_[0], disij)) {
                    heap_swap_top<C>(topk, val_, ids_, disij, id[k]);
                }
                code += dim;
            }
        }

        heap_reorder<C> (topk, val_, ids_);
    }
}
