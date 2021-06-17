#pragma once

#include "ivf/clustering.h"
#include "flat/flat.h"
#include "utils/distance.h"

#include <vector>
#include <algorithm>
#include <functional>
#include <omp.h>

template<typename T>
void ivf_flat_insert(int32_t nx, const T* x_in, int32_t dim,
                     int32_t nlist, const float* centroids,
                     std::vector<std::vector<T>> &codes,
                     std::vector<std::vector<int32_t>> &ids) {

    std::unique_ptr<int32_t []> assign(new int32_t[nx]);
    std::unique_ptr<float []> dis(new float[nx]);
    elkan_L2_assign(x_in, centroids, dim, nx, nlist, assign.get(), dis.get());
    dis = nullptr;

    codes.resize(nlist);
    ids.resize(nlist);

#pragma omp parallel
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // this thread is taking care of centroids c0:c1
        size_t c0 = (nlist * rank) / nt;
        size_t c1 = (nlist * (rank + 1)) / nt;

        for (int32_t i = 0; i < nx; i++) {
            int32_t ci = assign[i];
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
void ivf_flat_search(int32_t nq, const T* q_in, int32_t dim,
                     int32_t nlist, const float* centroids,
                     std::vector<std::vector<T>> &codes,
                     std::vector<std::vector<int32_t>> &ids,
                     int32_t nprobe, int32_t topk,
                     typename C::T * value,
                     typename C::TI * labels,
                     Computer<T, T, typename C::T> comptuer) {

    std::unique_ptr<int32_t[]> idx(new int32_t[nq * nprobe]);
    std::unique_ptr<float[]> coarse_dis(new float[nq * nprobe]);
    knn_1<CMax<float, int32_t>, T, float> (
        q_in, centroids, nq, nlist, dim, nprobe, coarse_dis.get(), idx.get(), L2sqr<const T, const float, float>);
    coarse_dis = nullptr;

#pragma omp parallel for
    for (int32_t i = 0; i < nq; i++) {
        auto *q_i = q_in + i * dim;

        auto * __restrict val_ = value  + i * topk;
        auto * __restrict ids_ = labels  + i * topk;

        // init heap
        heap_heapify<C>(topk, val_, ids_);

        for (int32_t j = 0; j < nprobe; j++) {
            auto idx_j = idx[i * nprobe + j];
            auto code = codes[idx_j].data();
            auto id = ids[idx_j].data();
            int32_t count = ids[idx_j].size();

            for (int32_t k = 0; k < count; k++) {
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