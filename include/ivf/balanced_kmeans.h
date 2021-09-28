#pragma once

#include "ivf/kmeans.h"

// Data type: T1, T2
// Distance type: R
// ID type int64_t
template <typename T1, typename T2, typename R>
void dynamic_assign(
        const T1 * x, // data base vector
        const T2 * y, // centroids vector
        int64_t dim, int64_t nx, int64_t ny,
        float weight, int64_t *ids, R *val
) {
    if (nx == 0 || ny == 0) {
        return;
    }
    int64_t* hassign = new int64_t [ny];
    float dist = 0.0;
    int64_t min = 0;
    float min_value = 0.0;
    memset(hassign, 0, sizeof(int64_t)*ny);

    for (int i = 0; i < nx; i++) {
        auto * __restrict x_in = x + i *dim;
        min_value = L2sqr<const T1, const T2, R>(x_in, y, dim) + weight * hassign[0];
        min =0;
        for (int j = 1; j< ny; j++) {
            auto * __restrict  y_in = y + j * dim;
            dist = L2sqr<const T1, const T2, R>(x_in, y_in, dim) + weight * hassign[j];
            if (dist < min_value) {
                min = j;
                min_value = dist;
            }
        }
        ids[i] = min;
        val[i] = min_value;
        hassign[min]++;
    }
}


template <typename T>
void balanced_kmeans (int64_t nx, const T* x_in, int64_t dim, int64_t k, float* centroids,
             float weight, bool kmpp = false, float avg_len = 0.0, int64_t niter = 10,
             int64_t seed = 1234) {

    if (k > 1000)
        nx = k * 40;
    std::cout << "new nx = " << nx << std::endl;
    const int64_t max_points_per_centroid = 256;
    const int64_t min_points_per_centroid = 39;

    if (nx < k) {
        printf("trained points is not enough %ld given %ld\n", k, nx);
        return;
    }

    if (nx == k) {
        for (int64_t i = 0; i < nx * dim; i++)
            centroids[i] = x_in[i];
        return;
    }

    if (nx < k * min_points_per_centroid) {
        printf("Too little trained points need %ld given %ld\n", k * min_points_per_centroid, nx);
    } else if (nx > k * max_points_per_centroid) {
        printf("Too many trained points need %ld given %ld\n", k * max_points_per_centroid, nx);
    }

    std::unique_ptr<int64_t []> hassign(new int64_t[k]);
  //  std::unique_ptr<float []> sum(new float[k * dim]);

    std::unique_ptr<int64_t []> assign(new int64_t[nx]);
    std::unique_ptr<float []> dis(new float[nx]);

    if (kmpp) {
        kmeanspp<T>(x_in, nx, dim, k, centroids);
    } else {
        rand_perm(assign.get(), nx, k, seed);
        for (int64_t i = 0; i < k; i++) {
            // std::cout<<i<<assign[i]<<std::endl;
            const T* x = x_in + assign[i] * dim;

            float* c = centroids + i * dim;

            for (int64_t d = 0; d < dim; d++){
                c[d] = x[d];

            }
        }
    }

    float err = std::numeric_limits<float>::max();
    for (int64_t i = 0; i < niter; i++) {
        // printf("iter %d ", i);
        if( weight!=0 && nx <= KMEANS_THRESHOLD ) {
           dynamic_assign<T, float, float>(x_in, centroids, dim, nx, k, weight, assign.get(), dis.get());
        } else {
            elkan_L2_assign<T, float, float>(x_in, centroids, dim, nx, k, assign.get(), dis.get());
        }

        compute_centroids<T>(dim, k, nx, x_in, assign.get(), hassign.get(), centroids, avg_len);

        int64_t split = split_clusters(dim, k, nx, hassign.get(), centroids);

        if (split != 0) {
            printf("split %ld\n", split);
        } else {
            float cur_err = 0.0;
            for (auto j = 0; j < nx; j ++)
                cur_err += dis[j];

            if (fabs(cur_err - err) < err * 0.01) {
                std::cout << "exit kmeans iteration after the " << i << "th iteration, err = " << err << ", cur_err = " << cur_err << std::endl;
                break;
            }
            err = cur_err;
        }
    }

    int empty_cnt = 0;
    int mx, mn;
    mx = mn = hassign[0];
    for (auto i = 0; i < k; i ++) {
        if (hassign[i] == 0)
            empty_cnt ++;
        if (hassign[i] > mx)
            mx = hassign[i];
        if (hassign[i] < mn)
            mn = hassign[i];
        // std::cout<<hassign[i]<<std::endl;
    }
    std::cout << "after the kmeans with nx = " << nx << ", k = " << k
              << ", has " << empty_cnt << " empty clusters,"
              << " max cluster: " << mx
              << " min cluster: " << mn
              << std::endl;
}