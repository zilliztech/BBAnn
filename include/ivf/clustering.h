#pragma once

#include <omp.h>

#include <string.h>
#include <assert.h>
#include <memory>
#include "util/distance.h"
#include "util/utils.h"
#include "util/random.h"


// Data type: T1, T2
// Distance type: R
// ID type int32_t

template<typename T1, typename T2, typename R>
void elkan_L2_assign (
        const T1 * x,
        const T2 * y,
        uint64_t dim, uint64_t nx, uint64_t ny,
        uint64_t *ids, R *val) {

    if (nx == 0 || ny == 0) {
        return;
    }

    const size_t bs_y = 1024;
    R *data = (R *) malloc((bs_y * (bs_y - 1) / 2) * sizeof (R));

    for (uint64_t j0 = 0; j0 < ny; j0 += bs_y) {
        uint64_t j1 = j0 + bs_y;
        if (j1 > ny) j1 = ny;

        auto Y = [&](uint64_t i, uint64_t j) -> R& {
            assert(i != j);
            i -= j0, j -= j0;
            return (i > j) ? data[j + i * (i - 1) / 2] : data[i + j * (j - 1) / 2];
        };

#pragma omp parallel
        {
            int nt = omp_get_num_threads();
            int rank = omp_get_thread_num();
            for (uint64_t i = j0 + 1 + rank; i < j1; i += nt) {
                const T2* y_i = y + i * dim;
                for (uint64_t j = j0; j < i; j++) {
                    const T2* y_j = y + j * dim;
                    Y(i, j) = L2sqr<const T2,const T2,R>(y_i, y_j, dim);
                }
            }
        }

#pragma omp parallel for
        for (uint64_t i = 0; i < nx; i++) {
            const T1* x_i = x + i * dim;

            uint64_t ids_i = j0;
            R val_i = L2sqr<const T1,const T2,R>(x_i, y + j0 * dim, dim);
            R val_i_time_4 = val_i * 4;
            for (uint64_t j = j0 + 1; j < j1; j++) {
                if (val_i_time_4 <= Y(ids_i, j)) {
                    continue;
                }
                const T2 *y_j = y + j * dim;
                R disij = L2sqr<const T1,const T2,R>(x_i, y_j, dim / 2);
                if (disij >= val_i) {
                    continue;
                }
                disij += L2sqr<const T1,const T2,R>(x_i + dim / 2, y_j + dim / 2, dim - dim / 2);
                if (disij < val_i) {
                    ids_i = j;
                    val_i = disij;
                    val_i_time_4 = val_i * 4;
                }
            }

            if (j0 == 0 || val[i] > val_i) {
                val[i] = val_i;
                ids[i] = ids_i;
            }
        }
    }

    free(data);
}

// avg_sqr_len:
//    0: not to normalize
//    else: normalize
template <typename T>
void compute_centroids (int32_t dim, int32_t k, int32_t n,
                       const T * x,
                       const uint64_t * assign,
                       int32_t * hassign,
                       float * centroids,
                       float avg_sqr_len = 0.0)
{
    memset(hassign, 0, sizeof(int32_t) * k);
    memset(centroids, 0, sizeof(float) * dim * k);

#pragma omp parallel
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // this thread is taking care of centroids c0:c1
        size_t c0 = (k * rank) / nt;
        size_t c1 = (k * (rank + 1)) / nt;

        for (int32_t i = 0; i < n; i++) {
            uint64_t ci = assign[i];
            if (ci >= c0 && ci < c1)  {
                float * c = centroids + ci * dim;
                const T * xi = x + i * dim;
                for (int32_t j = 0; j < dim; j++) {
                    c[j] += xi[j];
                }
                hassign[ci] ++;
            }
        }
    }

#pragma omp parallel for
    for (int32_t ci = 0; ci < k; ci++) {
        if (hassign[ci] == 0) {
            continue;
        }

        float *c = centroids + ci * dim;
        if (avg_sqr_len != 0.0) {
            float len = sqrt(avg_sqr_len / IP<float, float, double>(c, c, dim));
            for (int32_t j = 0; j < dim; j++){
                c[j] *= len;
            }
        } else {
            float norm = 1.0 / hassign[ci];
            for (int32_t j = 0; j < dim; j++) {
                c[j] *= norm;
            }
        }
    }
}

int32_t split_clusters (int32_t dim, int32_t k, int32_t n,
                        int32_t * hassign, float * centroids)
{
    const double EPS = (1 / 1024.);
    /* Take care of void clusters */
    int32_t nsplit = 0;

    for (int32_t ci = 0; ci < k; ci++) {
        if (hassign[ci] == 0) { /* need to redefine a centroid */
            int32_t cj;
            for (cj = 0; 1; cj = (cj + 1) % k) {
                /* probability to pick this cluster for split */
                float p = (hassign[cj] - 1.0) / (float) (n - k);
                float r = rand_float();
                if (r < p) {
                    break; /* found our cluster to be split */
                }
            }
            memcpy (centroids+ci*dim, centroids+cj*dim, sizeof(float) * dim);

            /* small symmetric pertubation */
            for (int32_t j = 0; j < dim; j++) {
                if (j % 2 == 0) {
                    centroids[ci * dim + j] *= 1 + EPS;
                    centroids[cj * dim + j] *= 1 - EPS;
                } else {
                    centroids[ci * dim + j] *= 1 - EPS;
                    centroids[cj * dim + j] *= 1 + EPS;
                }
            }

            /* assume even split of the cluster */
            hassign[ci] = hassign[cj] / 2;
            hassign[cj] -= hassign[ci];
            nsplit++;
        }
    }

    return nsplit;
}


template <typename DATAT>
void kmeanspp(const DATAT* pdata, const uint64_t nb, const uint64_t dim,
              const uint64_t num_clusters, float*& centroids) {
    auto disf = L2sqr<const DATAT, float, float>;
    std::random_device rd;
    auto x = rd();
    std::mt19937 generator(x);
    std::uniform_int_distribution<uint64_t> distribution(0, nb - 1);

    uint64_t init_id = distribution(generator);
    std::vector<float> dist(nb, std::numeric_limits<float>::max());
    for (auto i = 0; i < dim; i ++)
        centroids[i] = (float)pdata[init_id * dim + i];

    for (auto i = 1; i < num_clusters; i ++) {
        double sumdx = 0.0;
#pragma omp parallel for schedule(static, 4096) reduction(+: sumdx)
        for (auto j = 0; j < nb; j ++) {
            auto dist_cj = disf(pdata + j * dim, centroids + (i - 1) * dim, dim);
            dist[j] = std::min(dist[j], dist_cj);
            sumdx += dist[j];
        }
        std::uniform_real_distribution<double> distridb(0, sumdx);
        auto prob = distridb(generator);
        for (auto j = 0; j < nb; j ++) {
            if (prob <= 0) {
                for (auto k = 0; k < dim; k ++) 
                    centroids[i * dim + k] = (float)pdata[j * dim + k];
                break;
            }
            prob -= dist[j];
        }
    }
}

// Data Type: T
// Distance Type: float
// Centroid Type: float

template <typename T>
void kmeans (int32_t nx, const T* x_in, int32_t dim, int32_t k, float* centroids,
             bool kmpp = false, bool normalize = false, int32_t niter = 10, 
             int32_t seed = 1234) {
    std::cout << "start do " << k << "-means on " << nx << " vectors" << std::endl;
    if (k > 1000)
        nx = k * 40;
    std::cout << "new nx = " << nx << std::endl;
    const int32_t max_points_per_centroid = 256;
    const int32_t min_points_per_centroid = 39;

    if (nx < k) {
        printf("trained points is not enough %d given %d\n", k, nx);
        return;
    }
    
    if (nx == k) {
        for (int32_t i = 0; i < nx * dim; i++)
            centroids[i] = x_in[i];
        return;
    }
    
    if (nx < k * min_points_per_centroid) {
        printf("Too little trained points need %d given %d\n", k * min_points_per_centroid, nx);
    } else if (nx > k * max_points_per_centroid) {
        printf("Too many trained points need %d given %d\n", k * max_points_per_centroid, nx);
    }

    std::unique_ptr<int32_t []> hassign(new int32_t[k]);
    std::unique_ptr<float []> sum(new float[k * dim]);

    std::unique_ptr<uint64_t []> assign(new uint64_t[nx]);
    std::unique_ptr<float []> dis(new float[nx]);

    float avg_sqr_len = 0;
    if (normalize) {
        float max_sqr_len = 0;
        float min_sqr_len = std::numeric_limits<float>::max();
        for (int32_t i = 0; i < nx; i++) {
            const T *p = x_in + i * dim;
            float sqr_len = IP<const T, const T, float>(p, p, dim);
            if (sqr_len > max_sqr_len) max_sqr_len = sqr_len;
            if (sqr_len < min_sqr_len) min_sqr_len = sqr_len;
            avg_sqr_len += sqr_len;
        }
        if (min_sqr_len / max_sqr_len > 0.81) {
            avg_sqr_len /= nx;
            printf("min_sqr_len %f max_sqr_len %f avg_sqr_len %lf\n", min_sqr_len, max_sqr_len, avg_sqr_len);
        } else {
            // it is inappropriate to normalize
            avg_sqr_len = 0;
            printf("min_sqr_len %f max_sqr_len %f, not to normalize\n", min_sqr_len, max_sqr_len);
        }
    }

    if (kmpp) {
        kmeanspp<T>(x_in, nx, dim, k, centroids);
    } else {
        rand_perm(assign.get(), nx, k, seed);
        for (int32_t i = 0; i < k; i++) {
            const T* x = x_in + assign[i] * dim;
            float* c = centroids + i * dim;
            for (int32_t d = 0; d < dim; d++){
                c[d] = x[d];
            }
        }
    }

    float err = std::numeric_limits<float>::max();
    for (int32_t i = 0; i < niter; i++) {
        // printf("iter %d ", i);

        elkan_L2_assign<T, float, float>(x_in, centroids, dim, nx, k, assign.get(), dis.get());
        compute_centroids<T>(dim, k, nx, x_in, assign.get(), hassign.get(), centroids, avg_sqr_len);

        int32_t split = split_clusters(dim, k, nx, hassign.get(), centroids);
        if (split != 0) {
            printf("split %d\n", split);
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
    }
    std::cout << "after the kmeans with nx = " << nx << ", k = " << k 
              << ", has " << empty_cnt << " empty clusters," 
              << " max cluster: " << mx
              << " min cluster: " << mn
              << std::endl;
}
