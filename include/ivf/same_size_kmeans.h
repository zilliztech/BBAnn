#pragma once

#include <omp.h>

#include "util/distance.h"
#include "util/random.h"
#include "util/utils.h"
#include <algorithm>
#include <assert.h>
#include <deque>
#include <memory>
#include <string.h>
#include <unistd.h>

// #define SSK_LOG

template <typename T>
void ssk_compute_dist_tab(int64_t nx, const T *x_in, int64_t dim, int64_t k,
                          const float *centroids, float *dis_tab) {
// #pragma omp parallel for
  for (int64_t i = 0; i < nx; ++i) {
    const T *x = x_in + i * dim;
    const int64_t ii = i * k;
    for (int64_t j = 0; j < k; ++j) {
      dis_tab[ii + j] =
          L2sqr<const T, const float, float>(x, centroids + j * dim, dim);
    }
  }
}

void ssk_init_assign(int64_t nx, int64_t k, uint64_t max_target_size,
                     const float *dis_tab, int64_t *hassign, int64_t *assign) {
  uint64_t remain = nx;
  std::vector<float> min_max_dist_diff(nx);
  std::vector<int64_t> min_cluster_ids(nx);
  bool idx = 0;
  // rotate vector each iteration
  std::vector<std::vector<int64_t>> points(2, std::vector<int64_t>(nx));

  for (int64_t i = 0; i < nx; ++i) {
    points[idx][i] = i;
  }

  while (remain) {
    for (int64_t i = 0; i < remain; ++i) {
      float min_dis = std::numeric_limits<float>::max();
      float max_dis = 0;
      int64_t min_cluster_id = -1;
      const auto x = points[idx][i];

      for (int64_t j = 0; j < k; ++j) {
        if (hassign[j] < max_target_size) {
          auto dis = dis_tab[x * k + j];
          if (dis < min_dis) {
            min_dis = dis;
            min_cluster_id = j;
          }
          max_dis = std::max(max_dis, dis);
        }
      }
      // this should not happen as the max_target_size is a ceiling
      // so there is at least one of the clusters could fit the vector
      assert(min_cluster_id != -1);
      min_cluster_ids[x] = min_cluster_id;
      min_max_dist_diff[x] = min_dis - max_dis;
    }

    std::sort(points[idx].begin(), points[idx].begin() + remain,
              [&](const auto &x, const auto &y) {
                return min_max_dist_diff[x] < min_max_dist_diff[y];
              });

    int64_t j = 0;
    for (int64_t i = 0; i < remain; ++i) {
      const auto x = points[idx][i];
      const auto c = min_cluster_ids[x];

      if (hassign[c] < max_target_size) {
        assign[x] = c;
        ++hassign[c];
      } else {
        points[!idx][j++] = x;
      }
    }

    remain = j;
    idx = !idx;
  }
}

void ssk_print_cluster_size_stats(int64_t k, const int64_t *hassign) {
  float mini = std::numeric_limits<float>::max(), maxi = 0, avg = 0;
  for (int64_t i = 0; i < k; ++i) {
    avg += hassign[i];
    mini = std::min(mini, 1.0f * hassign[i]);
    maxi = std::max(maxi, 1.0f * hassign[i]);
  }
  std::cout << "avg: " << avg / k << " min: " << mini << " max: " << maxi
            << std::endl;
}

template <typename T>
void same_size_kmeans(int64_t nx, const T *x_in, int64_t dim, int64_t k,
                      float *centroids, int64_t *assign, bool kmpp = false,
                      float avg_len = 0.0, int64_t niter = 10,
                      int64_t seed = 1234) {
  assert(x_in != nullptr);
  assert(centroids != nullptr);
  assert(assign != nullptr);

  assert(nx > 0);
  assert(k > 0);

  uint64_t max_target_size = (nx + k - 1) / k;
  uint64_t min_target_size = nx / k;

  // the nubmer of vectors in the cluster
  int64_t *hassign = new int64_t[k];

  memset(hassign, 0, sizeof(int64_t) * k);

  if (kmpp) {
    kmeanspp<T>(x_in, nx, dim, k, centroids);
  } else {
    rand_perm(assign, nx, k, seed);
    for (int64_t i = 0; i < k; i++) {
      const T *x = x_in + assign[i] * dim;
      float *c = centroids + i * dim;

      for (int64_t d = 0; d < dim; d++) {
        c[d] = x[d];
      }
    }
  }

  float *dis_tab = new float[nx * k];
  assert(dis_tab != nullptr);

  ssk_compute_dist_tab(nx, x_in, dim, k, centroids, dis_tab);

#ifdef SSK_LOG
  std::cout << "init compute dis_tab done" << std::endl;
#endif

  ssk_init_assign(nx, k, max_target_size, dis_tab, hassign, assign);

#ifdef SSK_LOG
  std::cout << "Initialization done" << std::endl;
#endif

  int64_t *xs = new int64_t[nx];
  for (int64_t i = 0; i < nx; ++i) {
    xs[i] = i;
  }

  int64_t *ks = new int64_t[k];
  for (int64_t i = 0; i < k; ++i) {
    ks[i] = i;
  }

  auto delta_cur_best = [&](const auto &x) {
    float min_dis = std::numeric_limits<float>::max();
    for (int64_t i = 0; i < k; ++i) {
      min_dis = std::min(min_dis, dis_tab[x * k + i]);
    }
    return dis_tab[x * k + assign[x]] - min_dis;
  };

  auto gain = [&](const auto &x, int64_t i) {
    return dis_tab[x * k + assign[x]] - dis_tab[x * k + i];
  };

  compute_centroids(dim, k, nx, x_in, assign, hassign, centroids, avg_len);

  std::vector<std::deque<int64_t>> transfer_lists(k);
  float err = std::numeric_limits<float>::max();

  for (int64_t iter = 0; iter < niter; ++iter) {
#ifdef SSK_LOG
    std::cout << "Start " << iter << "th iteration" << std::endl;
#endif

    int64_t transfer_cnt = 0;
    ssk_compute_dist_tab(nx, x_in, dim, k, centroids, dis_tab);

    std::sort(xs, xs + nx, [&](const auto &x, const auto &y) {
      return delta_cur_best(x) > delta_cur_best(y);
    });

    for (int64_t i = 0; i < nx; ++i) {
      const auto x = xs[i];
      int64_t x_cluster = assign[x];
      std::sort(ks, ks + k, [&](const auto &a, const auto &b) {
        return gain(x, a) > gain(x, b);
      });

      for (int64_t j = 0; j < k; ++j) {
        if (j == assign[x])
          continue;

        float x_gain = gain(x, j);

        for (int64_t v = 0; v < transfer_lists[j].size(); ++v) {
          const int64_t candidate = transfer_lists[j][v];

          if (x_gain + gain(candidate, x_cluster) > 0) {
            std::swap(assign[x], assign[candidate]);
            x_cluster = assign[x];
            transfer_lists[j].erase(transfer_lists[j].begin() + v);
            transfer_cnt += 2;

            x_gain = 0;
            break;
          }
        }

        if (x_gain > 0 && (hassign[x_cluster] > min_target_size &&
                           hassign[j] < max_target_size)) {
          --hassign[x_cluster];
          ++hassign[j];
          assign[x] = j;
          x_cluster = j;
          ++transfer_cnt;
        }
      }

      if (assign[x] != ks[0] &&
          dis_tab[x * k + assign[x]] > dis_tab[x * k + ks[0]]) {
        transfer_lists[assign[x]].push_back(x);
      }
    }

    int64_t skip_cnt = 0;
    for (auto &l : transfer_lists) {
      skip_cnt += l.size();
      l.clear();
    }

    float cur_err = 0.0;
    for (auto i = 0; i < nx; ++i) {
      cur_err += dis_tab[i * k + assign[i]];
    }
#ifdef SSK_LOG
    std::cout << "Transfered " << transfer_cnt << ", skipped " << skip_cnt
              << " points." << std::endl;
    std::cout << "Current Error: " << cur_err << std::endl;
#endif

    if (fabs(cur_err - err) < err * 0.01) {
#ifdef SSK_LOG
      std::cout << "exit kmeans iteration after the " << iter
                << "th iteration, err = " << err << ", cur_err = " << cur_err
                << std::endl;
#endif
      break;
    }
    err = cur_err;

    if (transfer_cnt == 0) {
#ifdef SSK_LOG
      std::cout << "No tranfer occurs in the last iteration. Terminate."
                << std::endl;
#endif
      break;
    }

    compute_centroids(dim, k, nx, x_in, assign, hassign, centroids, avg_len);
  }

  ssk_print_cluster_size_stats(k, hassign);

  delete[] xs;
  delete[] ks;

  delete[] dis_tab;
  delete[] hassign;
}
