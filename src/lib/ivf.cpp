#include "lib/ivf.h"
#include <omp.h>

#include <algorithm>
#include <assert.h>
#include <cstdint>
#include <deque>
#include <memory>
#include <string.h>
#include <unistd.h>

#include "util/constants.h"
#include "util/distance.h"
#include "util/file_handler.h"
#include "util/utils_inline.h"

namespace {

// avg_len:
//    0: not to normalize
//    else: normalize
template <typename T>
void compute_centroids(int64_t dim, int64_t k, int64_t n, const T *x,
                       const int64_t *assign, int64_t *hassign,
                       float *centroids, float avg_len = 0.0) {
  memset(hassign, 0, sizeof(int64_t) * k);
  memset(centroids, 0, sizeof(float) * dim * k);

#pragma omp parallel
  {
    int64_t nt = omp_get_num_threads();
    int64_t rank = omp_get_thread_num();

    // this thread is taking care of centroids c0:c1
    int64_t c0 = (k * rank) / nt;
    int64_t c1 = (k * (rank + 1)) / nt;

    for (int64_t i = 0; i < n; i++) {
      int64_t ci = assign[i];
      if (ci >= c0 && ci < c1) {
        float *c = centroids + ci * dim;
        const T *xi = x + i * dim;
        for (int64_t j = 0; j < dim; j++) {
          c[j] += xi[j];
        }
        hassign[ci]++;
      }
    }
  }

#pragma omp parallel for
  for (int64_t ci = 0; ci < k; ci++) {
    if (hassign[ci] == 0) {
      continue;
    }

    float *c = centroids + ci * dim;
    if (avg_len != 0.0) {
      float len = avg_len / sqrt(IP<float, float, double>(c, c, dim));
      for (int64_t j = 0; j < dim; j++) {
        c[j] *= len;
      }
    } else {
      float norm = 1.0 / hassign[ci];
      for (int64_t j = 0; j < dim; j++) {
        c[j] *= norm;
      }
    }
  }
}

template <typename T>
void find_nearest_large_bucket(const T *x, const float *centroids, int64_t nx,
                               int64_t k, int64_t dim, int64_t *hassign,
                               int64_t *transform_table,
                               std::vector<int64_t> &assign) {
  std::vector<int64_t> new_assign(nx, -1);
#pragma omp parallel for
  for (int i = 0; i < nx; i++) {
    auto *x_i = x + i * dim;
    int min_id = 0;
    float min_dist;
    float dist;
    if (transform_table[assign[i]] != -1) {
      new_assign[i] = transform_table[assign[i]];
    } else {
      min_dist = L2sqr<const T, const float, float>(
          x_i, centroids + transform_table[min_id] * dim, dim);

      for (int j = min_id; j < k; j++) {

        dist = L2sqr<const T, const float, float>(
            x_i, centroids + transform_table[j] * dim, dim);
        if (dist < min_dist) {
          min_dist = dist;
          min_id = j;
        }
      }
      new_assign[i] = min_id;
    }
  }
  assign.assign(new_assign.begin(), new_assign.end());
}

template <typename T>
void merge_clusters(LevelType level, int64_t dim, int64_t nx, int64_t &k,
                    const T *x, std::vector<int64_t> &assign,
                    std::vector<float> &centroids, float avg_len = 0.0) {

  int64_t *hassign = new int64_t[k];
  memset(hassign, 0, sizeof(int64_t) * k);
  for (int i = 0; i < nx; i++) {
    hassign[assign[i]]++;
  }

  int64_t large_bucket_min_limit;
  int64_t small_bucket_max_limit;
  // strategies should be changed according to different scenarios
  if (level == LevelType::FIRST_LEVEL) {

    large_bucket_min_limit = MAX_SAME_SIZE_THRESHOLD;
    small_bucket_max_limit = MAX_SAME_SIZE_THRESHOLD;

  } else {

    large_bucket_min_limit = MAX_SAME_SIZE_THRESHOLD;
    small_bucket_max_limit = MIN_SAME_SIZE_THRESHOLD;
  }

  // find the new k2 and centroids:
  int64_t new_k = 0;
  int64_t large_bucket_num = 0;
  int64_t middle_bucket_num = 0;
  int64_t *transform_table = new int64_t[k]; // old k to new k
  for (int i = 0; i < k; i++) {
    if (hassign[i] >= large_bucket_min_limit) {
      transform_table[i] = large_bucket_num;
      large_bucket_num++;
    } else {
      transform_table[i] = -1;
    }
  }
  new_k += large_bucket_num;
  for (int i = 0; i < k; i++) {
    if (hassign[i] >= small_bucket_max_limit && transform_table[i] == -1) {
      transform_table[i] = new_k;
      new_k++;
      middle_bucket_num++;
    }
  }
  if (new_k == k) {

    return;
  }
  new_k = new_k != 0 ? new_k : 1; // add a bucket for all small bucket

  int64_t *new_hassign = new int64_t[new_k];
  float *new_centroids = new float[dim * new_k];
  for (int i = 0; i < k; i++) {
    if (transform_table[i] != -1) {
      memcpy(new_centroids + transform_table[i] * dim,
             centroids.data() + i * dim, dim * sizeof(float));
    }
  }
  if (large_bucket_num) {

    find_nearest_large_bucket<T>(x, new_centroids, nx, large_bucket_num, dim,
                                 hassign, transform_table, assign);

    compute_centroids<T>(dim, new_k, nx, x, assign.data(), new_hassign,
                         new_centroids, avg_len);

  } else if (middle_bucket_num) {
    find_nearest_large_bucket<T>(x, new_centroids, nx, middle_bucket_num, dim,
                                 hassign, transform_table, assign);

    compute_centroids<T>(dim, new_k, nx, x, assign.data(), new_hassign,
                         new_centroids, avg_len);
  } else {

    float *__restrict merge_centroid = new_centroids;
    int64_t merge_centroid_id = 0;
    memset(merge_centroid, 0, sizeof(float) * dim);
#pragma omp parallel for
    for (int i = 0; i < nx; i++) {
      auto *__restrict x_in = x + i * dim;
      if (transform_table[assign[i]] == -1) {
        for (int d = 0; d < dim; d++) {
          merge_centroid[d] += x_in[d];
        }
        assign[i] = merge_centroid_id;
      } else {
        assign[i] = transform_table[assign[i]];
      }
    }

    if (avg_len != 0.0) {
      float len = avg_len / sqrt(IP<float, float, double>(merge_centroid,
                                                          merge_centroid, dim));
      for (int64_t j = 0; j < dim; j++) {
        merge_centroid[j] *= len;
      }
    } else {
      float norm = 1.0 / nx;
      for (int64_t j = 0; j < dim; j++) {
        merge_centroid[j] *= norm;
      }
    }
  }

  // update meta :
  k = new_k;
  centroids.assign(new_centroids, new_centroids + k * dim);

  delete[] new_centroids;
  delete[] new_hassign;
  delete[] transform_table;
  delete[] hassign;

  return;
}
} // namespace
// Data type: T1, T2
// Distance type: R
// ID type int64_t
template <typename T1, typename T2, typename R>
void dynamic_assign(const T1 *x, // data base vector
                    const T2 *y, // centroids vector
                    int64_t dim, int64_t nx, int64_t ny, float weight,
                    int64_t *ids, R *val) {
  if (nx == 0 || ny == 0) {
    return;
  }
  int64_t *hassign = new int64_t[ny];
  float dist = 0.0;
  int64_t min = 0;
  float min_value = 0.0;
  memset(hassign, 0, sizeof(int64_t) * ny);

  for (int i = 0; i < nx; i++) {
    auto *__restrict x_in = x + i * dim;
    min_value =
        L2sqr<const T1, const T2, R>(x_in, y, dim) + weight * hassign[0];
    min = 0;
    for (int j = 1; j < ny; j++) {
      auto *__restrict y_in = y + j * dim;
      dist =
          L2sqr<const T1, const T2, R>(x_in, y_in, dim) + weight * hassign[j];
      if (dist < min_value) {
        min = j;
        min_value = dist;
      }
    }
    ids[i] = min;
    val[i] = min_value;
    hassign[min]++;
  }

  delete[] hassign;
}

// Data type: T1, T2
// Distance type: R
// ID type int64_t
template <typename T1, typename T2, typename R>
void elkan_L2_assign(const T1 *x, const T2 *y, int64_t dim, int64_t nx,
                     int64_t ny, int64_t *ids, R *val) {

  if (nx == 0 || ny == 0) {
    return;
  }

  const size_t bs_y = 1024;
  R *data = (R *)malloc((bs_y * (bs_y - 1) / 2) * sizeof(R));

  for (int64_t j0 = 0; j0 < ny; j0 += bs_y) {
    int64_t j1 = j0 + bs_y;
    if (j1 > ny)
      j1 = ny;

    auto Y = [&](int64_t i, int64_t j) -> R & {
      assert(i != j);
      i -= j0, j -= j0;
      return (i > j) ? data[j + i * (i - 1) / 2] : data[i + j * (j - 1) / 2];
    };

#pragma omp parallel
    {
      int nt = omp_get_num_threads();
      int rank = omp_get_thread_num();
      for (int64_t i = j0 + 1 + rank; i < j1; i += nt) {
        const T2 *y_i = y + i * dim;
        for (int64_t j = j0; j < i; j++) {
          const T2 *y_j = y + j * dim;
          Y(i, j) = L2sqr<const T2, const T2, R>(y_i, y_j, dim);
        }
      }
    }

#pragma omp parallel for
    for (int64_t i = 0; i < nx; i++) {
      const T1 *x_i = x + i * dim;

      int64_t ids_i = j0;
      R val_i = L2sqr<const T1, const T2, R>(x_i, y + j0 * dim, dim);
      R val_i_time_4 = val_i * 4;
      for (int64_t j = j0 + 1; j < j1; j++) {
        if (val_i_time_4 <= Y(ids_i, j)) {
          continue;
        }
        const T2 *y_j = y + j * dim;
        R disij = L2sqr<const T1, const T2, R>(x_i, y_j, dim / 2);
        if (disij >= val_i) {
          continue;
        }
        disij += L2sqr<const T1, const T2, R>(x_i + dim / 2, y_j + dim / 2,
                                              dim - dim / 2);
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

inline int64_t split_clusters(int64_t dim, int64_t k, int64_t n,
                              int64_t *hassign, float *centroids) {
  const double EPS = (1 / 1024.);
  /* Take care of void clusters */
  int64_t nsplit = 0;

  for (int64_t ci = 0; ci < k; ci++) {
    if (hassign[ci] == 0) { /* need to redefine a centroid */
      int64_t cj;
      for (cj = 0; 1; cj = (cj + 1) % k) {
        /* probability to pick this cluster for split */
        float p = (hassign[cj] - 1.0) / (float)(n - k);
        float r = bbann::rand_float();
        if (r < p) {
          break; /* found our cluster to be split */
        }
      }
      memcpy(centroids + ci * dim, centroids + cj * dim, sizeof(float) * dim);

      /* small symmetric pertubation */
      for (int64_t j = 0; j < dim; j++) {
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

template <typename T>
int64_t split_clusters_half(int64_t dim, int64_t k, int64_t n, const T *x_in,
                            int64_t *hassign, int64_t *assign, float *centroids,
                            float avg_len = 0.0) {
  /* Take care of void clusters */
  int64_t nsplit = 0;
  bool set_hassign = (hassign == nullptr);
  if (set_hassign) {
    hassign = new int64_t[k];
    memset(hassign, 0, sizeof(int64_t) * k);
    for (int i = 0; i < n; i++) {
      hassign[assign[i]]++;
    }
  }

  for (int64_t ci = 0; ci < k; ci++) {
    if (hassign[ci] == 0) { /* need to redefine a centroid */
      int64_t cj;
      for (cj = 0; 1; cj = (cj + 1) % k) {
        /* probability to pick this cluster for split */
        float p = (hassign[cj] - 1.0) / (float)(n - k);
        float r = bbann::rand_float();
        if (r < p) {
          break; /* found our cluster to be split */
        }
      }
      int64_t split_point = hassign[cj] / 2;
      memset(centroids + ci * dim, 0, sizeof(float) * dim);
      memset(centroids + cj * dim, 0, sizeof(float) * dim);
      hassign[ci] = hassign[cj] = 0;

      for (int64_t i = 0; i < n; i++) {
        if (assign[i] == cj) {
          if (hassign[ci] < split_point) {
            hassign[ci]++;
            assign[i] = ci;
            for (int64_t j = 0; j < dim; j++) {
              centroids[ci * dim + j] += x_in[i * dim + j];
            }
          } else {
            hassign[cj]++;
            for (int64_t j = 0; j < dim; j++) {
              centroids[cj * dim + j] += x_in[i * dim + j];
            }
          }
        }
      }

      float leni, lenj;
      if (avg_len != 0.0) {
        leni = avg_len / sqrt(IP<float, float, double>(
                             centroids + ci * dim, centroids + ci * dim, dim));
        lenj = avg_len / sqrt(IP<float, float, double>(
                             centroids + cj * dim, centroids + cj * dim, dim));
      } else {
        leni = 1.0 / hassign[ci];
        lenj = 1.0 / hassign[cj];
      }
      for (int64_t j = 0; j < dim; j++) {
        centroids[ci * dim + j] *= leni;
        centroids[cj * dim + j] *= lenj;
      }
      nsplit++;
    }
  }
  if (set_hassign) {
    delete[] hassign;
    hassign = nullptr;
  }

  /* The distance between centroids and x_in change in the assign function*/
  return nsplit;
}

template <typename DATAT>
void kmeanspp(const DATAT *pdata, const int64_t nb, const int64_t dim,
              const int64_t num_clusters, float *&centroids) {
  auto disf = L2sqr<const DATAT, float, float>;
  std::random_device rd;
  auto x = rd();
  std::mt19937 generator(x);
  std::uniform_int_distribution<uint64_t> distribution(0, nb - 1);

  int64_t init_id = distribution(generator);
  std::vector<float> dist(nb, std::numeric_limits<float>::max());
  for (int64_t i = 0; i < dim; i++)
    centroids[i] = (float)pdata[init_id * dim + i];

  for (int64_t i = 1; i < num_clusters; i++) {
    double sumdx = 0.0;
#pragma omp parallel for schedule(static, 4096) reduction(+ : sumdx)
    for (int64_t j = 0; j < nb; j++) {
      float dist_cj = disf(pdata + j * dim, centroids + (i - 1) * dim, dim);
      dist[j] = std::min(dist[j], dist_cj);
      sumdx += dist[j];
    }
    std::uniform_real_distribution<double> distridb(0, sumdx);
    auto prob = distridb(generator);
    for (int64_t j = 0; j < nb; j++) {
      if (prob <= 0) {
        for (int64_t k = 0; k < dim; k++)
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

// avg_len:
//    0: not to normalize
//    else: normalize

template <typename T>
void kmeans(int64_t nx, const T *x_in, int64_t dim, int64_t k, float *centroids,
            bool kmpp, float avg_len, int64_t niter, int64_t seed) {
  clock_t start, end;
  start = clock();
  if (k > 1000)
    nx = k * 40;
  // std::cout << "new nx = " << nx << std::endl;
  const int64_t max_points_per_centroid = 256;
  const int64_t min_points_per_centroid = 39;

  if (nx < k) {
    // printf("trained points is not enough %ld given %ld\n", k, nx);
    return;
  }

  if (nx == k) {
    for (int64_t i = 0; i < nx * dim; i++)
      centroids[i] = x_in[i];
    return;
  }

  // if (nx < k * min_points_per_centroid) {
  //   printf("Too little trained points need %ld given %ld\n",
  //          k * min_points_per_centroid, nx);
  // } else if (nx > k * max_points_per_centroid) {
  //   printf("Too many trained points need %ld given %ld\n",
  //          k * max_points_per_centroid, nx);
  // }

  std::unique_ptr<int64_t[]> hassign(new int64_t[k]);

  std::unique_ptr<int64_t[]> assign(new int64_t[nx]);
  std::unique_ptr<float[]> dis(new float[nx]);

  if (kmpp) {
    kmeanspp<T>(x_in, nx, dim, k, centroids);
  } else {
    bbann::util::rand_perm(assign.get(), nx, k, seed);
    for (int64_t i = 0; i < k; i++) {
      const T *x = x_in + assign[i] * dim;

      float *c = centroids + i * dim;

      for (int64_t d = 0; d < dim; d++) {
        c[d] = x[d];
      }
    }
  }

  float err = std::numeric_limits<float>::max();
  for (int64_t i = 0; i < niter; i++) {

    elkan_L2_assign<T, float, float>(x_in, centroids, dim, nx, k, assign.get(),
                                     dis.get());
    compute_centroids<T>(dim, k, nx, x_in, assign.get(), hassign.get(),
                         centroids, avg_len);

    // int64_t split = split_clusters(dim, k, nx, hassign.get(), centroids);
    int64_t split = split_clusters_half(dim, k, nx, x_in, hassign.get(),
                                        assign.get(), centroids, avg_len);
    if (split != 0) {
      // printf("split %ld\n", split);
    } else {
      float cur_err = 0.0;
      for (auto j = 0; j < nx; j++)
        cur_err += dis[j];

      if (fabs(cur_err - err) < err * 0.01) {
        // std::cout << "exit kmeans iteration after the " << i << "th
        // iteration, err = " << err << ", cur_err = " << cur_err << std::endl;
        break;
      }
      err = cur_err;
    }
  }

  int empty_cnt = 0;
  int mx, mn;
  mx = mn = hassign[0];
  for (auto i = 0; i < k; i++) {
    if (hassign[i] == 0)
      empty_cnt++;
    if (hassign[i] > mx)
      mx = hassign[i];
    if (hassign[i] < mn)
      mn = hassign[i];
    // std::cout<<hassign[i]<<std::endl;
  }
  end = clock();
  // std::cout << "after the kmeans with nx = " << nx << ", k = " << k << ", has "
  //           << empty_cnt << " empty clusters,"
  //           << " max cluster: " << mx << " min cluster: " << mn
  //           << " time spent " << ((double)end - start) / CLOCKS_PER_SEC * 1000
  //           << "ms" << std::endl;
}

template <typename T>
void non_recursive_multilevel_kmeans(
    uint32_t k1_id,          // the index of k1 round k-means
    int64_t cluster_size,    // num vectors in this cluster
    T *data,                 // buffer to place all vectors
    uint32_t *ids,           // buffer to place all ids
    int64_t round_offset,    // the offset of to clustering data in this round
    int64_t dim,             // the dimension of vector
    uint32_t threshold,      // determines when to stop recursive clustering
    const uint64_t blk_size, // general 4096, determines how many vectors can be
                             // placed in a block
    uint32_t
        &blk_num, // output variable, num block output in this round clustering
    IOWriter &data_writer,         // file writer 1: to output base vectors
    IOWriter &centroids_writer,    // file writer 2: to output centroid vectors
    IOWriter &centroids_id_writer, // file writer 3: to output centroid ids
    int64_t
        centroids_id_start_position, // the start position of all centroids id
    int level,         // n-th round recursive clustering, start with 0
    std::mutex &mutex, // mutex to protect write out centroids
    std::vector<ClusteringTask> &output_tasks, // output clustering tasks
    bool kmpp,                                 // k-means parameter
    float avg_len,                             // k-means parameter
    int64_t niter,                             // k-means parameter
    int64_t seed                               // k-means parameter
) {
  // move pointer to current round
  data = data + round_offset * dim;
  ids = ids + round_offset;

  float weight = 0;
  int64_t vector_size = sizeof(T) * dim;
  int64_t id_size = sizeof(uint32_t);

  // Step 0: set the num of cluster in this round clustering

  int64_t k2 = -1; // num cluster in this round clustering
  bool do_same_size_kmeans = (LevelType(level) >= LevelType ::BALANCE_LEVEL) ||
                             +(LevelType(level) == LevelType ::THIRTH_LEVEL &&
                               cluster_size >= MIN_SAME_SIZE_THRESHOLD &&
                               cluster_size <= MAX_SAME_SIZE_THRESHOLD);
  if (do_same_size_kmeans) {
    k2 = std::max((cluster_size + threshold - 1) / threshold, 1L);
  } else {
    k2 = int64_t(sqrt(cluster_size / threshold)) + 1;
    k2 = k2 < MAX_CLUSTER_K2 ? k2 : MAX_CLUSTER_K2;
  }
  assert(k2 != -1);
  // std::cout << "step 0: set k2: "
  //           << "[level " << level << "] "
  //           << "[cluster_size " << cluster_size << "] "
  //           << "[k2 " << k2 << "] "
  //           << "[do same size kmeans " << do_same_size_kmeans << "] "
  //           << std::endl;

  // Step 1: clustering

  std::vector<float> k2_centroids(k2 * dim, 0.0);
  std::vector<int64_t> cluster_id(cluster_size, -1);

  if (do_same_size_kmeans) {
    // use same size kmeans or graph partition
    k2 = std::max((cluster_size + threshold - 1) / threshold, 1L);
    same_size_kmeans<T>(cluster_size, data, dim, k2, k2_centroids.data(),
                        cluster_id.data(), kmpp, avg_len, niter, seed);
  } else {
    int64_t train_size = cluster_size;
    T *train_data = nullptr;
    if (cluster_size > k2 * K2_MAX_POINTS_PER_CENTROID) {
      train_size = k2 * K2_MAX_POINTS_PER_CENTROID;
      train_data = new T[train_size * dim];
      bbann::random_sampling_k2(data, cluster_size, dim, train_size, train_data,
                                seed);
    } else {
      train_data = data;
    }
    kmeans<T>(train_size, train_data, dim, k2, k2_centroids.data(), kmpp,
              avg_len, niter, seed);
    if (cluster_size > k2 * K2_MAX_POINTS_PER_CENTROID) {
      delete[] train_data;
    }

    // Dynamic balance constraint K-means:
    // balanced_kmeans<T>(cluster_size, data, dim, k2, k2_centroids, weight,
    // kmpp, avg_len, niter, seed);
    std::vector<float> dists(cluster_size, -1);
    if (weight != 0 && cluster_size <= KMEANS_THRESHOLD) {
      dynamic_assign<T, float, float>(data, k2_centroids.data(), dim,
                                      cluster_size, k2, weight,
                                      cluster_id.data(), dists.data());
    } else {
      elkan_L2_assign<T, float, float>(data, k2_centroids.data(), dim,
                                       cluster_size, k2, cluster_id.data(),
                                       dists.data());
    }

    // dists is useless, so delete first
    std::vector<float>().swap(dists);

    merge_clusters<T>((LevelType)level, dim, cluster_size, k2, data, cluster_id,
                      k2_centroids, avg_len);

    // split_clusters_half(dim, k2, cluster_size, data, nullptr,
    // cluster_id.data(), k2_centroids, avg_len);
  }
  // std::cout << "step 1: clustering: "
  //           << "cluster centroids be wrote into k2_centroids and cluster_id: "
  //           << std::endl;

  // Step 2: reorder data by cluster id

  std::vector<int64_t> bucket_pre_size(k2 + 1, 0);
  for (int i = 0; i < cluster_size; i++) {
    bucket_pre_size[cluster_id[i] + 1]++;
  }
  for (int i = 1; i <= k2; i++) {
    bucket_pre_size[i] += bucket_pre_size[i - 1];
  }

  // now, elems in bucket_pre_size is prefix sum

  // reorder thr data and ids by their cluster id
  T *x_temp = new T[cluster_size * dim];
  uint32_t *ids_temp = new uint32_t[cluster_size];
  int64_t offest;
  memcpy(x_temp, data, cluster_size * vector_size);
  memcpy(ids_temp, ids, cluster_size * id_size);
  for (int i = 0; i < cluster_size; i++) {
    offest = (bucket_pre_size[cluster_id[i]]++);
    ids[offest] = ids_temp[i];
    memcpy(data + offest * dim, x_temp + i * dim, vector_size);
  }
  delete[] x_temp;
  delete[] ids_temp;

  // std::cout << "step 2: reorder data by cluster id: " << std::endl;

  // Step 3: check all cluster, write out or generate new ClusteringTask

  int64_t bucket_size;
  int64_t bucket_offset;
  int entry_size = vector_size + id_size;

  char *data_blk_buf = new char[blk_size];
  for (int i = 0; i < k2; i++) {
    if (i == 0) {
      bucket_size = bucket_pre_size[i];
      bucket_offset = 0;
    } else {
      bucket_size = bucket_pre_size[i] - bucket_pre_size[i - 1];
      bucket_offset = bucket_pre_size[i - 1];
    }
    // std::cout<<"after kmeans : centroids i"<<i<<" has vectors
    // "<<(int)bucket_size<<std::endl;
    if (bucket_size <= threshold) {
      // write a blk to file
      // std::cout << bucket_size<<std::endl;
      memset(data_blk_buf, 0, blk_size);
      *reinterpret_cast<uint32_t *>(data_blk_buf) = bucket_size;
      char *beg_address = data_blk_buf + sizeof(uint32_t);

      for (int j = 0; j < bucket_size; j++) {
        memcpy(beg_address + j * entry_size, data + dim * (bucket_offset + j),
               vector_size);
        memcpy(beg_address + j * entry_size + vector_size,
               ids + bucket_offset + j, id_size);
      }

      // need a lock
      {
        std::lock_guard<std::mutex> lock(mutex);

        int64_t current_position = centroids_id_writer.get_position();
        assert(current_position != -1);
        // std::cout << "global_centroids_number: current_position: " <<
        // current_position << std::endl;

        // centroid id is uint32_t type
        int64_t blk_num =
            (current_position - centroids_id_start_position) / sizeof(uint32_t);

        // make sure type cast safety
        assert(blk_num >= 0);

        uint32_t global_id =
            bbann::util::gen_global_block_id(k1_id, (uint32_t)blk_num);

        // std::cout << "blk_size " << blk_size << std::endl;
        data_writer.write((char *)data_blk_buf, blk_size);
        // convert centroids to specified datatype
        if (sizeof(T) != sizeof(float)) {
          T* k2_centroids_T = new T[dim];
          for (int j = 0; j < dim; j++) {
              k2_centroids_T[j] = (T) k2_centroids[i * dim + j];
          }
          centroids_writer.write((char *) k2_centroids_T, sizeof(T) * dim);
          delete[] k2_centroids_T;
        } else {
          centroids_writer.write((char *) (k2_centroids.data() + i * dim), sizeof(float) * dim);
        }
        centroids_id_writer.write((char *)(&global_id), sizeof(uint32_t));
      }
    } else {
      output_tasks.emplace_back(
          ClusteringTask(round_offset + bucket_offset, bucket_size, level + 1));
    }
  }
  delete[] data_blk_buf;
  // std::cout << "step 3: write out and generate new ClusteringTask: "
  //           << "[output_tasks size " << output_tasks.size() << "]" << std::endl;
}

template <typename T>
void ssk_compute_dist_tab(int64_t nx, const T *x_in, int64_t dim, int64_t k,
                          const float *centroids, float *dis_tab) {
#pragma omp parallel for
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
                      float *centroids, int64_t *assign, bool kmpp,
                      float avg_len, int64_t niter, int64_t seed) {
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
    bbann::util::rand_perm(assign, nx, k, seed);
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

#ifdef SSK_LOG
  ssk_print_cluster_size_stats(k, hassign);
#endif

  delete[] xs;
  delete[] ks;

  delete[] dis_tab;
  delete[] hassign;
}

#define IVF(T)                                                                 \
  template void kmeans<T>(int64_t nx, const T *x_in, int64_t dim, int64_t k,   \
                          float *centroids, bool kmpp = false,                 \
                          float avg_len = 0.0, int64_t niter = 10,             \
                          int64_t seed = 1234);                                \
  template void non_recursive_multilevel_kmeans<T>(                            \
      uint32_t k1_id, int64_t cluster_size, T * data, uint32_t * ids,          \
      int64_t round_offset, int64_t dim, uint32_t threshold,                   \
      const uint64_t blk_size, uint32_t &blk_num, IOWriter &data_writer,       \
      IOWriter &centroids_writer, IOWriter &centroids_id_writer,               \
      int64_t centroids_id_start_position, int level, std::mutex &mutex,       \
      std::vector<ClusteringTask> &output_tasks, bool kmpp = false,            \
      float avg_len, int64_t niter, int64_t seed);                             \
  template void same_size_kmeans<T>(                                           \
      int64_t nx, const T *x_in, int64_t dim, int64_t k, float *centroids,     \
      int64_t *assign, bool kmpp = false, float avg_len, int64_t niter,        \
      int64_t seed);
#define IVF_T(T1,T2,R) \
template void elkan_L2_assign<T1, T2, R>(const T1 *x, const T2 *y, int64_t dim, \
                                         int64_t nx, int64_t ny, int64_t *ids, \
                                         R *val); 

IVF(uint8_t);
IVF(int8_t);
IVF(float);
IVF_T(uint8_t, float, uint32_t)
IVF_T(float, float, float)
IVF_T(int8_t, float, int32_t)
#undef IVF_T
#undef IVF
