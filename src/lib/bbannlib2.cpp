#include "lib/bbannlib2.h"
#include "ann_interface.h"
#include "util/TimeRecorder.h"
#include "util/file_handler.h"
#include "util/heap.h"
#include "util/utils_inline.h"
#include <iostream>
#include <libaio.h>
#include <map>
#include <omp.h>
#include <stdint.h>
#include <string>
namespace bbann {

template <typename T1, typename T2, typename R>
R L2sqr(T1 *a, T2 *b, size_t n) {
  size_t i = 0;
  R dis = 0, dif;
  switch (n & 7) {
  default:
    while (n > 7) {
      n -= 8;
      dif = (R)a[i] - (R)b[i];
      dis += dif * dif;
      i++;
    case 7:
      dif = (R)a[i] - (R)b[i];
      dis += dif * dif;
      i++;
    case 6:
      dif = (R)a[i] - (R)b[i];
      dis += dif * dif;
      i++;
    case 5:
      dif = (R)a[i] - (R)b[i];
      dis += dif * dif;
      i++;
    case 4:
      dif = (R)a[i] - (R)b[i];
      dis += dif * dif;
      i++;
    case 3:
      dif = (R)a[i] - (R)b[i];
      dis += dif * dif;
      i++;
    case 2:
      dif = (R)a[i] - (R)b[i];
      dis += dif * dif;
      i++;
    case 1:
      dif = (R)a[i] - (R)b[i];
      dis += dif * dif;
      i++;
    }
  }
  return dis;
}

template <typename T1, typename T2, typename R> R IP(T1 *a, T2 *b, size_t n) {
  size_t i = 0;
  R dis = 0;
  switch (n & 7) {
  default:
    while (n > 7) {
      n -= 8;
      dis += (R)a[i] * (R)b[i];
      i++;
    case 7:
      dis += (R)a[i] * (R)b[i];
      i++;
    case 6:
      dis += (R)a[i] * (R)b[i];
      i++;
    case 5:
      dis += (R)a[i] * (R)b[i];
      i++;
    case 4:
      dis += (R)a[i] * (R)b[i];
      i++;
    case 3:
      dis += (R)a[i] * (R)b[i];
      i++;
    case 2:
      dis += (R)a[i] * (R)b[i];
      i++;
    case 1:
      dis += (R)a[i] * (R)b[i];
      i++;
    }
  }
  return dis;
}

template <typename T1, typename T2, typename R>
std::function<R(const T1 *, const T2 *, int n)>
select_computer(MetricType metric_type) {
  switch (metric_type) {
  case MetricType::L2:
    return L2sqr<const T1, const T2, R>;
  case MetricType::IP:
    return IP<const T1, const T2, R>;
  }
  assert(false);
}

template <typename T>
void stat_length(T *x, int64_t n, int64_t dim, double &max_len, double &min_len,
                 double &avg_len) {
  double sum_len = 0;
  max_len = 0;
  min_len = std::numeric_limits<double>::max();

#pragma omp parallel for reduction(max:max_len) reduction(min:min_len) reduction(+:sum_len)
  for (int64_t i = 0; i < n; i++) {
    T *p = x + i * dim;
    double len = sqrt(IP<T, T, double>(p, p, dim));
    if (len > max_len)
      max_len = len;
    if (len < min_len)
      min_len = len;
    sum_len += len;
  }

  avg_len = sum_len / n;
}

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
}

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

template <typename T>
void reservoir_sampling(const std::string &data_file, const size_t sample_num,
                        T *sample_data) {
  assert(sample_data != nullptr);
  std::random_device rd;
  auto x = rd();
  std::mt19937 generator((unsigned)x);
  uint32_t nb, dim;
  size_t ntotal, ndims;
  IOReader reader(data_file);
  reader.read((char *)&nb, sizeof(uint32_t));
  reader.read((char *)&dim, sizeof(uint32_t));
  ntotal = nb;
  ndims = dim;
  std::unique_ptr<T[]> tmp_buf = std::make_unique<T[]>(ndims);
  for (size_t i = 0; i < sample_num; i++) {
    auto pi = sample_data + ndims * i;
    reader.read((char *)pi, ndims * sizeof(T));
  }
  for (size_t i = sample_num; i < ntotal; i++) {
    reader.read((char *)tmp_buf.get(), ndims * sizeof(T));
    std::uniform_int_distribution<size_t> distribution(0, i);
    size_t rand = (size_t)distribution(generator);
    if (rand < sample_num) {
      memcpy((char *)(sample_data + ndims * rand), tmp_buf.get(),
             ndims * sizeof(T));
    }
  }
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
        float r = util::rand_float();
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
void kmeans(int64_t nx, const T *x_in, int64_t dim, int64_t k, float *centroids,
            bool kmpp = false, float avg_len = 0.0, int64_t niter = 10,
            int64_t seed = 1234) {

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

  util::rand_perm(assign.get(), nx, k, seed);
  for (int64_t i = 0; i < k; i++) {
    const T *x = x_in + assign[i] * dim;

    float *c = centroids + i * dim;

    for (int64_t d = 0; d < dim; d++) {
      c[d] = x[d];
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
        // std::cout << "exit kmeans iteration after the " << i
        //           << "th iteration, err = " << err << ", cur_err = " <<
        //           cur_err
        //           << std::endl;
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
}

template <typename DATAT>
void train_cluster(const std::string &raw_data_bin_file,
                   const std::string &output_path, const int32_t K1,
                   float **centroids, double &avg_len) {
  TimeRecorder rc("train cluster");
  std::cout << "train_cluster parameters:" << std::endl;
  std::cout << " raw_data_bin_file: " << raw_data_bin_file
            << " output path: " << output_path << " K1: " << K1
            << " centroids: " << *centroids << std::endl;
  assert((*centroids) == nullptr);
  DATAT *sample_data = nullptr;
  uint32_t nb, dim;
  util::get_bin_metadata(raw_data_bin_file, nb, dim);
  int64_t sample_num = nb * consts::K1_SAMPLE_RATE;
  std::cout << "nb = " << nb << ", dim = " << dim
            << ", sample_num 4 K1: " << sample_num << std::endl;

  *centroids = new float[K1 * dim];
  sample_data = new DATAT[sample_num * dim];
  reservoir_sampling(raw_data_bin_file, sample_num, sample_data);
  rc.RecordSection("reservoir sample with sample rate: " +
                   std::to_string(consts::K1_SAMPLE_RATE) + " done");
  double mxl, mnl;
  int64_t stat_n = std::min(static_cast<int64_t>(1000000), sample_num);
  stat_length<DATAT>(sample_data, stat_n, dim, mxl, mnl, avg_len);
  rc.RecordSection("calculate " + std::to_string(stat_n) +
                   " vectors from sample_data done");
  std::cout << "max len: " << mxl << ", min len: " << mnl
            << ", average len: " << avg_len << std::endl;
  kmeans<DATAT>(sample_num, sample_data, dim, K1, *centroids, avg_len);
  rc.RecordSection("kmeans done");
  assert((*centroids) != nullptr);

  delete[] sample_data;
  rc.ElapseFromBegin("train cluster done.");
}

template <typename DATAT, typename DISTT>
void divide_raw_data(const BBAnnParameters para, const float *centroids) {
  TimeRecorder rc("divide raw data");
  std::cout << "divide_raw_data parameters:" << std::endl;
  std::cout << " raw_data_bin_file: " << para.dataFilePath
            << " output_path: " << para.indexPrefixPath
            << " centroids: " << centroids << " K1: " << para.K1 << std::endl;
  int K1 = para.K1;
  IOReader reader(para.dataFilePath);
  uint32_t nb, dim;
  reader.read((char *)&nb, sizeof(uint32_t));
  reader.read((char *)&dim, sizeof(uint32_t));
  uint32_t placeholder = 0, const_one = 1;
  std::vector<uint32_t> cluster_size(K1, 0);
  std::vector<std::ofstream> cluster_dat_writer(K1);
  std::vector<std::ofstream> cluster_ids_writer(K1);
  for (int i = 0; i < K1; i++) {
    cluster_dat_writer[i] = std::ofstream(
        getClusterRawDataFileName(para.indexPrefixPath, i), std::ios::binary);
    cluster_ids_writer[i] = std::ofstream(
        getClusterGlobalIdsFileName(para.indexPrefixPath, i), std::ios::binary);
    cluster_dat_writer[i].write((char *)&placeholder, sizeof(uint32_t));
    cluster_dat_writer[i].write((char *)&dim, sizeof(uint32_t));
    cluster_ids_writer[i].write((char *)&placeholder, sizeof(uint32_t));
    cluster_ids_writer[i].write((char *)&const_one, sizeof(uint32_t));
  }

  int64_t block_size = 1000000;
  assert(nb > 0);
  int64_t block_num = (nb - 1) / block_size + 1;
  std::vector<int64_t> cluster_id(block_size);
  std::vector<DISTT> dists(block_size);
  DATAT *block_buf = new DATAT[block_size * dim];
  for (int64_t i = 0; i < block_num; i++) {
    TimeRecorder rci("batch-" + std::to_string(i));
    int64_t sp = i * block_size;
    int64_t ep = std::min((int64_t)nb, sp + block_size);
    std::cout << "split the " << i << "th batch, start position = " << sp
              << ", end position = " << ep << std::endl;
    reader.read((char *)block_buf, (ep - sp) * dim * sizeof(DATAT));
    rci.RecordSection("read batch data done");
    elkan_L2_assign<const DATAT, const float, DISTT>(
        block_buf, centroids, dim, ep - sp, K1, cluster_id.data(),
        dists.data());
    rci.RecordSection("select file done");
    for (int64_t j = 0; j < ep - sp; j++) {
      int64_t cid = cluster_id[j];
      uint32_t uid = (uint32_t)(j + sp);
      cluster_dat_writer[cid].write((char *)(block_buf + j * dim),
                                    sizeof(DATAT) * dim);
      cluster_ids_writer[cid].write((char *)&uid, sizeof(uint32_t));
      cluster_size[cid]++;
    }
    rci.RecordSection("write done");
    rci.ElapseFromBegin("split batch " + std::to_string(i) + " done");
  }
  rc.RecordSection("split done");
  size_t sump = 0;
  std::cout << "split_raw_data done in ... seconds, show statistics:"
            << std::endl;
  for (int i = 0; i < K1; i++) {
    uint32_t cis = cluster_size[i];
    cluster_dat_writer[i].seekp(0);
    cluster_dat_writer[i].write((char *)&cis, sizeof(uint32_t));
    cluster_dat_writer[i].close();
    cluster_ids_writer[i].seekp(0);
    cluster_ids_writer[i].write((char *)&cis, sizeof(uint32_t));
    cluster_ids_writer[i].close();
    std::cout << "cluster-" << i << " has " << cis << " points." << std::endl;
    sump += cis;
  }
  rc.RecordSection("rewrite header done");
  std::cout << "total points num: " << sump << std::endl;

  delete[] block_buf;
  block_buf = nullptr;
  rc.ElapseFromBegin("split_raw_data totally done");
}

template <typename T>
void recursive_kmeans(uint32_t k1_id, uint32_t cluster_size, T *data,
                      uint32_t *ids, int64_t dim, uint32_t threshold,
                      const uint64_t blk_size, uint32_t &blk_num,
                      IOWriter &data_writer, IOWriter &centroids_writer,
                      IOWriter &centroids_id_writer, bool kmpp = false,
                      float avg_len = 0.0, int64_t niter = 10,
                      int64_t seed = 1234) {

  float weight = 0;
  int vector_size = sizeof(T) * dim;
  int id_size = sizeof(uint32_t);
  int k2;
  if (weight != 0 && cluster_size < consts::KMEANS_THRESHOLD) {
    k2 = int(sqrt(cluster_size / threshold)) + 1;
  } else {
    k2 = int(cluster_size / threshold) + 1;
  }

  k2 = k2 < consts::MAX_CLUSTER_K2 ? k2 : consts::MAX_CLUSTER_K2;
  float *k2_centroids = new float[k2 * dim];

  kmeans<T>(cluster_size, data, dim, k2, k2_centroids, kmpp, avg_len, niter,
            seed);
  // Dynamic balance constraint K-means:
  // balance_kmeans<T>(cluster_size, data, dim, k2, k2_centroids, weight, kmpp,
  // avg_len, niter, seed);
  std::vector<int64_t> cluster_id(cluster_size, -1);
  std::vector<float> dists(cluster_size, -1);
  std::vector<float> bucket_pre_size(k2 + 1, 0);
  if (weight != 0 && cluster_size <= consts::KMEANS_THRESHOLD) {
    dynamic_assign<T, float, float>(data, k2_centroids, dim, cluster_size, k2,
                                    weight, cluster_id.data(), dists.data());
  } else {
    elkan_L2_assign<T, float, float>(data, k2_centroids, dim, cluster_size, k2,
                                     cluster_id.data(), dists.data());
  }

  split_clusters_half(dim, k2, cluster_size, data, nullptr, cluster_id.data(),
                      k2_centroids, avg_len);

  // dists is useless, so delete first
  std::vector<float>().swap(dists);

  for (int i = 0; i < cluster_size; i++) {
    bucket_pre_size[cluster_id[i] + 1]++;
  }
  for (int i = 1; i <= k2; i++) {
    bucket_pre_size[i] += bucket_pre_size[i - 1];
  }

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

  int64_t bucket_size;
  int64_t bucket_offest;
  int entry_size = vector_size + id_size;
  uint32_t global_id;

  char *data_blk_buf = new char[blk_size];
  for (int i = 0; i < k2; i++) {
    if (i == 0) {
      bucket_size = bucket_pre_size[i];
      bucket_offest = 0;
    } else {
      bucket_size = bucket_pre_size[i] - bucket_pre_size[i - 1];
      bucket_offest = bucket_pre_size[i - 1];
    }
    // std::cout<<"after kmeans : centroids i"<<i<<" has vectors
    // "<<(int)bucket_size<<std::endl;
    if (bucket_size <= threshold) {
      // write a blk to file
      memset(data_blk_buf, 0, blk_size);
      *reinterpret_cast<uint32_t *>(data_blk_buf) = bucket_size;
      char *beg_address = data_blk_buf + sizeof(uint32_t);

      for (int j = 0; j < bucket_size; j++) {
        memcpy(beg_address + j * entry_size, data + dim * (bucket_offest + j),
               vector_size);
        memcpy(beg_address + j * entry_size + vector_size,
               ids + bucket_offest + j, id_size);
      }
      global_id = util::gen_global_block_id(k1_id, blk_num);

      data_writer.write((char *)data_blk_buf, blk_size);
      centroids_writer.write((char *)(k2_centroids + i * dim),
                             sizeof(float) * dim);
      centroids_id_writer.write((char *)(&global_id), sizeof(uint32_t));
      blk_num++;

    } else {
      recursive_kmeans(k1_id, (uint32_t)bucket_size, data + bucket_offest * dim,
                       ids + bucket_offest, dim, threshold, blk_size, blk_num,
                       data_writer, centroids_writer, centroids_id_writer, kmpp,
                       avg_len, niter, seed);
    }
  }
  delete[] data_blk_buf;
  delete[] k2_centroids;
}

template <typename DATAT, typename DISTT>
void hierarchical_clusters(const BBAnnParameters para, const double avg_len) {
  TimeRecorder rc("hierarchical clusters");
  std::cout << "hierarchical clusters parameters:" << std::endl;
  std::cout << " output_path: " << para.indexPrefixPath
            << " vector avg length: " << avg_len
            << " block size: " << para.blockSize << std::endl;
  int K1 = para.K1;
  uint32_t cluster_size, cluster_dim, ids_size, ids_dim;
  uint32_t entry_num;

  std::string bucket_centroids_file =
      para.indexPrefixPath + "bucket-centroids.bin";
  std::string bucket_centroids_id_file =
      para.indexPrefixPath + "cluster-combine_ids.bin";
  uint32_t placeholder = 1;
  uint32_t global_centroids_number = 0;
  uint32_t centroids_dim = 0;

  {
    IOWriter centroids_writer(bucket_centroids_file);
    IOWriter centroids_id_writer(bucket_centroids_id_file);
    centroids_writer.write((char *)&placeholder, sizeof(uint32_t));
    centroids_writer.write((char *)&placeholder, sizeof(uint32_t));
    centroids_id_writer.write((char *)&placeholder, sizeof(uint32_t));
    centroids_id_writer.write((char *)&placeholder, sizeof(uint32_t));

    for (uint32_t i = 0; i < K1; i++) {
      TimeRecorder rci("train-cluster-" + std::to_string(i));
      IOReader data_reader(getClusterRawDataFileName(para.indexPrefixPath, i));
      IOReader ids_reader(getClusterGlobalIdsFileName(para.indexPrefixPath, i));

      data_reader.read((char *)&cluster_size, sizeof(uint32_t));
      data_reader.read((char *)&cluster_dim, sizeof(uint32_t));
      ids_reader.read((char *)&ids_size, sizeof(uint32_t));
      ids_reader.read((char *)&ids_dim, sizeof(uint32_t));
      entry_num = (para.blockSize - sizeof(uint32_t)) /
                  (cluster_dim * sizeof(DATAT) + ids_dim * sizeof(uint32_t));
      centroids_dim = cluster_dim;
      assert(cluster_size == ids_size);
      assert(ids_dim == 1);
      assert(entry_num > 0);

      DATAT *datai = new DATAT[cluster_size * cluster_dim];
      uint32_t *idi = new uint32_t[ids_size * ids_dim];
      uint32_t blk_num = 0;
      data_reader.read((char *)datai,
                       cluster_size * cluster_dim * sizeof(DATAT));
      ids_reader.read((char *)idi, ids_size * ids_dim * sizeof(uint32_t));

      IOWriter data_writer(getClusterRawDataFileName(para.indexPrefixPath, i),
                           ioreader::MEGABYTE * 100);
      recursive_kmeans<DATAT>(i, cluster_size, datai, idi, cluster_dim,
                              entry_num, para.blockSize, blk_num, data_writer,
                              centroids_writer, centroids_id_writer, false,
                              avg_len);

      global_centroids_number += blk_num;

      delete[] datai;
      delete[] idi;
    }
  }

  uint32_t centroids_id_dim = 1;
  std::ofstream centroids_meta_writer(bucket_centroids_file,
                                      std::ios::binary | std::ios::in);
  std::ofstream centroids_ids_meta_writer(bucket_centroids_id_file,
                                          std::ios::binary | std::ios::in);
  centroids_meta_writer.seekp(0);
  centroids_meta_writer.write((char *)&global_centroids_number,
                              sizeof(uint32_t));
  centroids_meta_writer.write((char *)&centroids_dim, sizeof(uint32_t));
  centroids_ids_meta_writer.seekp(0);
  centroids_ids_meta_writer.write((char *)&global_centroids_number,
                                  sizeof(uint32_t));
  centroids_ids_meta_writer.write((char *)&centroids_id_dim, sizeof(uint32_t));
  centroids_meta_writer.close();
  centroids_ids_meta_writer.close();

  std::cout << "hierarchical_clusters generate " << global_centroids_number
            << " centroids" << std::endl;
  return;
}

void build_graph(const std::string &index_path, const int hnswM,
                 const int hnswefC, MetricType metric_type) {
  TimeRecorder rc("create_graph_index");
  std::cout << "build hnsw parameters:" << std::endl;
  std::cout << " index_path: " << index_path << " hnsw.M: " << hnswM
            << " hnsw.efConstruction: " << hnswefC
            << " metric_type: " << (int)metric_type << std::endl;

  float *pdata = nullptr;
  uint32_t *pids = nullptr;
  uint32_t npts, ndim, nids, nidsdim;

  util::read_bin_file<float>(index_path + "bucket-centroids.bin", pdata, npts,
                             ndim);
  rc.RecordSection("load centroids of buckets done");
  std::cout << "there are " << npts << " of dimension " << ndim
            << " points of hnsw" << std::endl;
  assert(pdata != nullptr);
  hnswlib::SpaceInterface<float> *space;
  if (MetricType::L2 == metric_type) {
    space = new hnswlib::L2Space(ndim);
  } else if (MetricType::IP == metric_type) {
    space = new hnswlib::InnerProductSpace(ndim);
  } else {
    std::cout << "invalid metric_type = " << (int)metric_type << std::endl;
    return;
  }
  util::read_bin_file<uint32_t>(index_path + "cluster-combine_ids.bin", pids,
                                nids, nidsdim);
  rc.RecordSection("load combine ids of buckets done");
  std::cout << "there are " << nids << " of dimension " << nidsdim
            << " combine ids of hnsw" << std::endl;
  assert(pids != nullptr);
  assert(npts == nids);
  assert(nidsdim == 1);

  auto index_hnsw = std::make_shared<hnswlib::HierarchicalNSW<float>>(
      space, npts, hnswM, hnswefC);
  index_hnsw->addPoint(pdata, pids[0]);
#pragma omp parallel for
  for (int64_t i = 1; i < npts; i++) {
    index_hnsw->addPoint(pdata + i * ndim, pids[i]);
  }
  std::cout << "hnsw totally add " << npts << " points" << std::endl;
  rc.RecordSection("create index hnsw done");
  index_hnsw->saveIndex(index_path + "hnsw-index.bin");
  rc.RecordSection("hnsw save index done");
  delete[] pdata;
  pdata = nullptr;
  delete[] pids;
  pids = nullptr;
  rc.ElapseFromBegin("create index hnsw totally done");
}

void gather_buckets_stats(const ::std::string index_path, const int K1,
                          const uint64_t block_size) {
  char *buf = new char[block_size];
  uint64_t stats[3] = {0, 0,
                       std::numeric_limits<uint64_t>::max()}; // avg, max, min
  uint64_t bucket_cnt = 0;

  for (int i = 0; i < K1; ++i) {
    uint64_t file_size = util::fsize(getClusterRawDataFileName(index_path, i));
    auto fh = std::ifstream(getClusterRawDataFileName(index_path, i),
                            std::ios::binary);
    assert(!fh.fail());

    for (uint32_t j = 0; j * block_size < file_size; ++j) {
      fh.seekg(j * block_size);
      fh.read(buf, block_size);
      const uint64_t entry_num =
          static_cast<const uint64_t>(*reinterpret_cast<uint32_t *>(buf));

      stats[0] += entry_num;
      stats[1] = std::max(stats[1], entry_num);
      stats[2] = std::min(stats[2], entry_num);
      ++bucket_cnt;
    }
  }

  uint32_t cen_n, cen_dim;
  util::get_bin_metadata(index_path + "bucket-centroids.bin", cen_n, cen_dim);
  assert(cen_n == bucket_cnt);

  std::cout << "Total number of buckets: " << bucket_cnt << std::endl;
  std::cout << "#vectors in bucket avg: " << stats[0] * 1.0f / bucket_cnt
            << " max: " << stats[1] << " min: " << stats[2] << std::endl;

  delete[] buf;
}

template <typename dataT>
bool BBAnnIndex2<dataT>::LoadIndex(std::string &indexPathPrefix) {
  indexPrefix_ = indexPathPrefix;
  std::cout << "Loading: " << indexPrefix_;
  uint32_t bucket_num, dim;
  util::get_bin_metadata(getBucketCentroidsFileName(), bucket_num, dim);

  hnswlib::SpaceInterface<float> *space = nullptr;
  if (MetricType::L2 == metric_) {
    space = new hnswlib::L2Space(dim);
  } else if (MetricType::IP == metric_) {
    space = new hnswlib::InnerProductSpace(dim);
  } else {
    return false;
  }
  // load hnsw
  index_hnsw_ = std::make_shared<hnswlib::HierarchicalNSW<float>>(
      space, getHnswIndexFileName());
  indexPrefix_ = indexPathPrefix;
  return true;
}

template <typename DATAT, typename DISTT>
void search_bbann_queryonly(
    std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
    const BBAnnParameters para, const int topk, const DATAT *pquery,
    uint32_t *answer_ids, DISTT *answer_dists, uint32_t num_query,
    uint32_t dim) {
  TimeRecorder rc("search bigann");

  std::cout << "query numbers: " << num_query << " query dims: " << dim
            << std::endl;

  uint32_t *bucket_labels =
      new uint32_t[(int64_t)num_query * para.nProbe]; // 400K * nprobe

  // Search Graph====================================
  std::cout << "search graph parameters:" << std::endl;
  std::cout << " index_hnsw: " << index_hnsw << " num_query: " << num_query
            << " query dims: " << dim << " nprobe: " << para.nProbe
            << " refine_nprobe: " << para.efSearch
            << " pquery: " << static_cast<const void *>(pquery)
            << " bucket_labels: " << static_cast<void *>(bucket_labels)
            << std::endl;
  index_hnsw->setEf(para.efSearch);
#pragma omp parallel for
  for (int64_t i = 0; i < num_query; i++) {
    // auto queryi = pquery + i * dim;
    // todo: hnsw need to support query data is not float
    float *queryi = new float[dim];
    for (int j = 0; j < dim; j++)
      queryi[j] = (float)(*(pquery + i * dim + j));
    auto reti = index_hnsw->searchKnn(queryi, para.nProbe);
    auto p_labeli = bucket_labels + i * para.nProbe;
    while (!reti.empty()) {
      *p_labeli++ = reti.top().second;
      reti.pop();
    }
    delete[] queryi;
  }
  rc.ElapseFromBegin("search+graph+done.");
  rc.RecordSection("search buckets done.");

  uint32_t cid, bid;
  uint32_t gid;
  const uint32_t vec_size = sizeof(DATAT) * dim;
  const uint32_t entry_size = vec_size + sizeof(uint32_t);
  DATAT *vec;
  std::function<void(size_t, DISTT *, uint32_t *)> heap_heapify_func;
  std::function<bool(DISTT, DISTT)> cmp_func;
  std::function<void(size_t, DISTT *, uint32_t *, DISTT, uint32_t)>
      heap_swap_top_func;
  if (para.metric == MetricType::L2) {
    heap_heapify_func = heap_heapify<CMax<DISTT, uint32_t>>;
    cmp_func = CMax<DISTT, uint32_t>::cmp;
    heap_swap_top_func = heap_swap_top<CMax<DISTT, uint32_t>>;
  } else if (para.metric == MetricType::IP) {
    heap_heapify_func = heap_heapify<CMin<DISTT, uint32_t>>;
    cmp_func = CMin<DISTT, uint32_t>::cmp;
    heap_swap_top_func = heap_swap_top<CMin<DISTT, uint32_t>>;
  }

  // init answer heap
#pragma omp parallel for schedule(static, 128)
  for (int i = 0; i < num_query; i++) {
    auto ans_disi = answer_dists + topk * i;
    auto ans_idsi = answer_ids + topk * i;
    heap_heapify_func(topk, ans_disi, ans_idsi);
  }
  rc.RecordSection("heapify answers heaps");

  char *buf = new char[para.blockSize];
  auto dis_computer = select_computer<DATAT, DATAT, DISTT>(para.metric);

  /* flat */
  for (int64_t i = 0; i < num_query; ++i) {
    const auto ii = i * para.nProbe;
    const DATAT *q_idx = pquery + i * dim;

    for (int64_t j = 0; j < para.nProbe; ++j) {
      util::parse_global_block_id(bucket_labels[ii + j], cid, bid);
      auto fh =
          std::ifstream(getClusterRawDataFileName(para.indexPrefixPath, cid),
                        std::ios::binary);
      assert(!fh.fail());

      fh.seekg(bid * para.blockSize);
      fh.read(buf, para.blockSize);

      const uint32_t entry_num = *reinterpret_cast<uint32_t *>(buf);
      char *buf_begin = buf + sizeof(uint32_t);

      for (uint32_t k = 0; k < entry_num; ++k) {
        char *entry_begin = buf_begin + entry_size * k;
        vec = reinterpret_cast<DATAT *>(entry_begin);
        auto dis = dis_computer(vec, q_idx, dim);
        if (cmp_func(answer_dists[topk * i], dis)) {
          heap_swap_top_func(
              topk, answer_dists + topk * i, answer_ids + topk * i, dis,
              *reinterpret_cast<uint32_t *>(entry_begin + vec_size));
        }
      }
    }
  }

  delete[] buf;
  delete[] bucket_labels;
}

template <typename dataT>
void BBAnnIndex2<dataT>::BatchSearchCpp(const dataT *pquery, uint64_t dim,
                                        uint64_t numQuery, uint64_t knn,
                                        const BBAnnParameters para,
                                        uint32_t *answer_ids,
                                        distanceT *answer_dists) {
  std::cout << "Query: " << std::endl;

  search_bbann_queryonly<dataT, distanceT>(
      index_hnsw_, para, knn, pquery, answer_ids, answer_dists, numQuery, dim);
}

template <typename dataT>
void BBAnnIndex2<dataT>::BuildIndexImpl(const BBAnnParameters para) {
  auto index = std::make_unique<BBAnnIndex2<dataT>>(para.metric);
  index->BuildWithParameter(para);
}

template <typename dataT>
void BBAnnIndex2<dataT>::BuildWithParameter(const BBAnnParameters para) {
  std::cout << "Build start+ " << std::endl;
  TimeRecorder rc("build bigann");
  using distanceT = typename TypeWrapper<dataT>::distanceT;
  dataFilePath_ = para.dataFilePath;
  indexPrefix_ = para.indexPrefixPath;
  std::cout << "build bigann parameters:" << std::endl;
  std::cout << " raw_data_bin_file: " << dataFilePath_
            << " output_path: " << indexPrefix_ << " hnsw.M: " << para.hnswM
            << " hnsw.efConstruction: " << para.hnswefC << " K1: " << para.K1
            << std::endl;

  float *centroids = nullptr;
  double avg_len;
  // sampling and do K1-means to get the first round centroids
  train_cluster<dataT>(dataFilePath_, indexPrefix_, para.K1, &centroids,
                       avg_len);
  assert(centroids != nullptr);
  rc.RecordSection("train cluster to get " + std::to_string(para.K1) +
                   " centroids done.");

  std::function<void(const BBAnnParameters, const double)>
      hierarchical_clusters_func;
  std::function<void(const BBAnnParameters, const float *)>
      divide_raw_data_func;

  switch (para.metric) {
  case MetricType::L2: {
    divide_raw_data_func = divide_raw_data<dataT, distanceT>;
    hierarchical_clusters_func = hierarchical_clusters<dataT, distanceT>;
    break;
  }
  case MetricType::IP: {
    divide_raw_data_func = divide_raw_data<dataT, distanceT>;
    hierarchical_clusters_func = hierarchical_clusters<dataT, distanceT>;
    break;
  }
  default: {
    std::cerr << "Error" << std::endl;
    return;
  }
  }

  divide_raw_data_func(para, centroids);
  rc.RecordSection("divide raw data into " + std::to_string(para.K1) +
                   " clusters done");

  hierarchical_clusters_func(para, avg_len);
  rc.RecordSection("conquer each cluster into buckets done");

  build_graph(indexPrefix_, para.hnswM, para.hnswefC, para.metric);
  rc.RecordSection("build hnsw done.");

  gather_buckets_stats(indexPrefix_, para.K1, para.blockSize);
  rc.RecordSection("gather statistics done");

  delete[] centroids;
  rc.ElapseFromBegin("build bigann totally done.");
}

template <typename dataT>
std::tuple<std::vector<uint32_t>, std::vector<float>, std::vector<uint64_t>>
BBAnnIndex2<dataT>::RangeSearchCpp(const dataT *pquery, uint64_t dim,
                                   uint64_t numQuery, double radius,
                                   const BBAnnParameters para) {
  TimeRecorder rc("range search bbann");

  std::cout << "range search bigann parameters:" << std::endl;
  std::cout << " index_path: " << para.indexPrefixPath
            << " hnsw_ef: " << para.hnswefC << " radius: " << radius
            << " K1: " << para.K1 << std::endl;

  std::cout << "query numbers: " << numQuery << " query dims: " << dim
            << std::endl;
  /*
  for (int i = 0 ; i < numQuery; i++) {
    std::cout << "query " << i <<": ";
    for (int j = 0; j < dim; j++) {
      std::cout << pquery[i*dim + j] << " ";
    }
    std::cout << std::endl;
  }
   */

  std::vector<float> query_float;
  query_float.resize(numQuery * dim);
#pragma omp parallel for
  for (int64_t i = 0; i < numQuery * dim; i++) {
    query_float[i] = (float)pquery[i];
  }
  std::cout << " prepared query_float" << std::endl;
  // std::vector<uint32_t> *bucket_labels = new std::vector<uint32_t>[numQuery];
  std::vector<std::pair<uint32_t, uint32_t>> qid_bucketLabel;

  std::map<int, int> bucket_hit_cnt, hit_cnt_cnt, return_cnt;
  index_hnsw_->setEf(para.efSearch);
  // -- a function that conducts queries[a..b] and returns a list of <bucketid,
  // queryid> pairs; note: 1 bucketid may map to multiple queryid.
  auto run_hnsw_search = [&, this](int l,
                                   int r) -> std::vector<std::pair<int, int>> {
    std::vector<std::pair<int, int>> ret;
    for (int i = l; i < r; i++) {
      float *queryi = &query_float[i * dim];
      const auto reti = index_hnsw_->searchRange(queryi, para.nProbe, radius);
      for (auto const &[dist, bucket_label] : reti) {
        ret.emplace_back(std::make_pair(bucket_label, i));
      }
    }
    return ret;
  };
  int nparts_hnsw = 128;
  std::vector<std::pair<int, int>> bucketToQuery;
// #pragma omp parallel for
  for (int partID = 0; partID < nparts_hnsw; partID++) {
    int low = partID * numQuery / nparts_hnsw;
    int high = (partID + 1) * numQuery / nparts_hnsw;
    auto part = run_hnsw_search(low, high);
    bucketToQuery.insert(bucketToQuery.end(), part.begin(), part.end());
  }
  rc.RecordSection(" query hnsw done");
  sort(bucketToQuery.begin(), bucketToQuery.end());
  rc.RecordSection("sort query results done");

  const uint32_t vec_size = sizeof(dataT) * dim;
  const uint32_t entry_size = vec_size + sizeof(uint32_t);
  AIOBucketReader reader(para.indexPrefixPath, para.aio_EventsPerBatch);
  // -- a function that reads the file for bucketid/queryid in
  // bucketToQuery[a..b]
  //
  auto run_bucket_scan =
      [&, this, para, pquery](int l, int r) -> std::list<qidIdDistTupleType> {
    /* return a list of tuple <queryid, id, dist>:*/

    // auto cachereader =
    //    std::make_unique<CachedBucketReader>(para.indexPrefixPath);
    // std::vector<char> buf_v(para.blockSize);
    // char *buf = &buf_v[0];

    std::list<qidIdDistTupleType> ret;
    auto dis_computer = select_computer<dataT, dataT, distanceT>(para.metric);
    // auto reader = std::make_unique<CachedBucketReader>(para.indexPrefixPath);
    std::vector<uint32_t> bucketIds;
    bucketIds.reserve(r - l);
    for (int i = l; i < r; i++) {
      bucketIds.emplace_back(bucketToQuery[i].first);
    }
    // std::cout << std::endl;
    void *big_read_buf;
    if (posix_memalign(&big_read_buf, 512, para.blockSize * (r - l)) != 0) {
      std::cerr << " err allocating  buf" << std::endl;
      exit(-1);
    }
    auto resIds = reader.ReadToBuf(bucketIds, para.blockSize, big_read_buf);
    /*
    std::cout << "res size" << res.first.size() << std::endl;
    std::cout << "id size " << res.second.size() << std::endl;
    for (const auto id : res.second) {
      std::cout << id << " ";
    }
    std::cout << std::endl;
    std::cout << " expected id size" << r - l << std::endl;*/

    for (int i = l; i < r; i++) {
      const auto qid = bucketToQuery[i].second;
      const dataT *q_idx = pquery + qid * dim;
      // std::cout << "trying access" << res.second[i - l] << std::endl;
      // char *buf = &res.first[res.second[i - l] * para.blockSize];
      char *buf = (char *)big_read_buf + resIds[i - l] * para.blockSize;
      const uint32_t entry_num = *reinterpret_cast<uint32_t *>(buf);
      char *data_begin = buf + sizeof(uint32_t);

      for (uint32_t k = 0; k < entry_num; ++k) {
        char *entry_begin = data_begin + entry_size * k;
        auto dis =
            dis_computer(reinterpret_cast<dataT *>(entry_begin), q_idx, dim);
        if (dis < radius) {
          const uint32_t id =
              *reinterpret_cast<uint32_t *>(entry_begin + vec_size);
          ret.push_back(std::make_tuple(qid, id, dis));
        }
      }
    }
    std::cout << "Finished query number:" << r - l << std::endl;
    free(big_read_buf);
    std::cout << " relleased big_read_buf" << std::endl;
    return ret;
  };
  // execute queries with "index mod (1<<shift) == thid".
  auto run_bucket_scan_shift =
      [&, this, para, pquery](int thid,
                              int shift) -> std::list<qidIdDistTupleType> {
  /* return a list of tuple <queryid, id, dist>:*/
#define getIndex(i, shift, thid) ((i << shift) + thid)
    std::list<qidIdDistTupleType> ret;
    auto dis_computer = select_computer<dataT, dataT, distanceT>(para.metric);
    // auto reader = std::make_unique<CachedBucketReader>(para.indexPrefixPath);
    std::vector<uint32_t> bucketIds;
    int totalQuerySize = bucketToQuery.size();
    int thisQuerySize = totalQuerySize >> shift;
    bucketIds.reserve(thisQuerySize);
    for (int i = 0; i < thisQuerySize; i++) {
      bucketIds.emplace_back(bucketToQuery[getIndex(i, shift, thid)].first);
    }
    // std::cout << std::endl;
    void *big_read_buf;
    if (posix_memalign(&big_read_buf, 512, para.blockSize * thisQuerySize) !=
        0) {
      std::cerr << " err allocating  buf" << std::endl;
      exit(-1);
    }
    auto resIds = reader.ReadToBuf(bucketIds, para.blockSize, big_read_buf);

    for (int i = 0; i < thisQuerySize; i++) {
      const auto qid = bucketToQuery[getIndex(i, shift, thid)].second;
      const dataT *q_idx = pquery + qid * dim;
      char *buf = (char *)big_read_buf + resIds[i] * para.blockSize;
      const uint32_t entry_num = *reinterpret_cast<uint32_t *>(buf);
      char *data_begin = buf + sizeof(uint32_t);

      for (uint32_t k = 0; k < entry_num; ++k) {
        char *entry_begin = data_begin + entry_size * k;
        auto dis =
            dis_computer(reinterpret_cast<dataT *>(entry_begin), q_idx, dim);
        if (dis < radius) {
          const uint32_t id =
              *reinterpret_cast<uint32_t *>(entry_begin + vec_size);
          ret.push_back(std::make_tuple(qid, id, dis));
        }
      }
    }
    std::cout << "Finished query number:" << thisQuerySize << std::endl;
    free(big_read_buf);
    std::cout << " relleased big_read_buf" << std::endl;
    return ret;
  };

  std::cout << "Need to access bucket data for " << bucketToQuery.size()
            << " times, " << std::endl;
  uint32_t totQuery = 0;
  uint32_t totReturn = 0;
  int nparts_block = 64;
  std::vector<qidIdDistTupleType> ans_list;
#pragma omp parallel for
  for (uint64_t partID = 0; partID < nparts_block; partID++) {
    uint32_t low = partID * bucketToQuery.size() / nparts_block;
    uint32_t high = (partID + 1) * bucketToQuery.size() / nparts_block;
    totQuery += (high - low);
    auto qid_id_dist = run_bucket_scan(low, high);
    // auto qid_id_dist = run_bucket_scan_shift(partID, 6);
    totReturn += qid_id_dist.size();
    std::move(qid_id_dist.begin(), qid_id_dist.end(),
              std::back_inserter(ans_list));
    std::cout << "finished " << totQuery << "queries, returned " << totReturn
              << " answers" << std::endl;
  }
  rc.RecordSection("scan blocks done");
  sort(ans_list.begin(), ans_list.end());
  std::vector<uint32_t> ids;
  std::vector<float> dists;
  std::vector<uint64_t> lims(numQuery + 1);
  int lims_index = 0;
  for (auto const &[qid, ansid, dist] : ans_list) {
    // std::cout << qid << " " << ansid << " " << dist << std::endl;
    while (lims_index < qid) {
      lims[lims_index] = ids.size();
      // std::cout << "lims" << lims_index << "!" << lims[lims_index] <<
      // std::endl;
      lims_index++;
    }
    if (lims[qid] == 0 && qid == lims_index) {
      lims[qid] = ids.size();
      // std::cout << "lims" << qid << " " << lims[qid] << std::endl;
      lims_index++;
    }
    ids.push_back(ansid);
    dists.push_back(dist);
    // std::cout << "ansid " << ansid << " " << ids[lims[qid]] << std::endl;
  }
  while (lims_index <= numQuery) {
    lims[lims_index] = ids.size();
    lims_index++;
  }

  rc.RecordSection("format answer done");

  rc.ElapseFromBegin("range search bbann totally done");
  return std::make_tuple(ids, dists, lims);
}

#define BBANNLIB_DECL(dataT)                                                   \
  template bool BBAnnIndex2<dataT>::LoadIndex(std::string &indexPathPrefix);   \
  template void BBAnnIndex2<dataT>::BatchSearchCpp(                            \
      const dataT *pquery, uint64_t dim, uint64_t numQuery, uint64_t knn,      \
      const BBAnnParameters para, uint32_t *answer_ids,                        \
      distanceT *answer_dists);                                                \
  template void BBAnnIndex2<dataT>::BuildIndexImpl(                            \
      const BBAnnParameters para);                                             \
  template std::tuple<std::vector<uint32_t>, std::vector<float>,               \
                      std::vector<uint64_t>>                                   \
  BBAnnIndex2<dataT>::RangeSearchCpp(const dataT *pquery, uint64_t dim,        \
                                     uint64_t numQuery, double radius,         \
                                     const BBAnnParameters para);

BBANNLIB_DECL(float);
BBANNLIB_DECL(uint8_t);
BBANNLIB_DECL(int8_t);

#undef BBANNLIB_DECL

} // namespace bbann