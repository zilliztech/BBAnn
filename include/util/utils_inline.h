#pragma once
#include "util/defines.h"
#include "util/distance.h"
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

namespace bbann {
namespace util {

inline int round_up_div(int x, int y) { return (x + y - 1) / y; }

inline int get_max_events_num_of_aio() {
  auto file = "/proc/sys/fs/aio-max-nr";
  auto f = std::ifstream(file);

  if (!f.good()) {
    return 1024;
  }

  int num;
  f >> num;
  f.close();
  return num;
}

inline void get_bin_metadata(const std::string &bin_file, uint32_t &nrows,
                             uint32_t &ncols) {
  std::ifstream reader(bin_file, std::ios::binary);
  reader.read((char *)&nrows, sizeof(uint32_t));
  reader.read((char *)&ncols, sizeof(uint32_t));
  reader.close();
  std::cout << "get meta from " << bin_file << ", nrows = " << nrows
            << ", ncols = " << ncols << std::endl;
}

inline void set_bin_metadata(const std::string &bin_file, const uint32_t &nrows,
                             const uint32_t &ncols) {
  std::ofstream writer(bin_file, std::ios::binary | std::ios::in);
  writer.seekp(0);
  writer.write((char *)&nrows, sizeof(uint32_t));
  writer.write((char *)&ncols, sizeof(uint32_t));
  writer.close();
  std::cout << "set meta to " << bin_file << ", nrows = " << nrows
            << ", ncols = " << ncols << std::endl;
}

template <typename T>
inline void read_bin_file(const std::string &file_name, T *&data, uint32_t &n,
                          uint32_t &dim) {
  std::ifstream reader(file_name, std::ios::binary);

  reader.read((char *)&n, sizeof(uint32_t));
  reader.read((char *)&dim, sizeof(uint32_t));
  if (data == nullptr) {
    data = new T[(uint64_t)n * (uint64_t)dim];
  }
  reader.read((char *)data, sizeof(T) * (uint64_t)n * dim);

  reader.close();
  std::cout << "read binary file from " << file_name
            << " done in ... seconds, n = " << n << ", dim = " << dim
            << std::endl;
}

inline uint64_t fsize(const std::string filename) {
  struct stat st;
  stat(filename.c_str(), &st);
  return st.st_size;
}
inline void rand_perm(int64_t *perm, int64_t n, int64_t k, int64_t seed) {
  std::mt19937 generator(seed);

  for (int64_t i = 0; i < n; i++) {
    perm[i] = i;
  }

  for (int64_t i = 0; i < k; i++) {
    int64_t i2 = i + generator() % (n - i);
    std::swap(perm[i], perm[i2]);
  }
}

inline uint32_t gen_global_block_id(const uint32_t cid, const uint32_t bid) {
  uint32_t ret = 0;
  ret |= (cid & 0xff);
  ret <<= 24;
  ret |= (bid & 0xffffff);
  return ret;
}

inline float rand_float() {
  static std::mt19937 generator(1234);
  return generator() / (float)generator.max();
}

inline int rand_int() {
  static std::mt19937 generator(3456);
  return generator() & 0x7fffffff;
}

inline void parse_global_block_id(uint32_t id, uint32_t &cid, uint32_t &bid) {
  bid = (id & 0xffffff);
  id >>= 24;
  cid = (id & 0xff);
  return;
}

inline uint64_t gen_id(const uint32_t cid, const uint32_t bid,
                       const uint32_t off) {
  uint64_t ret = 0;
  ret |= (cid & 0xff);
  ret <<= 24;
  ret |= (bid & 0xffffff);
  ret <<= 32;
  ret |= (off & 0xffffffff);
  return ret;
}

inline void parse_id(uint64_t id, uint32_t &cid, uint32_t &bid, uint32_t &off) {
  off = (id & 0xffffffff);
  id >>= 32;
  bid = (id & 0xffffff);
  id >>= 24;
  cid = (id & 0xff);
}

template <typename T1, typename T2, typename R>
using Computer = std::function<R(const T1 *, const T2 *, int n)>;

template <typename T1, typename T2, typename R>
inline Computer<T1, T2, R> select_computer(MetricType metric_type) {
  switch (metric_type) {
  case MetricType::L2:
    return L2sqr<const T1, const T2, R>;
    break;
  case MetricType::IP:
    return IP<const T1, const T2, R>;
    break;
  }
}

} // namespace util

inline std::string getClusterRawDataFileName(std::string prefix,
                                             int cluster_id) {
  return prefix + "cluster-" + std::to_string(cluster_id) + "-raw_data.bin";
}
inline std::string getClusterGlobalIdsFileName(std::string prefix,
                                               int cluster_id) {
  return prefix + "cluster-" + std::to_string(cluster_id) + "-global_ids.bin";
}
inline std::string getSQMetaFileName(std::string prefix) {
  return prefix + "meta";
}

inline float rand_float() {
  static std::mt19937 generator(1234);
  return generator() / (float)generator.max();
}

inline int rand_int() {
  static std::mt19937 generator(3456);
  return generator() & 0x7fffffff;
}

template <typename T>
inline void random_sampling_k2(const T *data, const int64_t data_size,
                               const int64_t dim, const int64_t sample_size,
                               T *sample_data, int64_t seed = 1234) {
  std::vector<int> perm(data_size);
  for (int64_t i = 0; i < data_size; i++) {
    perm[i] = i;
  }
  std::shuffle(perm.begin(), perm.end(), std::default_random_engine(seed));
  for (int64_t i = 0; i < sample_size; i++) {
    memcpy(sample_data + i * dim, data + perm[i] * dim, dim * sizeof(T));
  }
  return;
}

template <typename T>
inline void transform_data(T *data, T *tdata, int64_t n, uint32_t dim) {
  for (auto i = 0; i < n; i++) {
    for (auto j = 0; j < dim; j++) {
      tdata[n * j + i] = data[i * dim + j];
    }
  }
}

template <typename T>
inline void train_code(T *max_len, T *min_len, T *data, int64_t n,
                       uint32_t dim) {
  float rs_arg = 0.0;
  int o;
  T *min, max;
  T *tdata = new T[n * dim];
  transform_data(data, tdata, n, dim);
  for (int d = 0; d < dim; d++) {
    auto *__restrict tdata_d = tdata + n * d;
    std::sort(tdata_d, tdata_d + n);
    o = int(rs_arg * n);
    if (o < 0)
      o = 0;
    if (o > n - o)
      o = n / 2;
    min_len[d] = tdata_d[o];
    max_len[d] = tdata_d[n - 1 - o];
  }
  delete[] tdata;
}

// For RS_opt
/*
template<typename T>
inline void train_code(T* max_len, T* min_len, T* data, int64_t n, uint32_t
dim){ float rs_arg = 0; int o; T vmin,vmax; T* tdata= new T[n * dim]; int k =
256; transform_data(data, tdata, n, dim); for (int d = 0; d < dim; d++) { float
a, b; float sx = 0; T* x = tdata + n * d;
        {
            vmin = std::numeric_limits<T>::max();
            vmax = std::numeric_limits<T>::min();
            for (auto i = 0; i < n; i++) {
                if (x[i] < vmin)
                    vmin = x[i];
                if (x[i] > vmax)
                    vmax = x[i];
                sx += x[i];
            }
            b = vmin;
            a = (vmax - vmin) / k;
        }

        int niter = 2000;
        float last_err = -1;
        int iter_last_err = 0;
        for (int it = 0; it < niter; it++) {
            float sn = 0, sn2 = 0, sxn = 0, err1 = 0;

            for (auto i = 0; i < n; i++) {
                float xi = x[i];
                float ni = floor((xi - b) / a + 0.5);
                if (ni < 0)
                    ni = 0;
                if (ni >= k)
                    ni = k - 1;
                err1 += std::sqrt(xi - (ni * a + b));
                sn += ni;
                sn2 += ni * ni;
                sxn += ni * xi;
            }

            if (err1 == last_err) {
                iter_last_err++;
                if (iter_last_err == 16)
                    break;
            } else {
                last_err = err1;
                iter_last_err = 0;
            }

            float det = std::sqrt(sn) - sn2 * n;

            b = (sn * sxn - sn2 * sx) / det;
            a = (sn * sx - n * sxn) / det;
        }
        min_len[d] = vmin;
        max_len[d] = vmax;
    }
    delete [] tdata;
}
*/

template <typename T>
inline void encode_uint8(T *max_len, T *min_len, T *data, uint8_t *code,
                         int64_t n, uint32_t dim) {
  for (int64_t i = 0; i < n; i++) {
    T x_temp;
    T *__restrict x = data + i * dim;
    uint8_t *__restrict y = code + i * dim;
    for (int d = 0; d < dim; d++) {
      x_temp = (x[d] - min_len[d]) / (max_len[d] - min_len[d]);
      if (x_temp < 0.0)
        x_temp = 0;
      if (x_temp > 1.0)
        x_temp = 1;
      y[d] = (uint8_t)(x_temp * 255);
    }
  }
}

template <typename T>
inline void decode_uint8(T *max_len, T *min_len, T *data, uint8_t *code,
                         int64_t n, uint32_t dim) {
  for (int64_t i = 0; i < n; i++) {
    auto *__restrict x = data + i * dim;
    auto *__restrict y = code + i * dim;
    for (int d = 0; d < dim; d++) {
      x[d] = min_len[d] + (y[d] + 0.5f) / 255.0f * (max_len[d] - min_len[d]);
    }
  }
}

template <typename T>
inline void encode_uint8_2(T *max_len, T *min_len, T *data, uint8_t *code,
                           int64_t n, uint32_t dim) {
  for (int64_t i = 0; i < n; i++) {
    auto *__restrict x = data + i * dim;
    auto *__restrict y = code + i * dim;
    for (int d = 0; d < dim; d++) {
      y[d] =
          (uint8_t)((x[d] - min_len[d]) / (max_len[d] - min_len[d] + 1) * 256);
    }
  }
}
template <typename T>
inline void decode_uint8_2(T *max_len, T *min_len, T *data, uint8_t *code,
                           int64_t n, uint32_t dim) {
  for (int64_t i = 0; i < n; i++) {
    auto *__restrict x = data + i * dim;
    auto *__restrict y = code + i * dim;
    for (int d = 0; d < dim; d++) {
      x[d] = y[d] * (max_len[d] - min_len[d] + 1) / 256 + min_len[d];
    }
  }
}

template <typename T>
inline void train_code_2(T *max_len, T *min_len, T *data, int64_t n,
                         uint32_t dim) {
  std::vector<T> tdata(dim * n);
  transform_data(data, tdata, n, dim);
  for (auto i = 0; i < dim; i++) {
    auto *__restrict tx = tdata + i * n;
    min_len[i] = std::min((float *)tx, (float *)(tx + n));
    max_len[i] = std::max((float *)tx, (float *)(tx + n));
  }
}

} // namespace bbann
