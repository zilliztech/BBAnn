#pragma once

#include <limits>

namespace sq_hnswlib {
template <typename T>
uint8_t *train_codebook(
    const std::string &filename, const int hnswM, const int hnswefC,
    MetricType metric_type,
    std::function<void(const std::string &, uint32_t &, uint32_t &)> read_meta,
    std::function<const std::string &, T *&, uint32_t &, uint32_t &,
                  const uint32_t &>
        read_bin_file_half_dimension,
    std::function<const T *, const int64_t, const int64_t, const int64_t,
                  float *&>
        kmeans) {

  uint32_t *pids = nullptr;
  uint32_t npts, ndim, nids, nidsdim, npts2;
  uint32_t total_n = 0;

  read_meta(filename, total_n, ndim);
  uint32_t half_dim = ndim / 2;

  float *pdata = new float[(uint64_t)total_n * (uint64_t)half_dim];
  float *codes = new float[256 * ndim];
  uint8_t *codebook = new uint8_t[(uint64_t)total_n * (uint64_t)ndim];
  memset(codebook, 0, sizeof(uint8_t) * (uint64_t)total_n * (uint64_t)ndim);

  read_bin_file_half_dimension(filename, pdata, npts, ndim, 0);

#pragma omp parallel for
  for (uint32_t i = 0; i < half_dim; ++i) {
    float *centers = new float[256];
    kmeans(total_n, pdata + i * total_n, 1, 256, centers);
    for (int j = 0; j < 256; j++) {
      codes[j * ndim + i] = centers[j];
    }
    for (uint32_t j = 0; j < total_n; ++j) {
      float min_dis = std::numeric_limits<float>::max();
      for (uint32_t k = 0; k < 256; ++k) {
        float diff = codes[k * ndim + i] - pdata[i * total_n + j];
        float now_dis = diff * diff;
        if (now_dis < min_dis) {
          min_dis = now_dis;
          uint32_t *p32 = &k;
          uint8_t *p8 = (uint8_t *)p32;
          codebook[j * ndim + i] = (uint8_t)(*(p8));
        }
      }
    }
  }

  delete[] pdata;
  pdata = nullptr;

  read_bin_file_half_dimension(filename, pdata, npts, ndim, half_dim);

#pragma omp parallel for
  for (uint32_t i = half_dim; i < ndim; ++i) {
    float *centers = new float[256];
    kmeans(total_n, pdata + (i - half_dim) * total_n, 1, 256, centers);
    for (int j = 0; j < 256; j++) {
      codes[j * ndim + i] = centers[j];
    }
    for (uint32_t j = 0; j < total_n; ++j) {
      float min_dis = std::numeric_limits<float>::max();
      for (uint32_t k = 0; k < 256; ++k) {
        float diff = codes[k * ndim + i] - pdata[(i - half_dim) * total_n + j];
        float now_dis = diff * diff;
        if (now_dis < min_dis) {
          min_dis = now_dis;
          uint32_t *p32 = &k;
          uint8_t *p8 = (uint8_t *)p32;
          codebook[j * ndim + i] = (uint8_t)(*(p8));
        }
      }
    }
  }
  delete[] pdata;
  pdata = nullptr;

  return codebook;
}

} // namespace sq_hnswlib
