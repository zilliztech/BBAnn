#pragma once
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
} // namespace util
} // namespace bbann
