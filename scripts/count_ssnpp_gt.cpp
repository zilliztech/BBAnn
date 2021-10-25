#include <fstream>
#include <iostream>
#include <map>
#include <stdint.h>
#include <string>
#include <vector>

std::string path = "/home/pat/datasets/Facebook-SimSearchNet++/ssnpp-10M";

template <typename FILE_IDT, typename IDT>
void read_comp_range_search(std::vector<std::vector<IDT>> &v,
                            const std::string &gt_file, int32_t &nq,
                            int32_t &total_res) {
  std::ifstream gin(gt_file, std::ios::binary);

  gin.read((char *)&nq, sizeof(int32_t));
  gin.read((char *)&total_res, sizeof(int32_t));

  std::cout << nq << " " << total_res << std::endl;

  v.resize(nq);
  int32_t n_results_per_query;
  uint64_t tot = 0;
  for (int i = 0; i < nq; ++i) {
    gin.read((char *)&n_results_per_query, sizeof(int32_t));
    v[i].resize(n_results_per_query);
    tot += n_results_per_query;
    // std::cout << n_results_per_query << std::endl;
  }
  std::cout << tot << std::endl;

  FILE_IDT t_id;
  for (uint32_t i = 0; i < nq; ++i) {
    for (uint32_t j = 0; j < v[i].size(); ++j) {
      gin.read((char *)&t_id, sizeof(FILE_IDT));
      v[i][j] = static_cast<IDT>(t_id);
    }
  }

  gin.close();
}

int main() {
  std::vector<std::vector<int32_t>> v;
  int32_t nq, total_res;
  read_comp_range_search<int32_t, int32_t>(v, path, nq, total_res);

  std::map<int, int> m;
  for (int i = 0; i < nq; ++i) {
    ++m[v[i].size()];
  }

  for (const auto x : m) {
    std::cout << x.first << " " << x.second << std::endl;
  }

  return 0;
}
