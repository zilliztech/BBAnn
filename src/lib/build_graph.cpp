#include <cstdlib>
#include <iostream>
#include <string>

#include "lib/algo.h"
#include "util/defines.h"

struct ee {
  ee(const std::string &entry_text, const std::string &exit_text)
      : exit_text_(exit_text) {
    std::cout << entry_text << std::endl << std::endl;
  }
  ~ee() { std::cout << std::endl << exit_text_ << std::endl; }
  std::string exit_text_;
};

int main(int argc, char **argv) {
  ee e("start build_graph", "finish build_graph");

  std::cout << "arguments: " << std::endl;
  for (int i = 0; i < argc; ++i) {
    std::cout << argv[i] << std::endl;
  }
  std::cout << std::endl;

  if (argc != 5) {
    std::cout
        << "usage: ./build_graph <output_path> <hnswM> <hnswefC> <metric_type>"
        << std::endl;
    std::cout << std::endl;
    std::cout
        << "       output_path: /path/to/index/files/, must be end with a slash"
        << std::endl;
    std::cout << "       hnswM: parameter M for HNSW" << std::endl;
    std::cout << "       hnswefC: parameter efC for HNSW" << std::endl;
    std::cout << "       metric_type: 1 for L2, 2 for IP" << std::endl;
    exit(-1);
  }

  std::string output_path(argv[1]);
  int hnswM = atoi(argv[2]);
  int hnswefC = atoi(argv[3]);
  MetricType metric_type = (MetricType)atoi(argv[4]);

  bbann::build_hnsw_sq(output_path, hnswM, hnswefC, metric_type);

  return 0;
}
