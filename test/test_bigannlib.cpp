#include "lib/bbannlib2.h"

using bbann::BBAnnIndex2;
using bbann::BBAnnParameters;
using bbann::MetricType;

int main() {
  BBAnnIndex2<float, BBAnnParameters>::index(MetricType::L2);
  index.LoadIndex("/data/catcat/BigAnn/benchmark/data/indices/T2/BBANN/"
                  "SSNPPDataset-10000000/None/");
  const dataT *pquery = query.data();
  std::vector<std::vector<uint32_t>> ids(numQuery);
  std::vector<std::vector<float>> dists(numQuery);
  std::vector<uint64_t> lims(numQuery + 1);
  
template <typename T>
inline void read_bin_file(const std::string &file_name, T *&data, uint32_t &n,
                          uint32_t &dim) 
                          
  index.RangeSearchCpp(pquery, dim, numQuery, radius, para, ids, dists, lims);
}