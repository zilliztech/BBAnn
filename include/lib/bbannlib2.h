#pragma once
#include "ann_interface.h"

#include "hnswlib/hnswalg.h"
#include "bbann.h"
#include <iostream>
#include <memory>
#include <stdint.h>
#include <string>
#include <tuple>

namespace bbann {

struct BBAnnParameters {
  std::string dataFilePath;
  std::string indexPrefixPath;
  std::string queryPath;
  std::string groundTruthFilePath;
  MetricType metric;
  int K = 20; // top k.
  int hnswM = 32;
  int hnswefC = 500;
  int K1 = 20;
  int blockSize = 1;
  int nProbe = 2;
  int efSearch = 250;
  int rangeSearchProbeCount = 20;
};

template <typename dataT>
struct BBAnnIndex2
    : public BuildIndexFactory<BBAnnIndex2<dataT>, BBAnnParameters>,
      public AnnIndexInterface<dataT, BBAnnParameters> {

  using parameterType = BBAnnParameters;
  using dataType = dataT;
  using distanceT = typename TypeWrapper<dataT>::distanceT;
  using qidIdDistTupleType = std::tuple<uint32_t, uint32_t, distanceT>;

public:
  BBAnnIndex2(MetricType metric) : metric_(metric) {
    std::cout << "BBAnnIndex constructor" << std::endl;
  }
  MetricType metric_;

  bool LoadIndex(std::string &indexPathPrefix);

  void BatchSearchCpp(const dataT *pquery, uint64_t dim, uint64_t numQuery,
                      uint64_t knn, const BBAnnParameters para,
                      uint32_t *answer_ids, distanceT *answer_dists) override;

  std::tuple<std::vector<uint32_t>, std::vector<float>, std::vector<uint64_t>>
 RangeSearchCpp(const dataT *pquery, uint64_t dim, uint64_t numQuery,
                      double radius, const BBAnnParameters para) override;
  std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw_;
  std::string indexPrefix_;
  std::string dataFilePath_;

  static void BuildIndexImpl(const BBAnnParameters para);
  void BuildWithParameter(const BBAnnParameters para);

  std::string getHnswIndexFileName() { return indexPrefix_ + "hnsw-index.bin"; }
  std::string getBucketCentroidsFileName() {
    return indexPrefix_ + "bucket-centroids.bin";
  }
};

} // namespace bbann