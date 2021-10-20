#pragma once
#include "ann_interface.h"

#include "hnswlib/hnswalg.h"
#include "lib/algo.h"
#include <iostream>
#include <memory>
#include <stdint.h>
#include <string>
#include <tuple>

namespace bbann {

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

  std::tuple<std::vector<uint32_t>, std::vector<distanceT>, std::vector<uint64_t>>
  RangeSearchCpp(const dataT *pquery, uint64_t dim, uint64_t numQuery,
                      double radius, const BBAnnParameters para) override;
  std::shared_ptr<hnswlib::HierarchicalNSW<distanceT>> index_hnsw_;

  std::string indexPrefix_;
  std::string dataFilePath_;

  static void BuildIndexImpl(const BBAnnParameters para);
  void BuildWithParameter(const BBAnnParameters para);

  std::string getHnswIndexFileName() { return indexPrefix_ + "hnsw-index.bin"; }
  std::string getBucketCentroidsFileName() {
    return indexPrefix_ + "bucket-centroids.bin";
  }
  std::string getClusterRawDataFileName(int cluster_id) {
    return indexPrefix_ + "cluster-" + std::to_string(cluster_id) +
           "-raw_data.bin";
  }
  std::string getClusterGlobalIdsFileName(int cluster_id) {
    return indexPrefix_ + "cluster-" + std::to_string(cluster_id) +
           "-global_ids.bin";
  }
};

} // namespace bbann