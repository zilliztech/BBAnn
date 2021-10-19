#pragma once
#include "ann_interface.h"

#include "hnswlib/hnswalg.h"
#include <iostream>
#include <memory>
#include <stdint.h>
#include <string>
#include <tuple>

namespace bbann {

namespace consts {
// num of clusters in the first round k-means
// constexpr static int K1 = 10;
// sample rate of the first round k-means
constexpr static float K1_SAMPLE_RATE = 0.01;
// sample rate of the pq train set
constexpr static float PQ_SAMPLE_RATE = 0.01;
// the threshold of the second round k-means, if the size of cluster is larger
// than this threshold, than do ((cluster size)/threshold)-means
constexpr static int SPLIT_THRESHOLD = 500;
// the max cluster number in hierarchical_cluster
constexpr static int MAX_CLUSTER_K2 = 500;

constexpr static int KMEANS_THRESHOLD = 2000;

} // namespace consts
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