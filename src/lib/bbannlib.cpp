#include "lib/bbannlib.h"
#include "ann_interface.h"
#include <stdint.h>
#include <string>

template <typename dataT, typename paraT>
bool BBAnnIndex<dataT, paraT>::LoadIndex(std::string &indexPathPrefix) {
  std::cout << "Loading: " << indexPathPrefix;

  std::string hnsw_index_file = indexPathPrefix + HNSW + INDEX + BIN;
  std::string bucket_centroids_file =
      indexPathPrefix + BUCKET + CENTROIDS + BIN;

  uint32_t bucket_num, dim;
  get_bin_metadata(bucket_centroids_file, bucket_num, dim);

  hnswlib::SpaceInterface<float> *space = nullptr;
  if (MetricType::L2 == metric_) {
    space = new hnswlib::L2Space(dim);
  } else if (MetricType::IP == metric_) {
    space = new hnswlib::InnerProductSpace(dim);
  }
  // load hnsw
  index_hnsw_ =
      std::make_shared<hnswlib::HierarchicalNSW<float>>(space, hnsw_index_file);
  indexPrefix_ = indexPathPrefix;
  return true;
}

template <typename dataT, typename paraT>
void BBAnnIndex<dataT, paraT>::BatchSearchCpp(const dataT *pquery, uint64_t dim,
                                              uint64_t numQuery, uint64_t knn,
                                              const paraT para,
                                              uint32_t *answer_ids,
                                              distanceT *answer_dists) {
  std::cout << "Query: " << std::endl;
  switch (para.metric) {
  case MetricType::L2: {
    Computer<dataT, dataT, distanceT> dis_computer =
        L2sqr<const dataT, const dataT, distanceT>;
    search_bbann_queryonly<dataT, distanceT, CMax<distanceT, uint32_t>>(
        indexPrefix_, para.nProbe, para.hnswefC, knn, index_hnsw_, para.K1,
        para.blockSize, dis_computer, pquery, answer_ids, answer_dists,
        numQuery, dim);
    break;
  }
  case MetricType::IP: {
    Computer<dataT, dataT, distanceT> dis_computer =
        IP<const dataT, const dataT, distanceT>;
    search_bbann_queryonly<dataT, distanceT, CMin<distanceT, uint32_t>>(
        indexPrefix_, para.nProbe, para.hnswefC, knn, index_hnsw_, para.K1,
        para.blockSize, dis_computer, pquery, answer_ids, answer_dists,
        numQuery, dim);
    break;
  }
  default:
    std::cerr << "not supported metric type"  << (int) para.metric << std::endl;
  }
}

template <typename dataT, typename paraT>
void BBAnnIndex<dataT, paraT>::BuildIndexImpl(const paraT para) {
  std::cout << __func__ << "Build start " << std::endl;
  using distanceT = typename TypeWrapper<dataT>::distanceT;
  switch (para.metric) {
  case MetricType::L2: {
    std::cout << "Build With L2" << std::endl;
    std::cout << "dataT" << typeid(dataT).name() << std::endl;
    std::cout << "distanceT" << typeid(distanceT).name() << std::endl;
    build_bbann<dataT, distanceT, CMax<distanceT, uint32_t>>(
        para.dataFilePath, para.indexPrefixPath, para.hnswM, para.hnswefC,
        para.metric, para.K1, para.blockSize);
    return;
  }
  case MetricType::IP: {
    build_bbann<dataT, distanceT, CMin<distanceT, uint32_t>>(
        para.dataFilePath, para.indexPrefixPath, para.hnswM, para.hnswefC,
        para.metric, para.K1, para.blockSize);
    return;
  }
  default:
    std::cerr << "not supported" << std::endl;
  }
}

template <typename dataT, typename paraT>
void BBAnnIndex<dataT, paraT>::RangeSearchCpp(const dataT *pquery, uint64_t dim, uint64_t numQuery,
                                              double radius, const paraT para,
                                              std::vector<std::vector<uint32_t>> ids,
                                              std::vector<std::vector<float>> dists,
                                              std::vector<uint64_t> lims) {
  Computer<dataT, dataT, float> dis_computer =
      L2sqr<const dataT, const dataT, float>;
  range_search_bbann<dataT, float>(
      indexPrefix_, para.hnswefC, radius, index_hnsw_, para.K1,
      para.blockSize, dis_computer, pquery, ids, dists, lims,
      numQuery, dim);
}

#define BBANNLIB_DECL(dataT, paraT)                                       \
  template bool BBAnnIndex<dataT, paraT>::LoadIndex(                      \
      std::string &indexPathPrefix);                                      \
  template void BBAnnIndex<dataT, paraT>::BatchSearchCpp(                 \
      const dataT *pquery, uint64_t dim, uint64_t numQuery, uint64_t knn, \
      const paraT para, uint32_t *answer_ids, distanceT *answer_dists);   \
  template void BBAnnIndex<dataT, paraT>::RangeSearchCpp(                 \
      const dataT *pquery, uint64_t dim, uint64_t numQuery,               \
      double radius, const paraT para,                                    \
      std::vector<std::vector<uint32_t>> ids,                             \
      std::vector<std::vector<float>> dists,                              \
      std::vector<uint64_t> lims);                                        \
  template void BBAnnIndex<dataT, paraT>::BuildIndexImpl(const paraT para);

BBANNLIB_DECL(float, BBAnnParameters);
BBANNLIB_DECL(uint8_t, BBAnnParameters);
BBANNLIB_DECL(int8_t, BBAnnParameters);

#undef BBANNLIB_DECL
