#pragma once
#include "hnswlib/hnswlib.h"
#include "util/TimeRecorder.h"
#include "util/defines.h"
#include <chrono>
#include <memory>
#include <string>
namespace bbann {
std::string Hello();

template <typename DATAT, typename DISTT>
void search_graph(std::shared_ptr<hnswlib::HierarchicalNSW<DISTT>> index_hnsw,
                  const int nq, const int dq, const int nprobe,
                  const int refine_nprobe, const DATAT *pquery,
                  uint32_t *buckets_label, float *centroids_dist);

template <typename DATAT>
void train_cluster(const std::string &raw_data_bin_file,
                   const std::string &output_path, const int32_t K1,
                   float **centroids, double &avg_len, bool vector_use_sq = false);

template <typename DATAT, typename DISTT>
void divide_raw_data(const BBAnnParameters para, const float *centroids);

template <typename T>
void reservoir_sampling(const std::string &data_file, const size_t sample_num,
                        T *sample_data);

template <typename DATAT, typename DISTT>
void hierarchical_clusters(const BBAnnParameters para, const double avg_len);

template <typename DATAT, typename DISTT>
void build_graph(const std::string &index_path, const int hnswM,
                 const int hnswefC, MetricType metric_type,
                 const uint64_t block_size, const int32_t sample);
template <typename DATAT, typename DISTT>
hnswlib::SpaceInterface<DISTT> *getDistanceSpace(MetricType metric_type,
                                                 uint32_t ndim);

} // namespace bbann