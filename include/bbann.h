#pragma once

#include <iostream>
#include <algorithm>
#include <string>

#include "util/defines.h"
#include "util/constants.h"
#include "util/utils.h"
#include "hnswlib/hnswlib.h"
#include "sq_hnswlib/hnswlib.h"
#include "util/statistics.h"
#include "util/heap.h"
#include "ivf/hierarchical_kmeans.h"
#include "util/TimeRecorder.h"
#include "flat/flat.h"
#include <list>
#include <thread>

template <typename DATAT>
void train_cluster(const std::string &raw_data_bin_file,
                   const std::string &output_path, const int32_t K1,
                   float **centroids, double &avg_len);

void build_graph(const std::string &index_path, const int hnswM,
                 const int hnswefC, MetricType metric_type);

template<typename DATAT, typename DISTT, typename HEAPT>
void build_bbann(const std::string& raw_data_bin_file,
                    const std::string& output_path,
                    const int hnswM,
                    const int hnswefC,
                    MetricType metric_type,
                    const int K1,
                    const uint64_t block_size);
                    

template<typename DATAT, typename DISTT, typename HEAPT>
void search_bbann(const std::string& index_path,
                   const std::string& query_bin_file,
                   const std::string& answer_bin_file,
                   const int nprobe,
                   const int hnsw_ef,
                   const int topk,
                   std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
                   const int K1,
                   const uint64_t block_size,
                   Computer<DATAT, DATAT, DISTT>& dis_computer);

template <typename DATAT, typename DISTT, typename HEAPT>
void search_bbann_queryonly(
    const std::string &index_path, const int nprobe, const int hnsw_ef,
    const int topk, std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
    const int K1, const uint64_t block_size,
    Computer<DATAT, DATAT, DISTT> &dis_computer,
    /* for IO */
    const DATAT *pquery, uint32_t *answer_ids, DISTT *answer_dists,
    uint32_t num_query, uint32_t dim);


template <typename DATAT, typename DISTT>
void range_search_bbann(
    const std::string &index_path,
    const int hnsw_ef,
    const float radius, std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
    const int K1, const uint64_t block_size,
    Computer<DATAT, DATAT, DISTT> &dis_computer,
    /* for IO */
    const DATAT *pquery, 
    std::vector<std::vector<uint32_t>> &ids,
    std::vector<std::vector<float>> &dists,
    std::vector<uint64_t> &lims,
    uint32_t nq, uint32_t dim);
