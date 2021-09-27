#pragma once

#include <algorithm>
#include <iostream>
#include <string>

#include "flat/flat.h"
#include "hnswlib/hnswlib.h"
#include "ivf/clustering.h"
#include "util/TimeRecorder.h"
#include "util/constants.h"
#include "util/defines.h"
#include "util/heap.h"
#include "util/statistics.h"
#include "util/utils.h"

template <typename DATAT, typename DISTT, typename HEAPT>
void build_bbann(const std::string &raw_data_bin_file,
                 const std::string &output_path, const int hnswM,
                 const int hnswefC, MetricType metric_type, const int K1,
                 const uint64_t block_size);

template <typename DATAT, typename DISTT, typename HEAPT>
void search_bbann(const std::string &index_path,
                  const std::string &query_bin_file,
                  const std::string &answer_bin_file, const int nprobe,
                  const int hnsw_ef, const int topk,
                  std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
                  const int K1, const uint64_t block_size,
                  Computer<DATAT, DATAT, DISTT> &dis_computer);

// Just do the search and store in answer_dists/answer_ids.
template <typename DATAT, typename DISTT, typename HEAPT>
void search_bbann_exec(
    const std::string &index_path, const int nprobe, const int hnsw_ef,
    const int topk, std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
    const int K1, const uint64_t block_size,
    Computer<DATAT, DATAT, DISTT> &dis_computer, uint32_t &nq, uint32_t &dq,
    const DATAT *pquery, uint32_t *bucket_labels, DISTT *answer_dists,
    uint32_t *answer_ids);