#pragma once

#include <iostream>
#include <algorithm>
#include <string>

#include "util/defines.h"
#include "util/constants.h"
#include "util/utils.h"
#include "hnswlib/hnswlib.h"
#include "util/statistics.h"
#include "util/heap.h"
#include "ivf/clustering.h"
#include "util/TimeRecorder.h"
#include "flat/flat.h"

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