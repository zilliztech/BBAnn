#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <mutex>
#include <algorithm>
#include "util/defines.h"
#include "util/constants.h"
#include "util/utils.h"
#include "util/heap.h"
#include "hnswlib/hnswlib.h"
#include "pq/pq.h"
#include <string>
#include "flat/flat.h"
#include "ivf/clustering.h"
#include "util/TimeRecorder.h"

template<typename DATAT, typename DISTT>
void search_disk_index_simple(const std::string& index_path, 
                              const std::string& query_bin_file,
                              const std::string& answer_bin_file,
                              const int topk,
                              const int nprobe,
                              const int PQM, const int PQnbits,
                              MetricType metric_type = MetricType::L2);


template<typename DATAT, typename DISTT>
void build_disk_index(const std::string& raw_data_file, const std::string& index_output_path,
                      const int hnswM, const int hnswefC, const int PQM, const int PQnbits,
                      MetricType metric_type = MetricType::L2);



template<typename DATAT, typename DISTT>
void split_raw_data(const std::string& raw_data_file, const std::string& index_output_path,
                    float* centroids, MetricType metric_type);


template<typename DATAT, typename DISTT>
void train_clusters(const std::string& cluster_path, uint32_t& graph_nb, uint32_t& graph_dim, 
                    ProductQuantizer<CMax<DISTT, uint32_t>, DATAT, uint8_t>* pq_quantizer,
                    MetricType metric_type);


void create_graph_index(const std::string& index_path, 
                        const int hnswM, const int hnswefC,
                        MetricType metric_type);





