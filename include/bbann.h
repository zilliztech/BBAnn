#pragma once

#include <iostream>
#include <string>

#include "util/constants.h"
#include "util/utils.h"
#include "hnswlib/hnswlib.h"
#include "util/TimeRecorder.h"

template<typename DATAT>
void train_cluster(const std::string& raw_data_bin_file,
                   const std::string& output_path,
                   const int K1,
                   float** centroids);


template<typename DATAT, typename DISTT, typename HEAPT>
void divide_raw_data(const std::string& raw_data_bin_file,
                     const std::string& output_path,
                     const float* centroids,
                     const uint32_t K1);


template<typename DATAT, typename DISTT, typename HEAPT>
void conquer_clusters(const std::string& output_path,
                      const int K1, const int threshold);

template<typename DATAT, typename DISTT, typename HEAPT>
void  hierarchical_clusters(const std::string& output_path,
                            const int K1, const double  avg_len, const int blk_size);

void build_graph(const std::string& index_path,
                 const int hnswM, const int hnswefC,
                 MetricType metric_type);


template<typename DISTT, typename HEAPT>
void save_answers(const std::string& answer_bin_file,
                  const int topk,
                  const uint32_t nq,
                  DISTT*& answer_dists,
                  uint32_t*& answer_ids,
                  bool use_comp_format = true);

template<typename DATAT, typename DISTT, typename HEAPT>
void build_bigann(const std::string& raw_data_bin_file,
                  const std::string& output_path,
                  const int hnswM, const int hnswefC,
                  const int K1, const int threshold,
                  MetricType metric_type,
                  QuantizerType quantizer_type);