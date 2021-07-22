#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <mutex>
#include <algorithm>
#include <string>

#include "util/defines.h"
#include "util/constants.h"
#include "util/utils.h"
#include "util/statistics.h"
#include "util/heap.h"
#include "hnswlib/hnswlib.h"
#include "pq/pq.h"
#include "pq/pq_residual.h"
#include "flat/flat.h"
#include "ivf/clustering.h"
#include "util/TimeRecorder.h"


// interfaces of the first version
template<typename DATAT, typename DISTT>
void search_disk_index_simple(const std::string& index_path, 
                              const std::string& query_bin_file,
                              const std::string& answer_bin_file,
                              const int topk,
                              const int refine_topk,
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





// interfaces of the second version


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


void build_graph(const std::string& index_path,
                 const int hnswM, const int hnswefC,
                 MetricType metric_type);


template<typename DATAT, typename DISTT, typename HEAPT>
void train_pq_quantizer(const std::string& raw_data_bin_file,
                     const std::string& output_path,
                     const int K1,
                     const int PQM, const int PQnbits);

template<typename DATAT, typename DISTT, typename HEAPT>
void train_pq_residual_quantizer(
        const std::string& raw_data_bin_file,
        const std::string& output_path,
        const int K1,
        const int PQM, const int PQnbits,
        MetricType metric_type);

template<typename DATAT, typename DISTT, typename HEAPT>
void build_bigann(const std::string& raw_data_bin_file,
                  const std::string& output_path,
                  const int hnswM, const int hnswefC,
                  const int PQM, const int PQnbits,
                  const int K1, const int threshold,
                  MetricType metric_type,
                  QuantizerType quantizer_type);

void load_pq_codebook(const std::string& index_path,
                      std::vector<std::vector<uint8_t>>& pq_codebook, 
                      const int K1);

void load_meta(const std::string& index_path,
               std::vector<std::vector<uint32_t>>& meta,
               const int K1);

template<typename DATAT>
void search_graph(std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
                  const int nq,
                  const int dq,
                  const int nprobe,
                  const int refine_nprobe,
                  const DATAT* pquery,
                  uint64_t* buckets_label);

template<typename DATAT, typename DISTT, typename HEAPTT>
void search_pq_quantizer(ProductQuantizer<HEAPTT, DATAT, uint8_t>& pq_quantizer,
                      const uint32_t nq,
                      const uint32_t dq,
                      float* ivf_centroids,
                      uint64_t* buckets_label,
                      const int nprobe,
                      const int refine_topk,
                      const int K1,
                      const DATAT* pquery,
                      std::vector<std::vector<uint8_t>>& pq_codebook,
                      std::vector<std::vector<uint32_t>>& meta,
                      DISTT*& pq_distance,
                      uint64_t*& pq_offsets,
                      PQ_Computer<DATAT>& pq_cmp);

template<typename DATAT, typename DISTT, typename HEAPT>
void refine(const std::string& index_path,
            const int K1,
            const uint32_t nq,
            const uint32_t dq,
            const int topk,
            const int refine_topk,
            uint64_t* pq_offsets,
            const DATAT* pquery,
            DISTT*& answer_dists,
            uint32_t*& answer_ids,
            Computer<DATAT, DATAT, DISTT>& dis_computer);

template<typename DATAT, typename DISTT, typename HEAPT>
void refine_c(const std::string& index_path,
            const int K1,
            const uint32_t nq,
            const uint32_t dq,
            const int topk,
            const int refine_topk,
            uint64_t* pq_offsets,
            const DATAT* pquery,
            DISTT*& answer_dists,
            uint32_t*& answer_ids,
            Computer<DATAT, DATAT, DISTT>& dis_computer);

template<typename DISTT, typename HEAPT>
void save_answers(const std::string& answer_bin_file,
                  const int topk,
                  const uint32_t nq,
                  DISTT*& answer_dists,
                  uint32_t*& answer_ids,
                  bool use_comp_format = true);

template<typename DATAT, typename DISTT, typename HEAPT, typename HEAPTT>
void search_bigann(const std::string& index_path,
                   const std::string& query_bin_file,
                   const std::string& answer_bin_file,
                   const int nprobe,
                   const int refine_nprobe,
                   const int topk,
                   const int refine_topk,
                   std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
                   PQResidualQuantizer<HEAPTT, DATAT, uint8_t>& pq_quantizer,
                   const int K1,
                   std::vector<std::vector<uint8_t>>& pq_codebook,
                   std::vector<std::vector<uint32_t>>& meta,
                   Computer<DATAT, DATAT, DISTT>& dis_computer);

template<typename DATAT, typename DISTT, typename HEAPT, typename HEAPTT>
void search_bigann(const std::string& index_path,
                   const std::string& query_bin_file,
                   const std::string& answer_bin_file,
                   const int nprobe,
                   const int refine_nprobe,
                   const int topk,
                   const int refine_topk,
                   std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
                   ProductQuantizer<HEAPTT, DATAT, uint8_t>& pq_quantizer,
                   const int K1,
                   PQ_Computer<DATAT>& pq_cmp,
                   std::vector<std::vector<uint8_t>>& pq_codebook,
                   std::vector<std::vector<uint32_t>>& meta,
                   Computer<DATAT, DATAT, DISTT>& dis_computer);



