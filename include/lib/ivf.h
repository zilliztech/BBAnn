#pragma once
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex> 
#include <vector>
#include <string.h>
#include <unistd.h>
#include "util/file_handler.h"



struct ClusteringTask{
    ClusteringTask(int64_t o, int64_t n, int64_t l)
    : offset(o), num_elems(n), level(l) {}
    int64_t offset;
    int64_t num_elems;
    int64_t level;
};

template <typename T1, typename T2, typename R>
void elkan_L2_assign(const T1 *x, const T2 *y, int64_t dim, int64_t nx,
                     int64_t ny, int64_t *ids, R *val);

template <typename T>
void kmeans(int64_t nx, const T *x_in, int64_t dim, int64_t k, float *centroids,
            bool kmpp = false, float avg_len = 0.0, int64_t niter = 10,
            int64_t seed = 1234);

template <typename T>
void non_recursive_multilevel_kmeans(
    uint32_t k1_id,          // the index of k1 round k-means
    int64_t cluster_size,    // num vectors in this cluster
    T *data,                 // buffer to place all vectors
    uint32_t *ids,           // buffer to place all ids
    int64_t round_offset,    // the offset of to clustering data in this round
    int64_t dim,             // the dimension of vector
    uint32_t threshold,      // determines when to stop recursive clustering
    const uint64_t blk_size, // general 4096, determines how many vectors can be
                             // placed in a block
    uint32_t
        &blk_num, // output variable, num block output in this round clustering
    IOWriter &data_writer,         // file writer 1: to output base vectors
    IOWriter &centroids_writer,    // file writer 2: to output centroid vectors
    IOWriter &centroids_id_writer, // file writer 3: to output centroid ids
    int64_t
        centroids_id_start_position, // the start position of all centroids id
    int level,         // n-th round recursive clustering, start with 0
    std::mutex &mutex, // mutex to protect write out centroids
    std::vector<ClusteringTask> &output_tasks, // output clustering tasks
    bool kmpp = false,                         // k-means parameter
    float avg_len = 0.0,                       // k-means parameter
    int64_t niter = 10,                        // k-means parameter
    int64_t seed = 1234                        // k-means parameter
);



template <typename T>
void same_size_kmeans(int64_t nx, const T *x_in, int64_t dim, int64_t k,
                      float *centroids, int64_t *assign, bool kmpp = false,
                      float avg_len = 0.0, int64_t niter = 10,
                      int64_t seed = 1234);
