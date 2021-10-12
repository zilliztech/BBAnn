#pragma once
#include <string>


// constant defines

constexpr static int PAGESIZE = 4096;

// units
constexpr static uint64_t KILOBYTE = 1024;
constexpr static uint64_t MEGABYTE = 1024 * 1024;
constexpr static uint64_t GIGABYTE = 1024 * 1024 * 1024;

// num of clusters in the first round k-means
// constexpr static int K1 = 10;
// sample rate of the first round k-means
constexpr static float K1_SAMPLE_RATE = 0.01;
// sample rate of the pq train set
constexpr static float PQ_SAMPLE_RATE = 0.01;
// limit the training size of the k2 clustering
constexpr static int K2_MAX_POINTS_PER_CENTROID = 256;
// the threshold of the second round k-means, if the size of cluster is larger than this threshold, than do ((cluster size)/threshold)-means
constexpr static int SPLIT_THRESHOLD = 500;
// the max cluster number in hierarchical_cluster
constexpr static int MAX_CLUSTER_K2 = 500;

constexpr static int KMEANS_THRESHOLD = 2000;
// if cluster size smaller than SAME_SIZE_THRESHOLD , use same size kmeans or graph partition
constexpr static int SAME_SIZE_THRESHOLD = 5000;

constexpr static int MIN_SAME_SIZE_THRESHOLD = 500;

constexpr static int MAX_SAME_SIZE_THRESHOLD = 1500;


// file prefix strings and suffix strings
// file name rule: prefix + file_name + file_type + suffix

// prefix
constexpr const char* GLOBAL = "global-";
constexpr const char* CLUSTER = "cluster-";
constexpr const char* BUCKET = "bucket-";

// file_name
constexpr const char* HNSW = "hnsw-";
constexpr const char* PQ = "pq-";

// file type
constexpr const char* CENTROIDS = "centroids";
constexpr const char* PQ_CENTROIDS = "pq_centroids";
constexpr const char* COMBINE_IDS = "combine_ids";
constexpr const char* GLOBAL_IDS = "global_ids";
constexpr const char* CODEBOOK = "codebook";
constexpr const char* RAWDATA = "raw_data";
constexpr const char* SAMPLEDATA = "sampledata";
constexpr const char* META = "meta";
constexpr const char* INDEX = "index";

// suffix
constexpr const char* BIN = ".bin";
constexpr const char* TEXT = ".txt";
