#pragma once
#include <string>


// constant defines

// units
constexpr static int KILOBYTE = 1024;
constexpr static int MEGABYTE = 1024 * 1024;
constexpr static int GIGABYTE = 1024 * 1024 * 1024;

// num of clusters in the first round k-means
// constexpr static int K1 = 10;
// sample rate of the first round k-means
constexpr static float K1_SAMPLE_RATE = 0.01;
// sample rate of the pq train set
constexpr static float PQ_SAMPLE_RATE = 0.1;
// the threshold of the second round k-means, if the size of cluster is larger than this threshold, than do ((cluster size)/threshold)-means
constexpr static int SPLIT_THRESHOLD = 500;




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

