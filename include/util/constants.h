#pragma once
#include <string>


// constant defines

// units
constexpr static int KILOBYTE = 1024;
constexpr static int MEGABYTE = 1024 * 1024;
constexpr static int GIGABYTE = 1024 * 1024 * 1024;

// num of clusters in the first round k-means
constexpr static int K1 = 64;
// sample rate of the first round k-means
constexpr static float K1_SAMPLE_RATE = 0.01;
// the threshold of the second round k-means, if the size of cluster is larger than this threshold, than do ((cluster size)/threshold)-means
constexpr static int SPLIT_THRESHOLD = 500;




// file prefix strings and suffix strings
// file name rule: prefix + file_name + file_type + suffix

// prefix
constexpr static std::string CLUSTER = "cluster-";
constexpr static std::string BUCKET = "bucket-";

// file_name
constexpr static std::string HNSW = "hnsw";
constexpr static std::string PQ = "pq";

// file type
constexpr static std::string CENTROIDS = "_centroids";
constexpr static std::string PQ_CENTROIDS = "_pq_centroids";
constexpr static std::string COMBINE_IDS = "_combine_ids";
constexpr static std::string GLOBAL_IDS = "_global_ids";
constexpr static std::string CODEBOOK = "_codebook";
constexpr static std::string RAWDATA = "_raw_data";
constexpr static std::string SAMPLEDATA = "_sampledata";
constexpr static std::string META = "_meta";
constexpr static std::string INDEX = "_index";

// suffix
constexpr static std::string BIN = ".bin";
constexpr static std::string TEXT = ".txt";

