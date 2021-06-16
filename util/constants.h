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
constexpr static std::string KMEANS1 = "KMEANS_ROUND1";
constexpr static std::string KMEANS2 = "KMEANS_ROUND2";
constexpr static std::string CENTROIDS = "CENTROIDS";

// file type
constexpr static std::string CENTROIDS = "CENTROIDS";
constexpr static std::string PQ_CENTROIDS = "PQ_CENTROIDS";
constexpr static std::string IDS = "IDS";
constexpr static std::string CODEBOOK = "CODEBOOK";
constexpr static std::string RAWDATA = "RAWDATA";
constexpr static std::string SAMPLEDATA = "SAMPLEDATA";
constexpr static std::string META = "META";

// suffix
constexpr static std::string BIN = "BIN";
constexpr static std::string TEXT = "TEXT";

