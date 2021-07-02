#include "diskann.h"


// build disk-based index 
// strategy is hnsw + ivf + pq + refine

/*
 * args:
 * 1. data type(string): float or uint8 or int8
 * 2. raw data binary file(string): file name of raw data, include file path and file name
 * 3. index path(string): a string end with '/' denotes the directory that where the index related file locates
 * 4. M(int): parameter 4 hnsw
 * 5. efConstruction(int): parameter 4 hnsw
 * 6. PQ.M(int): the number of codebook 4 each vector
 * 7. PQ.nbits(int): the number of sub-cluster 4 PQ codebook
 * 8. metric type(string): metric type
 * 9. K1(int): number of centroids of the first round kmeans
 * 10. bucket threshold(int): the threshold of spliting the cluster
 */

int main(int argc, char** argv) {
    if (argc != 11) {
        std::cout << "Usage: << " << argv[0]
                  << " data_type(float or uint8 or int8)"
                  << " binary raw data file"
                  << " index output path"
                  << " hnsw.M"
                  << " hnsw.efConstruction"
                  << " PQ.M"
                  << " PQ.nbits"
                  << " metric type(L2 or IP)"
                  << " K1"
                  << " bucket split threshold"
                  << std::endl;
        return 1;
    }
    // parse parameters
    std::string raw_data_bin_file(argv[2]);
    std::string output_path(argv[3]);
    int hnswM = std::stoi(argv[4]);
    int hnswefC = std::stoi(argv[5]);
    int PQM = std::stoi(argv[6]);
    int PQnbits = std::stoi(argv[7]);
    auto metric_type = get_metric_type_by_name(std::string(argv[8]));
    int K1 = std::stoi(argv[9]);
    int threshold = std::stoi(argv[10]);
    assert(PQnbits == 8);

    if ('/' != *output_path.rbegin())
        output_path += '/';

    if (argv[1] == std::string("float")) {
        if (MetricType::L2 == metric_type) {
            build_bigann<float, float, CMax<float, uint32_t>>
                (raw_data_bin_file, output_path, hnswM, hnswefC, PQM, PQnbits, K1, threshold, metric_type);
        } else if (MetricType::IP == metric_type) {
            build_bigann<float, float, CMin<float, uint32_t>>
                (raw_data_bin_file, output_path, hnswM, hnswefC, PQM, PQnbits, K1, threshold, metric_type);
        }
    } else if (argv[1] == std::string("uint8")) {
        if (MetricType::L2 == metric_type) {
            build_bigann<uint8_t, uint32_t, CMax<uint32_t, uint32_t>>
                (raw_data_bin_file, output_path, hnswM, hnswefC, PQM, PQnbits, K1, threshold, metric_type);
        } else if (MetricType::IP == metric_type) {
            build_bigann<uint8_t, uint32_t, CMin<uint32_t, uint32_t>>
                (raw_data_bin_file, output_path, hnswM, hnswefC, PQM, PQnbits, K1, threshold, metric_type);
        }
    } else if (argv[1] == std::string("int8")) {
        if (MetricType::L2 == metric_type) {
            build_bigann<int8_t, int32_t, CMax<int32_t, uint32_t>>
                (raw_data_bin_file, output_path, hnswM, hnswefC, PQM, PQnbits, K1, threshold, metric_type);
        } else if (MetricType::IP == metric_type) {
            build_bigann<int8_t, int32_t, CMin<int32_t, uint32_t>>
                (raw_data_bin_file, output_path, hnswM, hnswefC, PQM, PQnbits, K1, threshold, metric_type);
        }
    }
    return 0;
}
