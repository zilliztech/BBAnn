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
 * 6. metric type(string): metric type
 * 7. K1(int): number of centroids of the first round kmeans
 */

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cout << "Usage: << " << argv[0]
                  << " data_type(float or uint8 or int8)"
                  << " binary raw data file"
                  << " index output path"
                  << " hnsw.M"
                  << " hnsw.efConstruction"
                  << " metric type(L2 or IP)"
                  << " K1"
                  << std::endl;
        return 1;
    }

    // parse parameters
    std::string raw_data_bin_file(argv[2]);
    std::string output_path(argv[3]);
    int hnswM = std::stoi(argv[4]);
    int hnswefC = std::stoi(argv[5]);
    auto metric_type = get_metric_type_by_name(std::string(argv[6]));
    int K1 = std::stoi(argv[7]);

    if ('/' != *output_path.rbegin())
        output_path += '/';

    if (argv[1] == std::string("float")) {
        if (MetricType::L2 == metric_type) {
            build_bbann<float, float, CMax<float, uint32_t>>
                (raw_data_bin_file, output_path, hnswM, hnswefC, K1, metric_type);
        } else if (MetricType::IP == metric_type) {
            build_bbann<float, float, CMin<float, uint32_t>>
                (raw_data_bin_file, output_path, hnswM, hnswefC, K1, metric_type);
        }
    } else if (argv[1] == std::string("uint8")) {
        if (MetricType::L2 == metric_type) {
            build_bbann<uint8_t, uint32_t, CMax<uint32_t, uint32_t>>
                (raw_data_bin_file, output_path, hnswM, hnswefC, K1, metric_type);
        } else if (MetricType::IP == metric_type) {
            build_bbann<uint8_t, uint32_t, CMin<uint32_t, uint32_t>>
                (raw_data_bin_file, output_path, hnswM, hnswefC, K1, metric_type);
        }
    } else if (argv[1] == std::string("int8")) {
        if (MetricType::L2 == metric_type) {
            build_bbann<int8_t, int32_t, CMax<int32_t, uint32_t>>
                (raw_data_bin_file, output_path, hnswM, hnswefC, K1, metric_type);
        } else if (MetricType::IP == metric_type) {
            build_bbann<int8_t, int32_t, CMin<int32_t, uint32_t>>
                (raw_data_bin_file, output_path, hnswM, hnswefC, K1, metric_type);
        }
    }
    return 0;
}
