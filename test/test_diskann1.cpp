#include <iostream>
#include "diskann.h"

// this file test the disk-based bigann
// strategy is hnsw + ivf + pq + refine

/*
 * args:
 * 1. data type(string): float or uint8
 * 2. raw data binary file(string): file name of raw data, include file path and file name
 * 3. index path(string): a string end with '/' denotes the directory that where the index related file locates
 * 4. M(int): parameter 4 hnsw
 * 5. efConstruction(int): parameter 4 hnsw
 * 6. PQ.M(int): the number of codebook 4 each vector
 * 7. PQ.nbits(int): the number of sub-cluster 4 PQ codebook
 * 8. query data binary file(string): file name of query data, include file path and file name
 * 9. answer data binary file(string): file name of answer data, include file path and file name
 * 10. topk(int): the number of answers 4 each query
 * 11. nprobe(int): the number of buckets 4 index to search
 * 12. metric type(string): metric type
 */

int main(int argc, char** argv) {
    if (argc != 13) {
        std::cout << "Usage: << " << argv[0]
                  << " data_type(float or uint8)"
                  << " binary raw data file"
                  << " index output path"
                  << " hnsw.M"
                  << " hnsw.efConstruction"
                  << " PQ.M"
                  << " PQ.nbits"
                  << " binary query data file"
                  << " file to save answer"
                  << " topk"
                  << " nprobe"
                  << " metric type(L2 or IP)"
                  << std::endl;
        return 1;
    }
    // main parameters
    std::string raw_data_file(argv[2]);
    std::string output_path(argv[3]);
    int hnswM = std::stoi(argv[4]);
    int hnswefC = std::stoi(argv[5]);
    int PQM = std::stoi(argv[6]);
    int PQnbits = std::stoi(argv[7]);
    std::string query_data_file(argv[8]);
    std::string answer_file(argv[9]);
    int topk = std::stoi(argv[10]);
    int nprobe = std::stoi(argv[11]);
    std::string metric_type_str(argv[12]);
    auto metric_type = get_metric_type_by_name(metric_type_str);
    if (MetricType::None == metric_type) {
        std::cout << "invalid metric_type = " << metric_type_str << std::endl;
        return 1;
    }

    TimeRecorder rc("main");
    // todo: split search into two functions: load and query
    if (argv[1] == std::string("float")) {
        build_disk_index<float, float>
            (raw_data_file, output_path, hnswM, hnswefC, PQM, PQnbits, metric_type);

        rc.RecordSection("build_disk_index with data type float done");

        search_disk_index_simple<float, float>
            (output_path, query_data_file, answer_file, topk, nprobe, PQM, PQnbits, metric_type);
    } else if (argv[1] == std::string("uint8")) {
        build_disk_index<uint8_t, uint32_t>
            (raw_data_file, output_path, hnswM, hnswefC, PQM, PQnbits, metric_type);

        rc.RecordSection("build_disk_index with data type uint8 done");
        search_disk_index_simple<uint8_t, uint32_t>
            (output_path, query_data_file, answer_file, topk, nprobe, PQM, PQnbits, metric_type);
    }
    rc.ElapseFromBegin("main done");
    return 0;
}
