#include <iostream>
#include <string>
#include "util/defines.h"
#include "ivf/clusting.h"
#include "hnswlib/hnswlib.h"
#include "util/constants.h"
#include "util/utils.h""
#include "flat/flat.h"

template<typename T, typename T1, typename T2, typename R>
Computer<T1, T2, R> select_computer(MetricType metric_type) {
    switch (metric_type) {
        case MetricType::L2:
            break;
        case MetricType::IP:
            break;
    }
}


template<typename T>
void split_raw_data(const std::string& raw_data_file, const std::string& index_output_path,
                    const float* centroids, MetricType metric_type) {
    IOReader reader(raw_data_file);
    uint32_t nb, dim;
    reader.read((char*)&nb, sizeof(uint32_t));
    reader.read((char*)&dim, sizeof(uint32_t));
    std::vector<size_t > cluster_size(K1, 0);
    std::vector<std::ofstream> cluster_ids_writer(K1);
    std::vector<std::ofstream> cluster_dat_writer(K1);
    uint32_t placeholder = 1;
    uint32_t one = 1;
    for (auto i = 0; i < K1; i ++) {
        std::string cluster_raw_data_file_name = index_output_path + KMEANS1 + std::to_string(i) + "_" + RAWDATA + BIN;
        std::string cluster_ids_data_file_name = index_output_path + KMEANS1 + std::to_string(i) + "_" + IDS + BIN;
        cluster_dat_writer[i] = std::ofstream(cluster_raw_data_file_name, std::ios::binary);
        cluster_ids_writer[i] = std::ofstream(cluster_ids_data_file_name, std::ios::binary);
        cluster_dat_writer[i].write((char*)&placeholder, sizeof(uint32_t));
        cluster_dat_writer[i].write((char*)&dim, sizeof(uint32_t));
        cluster_ids_writer[i].write((char*)&placeholder, sizeof(uint32_t));
        cluster_ids_writer[i].write((char*)&one, sizeof(uint32_t));
        cluster_size[i] = 0;
    }
    size_t block_size = 1000000;
    size_t block_num = (nb - 1) / block_size + 1;
    std::vector<size_t> cluster_id(block_size);
    std::vector<T> dists(block_size);
    T* block_buf = new T[block_size * dim];
    for (auto i = 0; i < block_num; i ++) {
        auto sp = i * block_size;
        auto ep = std::min(nb, sp + block_size);
        reader.read((char*)block_buf, (ep - sp) * dim * sizeof(T));
        knn_2<CMin<T, size_t>, float, T>(centroids, block_buf, K1, ep - sp, dim, 1, dists.data(), cluster_id.data(), select_computer<T>(metric_type));
    }

    delete[] block_buf;
    block_buf = nullptr;
}

template<typename T>
bool build_disk_index(const std::string& raw_data_file, const std::string& index_output_path,
                      const std::string& index_build_parameters,
                      MetricType metric_type = MetricType::L2) {


    size_t nb, dim;
    get_bin_metadata(raw_data_file, size_t &nb, size_t &dim);
    std::cout << "read meta from " << raw_data_file << ", nb = " << nb << "dim = " << dim << std::endl;
    size_t sample_num = (size_t)(nb * K1_SAMPLE_RATE);
    T* sample_data;
    sample_data = new T[sample_num * dim];
    reservoir_sampling(raw_data_file, sample_num, sample_data);
    float* centroids = new float[dim * K1];
    kmeans<T>(sample_num, sample_data, (int32_t)dim, K1, centroids);
    delete[] sample_data;
    sample_data = nullptr;

    split_raw_data<T>(raw_data_file, index_output_path, centroids, metric_type);

    delete[] centroids;
    centroids = nullptr;
}


