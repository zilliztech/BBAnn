#include <iostream>
#include <algorithm>
#include <string>
#include "util/defines.h"
#include "ivf/clusting.h"
#include "hnswlib/hnswlib.h"
#include "util/constants.h"
#include "util/utils.h""
#include "flat/flat.h"

template<typename T1, typename T2, typename R>
Computer<T1, T2, R> select_computer(MetricType metric_type) {
    switch (metric_type) {
        case MetricType::L2:
            return L2sqr<T1, T2, R>;
            break;
        case MetricType::IP:
            return IP<T1, T2, R>;
            break;
    }
}


template<typename T>
void split_raw_data(const std::string& raw_data_file, const std::string& index_output_path,
                    const float* centroids, MetricType metric_type, DataType data_type) {
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
        std::string cluster_raw_data_file_name = index_output_path + CLUSTER + std::to_string(i) + "_" + RAWDATA + BIN;
        std::string cluster_ids_data_file_name = index_output_path + CLUSTER + std::to_string(i) + "_" + GLOBAL_IDS + BIN;
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
        if (DataType::FLOAT == data_type) {
            knn_2<CMin<T, size_t>, float, T> (
                centroids, block_buf, K1, ep - sp, dim, 1, 
                dists.data(), cluster_id.data(), select_computer<T, T, T>(metric_type));
        } else if (DataType::INT8 == data_type) {
            knn_2<CMin<int32_t, size_t>, float, T> (
                centroids, block_buf, K1, ep - sp, dim, 1, 
                dists.data(), cluster_id.data(), select_computer<T, T, int32_t>(metric_type));
        }
        for (auto j = 0; j < ep - sp; j ++) {
            auto cid = cluster_id[j];
            auto uid = (uint32_t)(j + sp);
            cluster_dat_writer[cid].write((char*)(block_buf + j * dim), sizeof(T) * dim);
            cluster_ids_writer[cid].write((char*)&uid, sizeof(uint32_t));
            cluster_size[cid] ++;
        }
    }
    size_t sump = 0;
    std::cout << "split_raw_data done in ... seconds, show statistics:" << std::endl;
    for (auto i = 0; i < K1; i ++) {
        uint32_t cis = cluster_size[i];
        cluster_dat_writer[i].seekp(0);
        cluster_dat_writer[i].write((char*)&cis, sizeof(uint32_t));
        cluster_dat_writer[i].close();
        cluster_ids_writer[i].seekp(0);
        cluster_ids_writer[i].write((char*)&cis, sizeof(uint32_t));
        cluster_ids_writer[i].close();
        std::cout << "cluster-" << i << " has " << cis << " points." << std::endl;
        sump += cis;
    }
    std::cout << "total points num: " << sump << std::endl;

    delete[] block_buf;
    block_buf = nullptr;
}

template<typename T>
void train_clusters(const std::string& cluster_path, uint32_t& graph_nb, uint32_t& graph_dim, MetricType metric_type, DataType data_type) {
    std::vector<uint32_t> cluster_id;
    std::vector<T> dists;
    uint32_t placeholder = 0;
    // centroids file of each buckets
    std::string bucket_centroids_file = cluster_path + CLUSTER + "_" + CENTROIDS + BIN;
    // centroid_id of each buckets, each of which is cid + bid + offset
    std::string bucket_ids_file = cluster_path + CLUSTER + "_" + COMBINE_IDS + BIN;
    IOWriter bucket_ids_writer(bucket_ids_file, MEGABYTE * 10);
    IOWriter bucket_ctd_writer(bucket_centroids_file, MEGABYTE * 100);
    bucket_ctd_writer.write((char*)&placeholder, sizeof(uint32_t));
    bucket_ctd_writer.write((char*)&placeholder, sizeof(uint32_t));
    bucket_ids_writer.write((char*)&placeholder, sizeof(uint32_t));
    bucket_ids_writer.write((char*)&placeholder, sizeof(uint32_t));
    graph_nb = 0;
    uint32_t bucket_id_dim = 1;
    for (auto i = 0; i < K1; i ++) {
        // raw_data_file, read by split order, write by buckets
        std::string data_file = cluster_path + CLUSTER + std::to_string(i) + "_" + RAWDATA + BIN;
        // global id file, read by split order, write by buckets
        std::string ids_file  = cluster_path + CLUSTER + std::to_string(i) + "_" + GLOBAL_IDS + BIN;
        // meta_file, record the size of each bucket in cluster i
        std::string meta_file = cluster_path + CLUSTER + std::to_string(i) + "_" + META + BIN;
    
        uint32_t cluster_size, cluster_dim, ids_size, ids_dim;
        IOReader data_reader(data_file);
        IOReader ids_reader(ids_file);
        data_reader.read((char*)&cluster_size, sizeof(uint32_t));
        data_reader.read((char*)&cluster_dim, sizeof(uint32_t));
        ids_reader.read((char*)&ids_size, sizeof(uint32_t));
        ids_reader.read((char*)&ids_dim, sizeof(uint32_t));
        assert(cluster_size == ids_size);
        assert(ids_dim == 1);
        T* datai = new T[cluster_size * cluster_dim];
        data_reader.read((char*)datai, cluster_size * cluster_dim * sizeof(T));
        uint32_t* idsi = new uint32_t[ids_size * ids_dim];
        ids_reader.read((char*)idsi, ids_size * ids_dim * sizeof(uint32_t));
        data_reader.close();
        ids_reader.close();

        auto K2 = cluster_size / SPLIT_THRESHOLD;
        std::cout << "cluster-" << i << " will split into " << K2 << " buckets." << std::endl;
        float* centroids_i = new float[K2 * cluster_dim];
        kmeans<T>(cluster_size, datai, (int32_t)cluster_dim, K2, centroids_i);
        cluster_id.resize(cluster_size);
        dists.resize(cluster_size);
        if (DataType::FLOAT == data_type) {
            knn_2<CMin<T, uint32_t>, float, T> (
                centroids, block_buf, K1, ep - sp, dim, 1, 
                dists.data(), cluster_id.data(), select_computer<T, T, T>(metric_type));
        } else if (DataType::INT8 == data_type) {
            knn_2<CMin<int32_t, uint32_t>, float, T> (
                centroids, block_buf, K1, ep - sp, dim, 1, 
                dists.data(), cluster_id.data(), select_computer<T, T, int32_t>(metric_type));
        }
        std::vector<uint32_t> buckets_size(K2 + 1, 0);
        std::vector<std::pair<uint32_t, uint32_t>> cluster_off;
        cluster_off.resize(cluster_size);
        for (auto j = 0; j < cluster_size; j ++) {
            buckets_size[cluster_id[j] + 1] ++;
        }

        // write meta file
        write_bin_file<uint32_t>(meta_file, buckets_size.data(), cluster_size, 1);

        for (auto j = 1; j <= cluster_size; j ++) {
            buckets_size[j] += buckets_size[j - 1];
        }
        for (auto j = 0; j < cluster_size; j ++) {
            cluster_off[j].first = buckets_size[cluster_id[j]] ++;
            cluster_off[j].second = j;
        }
        std::sort(cluster_off.begin(), cluster_off.end(), [](const auto &l, const auto &r) {
                return l.first < r.first;
                });

        // rewrite raw_data and global ids by bucket order
        IOWriter data_writer(data_file, MEGABYTE * 100);
        IOWriter ids_writer(ids_file, MEGABYTE * 10);
        data_writer.write((char*)&cluster_size, sizeof(uint32_t));
        data_writer.write((char*)&cluster_dim, sizeof(uint32_t));
        ids_writer.write((char*)&ids_size, sizeof(uint32_t));
        ids_writer.write((char*)&ids_dim, sizeof(uint32_t));
        for (auto j = 0; j < cluster_size; j ++) {
            auto ori_pos = cluster_off[j].second;
            data_writer.write((char*)(datai + ori_pos * cluster_dim), sizeof(T) * cluster_dim);
            ids_writer.write((char*)(idsi + ori_pos * ids_dim), sizeof(uint32_t) * ids_dim);
        }
        data_writer.close();
        ids_writer.close();

        // write buckets's centroids and combine ids
        // write_bin_file<float>(bucket_centroids_file, centroids_i, K2, cluster_dim);
        bucket_ctd_writer.write((char*)centroids_i, sizeof(float) * K2 * cluster_dim);
        for (auto j = 0; j < K2; j ++) {
            uint64_t gid = gen_id(i, j, buckets_size[j]);
            bucket_ids_writer.write((char*)&gid, sizeof(uint64_t));
        }

        graph_nb += K2;
        graph_dim = cluster_dim;

        delete[] datai;
        delete[] idsi;
        delete[] centroids_i;
    }
    bucket_ctd_writer.close();
    bucket_ids_writer.close();
    std::cout << "total bucket num = " << graph_nb << std::endl;
    set_bin_metadata(bucket_centroids_file, graph_nb, graph_dim);
    set_bin_metadata(bucket_ids_file, graph_nb, bucket_id_dim);
}

void create_graph_index(const std::string& index_path, std::vector<std::string>& params, MetricType metric_type) {
    hnswlib::SpaceInterface<float>* space;
    if (MetricType::L2 == metric_type) {
        space = new hnswlib::L2Space();
    } else if (MetricType::IP == metric_type) {
    } else {
        std::cout << "invalid metric_type = " << metric_type << std::endl;
        return;
    }
}

template<typename T>
bool build_disk_index(const std::string& raw_data_file, const std::string& index_output_path,
                      const std::string& index_build_parameters,
                      MetricType metric_type = MetricType::L2) {

    std::stringstream parser;
    parser << std::string(index_build_parameters);
    std::string              cur_param;
    std::vector<std::string> param_list;
    while (parser >> cur_param)
      param_list.push_back(cur_param);


    auto data_type = (DataType)std::stoi(param_list[0]);
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

    split_raw_data<T>(raw_data_file, index_output_path, centroids, metric_type, data_type);

    size_t graph_nb, graph_dim;
    train_clusters<T>(index_output_path, graph_nb, graph_dim, metric_type, data_type);

    create_graph_index(index_output_path, param_list, metric_type); // hard code hnsw

    delete[] centroids;
    centroids = nullptr;
    return true;
}


