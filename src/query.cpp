#include <iostream>
#include <vector>
#include <mutex>
#include "util/defines.h"
#include "util/constants.h"


// parameters: 
//      DataType    data_type(float or uint8)
//      string      query_bin_file
//      string      answer_bin_file
//      int         topk
//      int         nprobe
template<typename T>
void search_disk_index_simple(const std::string& index_path, const std::string& search_parameters,
                       MetricType metric_type = MetricType::L2) {

    std::stringstream parser;
    parser << std::string(search_parameters);
    std::string              cur_param;
    std::vector<std::string> param_list;
    while (parser >> cur_param)
        param_list.push_back(cur_param);

    // parameters
    DataType data_type = std::stoi(param_list[0];
    int topk = std::stoi(param_list[3]);
    int nprobe = std::stoi(param_list[4]);
    int refine_topk = topk;

    // files
    std::string hnsw_index_file = index_path + HNSW + INDEX + BIN;
    std::string pq_centroids_file = index_path + PQ + PQ_CENTROIDS + BIN;
    std::string pq_codebook_file = index_path + PQ + CODEBOOK + BIN;
    std::string query_bin_file = param_list[1];
    std::string answer_bin_file = param_list[2];

    // variables
    uint32_t num_queries, num_base, num_pq_centroids, num_pq_codebook;
    uint32_t dim_queries, dim_base, dim_pq_centroids, dim_pq_codebook;
    num_base = 0;

    hnswlib::SpaceInterface<float>* space;
    if (MetricType::L2 == metric_type) {
        space = new hnswlib::L2Space(dim);
    } else if (MetricType::IP == metric_type) {
        space = new hnswlib::InnerProductSpace(dim);
    } else {
        std::cout << "invalid metric_type = " << metric_type << std::endl;
        return;
    }

    T* pquery = nullptr;
    float* pq_centroids = nullptr;
    // read bin files
    read_bin_file<T>(query_bin_file, pquery, num_queries, dim_queries);
    // dim_pq_centroids = dim_base / PQM, num_pq_centroids = PQM * 256
    read_bin_file<float>(pq_centroids_file, pq_centroids, num_pq_centroids, dim_pq_centroids);

    // if T is uint8_t, distance type is uint32_t, force transfer to uint32_t from float, size is the same
    if (DataType::FLOAT == data_type) {
        // std::vector<std::vector<std::pair<float, uint32_t>>> 
        //    answers(num_queries, std::vector<std::pair<float, uint32_t>>(topk, std::pair<float, uint32_t>(0.0, 0)));
        float* answer_dists = new float[num_queries * topk];
        uint32_t* answer_ids = new uint32_t[num_queries * topk];
        float* pq_distance = new float[num_queries * refine_topk];
        CMin<float, uint32_t> heap_comare_class;
        auto dis_computer = select_computer<T, T, T>(metric_type);
    } else if (DataType::INT8 == data_type) {
        // std::vector<std::vector<std::pair<uint32_t, uint32_t>>> 
        //    answers(num_queries, std::vector<std::pair<uint32_t, uint32_t>>(topk, std::pair<uint32_t, uint32_t>(0, 0)));
        uint32_t* answer_dists = new uint32_t[num_queries * topk];
        uint32_t* answer_ids = new uint32_t[num_queries * topk];
        uint32_t* pq_distance = new uint32_t[num_queries * refine_topk];
        CMin<uint32_t, uint32_t> heap_comare_class;
        auto dis_computer = select_computer<T, T, int32_t>(metric_type);
    } else {
        std::cout << "error! invalid data_type = " << data_type << std::endl;
        return;
    }
    uint64_t* pq_offsets = new uint64_t[num_queries * refine_topk];

    // in-memory data
    std::vector<std::vector<uint32_t>> metas(K1); // bucket size of each cluster
    std::vector<std::vector<uint8_t>> pq_codebook(K1); // pq codebook

    auto index_hnsw = std::make_shared<hnswlib::HierarchicalNSW<float>>(space, hnsw_index_file);

    uint32_t PQM = dim_queries / dim_pq_centroids;

    for (auto i = 0; i < K1; i ++) {
        IOReader meta_reader(index_path + CLUSTER + std::to_string(i) + META + BIN, MEGABYTE * 10);
        uint32_t cluster_sizei, meta_dimi; // cluster_sizei is the number of buckets in cluster i
        meta_reader.read((char*)&cluster_sizei, sizeof(uint32_t));
        meta_reader.read((char*)&meta_dimi, sizeof(uint32_t));
        assert(meta_dimi == 1);
        metas[i].resize(cluster_sizei);
        meta_reader.read((char*)metas[i].data(), cluster_sizei * sizeof(uint32_t));
        IOReader pq_codebook_reader(index_path + CLUSTER + std::to_string(i) + PQ + CODEBOOK + BIN, MEGABYTE * 10);
        uint32_t pq_codebook_sizei, pqmi; // pq_codebook_sizei is the number of vectors in cluster i
        pq_codebook_reader.read((char*)&pq_codebook_sizei, sizeof(uint32_t));
        pq_codebook_reader.read((char*)&pqmi, sizeof(uint32_t));
        // assert(pq_codebook_sizei == cluster_sizei);
        assert(pqmi == PQM);
        pq_codebook[i].resize(cluster_sizei * pqmi);
        pq_codebook_reader.read((char*)pq_codebook[i].data(), cluster_sizei * pqmi * sizeof(uint8_t));
        num_base += pq_codebook_sizei;
    }
    std::cout << "num_base = " << num_base << std::endl;

    // do query
    // step1: select nprobe buckets

    uint64_t* p_labels = new uint64_t[num_queries * nprobe];
    // float* p_dists = new float[num_queries * nprobe];
#pragma omp parallel for
    for (auto i = 0; i < num_queries; i ++) {
        auto queryi = pquery + i * dim_queries;
        auto reti = index_hnsw->searchKnn(queryi, nprobe);
        auto p_labeli = p_labels + i * nprobe;
        int retnum = reti.size() - 1;
        while (!reti.empty()) {
            p_labeli[retnum] = reti.top().second;
            reti.pop();
            retnum --;
        }
    }

    // step2: pq search
#pragma omp parallel for
    for (auto i = 0; i < num_queries; i ++) {
        auto p_labeli = p_labels + i * nprobe;
        auto pq_offseti = pq_offsets + i * refine_topk;
        auto pq_distancei = pq_distance + i * refine_topk;
        uint32_t cid, bid, off;
        // todo: invoke xuweizi's pq interface
        ;
    }

    // refine
    std::sort(pq_offsets, pq_offsets + refine_topk * num_queries);
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> refine_records(K1);
    for (auto j = 0; j < refine_topk * num_queries; j ++) {
        auto cid, off, qid;
        parse_refine_id(pq_offsets[j], cid, off, qid);
        refine_records[cid].push_back(off, qid);
    }
    std::vector<std::ifstream> raw_data_file_handlers(K1);
    std::vector<std::ifstream> ids_data_file_handlers(K1);
    for (auto i = 0; i < K1; i ++) {
        std::string data_filei = index_output_path + CLUSTER + std::to_string(i) + RAWDATA + BIN;
        std::string ids_filei  = index_output_path + CLUSTER + std::to_string(i) + GLOBAL_IDS + BIN;
        raw_data_file_handlers[i] = std::ifstream(data_filei, std::binary);
        ids_data_file_handlers[i] = std::ifstream(ids_filei , std::binary);
    }

    // init answer heap
    for (auto i = 0; i < num_queries; i ++) {
        auto ans_disi = answer_dists + topk * i;
        auto ans_idsi = answer_ids + topk * i;
        heap_heapify<heap_comare_class>(topk, ans_disi, ans_idsi);
    }

    std::vector<std::mutex> mtx(num_queries);
#pragma omp parallel for
    for (auto i = 0; i < K1; i ++) {
        if (refine_records[i].size() == 0)
            continue;
        uint32_t pre_qid = num_queries + 1;
        uint32_t meta_bytes = 8; // pass meta
        T* data_bufi = new T[dim_queries];
        uint32_t global_id;
        for (auto j = 0; j < refine_records[i].size(); j ++) {
            if (refine_records[i][j].second != pre_qid) {
                pre_qid = refine_records[i][j].second;
                raw_data_file_handlers[i].seekp(meta_bytes + refine_records[i][j].first * dim_queries * sizeof(T));
                raw_data_file_handlers[i].read((char*)data_bufi, dim_queries * sizeof(T));
                ids_data_file_handlers[i].seekp(meta_bytes + refine_records[i][j].first * sizeof(uint32_t));
                ids_data_file_handlers[i].read((char*)&global_id, sizeof(uint32_t));
                assert(global_id >= 0);
                assert(global_id < num_base);
            }
            auto dis = dis_computer(data_bufi, pquery + pre_qid, dim_queries);
            std::unique<std::mutex> lk(mtx[pre_qid]);
            if (heap_comare_class::cmp(answer_dists + topk * pre_qid, dis)) {
                heap_swap_top<heap_comare_class>(topk, answer_dists + topk * pre_qid, answer_ids + topk * pre_qid, dis, global_id);
            }
        }

        delete[] data_bufi;
    }

    // write answers
    uint32_t ans_num = num_queries * topk;
    uint32_t ans_dim = 2;
    std::ofstream answer_writer(answer_bin_file, std::ios::binary);
    answer_writer.write((char*)&ans_num, sizeof(uint32_t));
    answer_writer.write((char*)&ans_dim, sizeof(uint32_t));
    for (auto i = 0; i < num_queries; i ++) {
        auto ans_disi = answer_dists + topk * i;
        auto ans_idsi = answer_ids + topk * i;
        for (int j = topk; j > 0; j --) {
            answer_writer.write((char*)ans_disi, 4);
            answer_writer.write((char*)ans_idsi, 4);
            heap_pop<heap_comare_class>(j, ans_disi, ans_idsi);
        }
    }
    answer_writer.close();

    for (auto i = 0; i < K1; i ++) {
        std::string data_filei = index_output_path + CLUSTER + std::to_string(i) + RAWDATA + BIN;
        std::string ids_filei  = index_output_path + CLUSTER + std::to_string(i) + GLOBAL_IDS + BIN;
        raw_data_file_handlers[i].close();
        ids_data_file_handlers[i].close();
    }

    delete[] pquery;
    pquery = nullptr;
    delete[] pq_centroids;
    pq_centroids = nullptr;
    delete[] p_labels;
    p_labels = nullptr;
    // delete[] p_dists;
    // p_dists = nullptr;
    delete[] pq_distance;
    pq_distance = nullptr;
    delete[] pq_offsets;
    pq_offsets = nullptr;
    delete[] answer_ids;
    answer_ids = nullptr;
    delete[] answer_dists;
    answer_dists = nullptr;
}


