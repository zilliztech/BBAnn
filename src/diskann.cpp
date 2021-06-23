#include "diskann.h"

template<typename DATAT, typename DISTT>
void split_raw_data(const std::string& raw_data_file, const std::string& index_output_path,
                    float* centroids, MetricType metric_type) {
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
        std::string cluster_raw_data_file_name = index_output_path + CLUSTER + std::to_string(i) + RAWDATA + BIN;
        std::string cluster_ids_data_file_name = index_output_path + CLUSTER + std::to_string(i) + GLOBAL_IDS + BIN;
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
    std::vector<DISTT> dists(block_size);
    DATAT* block_buf = new DATAT[block_size * dim];
    for (auto i = 0; i < block_num; i ++) {
        auto sp = i * block_size;
        auto ep = std::min((size_t)nb, sp + block_size);
        reader.read((char*)block_buf, (ep - sp) * dim * sizeof(DATAT));
        knn_2<CMin<DISTT, size_t>, float, DATAT> (
            centroids, block_buf, K1, ep - sp, dim, 1, 
            dists.data(), cluster_id.data(), select_computer<float, DATAT, DISTT>(metric_type));

        for (auto j = 0; j < ep - sp; j ++) {
            auto cid = cluster_id[j];
            auto uid = (uint32_t)(j + sp);
            cluster_dat_writer[cid].write((char*)(block_buf + j * dim), sizeof(DATAT) * dim);
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

template<typename DATAT, typename DISTT>
void train_clusters(const std::string& cluster_path, uint32_t& graph_nb, uint32_t& graph_dim, 
                    ProductQuantizer<CMin<DISTT, uint32_t>, DATAT, uint8_t>* pq_quantizer,
                    MetricType metric_type) {
    std::vector<uint32_t> cluster_id;
    std::vector<DISTT> dists;
    uint32_t placeholder = 0;
    // centroids file of each buckets, the input of graph index
    std::string bucket_centroids_file = cluster_path + BUCKET + CENTROIDS + BIN;
    // centroids file of pq quantizer
    std::string pq_centroids_file = cluster_path + PQ + PQ_CENTROIDS + BIN;
    // centroid_id of each buckets, each of which is cid + bid + offset
    std::string bucket_ids_file = cluster_path + CLUSTER + COMBINE_IDS + BIN;

    // save pq centroids
    pq_quantizer->save_centroids(pq_centroids_file);

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
        std::string data_file = cluster_path + CLUSTER + std::to_string(i) + RAWDATA + BIN;
        // global id file, read by split order, write by buckets
        std::string ids_file  = cluster_path + CLUSTER + std::to_string(i) + GLOBAL_IDS + BIN;
        // meta_file, record the size of each bucket in cluster i
        std::string meta_file = cluster_path + CLUSTER + std::to_string(i) + META + BIN;
        // pq_codebook_file, save codebook of each cluster
        std::string pq_codebook_file = cluster_path + CLUSTER + std::to_string(i) + PQ + CODEBOOK + BIN;
    
        uint32_t cluster_size, cluster_dim, ids_size, ids_dim;
        IOReader data_reader(data_file);
        IOReader ids_reader(ids_file);
        data_reader.read((char*)&cluster_size, sizeof(uint32_t));
        data_reader.read((char*)&cluster_dim, sizeof(uint32_t));
        ids_reader.read((char*)&ids_size, sizeof(uint32_t));
        ids_reader.read((char*)&ids_dim, sizeof(uint32_t));
        assert(cluster_size == ids_size);
        assert(ids_dim == 1);
        DATAT* datai = new DATAT[cluster_size * cluster_dim];
        data_reader.read((char*)datai, cluster_size * cluster_dim * sizeof(DATAT));
        uint32_t* idsi = new uint32_t[ids_size * ids_dim];
        ids_reader.read((char*)idsi, ids_size * ids_dim * sizeof(uint32_t));

        auto K2 = cluster_size / SPLIT_THRESHOLD;
        std::cout << "cluster-" << i << " will split into " << K2 << " buckets." << std::endl;
        float* centroids_i = new float[K2 * cluster_dim];
        kmeans<DATAT>(cluster_size, datai, (int32_t)cluster_dim, K2, centroids_i);
        cluster_id.resize(cluster_size);
        dists.resize(cluster_size);
        knn_2<CMin<DISTT, uint32_t>, float, DATAT> (
            centroids_i, datai, K2, cluster_size, cluster_dim, 1, 
            dists.data(), cluster_id.data(), select_computer<float, DATAT, DISTT>(metric_type));
        
        std::vector<uint32_t> buckets_size(K2 + 1, 0);
        std::vector<std::pair<uint32_t, uint32_t>> cluster_off;
        cluster_off.resize(cluster_size);
        for (auto j = 0; j < cluster_size; j ++) {
            buckets_size[cluster_id[j] + 1] ++;
        }

        // write meta file
        write_bin_file<uint32_t>(meta_file, &buckets_size[1], K2, 1);

        for (auto j = 1; j <= K2; j ++) {
            buckets_size[j] += buckets_size[j - 1];
        }

        // write buckets's centroids and combine ids
        // write_bin_file<float>(bucket_centroids_file, centroids_i, K2, cluster_dim);
        bucket_ctd_writer.write((char*)centroids_i, sizeof(float) * K2 * cluster_dim);

        for (auto j = 0; j < K2; j ++) {
            uint64_t gid = gen_id(i, j, buckets_size[j]);
            bucket_ids_writer.write((char*)&gid, sizeof(uint64_t));
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
            data_writer.write((char*)(datai + ori_pos * cluster_dim), sizeof(DATAT) * cluster_dim);
            ids_writer.write((char*)(idsi + ori_pos * ids_dim), sizeof(uint32_t) * ids_dim);
        }

        // load buckets_size from meta_file again
        // uint32_t meta_numi, meta_dimi;
        // read_bin_file<uint32_t>(meta_file, buckets_size.data(), meta_numi, meta_dimi);
        // assert(meta_numi == K2);
        // assert(meta_dimi == 1);
        graph_nb += K2;
        graph_dim = cluster_dim;

        // load rearranged raw_data again and encode them
        IOReader data_reader2(data_file);
        data_reader2.read((char*)&cluster_size, sizeof(uint32_t));
        data_reader2.read((char*)&cluster_dim, sizeof(uint32_t));
        data_reader2.read((char*)datai, cluster_size * cluster_dim * sizeof(DATAT));
        pq_quantizer->encode_vectors_and_save(cluster_size, datai, pq_codebook_file);

        delete[] datai;
        delete[] idsi;
        delete[] centroids_i;
    }

    std::cout << "total bucket num = " << graph_nb << std::endl;
    set_bin_metadata(bucket_centroids_file, graph_nb, graph_dim);
    set_bin_metadata(bucket_ids_file, graph_nb, bucket_id_dim);
}

void create_graph_index(const std::string& index_path, 
                        const int hnswM, const int hnswefC,
                        MetricType metric_type) {

    std::cout << "parameters of create hnsw: M = " << hnswM << ", efConstruction = " << hnswefC << std::endl;
    float* pdata = nullptr;
    uint64_t* pids = nullptr;
    uint32_t npts, ndim, nids, nidsdim;
    read_bin_file<float>(index_path + BUCKET + CENTROIDS + BIN, pdata, npts, ndim);
    std::cout << "there are " << npts << " of dimension " << ndim << " points of hnsw" << std::endl;
    assert(pdata != nullptr);
    hnswlib::SpaceInterface<float>* space;
    if (MetricType::L2 == metric_type) {
        space = new hnswlib::L2Space(ndim);
    } else if (MetricType::IP == metric_type) {
        space = new hnswlib::InnerProductSpace(ndim);
    } else {
        std::cout << "invalid metric_type = " << (int)metric_type << std::endl;
        return;
    }
    read_bin_file<uint64_t>(index_path + CLUSTER + COMBINE_IDS + BIN, pids, nids, nidsdim);
    std::cout << "there are " << nids << " of dimension " << nidsdim << " combine ids of hnsw" << std::endl;
    assert(pids != nullptr);
    assert(npts == nids);
    assert(nidsdim == 1);
    auto index_hnsw = std::make_shared<hnswlib::HierarchicalNSW<float>>(space, npts, hnswM, hnswefC);
    index_hnsw->addPoint(pdata, pids[0]);
#pragma omp parallel for
    for (auto i = 1; i < npts; i ++) {
        index_hnsw->addPoint(pdata + i * ndim, pids[i]);
    }
    index_hnsw->saveIndex(index_path + HNSW + INDEX + BIN);
    delete[] pdata;
    pdata = nullptr;
    delete[] pids;
    pids = nullptr;
}

template<typename DATAT, typename DISTT>
void build_disk_index(const std::string& raw_data_file, const std::string& index_output_path,
                      const int hnswM, const int hnswefC, const int PQM, const int PQnbits,
                      MetricType metric_type) {

    uint32_t nb, dim;
    get_bin_metadata(raw_data_file, nb, dim);
    std::cout << "read meta from " << raw_data_file << ", nb = " << nb << "dim = " << dim << std::endl;
    size_t sample_num = (size_t)(nb * K1_SAMPLE_RATE);
    DATAT* sample_data;
    sample_data = new DATAT[sample_num * dim];
    reservoir_sampling(raw_data_file, sample_num, sample_data);
    float* centroids = new float[dim * K1];
    kmeans<DATAT>(sample_num, sample_data, (int32_t)dim, K1, centroids);
    delete[] sample_data;
    sample_data = nullptr;

    // pq.train
    size_t pq_sample_num = (size_t)(nb * PQ_SAMPLE_RATE);
    DATAT* pq_sample_data = new DATAT[pq_sample_num * dim];
    reservoir_sampling(raw_data_file, pq_sample_num, pq_sample_data);
    ProductQuantizer<CMin<DISTT, uint32_t>, DATAT, uint8_t> pq_quantizer(dim, PQM, PQnbits);
    pq_quantizer.train(pq_sample_num, pq_sample_data);
    delete[] pq_sample_data;
    pq_sample_data = nullptr;

    uint32_t graph_nb, graph_dim;
    split_raw_data<DATAT, DISTT>(raw_data_file, index_output_path, centroids, metric_type);
    train_clusters<DATAT, DISTT>(index_output_path, graph_nb, graph_dim, &pq_quantizer, metric_type);

    create_graph_index(index_output_path, hnswM, hnswefC, metric_type); // hard code hnsw

    delete[] centroids;
    centroids = nullptr;
}

// parameters: 
//      DataType    data_type(float or uint8)
//      string      query_bin_file
//      string      answer_bin_file
//      int         topk
//      int         nprobe
template<typename DATAT, typename DISTT>
void search_disk_index_simple(const std::string& index_path, 
                              const std::string& query_bin_file,
                              const std::string& answer_bin_file,
                              const int topk,
                              const int nprobe,
                              const int PQM, const int PQnbits,
                              MetricType metric_type) {

    // parameters
    int refine_topk = topk;

    // files
    std::string hnsw_index_file = index_path + HNSW + INDEX + BIN;
    std::string pq_centroids_file = index_path + PQ + PQ_CENTROIDS + BIN;

    // variables
    uint32_t num_queries, num_base, num_pq_centroids, num_pq_codebook;
    uint32_t dim_queries, dim_base, dim_pq_centroids, dim_pq_codebook;
    num_base = 0;

    get_bin_metadata(query_bin_file, num_queries, dim_queries);
    hnswlib::SpaceInterface<float>* space;
    if (MetricType::L2 == metric_type) {
        space = new hnswlib::L2Space(dim_queries);
    } else if (MetricType::IP == metric_type) {
        space = new hnswlib::InnerProductSpace(dim_queries);
    } else {
        std::cout << "invalid metric_type = " << (int)metric_type << std::endl;
        return;
    }

    DATAT* pquery = nullptr;
    // float* pq_centroids = nullptr;
    // read bin files
    read_bin_file<DATAT>(query_bin_file, pquery, num_queries, dim_queries);
    // dim_pq_centroids = dim_base / PQM, num_pq_centroids = PQM * 256
    // read_bin_file<float>(pq_centroids_file, pq_centroids, num_pq_centroids, dim_pq_centroids);

    // if DATAT is uint8_t, distance type is uint32_t, force transfer to uint32_t from float, size is the same
    DISTT* pq_distance = new DISTT[num_queries * refine_topk];
    DISTT* answer_dists = new DISTT[num_queries * topk];
    uint32_t* answer_ids = new uint32_t[num_queries * topk];
    using heap_comare_class = CMin<DISTT, uint32_t>;
    auto dis_computer = select_computer<DATAT, DATAT, DISTT>(metric_type);
    PQ_Computer<DATAT> pq_cmp;
    if (MetricType::L2 == metric_type) {
        pq_cmp = L2sqr<const DATAT, const float, float>;
    } else if (MetricType::IP == metric_type) {
        pq_cmp = IP<const DATAT, const float, float>;
    } else {
        std::cout << "invalid metric_type = " << int(metric_type) << std::endl;
    }
    uint64_t* pq_offsets = new uint64_t[num_queries * refine_topk];

    // in-memory data
    std::vector<std::vector<uint32_t>> metas(K1); // bucket size of each cluster
    std::vector<std::vector<uint8_t>> pq_codebook(K1); // pq codebook

    auto index_hnsw = std::make_shared<hnswlib::HierarchicalNSW<float>>(space, hnsw_index_file);

    // uint32_t PQM = dim_queries / dim_pq_centroids;

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
    std::cout << "load meta and pq_codebook done, num_base = " << num_base << std::endl;

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

    ProductQuantizer<CMin<DISTT, uint64_t>, DATAT, uint8_t> pq_quantizer(dim_queries, PQM, PQnbits);
    pq_quantizer.load_centroids(pq_centroids_file);

    // step2: pq search
#pragma omp parallel for
    for (auto i = 0; i < num_queries; i ++) {
        ProductQuantizer<CMin<DISTT, uint64_t>, DATAT, uint8_t> pq_quantizer_copiesi(pq_quantizer);
        pq_quantizer_copiesi.cal_precompute_table(pquery + i * dim_queries, pq_cmp);
        auto p_labeli = p_labels + i * nprobe;
        auto pq_offseti = pq_offsets + i * refine_topk;
        auto pq_distancei = pq_distance + i * refine_topk;
        uint32_t cid, bid, off;
        for (auto j = 0; j < nprobe; j ++) {
            parse_id(p_labeli[j], cid, bid, off);
            pq_quantizer_copiesi.search(pquery + i * dim_queries,
                    pq_codebook[cid].data() + off * PQM, metas[cid][bid],
                    refine_topk, pq_distancei, pq_offseti, pq_cmp, 
                    j + 1 == nprobe, j == 0,
                    cid, off, i);
        }
        pq_quantizer_copiesi.reset();
    }

    // refine
    std::sort(pq_offsets, pq_offsets + refine_topk * num_queries);
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> refine_records(K1);
    for (auto j = 0; j < refine_topk * num_queries; j ++) {
        uint32_t cid, off, qid;
        parse_refine_id(pq_offsets[j], cid, off, qid);
        refine_records[cid].emplace_back(off, qid);
    }
    std::vector<std::ifstream> raw_data_file_handlers(K1);
    std::vector<std::ifstream> ids_data_file_handlers(K1);
    for (auto i = 0; i < K1; i ++) {
        std::string data_filei = index_path + CLUSTER + std::to_string(i) + RAWDATA + BIN;
        std::string ids_filei  = index_path + CLUSTER + std::to_string(i) + GLOBAL_IDS + BIN;
        raw_data_file_handlers[i] = std::ifstream(data_filei, std::ios::binary);
        ids_data_file_handlers[i] = std::ifstream(ids_filei , std::ios::binary);
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
        DATAT* data_bufi = new DATAT[dim_queries];
        uint32_t global_id;
        for (auto j = 0; j < refine_records[i].size(); j ++) {
            if (refine_records[i][j].second != pre_qid) {
                pre_qid = refine_records[i][j].second;
                raw_data_file_handlers[i].seekg(meta_bytes + refine_records[i][j].first * dim_queries * sizeof(DATAT));
                raw_data_file_handlers[i].read((char*)data_bufi, dim_queries * sizeof(DATAT));
                ids_data_file_handlers[i].seekg(meta_bytes + refine_records[i][j].first * sizeof(uint32_t));
                ids_data_file_handlers[i].read((char*)&global_id, sizeof(uint32_t));
                assert(global_id >= 0);
                assert(global_id < num_base);
            }
            auto dis = dis_computer(data_bufi, pquery + pre_qid, dim_queries);
            std::unique_lock<std::mutex> lk(mtx[pre_qid]);
            if (heap_comare_class::cmp(answer_dists[topk * pre_qid], dis)) {
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
        std::string data_filei = index_path + CLUSTER + std::to_string(i) + RAWDATA + BIN;
        std::string ids_filei  = index_path + CLUSTER + std::to_string(i) + GLOBAL_IDS + BIN;
        raw_data_file_handlers[i].close();
        ids_data_file_handlers[i].close();
    }

    delete[] pquery;
    pquery = nullptr;
    // delete[] pq_centroids;
    // pq_centroids = nullptr;
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

template
void search_disk_index_simple<float, float>(const std::string& index_path, 
                              const std::string& query_bin_file,
                              const std::string& answer_bin_file,
                              const int topk,
                              const int nprobe,
                              const int PQM, const int PQnbits,
                              MetricType metric_type);


template
void search_disk_index_simple<uint8_t, uint32_t>(const std::string& index_path, 
                              const std::string& query_bin_file,
                              const std::string& answer_bin_file,
                              const int topk,
                              const int nprobe,
                              const int PQM, const int PQnbits,
                              MetricType metric_type);


template
void build_disk_index<float, float>(const std::string& raw_data_file, const std::string& index_output_path,
                      const int hnswM, const int hnswefC, const int PQM, const int PQnbits,
                      MetricType metric_type);

template
void build_disk_index<uint8_t, uint32_t>(const std::string& raw_data_file, const std::string& index_output_path,
                      const int hnswM, const int hnswefC, const int PQM, const int PQnbits,
                      MetricType metric_type);


template
void split_raw_data<float, float>(const std::string& raw_data_file, const std::string& index_output_path,
                    float* centroids, MetricType metric_type);

template
void split_raw_data<uint8_t, uint32_t>(const std::string& raw_data_file, const std::string& index_output_path,
                    float* centroids, MetricType metric_type);


template
void train_clusters<float, float>(const std::string& cluster_path, uint32_t& graph_nb, uint32_t& graph_dim, 
                    ProductQuantizer<CMin<float, uint32_t>, float, uint8_t>* pq_quantizer,
                    MetricType metric_type);

template
void train_clusters<uint8_t, uint32_t>(const std::string& cluster_path, uint32_t& graph_nb, uint32_t& graph_dim, 
                    ProductQuantizer<CMin<uint32_t, uint32_t>, uint8_t, uint8_t>* pq_quantizer,
                    MetricType metric_type);



