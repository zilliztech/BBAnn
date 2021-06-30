#include "diskann.h"


template<typename DATAT, typename DISTT, typename HEAPT>
void train_cluster(const std::string& raw_data_bin_file,
                   const std::string& output_path,
                   const int K1,
                   const float*& centroids) {
    std::cout << "train_cluster parameters:" << std::endl;
    std::cout << " raw_data_bin_file: " << raw_data_bin_file
              << " output path: " << output_path
              << " K1: " << K1
              << " centroids: " << centroids
              << std::endl;
    assert(centroids == nullptr);
    DATAT* sample_data = nullptr;
    uint32_t nb, dim;
    get_bin_metadata(raw_data_bin_file, nb, dim);
    uint32_t sample_num = nb * K1_SAMPLE_RATE;
    std::cout << "nb = " << nb << ", dim = " << dim << ", sample_num 4 K1: " << sample_num << std::endl;

    centroids = new float[K1 * dim];
    sample_data = new DATAT[sample_num * dim];
    reservoir_sampling(raw_data_bin_file, sample_num, sample_data);
    kmeans<DATAT>(sample_num, sample_data, (int32_t)dim, K1, centroids, true);

    delete[] sample_data;
}

template<typename DATAT, typename DISTT, typename HEAPT>
void divide_raw_data(const std::string& raw_data_bin_file,
                     const std::string& output_path,
                     const float* centroids,
                     const uint32_t K1) {
    TimeRecorder rc("split_raw_data");
    IOReader reader(raw_data_bin_file);
    uint32_t nb, dim;
    reader.read((char*)&nb, sizeof(uint32_t));
    reader.read((char*)&dim, sizeof(uint32_t));
    uint32_t placeholder = 0, const_one = 1;
    std::vector<uint32_t> cluster_size(K1, 0);
    std::vector<std::ofstream> cluster_dat_writer(K1);
    std::vector<std::ofstream> cluster_ids_writer(K1);
    for (auto i = 0; i < K1; i ++) {
        std::string cluster_raw_data_file_name = output_path + CLUSTER + std::to_string(i) + RAWDATA + BIN;
        std::string cluster_ids_data_file_name = output_path + CLUSTER + std::to_string(i) + GLOBAL_IDS + BIN;
        cluster_dat_writer = std::ofstream(cluster_raw_data_file_name, std::ios::binary);
        cluster_ids_writer = std::ofstream(cluster_ids_data_file_name, std::ios::binary);
        cluster_dat_writer[i].write((char*)&placeholder, sizeof(uint32_t));
        cluster_dat_writer[i].write((char*)&dim, sizeof(uint32_t));
        cluster_ids_writer[i].write((char*)&placeholder, sizeof(uint32_t));
        cluster_ids_writer[i].write((char*)&const_one, sizeof(uint32_t));
        cluster_size[i] = 0;
    }

    uint32_t block_size = 1000000;
    assert(nb > 0);
    uint32_t block_num = (nb - 1) / block_size + 1;
    std::vector<uint32_t> cluster_id(block_size);
    std::vector<DISTT> dists(block_size);
    DATAT* block_buf = new DATAT[block_size * dim];
    for (auto i = 0; i < block_num; i ++) {
        TimeRecorder rci("block-" + std::to_string(i));
        auto sp = i * block_size;
        auto ep = std::min((size_t)nb, sp + block_size);
        std::cout << "split the " << i << "th block, start position = " << sp << ", end position = " << ep << std::endl;
        reader.read((char*)block_buf, (ep - sp) * dim * sizeof(DATAT));
        rci.RecordSection("read block data done");
        knn_1<HEAPT, DATAT, float> (
            block_buf, centroids, ep - sp, K1, dim, 1, 
            dists.data(), cluster_id.data(), L2sqr<DATAT, float, DISTT>);
        rci.RecordSection("select file done");
        for (auto j = 0; j < ep - sp; j ++) {
            auto cid = cluster_id[j];
            auto uid = (uint32_t)(j + sp);
            cluster_dat_writer[cid].write((char*)(block_buf + j * dim), sizeof(DATAT) * dim);
            cluster_ids_writer[cid].write((char*)&uid, sizeof(uint32_t));
            cluster_size[cid] ++;
        }
        rci.RecordSection("write done");
        rci.ElapseFromBegin("split block " + std::to_string(i) + " done");
    }
    rc.RecordSection("split done");
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
    rc.RecordSection("rewrite header done");
    std::cout << "total points num: " << sump << std::endl;

    delete[] block_buf;
    block_buf = nullptr;
    rc.ElapseFromBegin("split_raw_data totally done");
}

template<typename DATAT, typename DISTT, typename HEAPT>
void conquer_clusters(const std::string& output_path,
                      const int K1, const int threshold) {
    TimeRecorder rc("conquer clusters");
    std::cout << "conquer clusters parameters:" << std::endl;
    std::cout << " output_path: " << output_path
              << std::endl;
    std::vector<uint32_t> cluster_id;
    std::vector<DISTT> dists;
    uint32_t placeholder = 1;
    std::string bucket_centroids_file = output_path + BUCKET + CENTROIDS + BIN;
    std::string bucket_ids_file = output_path + CLUSTER + COMBINE_IDS + BIN;

    std::ofstream bucket_ids_writer(bucket_ids_file, std::ios::binary);
    std::ofstream bucket_ctd_writer(bucket_centroids_file, std::ios::binary);
    bucket_ctd_writer.write((char*)&placeholder, sizeof(uint32_t));
    bucket_ctd_writer.write((char*)&placeholder, sizeof(uint32_t));
    bucket_ids_writer.write((char*)&placeholder, sizeof(uint32_t));
    bucket_ids_writer.write((char*)&placeholder, sizeof(uint32_t));
    uint32_t graph_nb = 0, graph_dim;
    for (auto i = 0; i < K1; i ++) {
        TimeRecorder rci("train-cluster-" + std::to_string(i));
        std::string data_file = output_path + CLUSTER + std::to_string(i) + RAWDATA + BIN;
        std::string ids_file  = output_path + CLUSTER + std::to_string(i) + GLOBAL_IDS + BIN;
        std::string meta_file = output_path + CLUSTER + std::to_string(i) + META + BIN;

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

        auto K2 = (cluster_size - 1) / threshold + 1;
        std::cout << "cluster-" << i << " will split into " << K2 << " buckets." << std::endl;
        float* centroids_i = new float[K2 * cluster_dim];
        kmeans<DATAT>(cluster_size, datai, (int32_t)cluster_dim, K2, centroids_i, true);
        rci.RecordSection("kmeans done");
        cluster_id.resize(cluster_size);
        dists.resize(cluster_size);
        knn_1<HEAPT, DATAT, float> (
            datai, centroids_i, cluster_size, K2, cluster_dim, 1, 
            dists.data(), cluster_id.data(), L2sqr<DATAT, float, DISTT>);
        rci.RecordSection("assign done");
        std::vector<uint32_t> buckets_size(K2 + 1, 0);
        std::vector<std::pair<uint32_t, uint32_t>> cluster_off;
        cluster_off.resize(cluster_size);
        for (auto j = 0; j < cluster_size; j ++) {
            buckets_size[cluster_id[j] + 1] ++;
        }

        {// validate bucket size
            std::vector<int> empty_bkid;
            for (auto j = 1; j <= K2; j ++) {
                assert(buckets_size[j] >= 0);
                if (buckets_size[j] == 0)
                    empty_bkid.push_back(j - 1);
            }
            std::cout << "cluster-" << i << " has " << empty_bkid.size() << " empty buckets:" << std::endl;
            for (auto j = 0; j < empty_bkid.size(); j ++)
                std::cout << empty_bkid[j] << " ";
            std::cout << std::endl;
        }

        // write meta file
        write_bin_file<uint32_t>(meta_file, &buckets_size[1], K2, 1);
        rci.RecordSection("save meta into file: " + meta_file + " done");

        for (auto j = 1; j <= K2; j ++) {
            buckets_size[j] += buckets_size[j - 1];
        }

        // write buckets's centroids and combine ids
        // write_bin_file<float>(bucket_centroids_file, centroids_i, K2, cluster_dim);
        bucket_ctd_writer.write((char*)centroids_i, sizeof(float) * K2 * cluster_dim);
        rci.RecordSection("append centroids_i into bucket_centroids_file");

        for (auto j = 0; j < K2; j ++) {
            assert(buckets_size[j] <= cluster_size);
            uint64_t gid = gen_id(i, j, buckets_size[j]);
            bucket_ids_writer.write((char*)&gid, sizeof(uint64_t));
        }
        rci.RecordSection("append combine ids into bucket_ids_file");

        for (auto j = 0; j < cluster_size; j ++) {
            cluster_off[j].first = buckets_size[cluster_id[j]] ++;
            cluster_off[j].second = j;
        }
        std::sort(cluster_off.begin(), cluster_off.end(), [](const auto &l, const auto &r) {
                return l.first < r.first;
                });
        rci.RecordSection("sort done");

        // rewrite raw_data and global ids by bucket order
        { // data_writer will close and auto flush in de-constructor
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
        }
        rci.RecordSection("rearrange raw_data and global ids done");

        // load buckets_size from meta_file again
        // uint32_t meta_numi, meta_dimi;
        // read_bin_file<uint32_t>(meta_file, buckets_size.data(), meta_numi, meta_dimi);
        // assert(meta_numi == K2);
        // assert(meta_dimi == 1);
        graph_nb += K2;
        graph_dim = cluster_dim;

        delete[] datai;
        delete[] idsi;
        delete[] centroids_i;
        rci.ElapseFromBegin("done");
        rc.RecordSection("conquer the " + std::to_string(i) + "th cluster done");
    }
    bucket_ids_writer.close();
    bucket_ctd_writer.close();

    std::cout << "total bucket num = " << graph_nb << std::endl;
    set_bin_metadata(bucket_centroids_file, graph_nb, graph_dim);
    assert(1 == placeholder);
    set_bin_metadata(bucket_ids_file, graph_nb, placeholder);
    rc.ElapseFromBegin("conquer cluster totally done");
}

void build_graph(const std::string& index_path,
                 const int hnswM, const int hnswefC,
                 MetricType metric_type) {
    TimeRecorder rc("create_graph_index");
    std::cout << "build hnsw parameters:" << std::endl;
    std::cout << " index_path: " << index_path
              << " hnsw.M: " << hnswM 
              << " hnsw.efConstruction: " << hnswefC
              << " metric_type: " << metric_type
              << std::endl;

    float* pdata = nullptr;
    uint64_t* pids = nullptr;
    uint32_t npts, ndim, nids, nidsdim;

    read_bin_file<float>(index_path + BUCKET + CENTROIDS + BIN, pdata, npts, ndim);
    rc.RecordSection("load centroids of buckets done");
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
    rc.RecordSection("load combine ids of buckets done");
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
    std::cout << "hnsw totally add " << npts << " points" << std::endl;
    rc.RecordSection("create index hnsw done");
    index_hnsw->saveIndex(index_path + HNSW + INDEX + BIN);
    rc.RecordSection("hnsw save index done");
    delete[] pdata;
    pdata = nullptr;
    delete[] pids;
    pids = nullptr;
    rc.ElapseFromBegin("create index hnsw totally done");
}


/*
 * residual pq
{
    read bucket_centroids_file into bucket_centroids;
    int bucket_cnt = 0;
    prepare sample array;
    for (i = 0; i < K1; i ++) {
        read metai; => meta_size(bucket num in cluster i), meta_dim(1);
        read raw_datai; => cluster_size, cluster_dim;
        for (j = 0; j < meta_size; j ++) {
            for (k = 0; k < meta[j]; j ++) {
                read one vector;
                diff with bucket_centroids[bucket_cnt];
            }
            bucket_cnt ++;
        }
    }

    pq.train();

    for (i = 0; i < K1; i ++) {
    }
}
*/

template<typename DATAT, typename DISTT, typename HEAPT>
void train_quantizer(const std::string& raw_data_bin_file,
                     const std::string& output_path,
                     const int K1,
                     const int PQM, const int PQnbits) {
    TimeRecorder rc("train quantizer");
    std::cout << "train quantizer parameters:" << std::endl;
    std::cout << " raw_data_bin_file: " << raw_data_bin_file
              << " output_path: " << output_path
              << " PQM: " << PQM
              << " PQnbits: " << PQnbits
              << std::endl;

    uint32_t nb, dim;
    get_bin_metadata(raw_data_bin_file, nb, dim);
    uint32_t pq_sample_num = (uint32_t)(nb * PQ_SAMPLE_RATE);
    DATAT* pq_sample_data = new DATAT[pq_sample_num * dim];
    reservoir_sampling(raw_data_bin_file, pq_sample_num, pq_sample_data);
    rc.RecordSection("reservoir_sampling 4 pq train set done");
    ProductQuantizer<HEAPT, DATAT, uint8_t> pq_quantizer(dim, PQM, PQnbits);
    pq_quantizer.train(pq_sample_num, pq_sample_data);
    rc.RecordSection("pq quantizer train done");
    pq_quantizer.save_centroids(output_path + PQ_CENTROIDS + BIN);
    rc.RecordSection("pq quantizer save centroids done");
    for (auto i = 0; i < K1; i ++) {
        std::string data_file = output_path + CLUSTER + std::to_string(i) + RAWDATA + BIN;
        std::string pq_codebook_file = output_path + CLUSTER + std::to_string(i) + PQ + CODEBOOK + BIN;
        uint32_t cluster_size, cluster_dim;
        IOReader data_reader(data_file);
        data_reader.read((char*)&cluster_size, sizeof(uint32_t));
        data_reader.read((char*)&cluster_dim, sizeof(uint32_t));
        DATAT* datai = new DATAT[cluster_size * cluster_dim];
        pq_quantizer.encode_vectors_and_save(cluster_size, datai, pq_codebook_file);
        delete[] datai;
    }

    rc.ElapseFromBegin("train quantizer totally done.");
}

template<typename DATAT, typename DISTT, typename HEAPT>
void build_bigann(const std::string& raw_data_bin_file,
                  const std::string& output_path,
                  const int hnswM, const int hnswefC,
                  const int PQM, const int PQnbits,
                  const int K1, const int threshold,
                  MetricType metric_type) {
    TimeRecorder rc("build bigann");
    std::cout << "build bigann parameters:" << std::endl;
    std::cout << " raw_data_bin_file: " << raw_data_bin_file
              << " output_path: " << output_path
              << " hnsw.M: " << hnswM
              << " hnsw.efConstruction: " << hnswefC
              << " PQ.M: " << PQM
              << " PQ.nbits: " << PQnbits
              << " K1: " << K1
              << " bucket split threshold: " << threshold
              << std::endl;

    float* centroids = nullptr;
    // sampling and do K1-means to get the first round centroids
    train_cluster<DATAT, DISTT, HEAPT>(raw_data_bin_file, output_path, K1, centroids);
    assert(centroids != nullptr);
    rc.RecordSection("train cluster to get " + std::to_string(K1) + " centroids done.");

    divide_raw_data<DATAT, DISTT, HEAPT>(raw_data_bin_file, output_path, centroids, K1);
    rc.RecordSection("divide raw data into " + std::to_string(K1) + " clusters done");

    conquer_clusters<DATAT, DISTT, HEAPT>(output_path, K1, threshold);
    rc.RecordSection("conquer each cluster into buckets done");

    build_graph(output_path, hnswM, hnswefC, metric_type);
    rc.RecordSection("build hnsw done.");

    train_quantizer<DATAT, DISTT, HEAPT>(raw_data_bin_file, output_path, K1, PQM, PQnbits);
    rc.RecordSection("train quantizer done.");
    delete[] centroids;
    rc.ElapseFromBegin("build bigann totally done.");
}

void load_pq_codebook(const std::string& index_path,
                      std::vector<std::vector<uint8_t>>& pq_codebook, 
                      const int K1) {
    TimeRecorder rc("load pq codebook");
    std::cout << "load pq codebook parameters:" << std::endl;
    std::cout << " index path: " << index_path
              << " K1: " << K1 
              << std::endl;

    for (auto i = 0; i < K1; i ++) {
        std::ifstream reader(index_path + CLUSTER + std::to_string(i) + PQ + CODEBOOK + BIN, std::ios::binary);
        uint32_t sizei, dimi;
        reader((char*)&sizei, sizeof(uint32_t));
        reader((char*)&dimi, sizeof(uint32_t));
        pq_codebook[i].resize(sizei * dimi);
        reader.read((char*)pq_codebook[i].data(), sizei * dimi * sizeof(uint8_t));
        reader.close();
    }
    rc.ElapseFromBegin("load pq codebook done.");
}

void load_meta(const std::string& index_path,
               std::vector<std::vector<uint32_t>>& meta,
               const int K1) {
    TimeRecorder rc("load meta");
    std::cout << "load meta parameters:" << std::endl;
    std::cout << " index path: " << index_path
              << " K1: " << K1 
              << std::endl;

    for (auto i = 0; i < K1; i ++) {
        std::ifstream reader(index_path + CLUSTER + std::to_string(i) + META + BIN, std::ios::binary);
        uint32_t nmeta, dmeta;
        reader.read((char*)&nmeta, sizeof(uint32_t));
        reader.read((char*)&dmeta, sizeof(uint32_t));
        assert(1 == dmeta);
        meta[i].resize(nmeta);
        reader.read((char*)meta[i].data(), nmeta * dmeta * sizeof(uint32_t));
        reader.close();
    }
    rc.ElapseFromBegin("load meta done.");
}


template<typename DATAT, typename DISTT, typename HEAPT, typename HEAPTT>
void search_bigann(const std::string& index_path,
                   const std::string& query_bin_file,
                   const std::string& answer_bin_file,
                   const int nprobe,
                   const int topk,
                   const int refine_topk,
                   std::shared_ptr<hnswlib::HierarchicalNSW<DATAT>> index_hnsw,
                   ProductQuantizer<HEAPTT, DATAT, uint8_t>& pq_quantizer,
                   const int K1,
                   PQ_Computer<DATAT>& pq_cmp) {
    TimeRecorder rc("search bigann");
    std::cout << "search bigann parameters:" << std::endl;
    std::cout << " index_path: " << index_path
              << " query_bin_file: " << query_bin_file
              << " answer_bin_file: " << answer_bin_file
              << " nprobe: " << nprobe
              << " topk: " << topk
              << " refine topk: " << refine_topk
              << " K1: " << K1
              << std::endl;






    rc.ElapseFromBegin("search bigann totally done");
}


