#include "diskann.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cstdio>

template<typename DATAT>
void train_cluster(const std::string& raw_data_bin_file,
                   const std::string& output_path,
                   const int32_t K1,
                   float** centroids,
                   double& avg_len) {
    TimeRecorder rc("train cluster");
    std::cout << "train_cluster parameters:" << std::endl;
    std::cout << " raw_data_bin_file: " << raw_data_bin_file
              << " output path: " << output_path
              << " K1: " << K1
              << " centroids: " << *centroids
              << std::endl;
    assert((*centroids) == nullptr);
    DATAT* sample_data = nullptr;
    uint32_t nb, dim;
    get_bin_metadata(raw_data_bin_file, nb, dim);
    int64_t sample_num = nb * K1_SAMPLE_RATE;
    std::cout << "nb = " << nb << ", dim = " << dim << ", sample_num 4 K1: " << sample_num << std::endl;

    *centroids = new float[K1 * dim];
    sample_data = new DATAT[sample_num * dim];
    reservoir_sampling(raw_data_bin_file, sample_num, sample_data);
    rc.RecordSection("reservoir sample with sample rate: " + std::to_string(K1_SAMPLE_RATE) + " done");
    double mxl, mnl;
    int64_t stat_n = std::min(static_cast<int64_t>(1000000), sample_num);
    stat_length<DATAT>(sample_data, stat_n, dim, mxl, mnl, avg_len);
    rc.RecordSection("calculate " + std::to_string(stat_n) + " vectors from sample_data done");
    std::cout << "max len: " << mxl << ", min len: " << mnl << ", average len: " << avg_len << std::endl;
    kmeans<DATAT>(sample_num, sample_data, dim, K1, *centroids, false, avg_len);
    rc.RecordSection("kmeans done");
    assert((*centroids) != nullptr);

    delete[] sample_data;
    rc.ElapseFromBegin("train cluster done.");
}

template<typename DATAT, typename DISTT, typename HEAPT>
void divide_raw_data(const std::string& raw_data_bin_file,
                     const std::string& output_path,
                     const float* centroids,
                     const int32_t K1) {
    TimeRecorder rc("divide raw data");
    std::cout << "divide_raw_data parameters:" << std::endl;
    std::cout << " raw_data_bin_file: " << raw_data_bin_file
              << " output_path: " << output_path
              << " centroids: " << centroids
              << " K1: " << K1 
              << std::endl;
    IOReader reader(raw_data_bin_file);
    uint32_t nb, dim;
    reader.read((char*)&nb, sizeof(uint32_t));
    reader.read((char*)&dim, sizeof(uint32_t));
    uint32_t placeholder = 0, const_one = 1;
    std::vector<uint32_t> cluster_size(K1, 0);
    std::vector<std::ofstream> cluster_dat_writer(K1);
    std::vector<std::ofstream> cluster_ids_writer(K1);
    for (int i = 0; i < K1; i ++) {
        std::string cluster_raw_data_file_name = output_path + CLUSTER + std::to_string(i) + RAWDATA + BIN;
        std::string cluster_ids_data_file_name = output_path + CLUSTER + std::to_string(i) + GLOBAL_IDS + BIN;
        cluster_dat_writer[i] = std::ofstream(cluster_raw_data_file_name, std::ios::binary);
        cluster_ids_writer[i] = std::ofstream(cluster_ids_data_file_name, std::ios::binary);
        cluster_dat_writer[i].write((char*)&placeholder, sizeof(uint32_t));
        cluster_dat_writer[i].write((char*)&dim, sizeof(uint32_t));
        cluster_ids_writer[i].write((char*)&placeholder, sizeof(uint32_t));
        cluster_ids_writer[i].write((char*)&const_one, sizeof(uint32_t));
    }

    int64_t block_size = 1000000;
    assert(nb > 0);
    int64_t block_num = (nb - 1) / block_size + 1;
    std::vector<int64_t> cluster_id(block_size);
    std::vector<DISTT> dists(block_size);
    DATAT* block_buf = new DATAT[block_size * dim];
    for (int64_t i = 0; i < block_num; i ++) {
        TimeRecorder rci("block-" + std::to_string(i));
        int64_t sp = i * block_size;
        int64_t ep = std::min((int64_t)nb, sp + block_size);
        std::cout << "split the " << i << "th block, start position = " << sp << ", end position = " << ep << std::endl;
        reader.read((char*)block_buf, (ep - sp) * dim * sizeof(DATAT));
        rci.RecordSection("read block data done");
        elkan_L2_assign<const DATAT, const float, DISTT>(block_buf, centroids, dim, ep -sp, K1, cluster_id.data(), dists.data());
        //knn_1<HEAPT, DATAT, float> (
        //    block_buf, centroids, ep - sp, K1, dim, 1, 
        //    dists.data(), cluster_id.data(), L2sqr<const DATAT, const float, DISTT>);
        rci.RecordSection("select file done");
        for (int64_t j = 0; j < ep - sp; j ++) {
            int64_t cid = cluster_id[j];
            uint32_t uid = (uint32_t)(j + sp);
            // for debug
            /*
            if (0 == uid) {
                std::cout << "vector0 was split into cluster " << cid << std::endl;
                std::cout << " show vector0:" << std::endl;
                for (auto si = 0; si < dim; si ++) {
                    std::cout << *(block_buf + j * dim + si) << " ";
                }
                std::cout << std::endl;
            }
            */
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
    for (int i = 0; i < K1; i ++) {
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
                      const int K1, const double avg_len, const int threshold) {
    TimeRecorder rc("conquer clusters");
    std::cout << "conquer clusters parameters:" << std::endl;
    std::cout << " output_path: " << output_path
              << " vector avg length: " << avg_len
              << std::endl;
    std::vector<int64_t> cluster_id;
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
    for (int i = 0; i < K1; i ++) {
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
        DATAT* datai = new DATAT[(uint64_t)cluster_size * cluster_dim];
        data_reader.read((char*)datai, (uint64_t)cluster_size * cluster_dim * sizeof(DATAT));
        uint32_t* idsi = new uint32_t[(uint64_t)ids_size * ids_dim];
        ids_reader.read((char*)idsi, (uint64_t)ids_size * ids_dim * sizeof(uint32_t));

        int64_t K2 = (cluster_size - 1) / threshold + 1;
        std::cout << "cluster-" << i << " will split into " << K2 << " buckets." << std::endl;
        float* centroids_i = new float[K2 * cluster_dim];
        kmeans<DATAT>(cluster_size, datai, (int32_t)cluster_dim, K2, centroids_i, false, avg_len);
        rci.RecordSection("kmeans done");
        cluster_id.resize(cluster_size);
        dists.resize(cluster_size);
        elkan_L2_assign<>(datai, centroids_i, cluster_dim, cluster_size, K2, cluster_id.data(), dists.data());
        // knn_1<HEAPT, DATAT, float> (
        //     datai, centroids_i, cluster_size, K2, cluster_dim, 1, 
        //     dists.data(), cluster_id.data(), L2sqr<const DATAT, const float, DISTT>);
        rci.RecordSection("assign done");
        std::vector<uint32_t> buckets_size(K2 + 1, 0);
        std::vector<std::pair<uint32_t, uint32_t>> cluster_off;
        cluster_off.resize(cluster_size);
        for (int j = 0; j < cluster_size; j ++) {
            buckets_size[cluster_id[j] + 1] ++;
        }

        {// validate bucket size
            std::vector<int> empty_bkid;
            for (int j = 1; j <= K2; j ++) {
                assert(buckets_size[j] >= 0);
                if (buckets_size[j] == 0)
                    empty_bkid.push_back(j - 1);
            }
            std::cout << "cluster-" << i << " has " << empty_bkid.size() << " empty buckets:" << std::endl;
            for (int j = 0; j < empty_bkid.size(); j ++)
                std::cout << empty_bkid[j] << " ";
            std::cout << std::endl;
        }

        // write meta file
        write_bin_file<uint32_t>(meta_file, &buckets_size[1], K2, 1);
        rci.RecordSection("save meta into file: " + meta_file + " done");

        for (int j = 1; j <= K2; j ++) {
            buckets_size[j] += buckets_size[j - 1];
        }

        // write buckets's centroids and combine ids
        // write_bin_file<float>(bucket_centroids_file, centroids_i, K2, cluster_dim);
        bucket_ctd_writer.write((char*)centroids_i, K2 * cluster_dim * sizeof(float));
        rci.RecordSection("append centroids_i into bucket_centroids_file");

        for (int j = 0; j < K2; j ++) {
            assert(buckets_size[j] <= cluster_size);
            uint64_t gid = gen_id(i, j, buckets_size[j]);
            bucket_ids_writer.write((char*)&gid, sizeof(uint64_t));
        }
        rci.RecordSection("append combine ids into bucket_ids_file");

        for (int j = 0; j < cluster_size; j ++) {
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
            for (int j = 0; j < cluster_size; j ++) {
                uint64_t ori_pos = cluster_off[j].second;
                data_writer.write((char*)(datai + ori_pos * cluster_dim), sizeof(DATAT) * cluster_dim);
                ids_writer.write((char*)(idsi + ori_pos * ids_dim), sizeof(uint32_t) * ids_dim);
                // for debug
                /*
                if (*(idsi + ori_pos * ids_dim) == 0) {
                    std::cout << "vector0 is arranged to new pos: " << j << " in cluster " << i << std::endl;
                    std::cout << " show vector0:" << std::endl;
                    for (auto si = 0; si < cluster_dim; si ++) {
                        std::cout << *(datai + ori_pos * cluster_dim + si) << " ";
                    }
                    std::cout << std::endl;
                    v0cid = i;
                    v0pos = j;
                }
                */
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
              << " metric_type: " << (int)metric_type
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
    for (int64_t i = 1; i < npts; i ++) {
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

template<typename DATAT, typename DISTT, typename HEAPT>
void train_pq_quantizer(
                    const std::string& raw_data_bin_file,
                    const std::string& output_path,
                    const int K1,
                    const int PQM,
                    const int PQnbits,
                    MetricType metric_type) {
    TimeRecorder rc("train pq quantizer");
    std::cout << "train quantizer parameters:" << std::endl;
    std::cout << " raw_data_bin_file: " << raw_data_bin_file
              << " output_path: " << output_path
              << " PQM: " << PQM
              << " PQnbits: " << PQnbits
              << std::endl;

    uint32_t nb, dim;
    get_bin_metadata(raw_data_bin_file, nb, dim);
    int64_t pq_sample_num = 100000;
    DATAT* pq_sample_data = new DATAT[pq_sample_num * dim];
    reservoir_sampling(raw_data_bin_file, pq_sample_num, pq_sample_data);
    rc.RecordSection("reservoir_sampling 4 pq train set done");
    ProductQuantizer<HEAPT, DATAT, uint8_t> pq_quantizer(dim, PQM, PQnbits, metric_type);
    pq_quantizer.train(pq_sample_num, pq_sample_data);
    rc.RecordSection("pq quantizer train done");
    pq_quantizer.save_centroids(output_path + PQ_CENTROIDS + BIN);
    rc.RecordSection("pq quantizer save centroids done");
    float* precomputer_table = nullptr;
    for (int i = 0; i < K1; i ++) {
        std::string data_file = output_path + CLUSTER + std::to_string(i) + RAWDATA + BIN;
        std::string pq_codebook_file = output_path + CLUSTER + std::to_string(i) + PQ + CODEBOOK + BIN;
        uint32_t cluster_size, cluster_dim;
        IOReader data_reader(data_file);
        data_reader.read((char*)&cluster_size, sizeof(uint32_t));
        data_reader.read((char*)&cluster_dim, sizeof(uint32_t));
        DATAT* datai = new DATAT[(uint64_t)cluster_size * cluster_dim];
        data_reader.read((char*)datai, (uint64_t)cluster_size * cluster_dim * sizeof(DATAT));
        pq_quantizer.encode_vectors_and_save(precomputer_table, cluster_size, datai, pq_codebook_file);
        delete[] datai;
        rc.RecordSection("the " + std::to_string(i) + "th cluster encode and save codebook done.");
    }

    delete[] pq_sample_data;
    delete[] precomputer_table;
    rc.ElapseFromBegin("train quantizer totally done.");
}

template<typename DATAT, typename DISTT, typename HEAPT>
void train_pq_residual_quantizer(
        const std::string& raw_data_bin_file,
        const std::string& output_path,
        const int K1,
        const int PQM,
        const int PQnbits,
        MetricType metric_type) {
    
    TimeRecorder rc("train pq quantizer with residual");
    std::cout << "train quantizer parameters:" << std::endl;
    std::cout << " raw_data_bin_file: " << raw_data_bin_file
              << " output_path: " << output_path
              << " PQM: " << PQM
              << " PQnbits: " << PQnbits
              << std::endl;

    uint32_t nb, dim;
    get_bin_metadata(raw_data_bin_file, nb, dim);


    std::vector<std::vector<uint32_t> > metas(K1);
    load_meta_impl(output_path, metas, K1);
    assert(metas.size() == K1);

    int64_t pq_sample_num = std::min(
                                static_cast<int64_t>(65536), 
                                std::min(
                                    static_cast<int64_t>(nb * PQ_SAMPLE_RATE),
                                    static_cast<int64_t>(std::accumulate(metas[0].begin(), metas[0].end(), 0))));

    DATAT* sample_data = new DATAT[pq_sample_num * dim];
    assert(sample_data != nullptr);

    float* sample_ivf_cen = new float[pq_sample_num * dim];
    assert(sample_ivf_cen != nullptr);

    // read ivf centroids
    float *ivf_cen = nullptr;
    uint32_t ivf_n, ivf_dim;
    read_bin_file<float>(output_path + BUCKET + CENTROIDS + BIN, ivf_cen, ivf_n, ivf_dim);
    assert(ivf_dim == dim);
    assert(ivf_cen != nullptr);

    reservoir_sampling_residual<DATAT>(output_path, metas, ivf_cen, dim, pq_sample_num, sample_data, sample_ivf_cen, K1);
    rc.RecordSection("reservoir_sampling_residual for pq training set done");

    PQResidualQuantizer<HEAPT, DATAT, uint8_t> quantizer(dim, PQM, PQnbits, metric_type);
    quantizer.train(pq_sample_num, sample_data, sample_ivf_cen);
    rc.RecordSection("pq residual quantizer train done");

    delete[] sample_data;
    delete[] sample_ivf_cen;

    quantizer.save_centroids(output_path + PQ_CENTROIDS + BIN);
    rc.RecordSection("pq residual quantizer save centroids done");

    float* precompute_table = nullptr;
    uint64_t bucket_cnt = 0;
    for (int i = 0; i < K1; ++i) {
        std::string data_file = output_path + CLUSTER + std::to_string(i) + RAWDATA + BIN;
        std::string pq_codebook_file = output_path + CLUSTER + std::to_string(i) + PQ + CODEBOOK + BIN;
        uint32_t cluster_size, cluster_dim;

        IOReader data_reader(data_file);
        data_reader.read((char*)&cluster_size, sizeof(uint32_t));
        data_reader.read((char*)&cluster_dim, sizeof(uint32_t));

        DATAT* datai = new DATAT[cluster_size * cluster_dim];
        data_reader.read((char*)datai, cluster_size * cluster_dim * sizeof(DATAT));
        quantizer.encode_vectors_and_save(precompute_table, cluster_size, datai,
                                          ivf_cen + bucket_cnt * cluster_dim, metas[i],
                                          pq_codebook_file);
        bucket_cnt += metas[i].size();

        delete[] datai;
        rc.RecordSection("the " + std::to_string(i) + "th cluster encode and save codebook done.");
    }

    delete[] precompute_table;
    delete[] ivf_cen;

    rc.ElapseFromBegin("train pq residual quantizer totally done.");
}

// page alignment 4 raw data and uid in each cluster 4 better refine performance, uid follow after rawdata
template<typename DATAT>
void page_align(const std::string& raw_data_bin_file, const std::string& output_path, const int K1) {
    TimeRecorder rc("page align");
    uint32_t nb, dim;
    get_bin_metadata(raw_data_bin_file, nb, dim);
    uint32_t page_size = PAGESIZE;
    uint32_t vector_size = dim * sizeof(DATAT);
    uint32_t id_size = sizeof(uint32_t);
    uint32_t node_size = vector_size + id_size;
    // number of nodes per page
    uint32_t nnpp = page_size / node_size;
    assert(nnpp > 0);
    std::cout << "page size = " << page_size << ", number of nodes(vector + id) in per page: " << nnpp
              << std::endl;

    char* data_page_buf = new char[page_size];

    for (uint32_t i = 0; i < K1; i ++) {
        std::string cluster_raw_data_file_name = output_path + CLUSTER + std::to_string(i) + RAWDATA + BIN;
        std::string cluster_ids_data_file_name = output_path + CLUSTER + std::to_string(i) + GLOBAL_IDS + BIN;
        std::string cluster_raw_data_file_name2 = output_path + CLUSTER + std::to_string(i) + "-" + RAWDATA + BIN;
        uint32_t cluster_size, cluster_dim, ids_size, ids_dim;

        IOReader data_reader(cluster_raw_data_file_name, (uint64_t)(2) * GIGABYTE);
        IOReader ids_reader(cluster_ids_data_file_name);
        IOWriter data_writer(cluster_raw_data_file_name2, (uint64_t)(2) * GIGABYTE);

        data_reader.read((char*)&cluster_size, sizeof(uint32_t));
        data_reader.read((char*)&cluster_dim, sizeof(uint32_t));
        ids_reader.read((char*)&ids_size, sizeof(uint32_t));
        ids_reader.read((char*)&ids_dim, sizeof(uint32_t));
        // number of pages 4 node, the first page is meta page
        uint32_t npn = (cluster_size - 1) / nnpp + 2;
        std::cout << "there are " << npn << " pages 4 nodes in cluster " << std::to_string(i) << std::endl;

        memset(data_page_buf, 0, page_size);
        *(uint32_t*)(data_page_buf + 0 * sizeof(uint32_t)) = cluster_size;
        *(uint32_t*)(data_page_buf + 1 * sizeof(uint32_t)) = cluster_dim;
        *(uint32_t*)(data_page_buf + 2 * sizeof(uint32_t)) = page_size;
        *(uint32_t*)(data_page_buf + 3 * sizeof(uint32_t)) = nnpp;
        *(uint32_t*)(data_page_buf + 4 * sizeof(uint32_t)) = npn;
        *(uint32_t*)(data_page_buf + 5 * sizeof(uint32_t)) = i;
        data_writer.write(data_page_buf, page_size);

        uint32_t write_cnt = 0;
        for (auto j = 1; j < npn; j ++) {
            memset(data_page_buf, 0, sizeof(page_size));
            for (auto k = 0; k < nnpp && write_cnt < cluster_size; k ++) {
                data_reader.read(data_page_buf + k * node_size, vector_size);
                ids_reader.read(data_page_buf + k * node_size + vector_size, id_size);
                write_cnt ++;
            }
            data_writer.write(data_page_buf, page_size);
        }
        
        auto rm_dat_ret = std::remove(cluster_raw_data_file_name.c_str());
        if (rm_dat_ret == 0) {
            std::cout << "remove old raw data file: " << cluster_raw_data_file_name << " success." << std::endl;
        } else {
            std::cout << "remove old raw data file: " << cluster_raw_data_file_name << " failed." << std::endl;
        }
        auto rm_ids_ret = std::remove(cluster_ids_data_file_name.c_str());
        if (rm_ids_ret == 0) {
            std::cout << "remove old ids data file: " << cluster_ids_data_file_name << " success." << std::endl;
        } else {
            std::cout << "remove old ids data file: " << cluster_ids_data_file_name << " failed." << std::endl;
        }
        rc.RecordSection("cluster " + std::to_string(i) + " page aligned done.");
    }

    delete[] data_page_buf;
    rc.ElapseFromBegin("page align totally done.");
}


template<typename DATAT, typename DISTT, typename HEAPT>
void build_bigann(const std::string& raw_data_bin_file,
                  const std::string& output_path,
                  const int hnswM, const int hnswefC,
                  const int PQM, const int PQnbits,
                  const int K1, const int threshold,
                  MetricType metric_type,
                  QuantizerType quantizer_type) {
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
    double avg_len;
    // sampling and do K1-means to get the first round centroids
    train_cluster<DATAT>(raw_data_bin_file, output_path, K1, &centroids, avg_len);
    assert(centroids != nullptr);
    rc.RecordSection("train cluster to get " + std::to_string(K1) + " centroids done.");

    divide_raw_data<DATAT, DISTT, HEAPT>(raw_data_bin_file, output_path, centroids, K1);
    rc.RecordSection("divide raw data into " + std::to_string(K1) + " clusters done");

    conquer_clusters<DATAT, DISTT, HEAPT>(output_path, K1, avg_len, threshold);
    rc.RecordSection("conquer each cluster into buckets done");

    build_graph(output_path, hnswM, hnswefC, metric_type);
    rc.RecordSection("build hnsw done.");

    if (QuantizerType::PQ == quantizer_type) {
        train_pq_quantizer<DATAT, DISTT, HEAPT>(raw_data_bin_file, output_path, K1, PQM, PQnbits, metric_type);
    } else if (QuantizerType::PQRES == quantizer_type) {
        train_pq_residual_quantizer<DATAT, DISTT, HEAPT>(raw_data_bin_file, output_path, K1, PQM, PQnbits, metric_type);
    }
    rc.RecordSection("train quantizer done.");

    page_align<DATAT>(raw_data_bin_file, output_path, K1);
    rc.RecordSection("page align done.");

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

    for (int i = 0; i < K1; i ++) {
        std::ifstream reader(index_path + CLUSTER + std::to_string(i) + PQ + CODEBOOK + BIN, std::ios::binary);
        uint32_t sizei, dimi;
        reader.read((char*)&sizei, sizeof(uint32_t));
        reader.read((char*)&dimi, sizeof(uint32_t));
        pq_codebook[i].resize((uint64_t)sizei * dimi);
        reader.read((char*)pq_codebook[i].data(), (uint64_t)sizei * dimi * sizeof(uint8_t));
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

    load_meta_impl(index_path, meta, K1);

    rc.ElapseFromBegin("load meta done.");
}

template<typename DATAT>
void search_graph(std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
                  const int nq,
                  const int dq,
                  const int nprobe,
                  const int refine_nprobe,
                  const DATAT* pquery,
                  uint64_t* buckets_label) {

    TimeRecorder rc("search graph");
    std::cout << "search graph parameters:" << std::endl;
    std::cout << " index_hnsw: " << index_hnsw
              << " nq: " << nq
              << " dq: " << dq
              << " nprobe: " << nprobe
              << " refine_nprobe: " << refine_nprobe
              << " pquery: " << static_cast<const void *>(pquery)
              << " buckets_label: " << static_cast<void *>(buckets_label)
              << std::endl;
    index_hnsw->setEf(refine_nprobe);
#pragma omp parallel for
    for (int64_t i = 0; i < nq; i ++) {
        // auto queryi = pquery + i * dq;
        // todo: hnsw need to support query data is not float
        float* queryi = new float[dq];
        for (int j = 0; j < dq;j ++) 
            queryi[j] = (float)(*(pquery + i * dq + j));
        auto reti = index_hnsw->searchKnn(queryi, nprobe);
        auto p_labeli = buckets_label + i * nprobe;
        while (!reti.empty()) {
            *p_labeli++ = reti.top().second;
            reti.pop();
        }
        delete[] queryi;
    }
    rc.ElapseFromBegin("search graph done.");
}

template<typename DATAT, typename DISTT, typename HEAPTT>
void search_pq_quantizer(ProductQuantizer<HEAPTT, DATAT, uint8_t>& pq_quantizer,
                      const uint32_t nq,
                      const uint32_t dq,
                      uint64_t* buckets_label,
                      const int nprobe,
                      const int refine_topk,
                      const int K1,
                      const DATAT* pquery,
                      std::vector<std::vector<uint8_t>>& pq_codebook,
                      std::vector<std::vector<uint32_t>>& meta,
                      DISTT*& pq_distance,
                      uint64_t*& pq_offsets) {
    TimeRecorder rc("search quantizer");
    std::cout << "search quantizer parameters:" << std::endl;
    std::cout << " pq_quantizer:" << &pq_quantizer
              << " nq: " << nq
              << " dq: " << dq
              << " buckets_label: " << buckets_label
              << " nprobe: " << nprobe
              << " refine_topk: " << refine_topk
              << " K1: " << K1
              << " pquery: " << static_cast<const void *>(pquery)
              << " pq_distance: " << static_cast<void *>(pq_distance)
              << " pq_offsets: " << static_cast<void *>(pq_offsets)
              << std::endl;


    auto pqm = pq_quantizer.getM();
    rc.RecordSection("pqm = " + std::to_string(pqm));


#pragma omp parallel
    {
        float* precompute_table = nullptr;
#pragma omp for
        for (int i = 0; i < nq; i ++) {
            pq_quantizer.calc_precompute_table(precompute_table, pquery + i * dq);
            auto p_labeli = buckets_label + i * nprobe;
            auto pq_offseti = pq_offsets + i * refine_topk;
            auto pq_distancei = pq_distance + i * refine_topk;
            uint32_t cid, bid, off;
            for (int j = 0; j < nprobe; j ++) {
                parse_id(p_labeli[j], cid, bid, off);
                assert(cid < K1);
                pq_quantizer.search(precompute_table, pquery + i * dq,
                        pq_codebook[cid].data() + (int64_t)off * pqm, meta[cid][bid],
                        refine_topk, pq_distancei, pq_offseti,
                        j + 1 == nprobe, j == 0, cid, off, i);
            }
        }
        delete[] precompute_table;
    }

    /*
#pragma omp parallel for
    for (auto i = 0; i < nq; i ++) {
        ProductQuantizer<HEAPTT, DATAT, uint8_t> pq_quantizer_copiesi(pq_quantizer);
        pq_quantizer_copiesi.cal_precompute_table(pquery + i * dq, pq_cmp);
        auto p_labeli = buckets_label + i * nprobe;
        auto pq_offseti = pq_offsets + i * refine_topk;
        auto pq_distancei = pq_distance + i * refine_topk;
        uint32_t cid, bid, off;
        for (auto j = 0; j < nprobe; j ++) {
            parse_id(p_labeli[j], cid, bid, off);
            assert(cid < K1);
            pq_quantizer_copiesi.search(pquery + i * dq,
                    pq_codebook[cid].data() + off * pqm, meta[cid][bid],
                    refine_topk, pq_distancei, pq_offseti, pq_cmp,
                    j + 1 == nprobe, j == 0, cid, off, i);
        }
        pq_quantizer_copiesi.reset();
    }
    */
    rc.ElapseFromBegin("search quantizer done.");
}

template<typename DATAT, typename DISTT, typename HEAPTT>
void search_pq_residual_quantizer(
        PQResidualQuantizer<HEAPTT, DATAT, uint8_t>& quantizer,
        const uint32_t nq,
        const uint32_t dq,
        float* ivf_centroids,
        uint64_t* buckets_label,
        const int nprobe,
        const int refine_topk,
        const int K1,
        const DATAT* pquery,
        std::vector<std::vector<uint8_t>>& pq_codebook,
        std::vector<std::vector<uint32_t>>& meta,
        DISTT*& pq_distance,
        uint64_t*& pq_offsets) {
    TimeRecorder rc("search pq residual quantizer");
    std::cout << "search quantizer parameters:" << std::endl;
    std::cout << " quantizer:" << &quantizer
              << " nq: " << nq
              << " dq: " << dq
              << " buckets_label: " << buckets_label
              << " nprobe: " << nprobe
              << " refine_topk: " << refine_topk
              << " K1: " << K1
              << " pquery: " << static_cast<const void *>(pquery)
              << " pq_distance: " << static_cast<void *>(pq_distance)
              << " pq_offsets: " << static_cast<void *>(pq_offsets)
              << std::endl;



    auto code_num = quantizer.getCodeNum();

    rc.RecordSection("pq code num = " + std::to_string(code_num));

    std::vector<uint32_t> pre(K1);
    for (int i = 0; i < K1; ++i) {
        pre[i] = (i == 0 ? 0 : pre[i-1] + meta[i-1].size());
    }

#pragma omp parallel
    {
        float* precompute_table = nullptr;
#pragma omp for
        for (int64_t i = 0; i < nq; i ++) {
            quantizer.calc_precompute_table(precompute_table, pquery + i * dq);
            auto p_labeli = buckets_label + i * nprobe;
            auto pq_offseti = pq_offsets + i * refine_topk;
            auto pq_distancei = pq_distance + i * refine_topk;
            uint32_t cid, bid, off;
            for (auto j = 0; j < nprobe; j ++) {
                parse_id(p_labeli[j], cid, bid, off);
                assert(cid < K1);
                const float* cen = ivf_centroids + (pre[cid] + bid) * dq;

                quantizer.search(
                        precompute_table,
                        pquery + i * dq,
                        cen,
                        pq_codebook[cid].data() + (uint64_t)off * code_num,
                        meta[cid][bid],
                        refine_topk,
                        pq_distancei,
                        pq_offseti,
                        j + 1 == nprobe,
                        j == 0,
                        cid,
                        off,
                        i);
            }
        }
        delete[] precompute_table;
    }

    /*
#pragma omp parallel for
    for (auto i = 0; i < nq; i ++) {
        ProductQuantizer<HEAPTT, DATAT, uint8_t> pq_quantizer_copiesi(pq_quantizer);
        pq_quantizer_copiesi.cal_precompute_table(pquery + i * dq, pq_cmp);
        auto p_labeli = buckets_label + i * nprobe;
        auto pq_offseti = pq_offsets + i * refine_topk;
        auto pq_distancei = pq_distance + i * refine_topk;
        uint32_t cid, bid, off;
        for (auto j = 0; j < nprobe; j ++) {
            parse_id(p_labeli[j], cid, bid, off);
            assert(cid < K1);
            pq_quantizer_copiesi.search(pquery + i * dq,
                    pq_codebook[cid].data() + off * pqm, meta[cid][bid],
                    refine_topk, pq_distancei, pq_offseti, pq_cmp,
                    j + 1 == nprobe, j == 0, cid, off, i);
        }
        pq_quantizer_copiesi.reset();
    }
    */
    rc.ElapseFromBegin("search quantizer done.");
}

// cannot work
template<typename DATAT, typename DISTT, typename HEAPT>
void refine(const std::string& index_path,
            const int K1,
            const uint32_t nq,
            const uint32_t dq,
            const int topk,
            const int refine_topk,
            uint64_t* pq_offsets,
            const DATAT* pquery,
            DISTT*& answer_dists,
            uint32_t*& answer_ids,
            Computer<DATAT, DATAT, DISTT>& dis_computer) {
    TimeRecorder rc("refine");
    std::cout << "refine parameters:" << std::endl;
    std::cout << " index_path: " << index_path
              << " cluster size: " << K1
              << " number of query: " << nq
              << " dim of query: " << dq
              << " topk: " << topk
              << " refine_topk:" << refine_topk
              << " pq_offsets:" << static_cast<void *>(pq_offsets)
              << " pquery:" << static_cast<const void *>(pquery)
              << " answer_dists:" << static_cast<void *>(answer_dists)
              << " answer_ids:" << static_cast<void *>(answer_ids)
              << std::endl;
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> refine_records(K1);
    for (int64_t i = 0; i < nq; i ++) {
        auto pq_offseti = pq_offsets + i * refine_topk;
        for (int j = 0; j < refine_topk; j ++) {
            if (pq_offseti[j] == (uint64_t)(-1))
                continue;
            uint32_t cid, off, qid;
            parse_refine_id(pq_offseti[j], cid, off, qid);
            refine_records[cid].emplace_back(off, qid);
        }
    }
    rc.RecordSection("parse_refine_id done");

    std::vector<std::ifstream> raw_data_file_handlers(K1);
    std::vector<std::ifstream> ids_data_file_handlers(K1);
    for (int i = 0; i < K1; i ++) {
        std::string data_filei = index_path + CLUSTER + std::to_string(i) + RAWDATA + BIN;
        std::string ids_filei  = index_path + CLUSTER + std::to_string(i) + GLOBAL_IDS + BIN;
        raw_data_file_handlers[i] = std::ifstream(data_filei, std::ios::binary);
        ids_data_file_handlers[i] = std::ifstream(ids_filei , std::ios::binary);
        uint32_t clu_size, clu_dim, clu_id_size, clu_id_dim;
        raw_data_file_handlers[i].read((char*)&clu_size, sizeof(clu_size));
        raw_data_file_handlers[i].read((char*)&clu_dim, sizeof(clu_dim));
        ids_data_file_handlers[i].read((char*)&clu_id_size, sizeof(clu_id_size));
        ids_data_file_handlers[i].read((char*)&clu_id_dim, sizeof(clu_id_dim));
        std::cout << "cluster-" << i << " has " << clu_size << " vectors,"
                  << " has clu_dim = " << clu_dim
                  << " clu_id_size = " << clu_id_size
                  << " clu_id_dim = " << clu_id_dim
                  << std::endl;
    }
    rc.RecordSection("open rawdata and idsdata file handlers");

    // init answer heap
#pragma omp parallel for schedule (static, 128)
    for (int i = 0; i < nq; i ++) {
        auto ans_disi = answer_dists + topk * i;
        auto ans_idsi = answer_ids + topk * i;
        heap_heapify<HEAPT>(topk, ans_disi, ans_idsi);
    }
    rc.RecordSection("heapify answers heaps");

    // statistics
    std::vector<int> load_vectors(K1, 0);
    std::vector<std::mutex> mtx(nq);
#pragma omp parallel for
    for (int i = 0; i < K1; i ++) {
        if (refine_records[i].size() == 0)
            continue;
        std::sort(refine_records[i].begin(), refine_records[i].end(), [](const auto &l, const auto &r) {
            return l.first < r.first;
        });
        uint64_t pre_off = refine_records[i][0].first;
        uint32_t meta_bytes = 8; // pass meta
        DATAT* data_bufi = new DATAT[dq];
        uint32_t global_id;
        raw_data_file_handlers[i].seekg(meta_bytes + pre_off * dq * sizeof(DATAT), raw_data_file_handlers[i].beg);
        raw_data_file_handlers[i].read((char*)data_bufi, dq * sizeof(DATAT));
        ids_data_file_handlers[i].seekg(meta_bytes + pre_off * sizeof(uint32_t), ids_data_file_handlers[i].beg);
        ids_data_file_handlers[i].read((char*)&global_id, sizeof(uint32_t));
        load_vectors[i] ++;
        // for debug
        for (int j = 0; j < refine_records[i].size(); j ++) {
            if (refine_records[i][j].first != pre_off) {
                pre_off = refine_records[i][j].first;
                raw_data_file_handlers[i].seekg(meta_bytes + pre_off * dq * sizeof(DATAT), raw_data_file_handlers[i].beg);
                raw_data_file_handlers[i].read((char*)data_bufi, dq * sizeof(DATAT));
                ids_data_file_handlers[i].seekg(meta_bytes + pre_off * sizeof(uint32_t), ids_data_file_handlers[i].beg);
                ids_data_file_handlers[i].read((char*)&global_id, sizeof(uint32_t));
                assert(global_id >= 0);
                load_vectors[i] ++;
            }
            uint32_t qid = refine_records[i][j].second;
            auto dis = dis_computer(data_bufi, pquery + qid * dq, dq);
            std::unique_lock<std::mutex> lk(mtx[qid]);
            if (HEAPT::cmp(answer_dists[topk * qid], dis)) {
                heap_swap_top<HEAPT>(topk, answer_dists + topk * qid, answer_ids + topk * qid, dis, global_id);
            }
        }

        delete[] data_bufi;
    }
    rc.RecordSection("calculate done.");
    int tot = 0;
    std::cout << "show load refine vectors of each cluster:" << std::endl;
    for (int i = 0; i < K1; i ++) {
        std::cout << "cluster-" << i << ": " << load_vectors[i] << "/" << refine_topk << std::endl;
        tot += load_vectors[i];
    }
    std::cout << "total load refine vectors: " << tot << "/" << refine_topk * nq << std::endl;

    for (int i = 0; i < K1; i ++) {
        std::string data_filei = index_path + CLUSTER + std::to_string(i) + RAWDATA + BIN;
        std::string ids_filei  = index_path + CLUSTER + std::to_string(i) + GLOBAL_IDS + BIN;
        raw_data_file_handlers[i].close();
        ids_data_file_handlers[i].close();
    }
    rc.ElapseFromBegin("refine done.");
}

template<typename DATAT, typename DISTT, typename HEAPT>
void aligned_refine(const std::string& index_path,
            const int K1,
            const uint32_t nq,
            const uint32_t dq,
            const int topk,
            const int refine_topk,
            uint64_t* pq_offsets,
            const DATAT* pquery,
            DISTT*& answer_dists,
            uint32_t*& answer_ids,
            Computer<DATAT, DATAT, DISTT>& dis_computer) {
    TimeRecorder rc("aligned refine");
    std::cout << "aligned refine parameters:" << std::endl;
    std::cout << " index_path: " << index_path
              << " cluster size: " << K1
              << " number of query: " << nq
              << " dim of query: " << dq
              << " topk: " << topk
              << " refine_topk:" << refine_topk
              << " pq_offsets:" << pq_offsets
              << " pquery:" << static_cast<const void *>(pquery)
              << " answer_dists:" << static_cast<void *>(answer_dists)
              << " answer_ids:" << static_cast<void *>(answer_ids)
              << std::endl;
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> refine_records(K1);
    for (int64_t i = 0; i < nq; i ++) {
        auto pq_offseti = pq_offsets + i * refine_topk;
        for (int j = 0; j < refine_topk; j ++) {
            if (pq_offseti[j] == (uint64_t)(-1))
                continue;
            uint32_t cid, off, qid;
            parse_refine_id(pq_offseti[j], cid, off, qid);
            refine_records[cid].emplace_back(off, qid);
        }
    }
    rc.RecordSection("parse_refine_id done");

    uint32_t page_size = PAGESIZE;
    uint32_t nnpp = 0;
    uint32_t npn = 0;

    char* dat_buf = new char[page_size];
    memset(dat_buf, 0, page_size);

    std::vector<std::ifstream> raw_data_file_handlers(K1);
    for (int i = 0; i < K1; i ++) {
        std::string aligned_data_filei = index_path + CLUSTER + std::to_string(i) + "-" + RAWDATA + BIN;
        raw_data_file_handlers[i] = std::ifstream(aligned_data_filei, std::ios::binary);
        uint32_t clu_size, clu_dim;
        uint32_t ps, check;
        raw_data_file_handlers[i].read(dat_buf, page_size);
        memcpy(&clu_size, dat_buf + 0 * sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&clu_dim , dat_buf + 1 * sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&ps      , dat_buf + 2 * sizeof(uint32_t), sizeof(uint32_t));
        assert(ps == page_size);
        memcpy(&nnpp    , dat_buf + 3 * sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&npn     , dat_buf + 4 * sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&check   , dat_buf + 5 * sizeof(uint32_t), sizeof(uint32_t));
        assert(check == i);
        std::cout << "cluster-" << i << " has " << clu_size << " vectors,"
                  << " clu_dim = " << clu_dim
                  << std::endl;
        std::cout << "main meta: " << std::endl;
        std::cout << " number of nodes in per page: " << nnpp
                  << " number of pages 4 node(rawdata + uid): " << npn
                  << std::endl;
    }
    rc.RecordSection("open rawdata and idsdata file handlers");

    // init answer heap
#pragma omp parallel for schedule (static, 128)
    for (int i = 0; i < nq; i ++) {
        auto ans_disi = answer_dists + topk * i;
        auto ans_idsi = answer_ids + topk * i;
        heap_heapify<HEAPT>(topk, ans_disi, ans_idsi);
    }
    rc.RecordSection("heapify answers heaps");

    uint32_t vector_size = dq * sizeof(DATAT);
    uint32_t node_size = vector_size + sizeof(uint32_t);

    std::vector<refine_stat> refine_statastics(K1);
    std::vector<std::mutex> mtx(nq);
#pragma omp parallel for
    for (int i = 0; i < K1; i ++) {
        if (refine_records[i].size() == 0)
            continue;
        std::sort(refine_records[i].begin(), refine_records[i].end(), [](const auto &l, const auto &r) {
                return l.first < r.first;
                });
        uint32_t pre_off = refine_records[i][0].first;
        uint64_t pn;
        pn = pre_off / nnpp;
        char* dat_bufi = new char[page_size];
        uint32_t global_id;
        raw_data_file_handlers[i].seekg((pn + 1) * page_size, raw_data_file_handlers[i].beg);
        raw_data_file_handlers[i].read(dat_bufi, page_size);
        refine_statastics[i].vector_load_cnt = 1;
        refine_statastics[i].id_load_cnt = 1;
        for (int j = 0; j < refine_records[i].size(); j ++) {
            int64_t refine_off = refine_records[i][j].first;
            if (pre_off != refine_off) {
                refine_statastics[i].different_offset_cnt ++;
            }
            pre_off = refine_off;
            if (refine_off >= (pn + 1) * nnpp) {
                pn = refine_off / nnpp;
                raw_data_file_handlers[i].seekg((pn + 1) * page_size, raw_data_file_handlers[i].beg);
                raw_data_file_handlers[i].read(dat_bufi, page_size);
                refine_statastics[i].vector_load_cnt ++;
            } else {
                refine_statastics[i].vector_page_hit_cnt ++;
            }
            uint32_t qid = refine_records[i][j].second;
            auto dis = dis_computer((DATAT*)(dat_bufi + (refine_off % nnpp) * node_size), pquery + qid * dq, dq);
            std::unique_lock<std::mutex> lk(mtx[qid]);
            if (HEAPT::cmp(answer_dists[topk * qid], dis)) {
                heap_swap_top<HEAPT>(topk, answer_dists + topk * qid, answer_ids + topk * qid, dis, *((uint32_t*)(dat_bufi + (refine_off % nnpp) * node_size + vector_size)));
            }
        }

        delete[] dat_bufi;
    }
    int64_t vector_load_tot = 0;
    int64_t id_load_tot = 0;
    int64_t vector_page_hit_tot = 0;
    int64_t id_page_hit_tot = 0;
    int64_t different_offset_tot = 0;
    for (auto i = 0; i < K1; i ++) {
        vector_load_tot += refine_statastics[i].vector_load_cnt;
        id_load_tot += refine_statastics[i].id_load_cnt;
        vector_page_hit_tot += refine_statastics[i].vector_page_hit_cnt - 1;
        id_page_hit_tot += refine_statastics[i].id_page_hit_cnt - 1;
        different_offset_tot += refine_statastics[i].different_offset_cnt;
    }
    std::cout << "total refine vectors: " << (int64_t)nq * refine_topk
              << ", load vector pages: " << vector_load_tot << ", load ids pages: " << id_load_tot
              << ", vector page hit: " << vector_page_hit_tot << ", ids page hit: " << id_page_hit_tot
              << ", different offsets: " << different_offset_tot << std::endl;
    rc.RecordSection("calculate done.");

    for (int i = 0; i < K1; i ++) {
        std::string data_filei = index_path + CLUSTER + std::to_string(i) + "-" + RAWDATA + BIN;
        raw_data_file_handlers[i].close();
    }

    delete[] dat_buf;
    rc.ElapseFromBegin("aligned refine done.");
}

// cannot work
template<typename DATAT, typename DISTT, typename HEAPT>
void aligned_refine_c(const std::string& index_path,
                    const int K1,
                    const uint32_t nq,
                    const uint32_t dq,
                    const int topk,
                    const int refine_topk,
                    uint64_t* pq_offsets,
                    const DATAT* pquery,
                    DISTT*& answer_dists,
                    uint32_t*& answer_ids,
                    Computer<DATAT, DATAT, DISTT>& dis_computer) {
    TimeRecorder rc("aligned aligned_refine_c with C-style interface: open(), pread(), close(). Here with O_RDONLY | O_DIRECT");
    std::cout << "aligned refine parameters:" << std::endl;
    std::cout << " index_path: " << index_path
              << " cluster size: " << K1
              << " number of query: " << nq
              << " dim of query: " << dq
              << " topk: " << topk
              << " refine_topk:" << refine_topk
              << " pq_offsets:" << pq_offsets
              << " pquery:" << static_cast<const void *>(pquery)
              << " answer_dists:" << static_cast<void *>(answer_dists)
              << " answer_ids:" << static_cast<void *>(answer_ids)
              << std::endl;
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> refine_records(K1);
    for (int64_t i = 0; i < nq; i ++) {
        auto pq_offseti = pq_offsets + i * refine_topk;
        for (int j = 0; j < refine_topk; j ++) {
            if (pq_offseti[j] == (uint64_t)(-1))
                continue;
            uint32_t cid, off, qid;
            parse_refine_id(pq_offseti[j], cid, off, qid);
            refine_records[cid].emplace_back(off, qid);
        }
    }
    rc.RecordSection("parse_refine_id done");

    uint32_t page_size = PAGESIZE;
    uint32_t nvpp = 0;
    uint32_t nipp = 0;
    uint32_t npv = 0;
    uint32_t npi = 0;

    char* dat_buf = new char[page_size];
    char* ids_buf = new char[page_size];
    memset(dat_buf, 0, page_size);
    memset(ids_buf, 0, page_size);

    std::vector<int> raw_data_file_fds;
    raw_data_file_fds.reserve(K1);
    std::vector<int> ids_data_file_fds;
    ids_data_file_fds.reserve(K1);
    for (int i = 0; i < K1; i ++) {
        std::string aligned_data_filei = index_path + CLUSTER + "-" + std::to_string(i) + RAWDATA + BIN;
        std::string aligned_ids_filei  = index_path + CLUSTER + "-" + std::to_string(i) + GLOBAL_IDS + BIN;
        raw_data_file_fds.emplace_back(open(aligned_data_filei.c_str(), O_RDONLY));
        assert(raw_data_file_fds.back() != -1);
        ids_data_file_fds.emplace_back(open(aligned_ids_filei.c_str(), O_RDONLY));
        assert(ids_data_file_fds.back() != -1);

        uint32_t clu_size, clu_dim, clu_id_size, clu_id_dim;
        uint32_t ps, check;
        int pread_size = pread(raw_data_file_fds.back(), dat_buf, page_size, 0);
        assert(pread_size == page_size);
        pread_size = pread(ids_data_file_fds.back(), ids_buf, page_size, 0);
        assert(pread_size == page_size);

        memcpy(&clu_size, dat_buf + 0 * sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&clu_dim , dat_buf + 1 * sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&ps      , dat_buf + 2 * sizeof(uint32_t), sizeof(uint32_t));
        assert(ps == page_size);
        memcpy(&nvpp    , dat_buf + 3 * sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&npv     , dat_buf + 4 * sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&check   , dat_buf + 5 * sizeof(uint32_t), sizeof(uint32_t));
        assert(check == i);
        memcpy(&clu_id_size, ids_buf + 0 * sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&clu_id_dim , ids_buf + 1 * sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&ps         , ids_buf + 2 * sizeof(uint32_t), sizeof(uint32_t));
        assert(ps == page_size);
        memcpy(&nipp    , ids_buf + 3 * sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&npi     , ids_buf + 4 * sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&check   , ids_buf + 5 * sizeof(uint32_t), sizeof(uint32_t));
        assert(check == i);
        std::cout << "cluster-" << i << " has " << clu_size << " vectors,"
                  << " has clu_dim = " << clu_dim
                  << " clu_id_size = " << clu_id_size
                  << " clu_id_dim = " << clu_id_dim
                  << std::endl;
        std::cout << "main meta: " << std::endl;
        std::cout << " number of vectors in per page: " << nvpp
                  << " number of pages 4 vector: " << npv
                  << " number of ids in per page: " << nipp
                  << " number of pages 4 id: " << npi
                  << std::endl;
    }
    rc.RecordSection("open rawdata and idsdata file handlers");

    // init answer heap
#pragma omp parallel for schedule (static, 128)
    for (int i = 0; i < nq; i ++) {
        auto ans_disi = answer_dists + topk * i;
        auto ans_idsi = answer_ids + topk * i;
        heap_heapify<HEAPT>(topk, ans_disi, ans_idsi);
    }
    rc.RecordSection("heapify answers heaps");

    uint32_t vector_size = dq * sizeof(DATAT);

    std::vector<refine_stat> refine_statastics(K1);
    std::vector<std::mutex> mtx(nq);
#pragma omp parallel for
    for (int i = 0; i < K1; i ++) {
        if (refine_records[i].size() == 0)
            continue;
        std::sort(refine_records[i].begin(), refine_records[i].end(), [](const auto &l, const auto &r) {
            return l.first < r.first;
        });
        uint32_t pre_off = refine_records[i][0].first;
        uint64_t pv, pi;
        pv = pre_off / nvpp;
        pi = pre_off / nipp;
        char* dat_bufi = new char[page_size];
        char* ids_bufi = new char[page_size];
        uint32_t global_id;

        int pread_size = pread(raw_data_file_fds[i], dat_bufi, page_size, (pv + 1) * page_size);
        assert(pread_size == dq * sizeof(DATAT));
        pread_size = pread(ids_data_file_fds[i], ids_bufi, page_size, (pi + 1) * page_size);
        assert(pread_size == sizeof(uint32_t));

        refine_statastics[i].vector_load_cnt = 1;
        refine_statastics[i].id_load_cnt = 1;
        for (int j = 0; j < refine_records[i].size(); j ++) {
            int64_t refine_off = refine_records[i][j].first;
            if (pre_off != refine_off) {
                refine_statastics[i].different_offset_cnt ++;
            }
            pre_off = refine_off;
            if (refine_off >= (pv + 1) * nvpp) {
                pv = refine_off / nvpp;
                pread_size = pread(raw_data_file_fds[i], dat_bufi, page_size, (pv + 1) * page_size);
                assert(pread_size == page_size);
                refine_statastics[i].vector_load_cnt ++;
            } else {
                refine_statastics[i].vector_page_hit_cnt ++;
            }
            if (refine_off >= (pi + 1) * nipp) {
                pi = refine_off / nipp;
                pread_size = pread(ids_data_file_fds[i], ids_bufi, page_size, (pi + 1) * page_size);
                assert(pread_size == sizeof(uint32_t));
                refine_statastics[i].id_load_cnt ++;
            } else {
                refine_statastics[i].id_page_hit_cnt ++;
            }
            uint32_t qid = refine_records[i][j].second;
            auto dis = dis_computer((DATAT*)(dat_bufi + (refine_off % nvpp) * vector_size), pquery + qid * dq, dq);
            std::unique_lock<std::mutex> lk(mtx[qid]);
            if (HEAPT::cmp(answer_dists[topk * qid], dis)) {
                heap_swap_top<HEAPT>(topk, answer_dists + topk * qid, answer_ids + topk * qid, dis, *((uint32_t*)(ids_bufi + (refine_off % nipp) * sizeof(uint32_t))));
            }
        }

        delete[] dat_bufi;
        delete[] ids_bufi;
    }
    int64_t vector_load_tot = 0;
    int64_t id_load_tot = 0;
    int64_t vector_page_hit_tot = 0;
    int64_t id_page_hit_tot = 0;
    int64_t different_offset_tot = 0;
    for (auto i = 0; i < K1; i ++) {
        vector_load_tot += refine_statastics[i].vector_load_cnt;
        id_load_tot += refine_statastics[i].id_load_cnt;
        vector_page_hit_tot += refine_statastics[i].vector_page_hit_cnt - 1;
        id_page_hit_tot += refine_statastics[i].id_page_hit_cnt - 1;
        different_offset_tot += refine_statastics[i].different_offset_cnt;
    }
    std::cout << "total refine vectors: " << (int64_t)nq * refine_topk
              << ", load vector pages: " << vector_load_tot << ", load ids pages: " << id_load_tot
              << ", vector page hit: " << vector_page_hit_tot << ", ids page hit: " << id_page_hit_tot
              << ", different offsets: " << different_offset_tot << std::endl;
    rc.RecordSection("calculate done.");

    for (int i = 0; i < K1; i ++) {
        std::string data_filei = index_path + CLUSTER + "-" + std::to_string(i) + RAWDATA + BIN;
        std::string ids_filei  = index_path + CLUSTER + "-" + std::to_string(i) + GLOBAL_IDS + BIN;
        close(raw_data_file_fds[i]);
        close(ids_data_file_fds[i]);
    }

    delete[] dat_buf;
    delete[] ids_buf;
    rc.ElapseFromBegin("aligned refine done.");
}

template<typename DATAT, typename DISTT, typename HEAPT>
void refine_c(const std::string& index_path,
            const int K1,
            const uint32_t nq,
            const uint32_t dq,
            const int topk,
            const int refine_topk,
            uint64_t* pq_offsets,
            const DATAT* pquery,
            DISTT*& answer_dists,
            uint32_t*& answer_ids,
            Computer<DATAT, DATAT, DISTT>& dis_computer) {
    TimeRecorder rc("refine_c with C-style interface: open(), pread(), close(). Here with O_RDONLY | O_DIRECT");
    std::cout << "refine parameters:" << std::endl;
    std::cout << " index_path: " << index_path
              << " cluster size: " << K1
              << " number of query: " << nq
              << " dim of query: " << dq
              << " topk: " << topk
              << " refine_topk:" << refine_topk
              << " pq_offsets:" << pq_offsets
              << " pquery:" << static_cast<const void *>(pquery)
              << " answer_dists:" << static_cast<void *>(answer_dists)
              << " answer_ids:" << static_cast<void *>(answer_ids)
              << std::endl;
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> refine_records(K1);
    for (int64_t i = 0; i < nq; i ++) {
        auto pq_offseti = pq_offsets + i * refine_topk;
        for (int j = 0; j < refine_topk; j ++) {
            if (pq_offseti[j] == (uint64_t)(-1))
                continue;
            uint32_t cid, off, qid;
            parse_refine_id(pq_offseti[j], cid, off, qid);
            refine_records[cid].emplace_back(off, qid);
        }
    }
    rc.RecordSection("parse_refine_id done");

    std::vector<int> raw_data_file_fds;
    raw_data_file_fds.reserve(K1);
    std::vector<int> ids_data_file_fds;
    ids_data_file_fds.reserve(K1);
    constexpr size_t O_DIRECT_ALIGNMENT = 512;
    for (int i = 0; i < K1; i ++) {
        std::string data_filei = index_path + CLUSTER + std::to_string(i) + RAWDATA + BIN;
        std::string ids_filei  = index_path + CLUSTER + std::to_string(i) + GLOBAL_IDS + BIN;
        raw_data_file_fds.emplace_back(open(data_filei.c_str(), O_RDONLY | O_DIRECT));
        assert(raw_data_file_fds.back() != -1);
        ids_data_file_fds.emplace_back(open(ids_filei.c_str(), O_RDONLY | O_DIRECT));
        assert(ids_data_file_fds.back() != -1);

        void *clu_size;
        int posix_memalign_size = posix_memalign(&clu_size, O_DIRECT_ALIGNMENT, sizeof(uint32_t));
        assert(posix_memalign_size == sizeof(uint32_t));
        void *clu_dim;
        posix_memalign_size = posix_memalign(&clu_dim, O_DIRECT_ALIGNMENT, sizeof(uint32_t));
        assert(posix_memalign_size == sizeof(uint32_t));
        void *clu_id_size;
        posix_memalign_size = posix_memalign(&clu_id_size, O_DIRECT_ALIGNMENT, sizeof(uint32_t));
        assert(posix_memalign_size == sizeof(uint32_t));
        void *clu_id_dim;
        posix_memalign_size = posix_memalign(&clu_id_dim, O_DIRECT_ALIGNMENT, sizeof(uint32_t));
        assert(posix_memalign_size == sizeof(uint32_t));

        int pread_size = pread(raw_data_file_fds.back(), clu_size, sizeof(uint32_t), 0);
        assert(pread_size == sizeof(uint32_t));
        pread_size = pread(raw_data_file_fds.back(), clu_dim, sizeof(uint32_t), 0 + sizeof(uint32_t));
        assert(pread_size == sizeof(uint32_t));
        pread_size = pread(ids_data_file_fds.back(), clu_id_size, sizeof(uint32_t), 0);
        assert(pread_size == sizeof(uint32_t));
        pread_size = pread(ids_data_file_fds.back(), clu_id_dim, sizeof(uint32_t), 0 + sizeof(uint32_t));
        assert(pread_size == sizeof(uint32_t));

        std::cout << "cluster-" << i << " has " << *static_cast<uint32_t *>(clu_size) << " vectors,"
                  << " has clu_dim = " << *static_cast<uint32_t *>(clu_dim)
                  << " clu_id_size = " << *static_cast<uint32_t *>(clu_id_size)
                  << " clu_id_dim = " << *static_cast<uint32_t *>(clu_id_dim)
                  << std::endl;
        free(clu_size);
        free(clu_dim);
        free(clu_id_size);
        free(clu_id_dim);
    }
    rc.RecordSection("open rawdata and idsdata file FDs with open()");

    // init answer heap
#pragma omp parallel for schedule (static, 128)
    for (int i = 0; i < nq; i ++) {
        auto ans_disi = answer_dists + topk * i;
        auto ans_idsi = answer_ids + topk * i;
        heap_heapify<HEAPT>(topk, ans_disi, ans_idsi);
    }
    rc.RecordSection("heapify answers heaps");

    // statistics
    std::vector<int> load_vectors(K1, 0);
    std::vector<std::mutex> mtx(nq);
#pragma omp parallel for
    for (int i = 0; i < K1; i ++) {
        if (refine_records[i].size() == 0)
            continue;
        std::sort(refine_records[i].begin(), refine_records[i].end(), [](const auto &l, const auto &r) {
                return l.first < r.first;
                });
        uint64_t pre_off = refine_records[i][0].first;
        uint32_t meta_bytes = 8; // pass meta
        void *data_bufi;
        int posix_memalign_size = posix_memalign(&data_bufi, O_DIRECT_ALIGNMENT, dq * sizeof(DATAT));
        assert(posix_memalign_size == dq * sizeof(DATAT));
        void *global_id;
        posix_memalign_size = posix_memalign(&global_id, O_DIRECT_ALIGNMENT, sizeof(uint32_t));
        assert(posix_memalign_size == sizeof(uint32_t));

        int pread_size = pread(raw_data_file_fds[i], data_bufi, dq * sizeof(DATAT), meta_bytes + pre_off * dq * sizeof(DATAT));
        assert(pread_size == dq * sizeof(DATAT));
        pread_size = pread(ids_data_file_fds[i], global_id, sizeof(uint32_t), meta_bytes + pre_off * sizeof(uint32_t));
        assert(pread_size == sizeof(uint32_t));
        load_vectors[i] ++;
        // for debug
        for (int j = 0; j < refine_records[i].size(); j ++) {
            if (refine_records[i][j].first != pre_off) {
                pre_off = refine_records[i][j].first;
                pread_size = pread(raw_data_file_fds[i], data_bufi, dq * sizeof(DATAT), meta_bytes + pre_off * dq * sizeof(DATAT));
                assert(pread_size == dq * sizeof(DATAT));
                pread_size = pread(ids_data_file_fds[i], global_id, sizeof(uint32_t), meta_bytes + pre_off * sizeof(uint32_t));
                assert(pread_size == sizeof(uint32_t));
                assert(*static_cast<uint32_t *>(global_id) >= 0);
                load_vectors[i] ++;
            }
            uint32_t qid = refine_records[i][j].second;
            auto dis = dis_computer(static_cast<DATAT*>(data_bufi), pquery + qid * dq, dq);
            std::unique_lock<std::mutex> lk(mtx[qid]);
            if (HEAPT::cmp(answer_dists[topk * qid], dis)) {
                heap_swap_top<HEAPT>(topk, answer_dists + topk * qid, answer_ids + topk * qid, dis, *static_cast<uint32_t *>(global_id));
            }
        }
        free(data_bufi);
        free(global_id);
    }
    rc.RecordSection("calculate done.");
    int tot = 0;
    std::cout << "show load refine vectors of each cluster:" << std::endl;
    for (int i = 0; i < K1; i ++) {
        std::cout << "cluster-" << i << ": " << load_vectors[i] << "/" << refine_topk << std::endl;
        tot += load_vectors[i];
    }
    std::cout << "total load refine vectors: " << tot << "/" << refine_topk * nq << std::endl;

    for (int i = 0; i < K1; i ++) {
        std::string data_filei = index_path + CLUSTER + std::to_string(i) + RAWDATA + BIN;
        std::string ids_filei  = index_path + CLUSTER + std::to_string(i) + GLOBAL_IDS + BIN;
        close(raw_data_file_fds[i]);
        close(ids_data_file_fds[i]);
    }
    rc.ElapseFromBegin("refine_c with C-style interface done.");
}

template<typename DISTT, typename HEAPT>
void save_sift_answer(const std::string& answer_bin_file,
                  const int topk,
                  const uint32_t nq,
                  DISTT*& answer_dists,
                  uint32_t*& answer_ids) {
    std::ofstream answer_writer(answer_bin_file, std::ios::binary);
    for (int i = 0; i < nq; i ++) {
        auto ans_disi = answer_dists + topk * i;
        auto ans_idsi = answer_ids + topk * i;
        answer_writer.write((char*)&topk, sizeof(uint32_t));
        for (int j = topk; j > 0; j --) {
            answer_writer.write((char*)ans_idsi, sizeof(uint32_t));
            answer_writer.write((char*)ans_disi, sizeof(DISTT));
            heap_pop<HEAPT>(j, ans_disi, ans_idsi);
        }
    }
    answer_writer.close();
}

template<typename DISTT, typename HEAPT>
void save_comp_answer(const std::string& answer_bin_file,
                  const int topk,
                  const uint32_t nq,
                  DISTT*& answer_dists,
                  uint32_t*& answer_ids) {
    std::ofstream answer_writer(answer_bin_file, std::ios::binary);
    answer_writer.write((char*)&nq, sizeof(uint32_t));
    answer_writer.write((char*)&topk, sizeof(uint32_t));

    for (int i = 0; i < nq; i ++) {
        auto ans_disi = answer_dists + topk * i;
        auto ans_idsi = answer_ids + topk * i;
        heap_reorder<HEAPT>(topk, ans_disi, ans_idsi);
    }

    uint32_t tot = nq * topk;
    answer_writer.write((char*)answer_ids, tot * sizeof(uint32_t));
    answer_writer.write((char*)answer_dists, tot * sizeof(DISTT));

    answer_writer.close();
}

template<typename DISTT, typename HEAPT>
void save_answers(const std::string& answer_bin_file,
                  const int topk,
                  const uint32_t nq,
                  DISTT*& answer_dists,
                  uint32_t*& answer_ids,
                  bool use_comp_format) {
    TimeRecorder rc("save answers");
    std::cout << "save answer parameters:" << std::endl;
    std::cout << " answer_bin_file: " << answer_bin_file
              << " topk: " << topk
              << " nq: " << nq
              << " answer_dists: " << answer_dists
              << " answer_ids: " << answer_ids
              << " use_comp_format: " << use_comp_format
              << std::endl;

    auto f = use_comp_format ? save_comp_answer<DISTT, HEAPT> : save_sift_answer<DISTT, HEAPT>;

    f(answer_bin_file, topk, nq, answer_dists, answer_ids);

    rc.ElapseFromBegin("save answers done.");
}

template<typename DATAT, typename DISTT, typename HEAPT, typename HEAPTT>
void search_bigann(const std::string& index_path,
                   const std::string& query_bin_file,
                   const std::string& answer_bin_file,
                   const int nprobe,
                   const int refine_nprobe,
                   const int topk,
                   const int refine_topk,
                   std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
                   PQResidualQuantizer<HEAPTT, DATAT, uint8_t>& quantizer,
                   const int K1,
                   std::vector<std::vector<uint8_t>>& pq_codebook,
                   std::vector<std::vector<uint32_t>>& meta,
                   Computer<DATAT, DATAT, DISTT>& dis_computer) {
    TimeRecorder rc("search bigann");

    std::cout << "search bigann parameters:" << std::endl;
    std::cout << " index_path: " << index_path
              << " query_bin_file: " << query_bin_file
              << " answer_bin_file: " << answer_bin_file
              << " nprobe: " << nprobe
              << " refine_nprobe: " << refine_nprobe
              << " topk: " << topk
              << " refine topk: " << refine_topk
              << " K1: " << K1
              << std::endl;


    DATAT* pquery = nullptr;
    DISTT* answer_dists = nullptr;
    uint32_t* answer_ids = nullptr;
    DISTT* pq_distance = nullptr;
    uint64_t* pq_offsets = nullptr;
    uint64_t* p_labels = nullptr;

    uint32_t nq, dq;

    read_bin_file<DATAT>(query_bin_file, pquery, nq, dq);
    rc.RecordSection("load query done.");

    std::cout << "query numbers: " << nq << " query dims: " << dq << std::endl;

    pq_distance = new DISTT[(int64_t)nq * refine_topk];
    answer_dists = new DISTT[(int64_t)nq * topk];
    answer_ids = new uint32_t[(int64_t)nq * topk];
    pq_offsets = new uint64_t[(int64_t)nq * refine_topk];
    p_labels = new uint64_t[(int64_t)nq * nprobe];

    search_graph<DATAT>(index_hnsw, nq, dq, nprobe, refine_nprobe, pquery, p_labels);
    rc.RecordSection("search buckets done.");

    float* ivf_centroids = nullptr;
    uint32_t c_n, c_dim;
    read_bin_file<float>(index_path + BUCKET + CENTROIDS + BIN, ivf_centroids, c_n, c_dim);
    search_pq_residual_quantizer<DATAT, DISTT, HEAPTT>(quantizer, nq, dq, ivf_centroids, p_labels, nprobe, 
                     refine_topk, K1, pquery, pq_codebook, 
                     meta, pq_distance, pq_offsets);
    delete[] ivf_centroids;

    rc.RecordSection("pq residual search done.");

    // refine<DATAT, DISTT, HEAPT>(index_path, K1, nq, dq, topk, refine_topk, pq_offsets, pquery, answer_dists, answer_ids, dis_computer);
    aligned_refine<DATAT, DISTT, HEAPT>(index_path, K1, nq, dq, topk, refine_topk, pq_offsets, pquery, answer_dists, answer_ids, dis_computer);  // refine with C++ std::ifstream
//    aligned_refine_c<DATAT, DISTT, HEAPT>(index_path, K1, nq, dq, topk, refine_topk, pq_offsets, pquery, answer_dists, answer_ids, dis_computer);  // refine_c with C open(), pread(), close()
    rc.RecordSection("refine done");
    // write answers
    save_answers<DISTT, HEAPT>(answer_bin_file, topk, nq, answer_dists, answer_ids, true);
    rc.RecordSection("write answers done");

    delete[] pquery;
    delete[] p_labels;
    delete[] pq_distance;
    delete[] pq_offsets;
    delete[] answer_ids;
    delete[] answer_dists;
    rc.ElapseFromBegin("search bigann totally done");
}

template<typename DATAT, typename DISTT, typename HEAPT, typename HEAPTT>
void search_bigann(const std::string& index_path,
                   const std::string& query_bin_file,
                   const std::string& answer_bin_file,
                   const int nprobe,
                   const int refine_nprobe,
                   const int topk,
                   const int refine_topk,
                   std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
                   ProductQuantizer<HEAPTT, DATAT, uint8_t>& pq_quantizer,
                   const int K1,
                   std::vector<std::vector<uint8_t>>& pq_codebook,
                   std::vector<std::vector<uint32_t>>& meta,
                   Computer<DATAT, DATAT, DISTT>& dis_computer) {
    // std::cout << "show pq centroids at the begining:" << std::endl;
    // pq_quantizer.show_centroids();
    TimeRecorder rc("search bigann");
    std::cout << "search bigann parameters:" << std::endl;
    std::cout << " index_path: " << index_path
              << " query_bin_file: " << query_bin_file
              << " answer_bin_file: " << answer_bin_file
              << " nprobe: " << nprobe
              << " refine_nprobe: " << refine_nprobe
              << " topk: " << topk
              << " refine topk: " << refine_topk
              << " K1: " << K1
              << std::endl;


    DATAT* pquery = nullptr;
    DISTT* answer_dists = nullptr;
    uint32_t* answer_ids = nullptr;
    DISTT* pq_distance = nullptr;
    uint64_t* pq_offsets = nullptr;
    uint64_t* p_labels = nullptr;

    uint32_t nq, dq;

    read_bin_file<DATAT>(query_bin_file, pquery, nq, dq);
    rc.RecordSection("load query done.");
    // for debug
    // nq = 1;
    // int topk = 10;
    // int refine_topk = 20;
    // int nprobe = 5;
    // int refine_nprobe = 10;

    std::cout << "query numbers: " << nq << " query dims: " << dq << std::endl;

    pq_distance = new DISTT[(int64_t)nq * refine_topk];
    answer_dists = new DISTT[(int64_t)nq * topk];
    answer_ids = new uint32_t[(int64_t)nq * topk];
    pq_offsets = new uint64_t[(int64_t)nq * refine_topk];
    p_labels = new uint64_t[(int64_t)nq * nprobe];

    search_graph<DATAT>(index_hnsw, nq, dq, nprobe, refine_nprobe, pquery, p_labels);
    rc.RecordSection("search buckets done.");
    /*
    {// for debug
        std::cout << "show details after search_graph:" << std::endl;
        for (auto i = 0; i < nq; i ++) {
            auto p_labeli = p_labels + i * nprobe;
            uint32_t cid, bid, off;
            std::cout << "cluster info of the " << i << "th query:";
            for (auto j = 0; j < nprobe; j ++) {
                parse_id(p_labeli[j], cid, bid, off);
                assert(cid < K1);
                std::cout << "(" << cid << ", " << bid << ", " << off << ") ";
            }
            std::cout << std::endl;
        }

    }
    */

    search_pq_quantizer<DATAT, DISTT, HEAPTT>(pq_quantizer, nq, dq, p_labels, nprobe, 
                         refine_topk, K1, pquery, pq_codebook, 
                         meta, pq_distance, pq_offsets);
    rc.RecordSection("pq search done.");
    /*
    {// for debug
        std::cout << "show details after search quantizer:" << std::endl;
        for (auto i = 0; i < nq; i ++) {
            auto pq_offseti = pq_offsets + i * refine_topk;
            auto pq_distancei = pq_distance + i * refine_topk;
            std::cout << "refine info of the " << i << "th query:";
            for (auto j = 0; j < refine_topk; j ++) {
                if (pq_offseti[j] == (uint64_t)(-1))
                    continue;
                uint32_t cid, off, qid;
                parse_refine_id(pq_offseti[j], cid, off, qid);
                std::cout << "(" << cid << ", " << off << ", " << qid << ", pqdis: " << pq_distancei[j] << ") " << std::endl;
            }
            std::cout << std::endl;
        }
    }
    */


    aligned_refine<DATAT, DISTT, HEAPT>(index_path, K1, nq, dq, topk, refine_topk, pq_offsets, pquery, answer_dists, answer_ids, dis_computer);  // refine with C++ std::ifstream
//    aligned_refine_c<DATAT, DISTT, HEAPT>(index_path, K1, nq, dq, topk, refine_topk, pq_offsets, pquery, answer_dists, answer_ids, dis_computer);  // refine_c with C open(), pread(), close()
//    refine_c<DATAT, DISTT, HEAPT>(index_path, K1, nq, dq, topk, refine_topk, pq_offsets, pquery, answer_dists, answer_ids, dis_computer);  // refine_c with C open(), pread(), close()
    rc.RecordSection("refine done");
    // write answers
    save_answers<DISTT, HEAPT>(answer_bin_file, topk, nq, answer_dists, answer_ids, true);
    rc.RecordSection("write answers done");

    delete[] pquery;
    delete[] p_labels;
    delete[] pq_distance;
    delete[] pq_offsets;
    delete[] answer_ids;
    delete[] answer_dists;
    rc.ElapseFromBegin("search bigann totally done");
}

template 
void search_bigann<float, float, CMax<float, uint32_t>, CMax<float, uint64_t>>(const std::string& index_path,
                   const std::string& query_bin_file,
                   const std::string& answer_bin_file,
                   const int nprobe,
                   const int refine_nprobe,
                   const int topk,
                   const int refine_topk,
                   std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
                   PQResidualQuantizer<CMax<float, uint64_t>, float, uint8_t>& pq_quantizer,
                   const int K1,
                   std::vector<std::vector<uint8_t>>& pq_codebook,
                   std::vector<std::vector<uint32_t>>& meta,
                   Computer<float, float, float>& dis_computer);

template 
void search_bigann<float, float, CMin<float, uint32_t>, CMin<float, uint64_t>>(const std::string& index_path,
                   const std::string& query_bin_file,
                   const std::string& answer_bin_file,
                   const int nprobe,
                   const int refine_nprobe,
                   const int topk,
                   const int refine_topk,
                   std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
                   PQResidualQuantizer<CMin<float, uint64_t>, float, uint8_t>& pq_quantizer,
                   const int K1,
                   std::vector<std::vector<uint8_t>>& pq_codebook,
                   std::vector<std::vector<uint32_t>>& meta,
                   Computer<float, float, float>& dis_computer);

template 
void search_bigann<uint8_t, uint32_t, CMax<uint32_t, uint32_t>, CMax<uint32_t, uint64_t>>(const std::string& index_path,
                   const std::string& query_bin_file,
                   const std::string& answer_bin_file,
                   const int nprobe,
                   const int refine_nprobe,
                   const int topk,
                   const int refine_topk,
                   std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
                   PQResidualQuantizer<CMax<uint32_t, uint64_t>, uint8_t, uint8_t>& pq_quantizer,
                   const int K1,
                   std::vector<std::vector<uint8_t>>& pq_codebook,
                   std::vector<std::vector<uint32_t>>& meta,
                   Computer<uint8_t, uint8_t, uint32_t>& dis_computer);

template 
void search_bigann<uint8_t, uint32_t, CMin<uint32_t, uint32_t>, CMin<uint32_t, uint64_t>>(const std::string& index_path,
                   const std::string& query_bin_file,
                   const std::string& answer_bin_file,
                   const int nprobe,
                   const int refine_nprobe,
                   const int topk,
                   const int refine_topk,
                   std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
                   PQResidualQuantizer<CMin<uint32_t, uint64_t>, uint8_t, uint8_t>& pq_quantizer,
                   const int K1,
                   std::vector<std::vector<uint8_t>>& pq_codebook,
                   std::vector<std::vector<uint32_t>>& meta,
                   Computer<uint8_t, uint8_t, uint32_t>& dis_computer);

template 
void search_bigann<int8_t, int32_t, CMax<int32_t, uint32_t>, CMax<int32_t, uint64_t>>(const std::string& index_path,
                   const std::string& query_bin_file,
                   const std::string& answer_bin_file,
                   const int nprobe,
                   const int refine_nprobe,
                   const int topk,
                   const int refine_topk,
                   std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
                   PQResidualQuantizer<CMax<int32_t, uint64_t>, int8_t, uint8_t>& pq_quantizer,
                   const int K1,
                   std::vector<std::vector<uint8_t>>& pq_codebook,
                   std::vector<std::vector<uint32_t>>& meta,
                   Computer<int8_t, int8_t, int32_t>& dis_computer);

template 
void search_bigann<int8_t, int32_t, CMin<int32_t, uint32_t>, CMin<int32_t, uint64_t>>(const std::string& index_path,
                   const std::string& query_bin_file,
                   const std::string& answer_bin_file,
                   const int nprobe,
                   const int refine_nprobe,
                   const int topk,
                   const int refine_topk,
                   std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
                   PQResidualQuantizer<CMin<int32_t, uint64_t>, int8_t, uint8_t>& pq_quantizer,
                   const int K1,
                   std::vector<std::vector<uint8_t>>& pq_codebook,
                   std::vector<std::vector<uint32_t>>& meta,
                   Computer<int8_t, int8_t, int32_t>& dis_computer);

template 
void search_bigann<float, float, CMax<float, uint32_t>, CMax<float, uint64_t>>(const std::string& index_path,
                   const std::string& query_bin_file,
                   const std::string& answer_bin_file,
                   const int nprobe,
                   const int refine_nprobe,
                   const int topk,
                   const int refine_topk,
                   std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
                   ProductQuantizer<CMax<float, uint64_t>, float, uint8_t>& pq_quantizer,
                   const int K1,
                   std::vector<std::vector<uint8_t>>& pq_codebook,
                   std::vector<std::vector<uint32_t>>& meta,
                   Computer<float, float, float>& dis_computer);

template 
void search_bigann<float, float, CMin<float, uint32_t>, CMin<float, uint64_t>>(const std::string& index_path,
                   const std::string& query_bin_file,
                   const std::string& answer_bin_file,
                   const int nprobe,
                   const int refine_nprobe,
                   const int topk,
                   const int refine_topk,
                   std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
                   ProductQuantizer<CMin<float, uint64_t>, float, uint8_t>& pq_quantizer,
                   const int K1,
                   std::vector<std::vector<uint8_t>>& pq_codebook,
                   std::vector<std::vector<uint32_t>>& meta,
                   Computer<float, float, float>& dis_computer);

template 
void search_bigann<uint8_t, uint32_t, CMax<uint32_t, uint32_t>, CMax<uint32_t, uint64_t>>(const std::string& index_path,
                   const std::string& query_bin_file,
                   const std::string& answer_bin_file,
                   const int nprobe,
                   const int refine_nprobe,
                   const int topk,
                   const int refine_topk,
                   std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
                   ProductQuantizer<CMax<uint32_t, uint64_t>, uint8_t, uint8_t>& pq_quantizer,
                   const int K1,
                   std::vector<std::vector<uint8_t>>& pq_codebook,
                   std::vector<std::vector<uint32_t>>& meta,
                   Computer<uint8_t, uint8_t, uint32_t>& dis_computer);

template 
void search_bigann<uint8_t, uint32_t, CMin<uint32_t, uint32_t>, CMin<uint32_t, uint64_t>>(const std::string& index_path,
                   const std::string& query_bin_file,
                   const std::string& answer_bin_file,
                   const int nprobe,
                   const int refine_nprobe,
                   const int topk,
                   const int refine_topk,
                   std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
                   ProductQuantizer<CMin<uint32_t, uint64_t>, uint8_t, uint8_t>& pq_quantizer,
                   const int K1,
                   std::vector<std::vector<uint8_t>>& pq_codebook,
                   std::vector<std::vector<uint32_t>>& meta,
                   Computer<uint8_t, uint8_t, uint32_t>& dis_computer);

template 
void search_bigann<int8_t, int32_t, CMax<int32_t, uint32_t>, CMax<int32_t, uint64_t>>(const std::string& index_path,
                   const std::string& query_bin_file,
                   const std::string& answer_bin_file,
                   const int nprobe,
                   const int refine_nprobe,
                   const int topk,
                   const int refine_topk,
                   std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
                   ProductQuantizer<CMax<int32_t, uint64_t>, int8_t, uint8_t>& pq_quantizer,
                   const int K1,
                   std::vector<std::vector<uint8_t>>& pq_codebook,
                   std::vector<std::vector<uint32_t>>& meta,
                   Computer<int8_t, int8_t, int32_t>& dis_computer);

template 
void search_bigann<int8_t, int32_t, CMin<int32_t, uint32_t>, CMin<int32_t, uint64_t>>(const std::string& index_path,
                   const std::string& query_bin_file,
                   const std::string& answer_bin_file,
                   const int nprobe,
                   const int refine_nprobe,
                   const int topk,
                   const int refine_topk,
                   std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
                   ProductQuantizer<CMin<int32_t, uint64_t>, int8_t, uint8_t>& pq_quantizer,
                   const int K1,
                   std::vector<std::vector<uint8_t>>& pq_codebook,
                   std::vector<std::vector<uint32_t>>& meta,
                   Computer<int8_t, int8_t, int32_t>& dis_computer);

template
void save_answers<float, CMax<float, uint32_t>>(const std::string& answer_bin_file,
                  const int topk,
                  const uint32_t nq,
                  float*& answer_dists,
                  uint32_t*& answer_ids,
                  bool use_comp_format = true);

template
void save_answers<float, CMin<float, uint32_t>>(const std::string& answer_bin_file,
                  const int topk,
                  const uint32_t nq,
                  float*& answer_dists,
                  uint32_t*& answer_ids,
                  bool use_comp_format = true);

template
void save_answers<uint32_t, CMax<uint32_t, uint32_t>>(const std::string& answer_bin_file,
                  const int topk,
                  const uint32_t nq,
                  uint32_t*& answer_dists,
                  uint32_t*& answer_ids,
                  bool use_comp_format = true);

template
void save_answers<uint32_t, CMin<uint32_t, uint32_t>>(const std::string& answer_bin_file,
                  const int topk,
                  const uint32_t nq,
                  uint32_t*& answer_dists,
                  uint32_t*& answer_ids,
                  bool use_comp_format = true);

template
void save_answers<int32_t, CMax<int32_t, uint32_t>>(const std::string& answer_bin_file,
                  const int topk,
                  const uint32_t nq,
                  int32_t*& answer_dists,
                  uint32_t*& answer_ids,
                  bool use_comp_format = true);

template
void save_answers<int32_t, CMin<int32_t, uint32_t>>(const std::string& answer_bin_file,
                  const int topk,
                  const uint32_t nq,
                  int32_t*& answer_dists,
                  uint32_t*& answer_ids,
                  bool use_comp_format = true);

template
void search_graph<float>(std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
                  const int nq,
                  const int dq,
                  const int nprobe,
                  const int refine_nprobe,
                  const float* pquery,
                  uint64_t* buckets_label);


template
void search_graph<uint8_t>(std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
                  const int nq,
                  const int dq,
                  const int nprobe,
                  const int refine_nprobe,
                  const uint8_t* pquery,
                  uint64_t* buckets_label);

template
void search_graph<int8_t>(std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
                  const int nq,
                  const int dq,
                  const int nprobe,
                  const int refine_nprobe,
                  const int8_t* pquery,
                  uint64_t* buckets_label);


template
void build_bigann<float, float, CMax<float, uint32_t>>
                 (const std::string& raw_data_bin_file,
                  const std::string& output_path,
                  const int hnswM, const int hnswefC,
                  const int PQM, const int PQnbits,
                  const int K1, const int threshold,
                  MetricType metric_type,
                  QuantizerType quantizer_type);

template
void build_bigann<float, float, CMin<float, uint32_t>>
                 (const std::string& raw_data_bin_file,
                  const std::string& output_path,
                  const int hnswM, const int hnswefC,
                  const int PQM, const int PQnbits,
                  const int K1, const int threshold,
                  MetricType metric_type,
                  QuantizerType quantizer_type);

template
void build_bigann<uint8_t, uint32_t, CMax<uint32_t, uint32_t>>
                 (const std::string& raw_data_bin_file,
                  const std::string& output_path,
                  const int hnswM, const int hnswefC,
                  const int PQM, const int PQnbits,
                  const int K1, const int threshold,
                  MetricType metric_type,
                  QuantizerType quantizer_type);

template
void build_bigann<uint8_t, uint32_t, CMin<uint32_t, uint32_t>>
                 (const std::string& raw_data_bin_file,
                  const std::string& output_path,
                  const int hnswM, const int hnswefC,
                  const int PQM, const int PQnbits,
                  const int K1, const int threshold,
                  MetricType metric_type,
                  QuantizerType quantizer_type);

template
void build_bigann<int8_t, int32_t, CMax<int32_t, uint32_t>>
                 (const std::string& raw_data_bin_file,
                  const std::string& output_path,
                  const int hnswM, const int hnswefC,
                  const int PQM, const int PQnbits,
                  const int K1, const int threshold,
                  MetricType metric_type,
                  QuantizerType quantizer_type);

template
void build_bigann<int8_t, int32_t, CMin<int32_t, uint32_t>>
                 (const std::string& raw_data_bin_file,
                  const std::string& output_path,
                  const int hnswM, const int hnswefC,
                  const int PQM, const int PQnbits,
                  const int K1, const int threshold,
                  MetricType metric_type,
                  QuantizerType quantizer_type);


template
void train_cluster<float>(const std::string& raw_data_bin_file,
                   const std::string& output_path,
                   const int K1,
                   float** centroids,
                   double& avg_len);

template
void train_cluster<uint8_t>(const std::string& raw_data_bin_file,
                   const std::string& output_path,
                   const int K1,
                   float** centroids,
                   double& avg_len);

template
void train_cluster<int8_t>(const std::string& raw_data_bin_file,
                   const std::string& output_path,
                   const int K1,
                   float** centroids,
                   double& avg_len);



















