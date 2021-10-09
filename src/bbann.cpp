#include "bbann.h"

template <typename DATAT>
void train_cluster(const std::string &raw_data_bin_file,
                   const std::string &output_path, const int32_t K1,
                   float **centroids, double &avg_len) {
  TimeRecorder rc("train cluster");
  std::cout << "train_cluster parameters:" << std::endl;
  std::cout << " raw_data_bin_file: " << raw_data_bin_file
            << " output path: " << output_path << " K1: " << K1
            << " centroids: " << *centroids << std::endl;
  assert((*centroids) == nullptr);
  DATAT *sample_data = nullptr;
  uint32_t nb, dim;
  get_bin_metadata(raw_data_bin_file, nb, dim);
  int64_t sample_num = nb * K1_SAMPLE_RATE;
  std::cout << "nb = " << nb << ", dim = " << dim
            << ", sample_num 4 K1: " << sample_num << std::endl;

  *centroids = new float[K1 * dim];
  sample_data = new DATAT[sample_num * dim];
  reservoir_sampling(raw_data_bin_file, sample_num, sample_data);
  rc.RecordSection("reservoir sample with sample rate: " +
                   std::to_string(K1_SAMPLE_RATE) + " done");
  double mxl, mnl;
  int64_t stat_n = std::min(static_cast<int64_t>(1000000), sample_num);
  stat_length<DATAT>(sample_data, stat_n, dim, mxl, mnl, avg_len);
  rc.RecordSection("calculate " + std::to_string(stat_n) +
                   " vectors from sample_data done");
  std::cout << "max len: " << mxl << ", min len: " << mnl
            << ", average len: " << avg_len << std::endl;
  kmeans<DATAT>(sample_num, sample_data, dim, K1, *centroids, false, avg_len);
  rc.RecordSection("kmeans done");
  assert((*centroids) != nullptr);

  delete[] sample_data;
  rc.ElapseFromBegin("train cluster done.");
}

template <typename DATAT, typename DISTT, typename HEAPT>
void divide_raw_data(const std::string &raw_data_bin_file,
                     const std::string &output_path, const float *centroids,
                     const int32_t K1) {
  TimeRecorder rc("divide raw data");
  std::cout << "divide_raw_data parameters:" << std::endl;
  std::cout << " raw_data_bin_file: " << raw_data_bin_file
            << " output_path: " << output_path << " centroids: " << centroids
            << " K1: " << K1 << std::endl;
  IOReader reader(raw_data_bin_file);
  uint32_t nb, dim;
  reader.read((char *)&nb, sizeof(uint32_t));
  reader.read((char *)&dim, sizeof(uint32_t));
  uint32_t placeholder = 0, const_one = 1;
  std::vector<uint32_t> cluster_size(K1, 0);
  std::vector<std::ofstream> cluster_dat_writer(K1);
  std::vector<std::ofstream> cluster_ids_writer(K1);
  for (int i = 0; i < K1; i++) {
    std::string cluster_raw_data_file_name =
        output_path + CLUSTER + std::to_string(i) + "-" + RAWDATA + BIN;
    std::string cluster_ids_data_file_name =
        output_path + CLUSTER + std::to_string(i) + "-" + GLOBAL_IDS + BIN;
    cluster_dat_writer[i] =
        std::ofstream(cluster_raw_data_file_name, std::ios::binary);
    cluster_ids_writer[i] =
        std::ofstream(cluster_ids_data_file_name, std::ios::binary);
    cluster_dat_writer[i].write((char *)&placeholder, sizeof(uint32_t));
    cluster_dat_writer[i].write((char *)&dim, sizeof(uint32_t));
    cluster_ids_writer[i].write((char *)&placeholder, sizeof(uint32_t));
    cluster_ids_writer[i].write((char *)&const_one, sizeof(uint32_t));
  }

  int64_t block_size = 1000000;
  assert(nb > 0);
  int64_t block_num = (nb - 1) / block_size + 1;
  std::vector<int64_t> cluster_id(block_size);
  std::vector<DISTT> dists(block_size);
  DATAT *block_buf = new DATAT[block_size * dim];
  for (int64_t i = 0; i < block_num; i++) {
    TimeRecorder rci("batch-" + std::to_string(i));
    int64_t sp = i * block_size;
    int64_t ep = std::min((int64_t)nb, sp + block_size);
    std::cout << "split the " << i << "th batch, start position = " << sp
              << ", end position = " << ep << std::endl;
    reader.read((char *)block_buf, (ep - sp) * dim * sizeof(DATAT));
    rci.RecordSection("read batch data done");
    elkan_L2_assign<const DATAT, const float, DISTT>(
        block_buf, centroids, dim, ep - sp, K1, cluster_id.data(),
        dists.data());
    rci.RecordSection("select file done");
    for (int64_t j = 0; j < ep - sp; j++) {
      int64_t cid = cluster_id[j];
      uint32_t uid = (uint32_t)(j + sp);
      // for debug
      /*
      if (0 == uid) {
          std::cout << "vector0 was split into cluster " << cid << std::endl;
          std::cout << " show vector0:" << std::endl;
          for (auto si = 0; si < dim; si ++) {
              std::cout << (int)(*(block_buf + j * dim + si)) << " ";
          }
          std::cout << std::endl;
      }
      */
      cluster_dat_writer[cid].write((char *)(block_buf + j * dim),
                                    sizeof(DATAT) * dim);
      cluster_ids_writer[cid].write((char *)&uid, sizeof(uint32_t));
      cluster_size[cid]++;
    }
    rci.RecordSection("write done");
    rci.ElapseFromBegin("split batch " + std::to_string(i) + " done");
  }
  rc.RecordSection("split done");
  size_t sump = 0;
  std::cout << "split_raw_data done in ... seconds, show statistics:"
            << std::endl;
  for (int i = 0; i < K1; i++) {
    uint32_t cis = cluster_size[i];
    cluster_dat_writer[i].seekp(0);
    cluster_dat_writer[i].write((char *)&cis, sizeof(uint32_t));
    cluster_dat_writer[i].close();
    cluster_ids_writer[i].seekp(0);
    cluster_ids_writer[i].write((char *)&cis, sizeof(uint32_t));
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

template <typename DATAT, typename DISTT, typename HEAPT>
void hierarchical_clusters(const std::string &output_path, const int K1,
                           const double avg_len, const int blk_size) {
  TimeRecorder rc("hierarchical clusters");
  std::cout << "hierarchical clusters parameters:" << std::endl;
  std::cout << " output_path: " << output_path
            << " vector avg length: " << avg_len << " block size: " << blk_size
            << std::endl;

  uint32_t cluster_size, cluster_dim, ids_size, ids_dim;
  uint32_t entry_num;

  std::string bucket_centroids_file = output_path + BUCKET + CENTROIDS + BIN;
  std::string bucket_centroids_id_file =
      output_path + CLUSTER + COMBINE_IDS + BIN;
  uint32_t placeholder = 1;
  uint32_t global_centroids_number = 0;
  uint32_t centroids_dim = 0;

  {
    IOWriter centroids_writer(bucket_centroids_file);
    IOWriter centroids_id_writer(bucket_centroids_id_file);
    centroids_writer.write((char *)&placeholder, sizeof(uint32_t));
    centroids_writer.write((char *)&placeholder, sizeof(uint32_t));
    centroids_id_writer.write((char *)&placeholder, sizeof(uint32_t));
    centroids_id_writer.write((char *)&placeholder, sizeof(uint32_t));

    for (uint32_t i = 0; i < K1; i++) {
      TimeRecorder rci("train-cluster-" + std::to_string(i));
      std::string data_file =
          output_path + CLUSTER + std::to_string(i) + "-" + RAWDATA + BIN;
      std::string ids_file =
          output_path + CLUSTER + std::to_string(i) + "-" + GLOBAL_IDS + BIN;
      IOReader data_reader(data_file);
      IOReader ids_reader(ids_file);

      data_reader.read((char *)&cluster_size, sizeof(uint32_t));
      data_reader.read((char *)&cluster_dim, sizeof(uint32_t));
      ids_reader.read((char *)&ids_size, sizeof(uint32_t));
      ids_reader.read((char *)&ids_dim, sizeof(uint32_t));
      entry_num = (blk_size - sizeof(uint32_t)) /
                  (cluster_dim * sizeof(DATAT) + ids_dim * sizeof(uint32_t));
      centroids_dim = cluster_dim;
      assert(cluster_size == ids_size);
      assert(ids_dim == 1);
      assert(entry_num > 0);


      int64_t data_size = static_cast<int64_t>(cluster_size);

      DATAT *datai = new DATAT[data_size * cluster_dim * 1ULL];

      uint32_t *idi = new uint32_t[ids_size * ids_dim];
      uint32_t blk_num = 0;
      data_reader.read((char *)datai,
                       data_size * cluster_dim * sizeof(DATAT));
      ids_reader.read((char *)idi, ids_size * ids_dim * sizeof(uint32_t));

      IOWriter data_writer(data_file, MEGABYTE * 100);
      recursive_kmeans<DATAT>(i, data_size, datai, idi, cluster_dim,
                              entry_num, blk_size, blk_num, data_writer,
                              centroids_writer, centroids_id_writer, 0, false,
                              avg_len);

      global_centroids_number += blk_num;

      delete[] datai;
      delete[] idi;
    }
  }

  uint32_t centroids_id_dim = 1;
  std::ofstream centroids_meta_writer(bucket_centroids_file,
                                      std::ios::binary | std::ios::in);
  std::ofstream centroids_ids_meta_writer(bucket_centroids_id_file,
                                          std::ios::binary | std::ios::in);
  centroids_meta_writer.seekp(0);
  centroids_meta_writer.write((char *)&global_centroids_number,
                              sizeof(uint32_t));
  centroids_meta_writer.write((char *)&centroids_dim, sizeof(uint32_t));
  centroids_ids_meta_writer.seekp(0);
  centroids_ids_meta_writer.write((char *)&global_centroids_number,
                                  sizeof(uint32_t));
  centroids_ids_meta_writer.write((char *)&centroids_id_dim, sizeof(uint32_t));
  centroids_meta_writer.close();
  centroids_ids_meta_writer.close();

  std::cout << "hierarchical_clusters generate " << global_centroids_number
            << " centroids" << std::endl;
  return;
}

void build_graph(const std::string &index_path, const int hnswM,
                 const int hnswefC, MetricType metric_type) {
  TimeRecorder rc("create_graph_index");
  std::cout << "build hnsw parameters:" << std::endl;
  std::cout << " index_path: " << index_path << " hnsw.M: " << hnswM
            << " hnsw.efConstruction: " << hnswefC
            << " metric_type: " << (int)metric_type << std::endl;

  float *pdata = nullptr;
  uint32_t *pids = nullptr;
  uint32_t npts, ndim, nids, nidsdim;

  read_bin_file<float>(index_path + BUCKET + CENTROIDS + BIN, pdata, npts,
                       ndim);
  rc.RecordSection("load centroids of buckets done");
  std::cout << "there are " << npts << " of dimension " << ndim
            << " points of hnsw" << std::endl;
  assert(pdata != nullptr);
  hnswlib::SpaceInterface<float> *space;
  if (MetricType::L2 == metric_type) {
    space = new hnswlib::L2Space(ndim);
  } else if (MetricType::IP == metric_type) {
    space = new hnswlib::InnerProductSpace(ndim);
  } else {
    std::cout << "invalid metric_type = " << (int)metric_type << std::endl;
    return;
  }
  read_bin_file<uint32_t>(index_path + CLUSTER + COMBINE_IDS + BIN, pids, nids,
                          nidsdim);
  rc.RecordSection("load combine ids of buckets done");
  std::cout << "there are " << nids << " of dimension " << nidsdim
            << " combine ids of hnsw" << std::endl;
  assert(pids != nullptr);
  assert(npts == nids);
  assert(nidsdim == 1);

  auto index_hnsw = std::make_shared<hnswlib::HierarchicalNSW<float>>(
      space, npts, hnswM, hnswefC);
  index_hnsw->addPoint(pdata, pids[0]);
#pragma omp parallel for
  for (int64_t i = 1; i < npts; i++) {
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

void gather_buckets_stats(const ::std::string index_path, const int K1,
                          const uint64_t block_size) {
  char *buf = new char[block_size];
  uint64_t stats[3] = {0, 0,
                       std::numeric_limits<uint64_t>::max()}; // avg, max, min
  uint64_t bucket_cnt = 0;

  for (int i = 0; i < K1; ++i) {
    std::string cluster_file_path =
        index_path + CLUSTER + std::to_string(i) + "-" + RAWDATA + BIN;
    uint64_t file_size = fsize(cluster_file_path);
    auto fh = std::ifstream(cluster_file_path, std::ios::binary);
    assert(!fh.fail());

    for (uint32_t j = 0; j * block_size < file_size; ++j) {
      fh.seekg(j * block_size);
      fh.read(buf, block_size);
      const uint64_t entry_num =
          static_cast<const uint64_t>(*reinterpret_cast<uint32_t *>(buf));

      stats[0] += entry_num;
      stats[1] = std::max(stats[1], entry_num);
      stats[2] = std::min(stats[2], entry_num);
      ++bucket_cnt;
    }
  }

  uint32_t cen_n, cen_dim;
  get_bin_metadata(index_path + BUCKET + CENTROIDS + BIN, cen_n, cen_dim);
  assert(cen_n == bucket_cnt);

  std::cout << "Total number of buckets: " << bucket_cnt << std::endl;
  std::cout << "#vectors in bucket avg: " << stats[0] * 1.0f / bucket_cnt
            << " max: " << stats[1] << " min: " << stats[2] << std::endl;

  delete[] buf;
}

template <typename DATAT, typename DISTT, typename HEAPT>
void build_bbann(const std::string &raw_data_bin_file,
                 const std::string &output_path, const int hnswM,
                 const int hnswefC, MetricType metric_type, const int K1,
                 const uint64_t block_size) {
  TimeRecorder rc("build bigann");
  std::cout << "build bigann parameters:" << std::endl;
  std::cout << " raw_data_bin_file: " << raw_data_bin_file
            << " output_path: " << output_path << " hnsw.M: " << hnswM
            << " hnsw.efConstruction: " << hnswefC << " K1: " << K1
            << std::endl;

  float *centroids = nullptr;
  double avg_len;
  // sampling and do K1-means to get the first round centroids
  train_cluster<DATAT>(raw_data_bin_file, output_path, K1, &centroids, avg_len);
  assert(centroids != nullptr);
  rc.RecordSection("train cluster to get " + std::to_string(K1) +
                   " centroids done.");

  divide_raw_data<DATAT, DISTT, HEAPT>(raw_data_bin_file, output_path,
                                       centroids, K1);
  rc.RecordSection("divide raw data into " + std::to_string(K1) +
                   " clusters done");

  hierarchical_clusters<DATAT, DISTT, HEAPT>(output_path, K1, avg_len,
                                             block_size);
  rc.RecordSection("conquer each cluster into buckets done");

  build_graph(output_path, hnswM, hnswefC, metric_type);
  rc.RecordSection("build hnsw done.");

  gather_buckets_stats(output_path, K1, block_size);
  rc.RecordSection("gather statistics done");

  delete[] centroids;
  rc.ElapseFromBegin("build bigann totally done.");
}

template <typename DATAT>
void search_graph(std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
                  const int nq, const int dq, const int nprobe,
                  const int refine_nprobe, const DATAT *pquery,
                  uint32_t *buckets_label, float *centroids_dist = nullptr) {

  TimeRecorder rc("search graph");
  std::cout << "search graph parameters:" << std::endl;
  std::cout << " index_hnsw: " << index_hnsw << " nq: " << nq << " dq: " << dq
            << " nprobe: " << nprobe << " refine_nprobe: " << refine_nprobe
            << " pquery: " << static_cast<const void *>(pquery)
            << " buckets_label: " << static_cast<void *>(buckets_label)
            << std::endl;
  index_hnsw->setEf(refine_nprobe);
  bool set_distance = centroids_dist!= nullptr;
#pragma omp parallel for
  for (int64_t i = 0; i < nq; i++) {
    // auto queryi = pquery + i * dq;
    // todo: hnsw need to support query data is not float
    float *queryi = new float[dq];
    float *queryi_dist = set_distance ? centroids_dist + i * nprobe : nullptr;

    for (int j = 0; j < dq; j++)
      queryi[j] = (float)(*(pquery + i * dq + j));
    auto reti = index_hnsw->searchKnn(queryi, nprobe);
    auto p_labeli = buckets_label + i * nprobe;
    while (!reti.empty()) {
      *p_labeli++ = reti.top().second;
      if(set_distance) {
          *queryi_dist++ = reti.top().first;
      }
      reti.pop();
    }
    delete[] queryi;
  }
  rc.ElapseFromBegin("search graph done.");
}

template <typename DISTT, typename HEAPT>
void save_answers(const std::string &answer_bin_file, const int topk,
                  const uint32_t nq, DISTT *&answer_dists,
                  uint32_t *&answer_ids) {
  TimeRecorder rc("save answers");
  std::cout << "save answer parameters:" << std::endl;
  std::cout << " answer_bin_file: " << answer_bin_file << " topk: " << topk
            << " nq: " << nq << " answer_dists: " << answer_dists
            << " answer_ids: " << answer_ids << std::endl;

  std::ofstream answer_writer(answer_bin_file, std::ios::binary);
  answer_writer.write((char *)&nq, sizeof(uint32_t));
  answer_writer.write((char *)&topk, sizeof(uint32_t));

  for (int i = 0; i < nq; i++) {
    auto ans_disi = answer_dists + topk * i;
    auto ans_idsi = answer_ids + topk * i;
    heap_reorder<HEAPT>(topk, ans_disi, ans_idsi);
  }

  uint32_t tot = nq * topk;
  answer_writer.write((char *)answer_ids, tot * sizeof(uint32_t));
  answer_writer.write((char *)answer_dists, tot * sizeof(DISTT));

  answer_writer.close();

  rc.ElapseFromBegin("save answers done.");
}

template <typename DATAT>
void search_flat(const std::string &index_path, const DATAT *pquery,
                 const uint32_t nq, const uint32_t dq, const int nprobe,
                 uint32_t *labels) {
  std::string bucket_centroids_file = index_path + BUCKET + CENTROIDS + BIN;
  std::string bucket_centroids_id_file =
      index_path + CLUSTER + COMBINE_IDS + BIN;

  float *cen_data = nullptr;
  uint32_t *cen_ids = nullptr;

  uint32_t cen_n, dim, temp;
  read_bin_file<float>(bucket_centroids_file, cen_data, cen_n, dim);
  std::cout << cen_n << " " << dim << std::endl;
  assert(dim == dq);
  read_bin_file<uint32_t>(bucket_centroids_id_file, cen_ids, cen_n, temp);
  std::cout << cen_n << " " << temp << std::endl;

  // for (int i = 0; i < cen_n; ++i) {
  //     uint32_t cid, bid;
  //     parse_global_block_id(cen_ids[i], cid, bid);
  //     std::cout << cen_ids[i] << " " << cid << " " << bid << std::endl;
  // }

  float *values = new float[(int64_t)nq * nprobe];

  knn_1<CMax<float, uint32_t>, DATAT, float>(
      pquery, cen_data, nq, cen_n, dim, nprobe, values, labels,
      L2sqr<const DATAT, const float, float>);

  for (uint64_t i = 0; i < (uint64_t)nq * nprobe; ++i) {
    labels[i] = cen_ids[labels[i]];
  }

  delete[] values;
}

void check_answer_by_id(const uint32_t *answer_ids, const uint32_t nq,
                        const int topk) {
  for (uint32_t i = 0; i < nq; ++i) {
    bool f = true;
    auto pos = -1;
    for (int j = 0; j < topk; ++j) {
      // std::cout << answer_ids[i*topk+j] << " ";
      if (answer_ids[i * topk + j] == i) {
        f = false;
        pos = j;
        break;
      }
    }
    // std::cout << std::endl;

    if (f) {
      std::cout << "Cound not find " << i << "th base vector" << std::endl;
    } else {
      std::cout << "Found " << i << "th base vector at top " << pos
                << std::endl;
    }
  }
}

template <typename DATAT>
void gather_vec_searched_per_query(const std::string index_path,
                                   const DATAT *pquery, const int nq,
                                   const int nprobe, const uint32_t dq,
                                   const uint64_t block_size, char *buf,
                                   const uint32_t *bucket_labels) {
  std::unordered_map<uint32_t, uint64_t> vec_per_q_mp;
  uint32_t cid, bid;

  for (int64_t i = 0; i < nq; ++i) {
    const auto ii = i * nprobe;
    const DATAT *q_idx = pquery + i * dq;

    for (int64_t j = 0; j < nprobe; ++j) {
      parse_global_block_id(bucket_labels[ii + j], cid, bid);
      std::string cluster_file_path =
          index_path + CLUSTER + std::to_string(cid) + "-" + RAWDATA + BIN;
      auto fh = std::ifstream(cluster_file_path, std::ios::binary);
      assert(!fh.fail());

      fh.seekg(bid * block_size);
      fh.read(buf, block_size);

      const uint32_t entry_num = *reinterpret_cast<uint32_t *>(buf);
      vec_per_q_mp[i] += entry_num;
    }
  }

  uint64_t stats[3] = {0, 0,
                       std::numeric_limits<uint64_t>::max()}; // avg, max, min
  for (const auto &[q, cnt] : vec_per_q_mp) {
    stats[0] += cnt;
    stats[1] = std::max(stats[1], cnt);
    stats[2] = std::min(stats[2], cnt);
  }
  std::cout << "#vec/query avg: " << stats[0] * 1.0f / nq
            << " max: " << stats[1] << " min: " << stats[2] << std::endl;
}

template <typename DATAT, typename DISTT, typename HEAPT>
void search_bbann(
    const std::string &index_path, const std::string &query_bin_file,
    const std::string &answer_bin_file, const int nprobe, const int hnsw_ef,
    const int topk, std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
    const int K1, const uint64_t block_size,
    Computer<DATAT, DATAT, DISTT> &dis_computer) {
  TimeRecorder rc("search bigann");

  std::cout << "search bigann parameters:" << std::endl;
  std::cout << " index_path: " << index_path
            << " query_bin_file: " << query_bin_file
            << " answer_bin_file: " << answer_bin_file << " nprobe: " << nprobe
            << " hnsw_ef: " << hnsw_ef << " topk: " << topk << " K1: " << K1
            << std::endl;

  DATAT *pquery = nullptr;
  DISTT *answer_dists = nullptr;
  uint32_t *answer_ids = nullptr;
  uint32_t *bucket_labels = nullptr;

  uint32_t nq, dq;

  read_bin_file<DATAT>(query_bin_file, pquery, nq, dq);
  rc.RecordSection("load query done.");

  std::cout << "query numbers: " << nq << " query dims: " << dq << std::endl;

  answer_dists = new DISTT[(int64_t)nq * topk];
  answer_ids = new uint32_t[(int64_t)nq * topk];
  bucket_labels = new uint32_t[(int64_t)nq * nprobe]; // 400K * nprobe

  // cid -> [bid -> [qid]]
  std::unordered_map<uint32_t,
                     std::unordered_map<uint32_t, std::unordered_set<uint32_t>>>
      mp;

  search_graph<DATAT>(index_hnsw, nq, dq, nprobe, hnsw_ef, pquery,
                      bucket_labels, nullptr);
  // search_flat<DATAT>(index_path, pquery, nq, dq, nprobe, bucket_labels);
  rc.RecordSection("search buckets done.");

  uint32_t cid, bid;
  uint32_t dim = dq, gid;
  const uint32_t vec_size = sizeof(DATAT) * dim;
  const uint32_t entry_size = vec_size + sizeof(uint32_t);
  DATAT *vec;

  // init answer heap
#pragma omp parallel for schedule(static, 128)
  for (int i = 0; i < nq; i++) {
    auto ans_disi = answer_dists + topk * i;
    auto ans_idsi = answer_ids + topk * i;
    heap_heapify<HEAPT>(topk, ans_disi, ans_idsi);
  }
  rc.RecordSection("heapify answers heaps");

  char *buf = new char[block_size];

  /* flat */
  for (int64_t i = 0; i < nq; ++i) {
    const auto ii = i * nprobe;
    const DATAT *q_idx = pquery + i * dq;

    for (int64_t j = 0; j < nprobe; ++j) {
      parse_global_block_id(bucket_labels[ii + j], cid, bid);
      std::string cluster_file_path =
          index_path + CLUSTER + std::to_string(cid) + "-" + RAWDATA + BIN;
      auto fh = std::ifstream(cluster_file_path, std::ios::binary);
      assert(!fh.fail());

      fh.seekg(bid * block_size);
      fh.read(buf, block_size);

      const uint32_t entry_num = *reinterpret_cast<uint32_t *>(buf);
      char *buf_begin = buf + sizeof(uint32_t);

      for (uint32_t k = 0; k < entry_num; ++k) {
        char *entry_begin = buf_begin + entry_size * k;
        vec = reinterpret_cast<DATAT *>(entry_begin);
        auto dis = dis_computer(vec, q_idx, dim);
        if (HEAPT::cmp(answer_dists[topk * i], dis)) {
          heap_swap_top<HEAPT>(
              topk, answer_dists + topk * i, answer_ids + topk * i, dis,
              *reinterpret_cast<uint32_t *>(entry_begin + vec_size));
        }
      }
    }
  }

  /* remove duplications + parallel on queries when searching block */
  //     for (int64_t i = 0; i < nq; ++i) {
  //         const auto ii = i * nprobe;
  //         for (int64_t j = 0; j < nprobe; ++j) {
  //             parse_global_block_id(bucket_labels[ii+j], cid, bid);
  //             mp[cid][bid].insert(i);
  //         }
  //     }

  //     for (const auto& [cid, bid_mp] : mp) {
  //         std::string cluster_file_path = index_path + CLUSTER +
  //         std::to_string(cid) + "-" + RAWDATA + BIN; auto fh =
  //         std::ifstream(cluster_file_path, std::ios::binary);
  //         assert(!fh.fail());

  //         for (const auto& [bid, qs] : bid_mp) {
  //             fh.seekg(bid * block_size);
  //             fh.read(buf, block_size);
  //             const uint32_t entry_num = *reinterpret_cast<uint32_t*>(buf);
  //             char* buf_begin = buf + sizeof(uint32_t);
  //     /*
  //      * A few options here. We can parallelize cid, bid, qs,
  //      dis_calculation.
  //      * Parallelizing qs is the only option that does not require locks, but
  //      having
  //      * potential waste of resources when qs is less than number of threads
  //      available.
  //      *
  //      * TODO: need more investigation here
  //      *
  //      * Other proposal like make hnsw_search, read_data and calculation as a
  //      pipeline
  //      * may also worth investigating.
  //      */
  //             std::vector<uint32_t> qv(qs.begin(), qs.end());

  // #pragma omp parallel for
  //             for (uint32_t i = 0; i < qv.size(); ++i) {
  //                 const uint32_t qid = qv[i];
  //                 const DATAT* q_idx = pquery + qid * dq;
  //                 for (uint32_t j = 0; j < entry_num; ++j) {
  //                     char *entry_begin = buf_begin + entry_size * j;
  //                     vec = reinterpret_cast<DATAT*>(entry_begin);
  //                     auto dis = dis_computer(vec, q_idx, dim);
  //                     if (HEAPT::cmp(answer_dists[topk * qid], dis)) {
  //                         heap_swap_top<HEAPT>(topk,
  //                                              answer_dists + topk * qid,
  //                                              answer_ids + topk * qid,
  //                                              dis,
  //                                              *reinterpret_cast<uint32_t*>(entry_begin
  //                                              + vec_size));
  //                     }
  //                 }
  //             }
  //         }
  //     }
  rc.RecordSection("scan blocks done");

  // write answers
  save_answers<DISTT, HEAPT>(answer_bin_file, topk, nq, answer_dists,
                             answer_ids);
  rc.RecordSection("write answers done");

  gather_vec_searched_per_query(index_path, pquery, nq, nprobe, dq, block_size,
                                buf, bucket_labels);
  rc.RecordSection("gather statistics done");

  delete[] buf;
  delete[] bucket_labels;
  delete[] pquery;
  delete[] answer_ids;
  delete[] answer_dists;

  rc.ElapseFromBegin("search bigann totally done");
}

template <typename DATAT, typename DISTT, typename HEAPT>
void dynamic_search_bbann(
        const std::string &index_path, const std::string &query_bin_file,
        const std::string &answer_bin_file, const int nprobe, const int hnsw_ef,
        const int topk, std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
        const int K1, const uint64_t block_size,
        Computer<DATAT, DATAT, DISTT> &dis_computer) {
  TimeRecorder rc("dynamic search bigann");

  std::cout << "search bigann parameters:" << std::endl;
  std::cout << " index_path: " << index_path
            << " query_bin_file: " << query_bin_file
            << " answer_bin_file: " << answer_bin_file << " nprobe: " << nprobe
            << " hnsw_ef: " << hnsw_ef << " topk: " << topk << " K1: " << K1
            << std::endl;

  DATAT *pquery = nullptr;
  float *centroids = nullptr;
  float *centroids_dists = nullptr;
  DISTT *answer_dists = nullptr;
  uint32_t *answer_ids = nullptr;
  uint32_t *bucket_labels = nullptr;

  uint32_t nq, dq;
  uint32_t nc, dc;

  read_bin_file<DATAT>(query_bin_file, pquery, nq, dq);
  rc.RecordSection("load query done.");
  //load centroids ：
  std::string bucket_centroids_file = index_path + BUCKET + CENTROIDS + BIN;
  read_bin_file<float>(bucket_centroids_file, centroids, nc, dc);
  rc.RecordSection("load centroids done.");

  std::cout << "query numbers: " << nq << " query dims: " << dq << std::endl;

  answer_dists = new DISTT[(int64_t)nq * topk];
  answer_ids = new uint32_t[(int64_t)nq * topk];

  const int graph_search_topk = 200;
  bucket_labels = new uint32_t[(int64_t)nq * graph_search_topk];

  // cid -> [bid -> [qid]]
  std::unordered_map<uint32_t,
          std::unordered_map<uint32_t, std::unordered_set<uint32_t>>>
                                       mp;

  centroids_dists = new float[(int64_t)nq*graph_search_topk*1ULL];

  search_graph<DATAT>(index_hnsw, nq, dq, graph_search_topk, graph_search_topk * 5, pquery,
                      bucket_labels, centroids_dists);

  // search_flat<DATAT>(index_path, pquery, nq, dq, nprobe, bucket_labels);
  rc.RecordSection("search buckets done.");

  uint32_t cid, bid;
  uint32_t dim = dq, gid;
  const uint32_t vec_size = sizeof(DATAT) * dim;
  const uint32_t entry_size = vec_size + sizeof(uint32_t);
  DATAT *vec;

  // init answer heap
#pragma omp parallel for schedule(static, 128)
  for (int i = 0; i < nq; i++) {
    auto ans_disi = answer_dists + topk * i;
    auto ans_idsi = answer_ids + topk * i;
    heap_heapify<HEAPT>(topk, ans_disi, ans_idsi);
  }
  rc.RecordSection("heapify answers heaps");

  char *buf = new char[block_size];
  int64_t total_vector =  0;
  /* flat */
  for (int64_t i = 0; i < nq; ++i) {
    const auto ii = i * graph_search_topk;
    const DATAT *q_idx = pquery + i * dq;
    const float *c_dist = centroids_dists + i * graph_search_topk;
    float search_threshold = c_dist[0];
    for (int64_t j = 0; j < graph_search_topk; ++j) {
      search_threshold = search_threshold < c_dist[j] ? search_threshold : c_dist[j];
    }
    search_threshold *= (1 + SEARCH_PRUNING_RATE);
    // std::cout<<i<<" search_threshold "<<search_threshold<<std::endl;
    for (int64_t j = 0; j < graph_search_topk; ++j) {
      //    std::cout<<"j"<<" ";
      //  std::cout<<c_dist[j]<<std::endl;
      if(c_dist[j] <= search_threshold) {

        parse_global_block_id(bucket_labels[ii + j], cid, bid);
        std::string cluster_file_path =
                index_path + CLUSTER + std::to_string(cid) + "-" + RAWDATA + BIN;
        auto fh = std::ifstream(cluster_file_path, std::ios::binary);
        assert(!fh.fail());

        fh.seekg(bid * block_size);
        fh.read(buf, block_size);

        const uint32_t entry_num = *reinterpret_cast<uint32_t *>(buf);
        char *buf_begin = buf + sizeof(uint32_t);
        total_vector += entry_num;
        for (uint32_t k = 0; k < entry_num; ++k) {
          char *entry_begin = buf_begin + entry_size * k;
          vec = reinterpret_cast<DATAT *>(entry_begin);
          auto dis = dis_computer(vec, q_idx, dim);
          if (HEAPT::cmp(answer_dists[topk * i], dis)) {
            heap_swap_top<HEAPT>(
                    topk, answer_dists + topk * i, answer_ids + topk * i, dis,
                    *reinterpret_cast<uint32_t *>(entry_begin + vec_size));
          }
        }
        // std::cout<<"search block"<<std::endl;
      }
    }
  }
  std::cout<<"total access vector :"<<total_vector<<std::endl;
  rc.RecordSection("scan blocks done");

  // write answers
  save_answers<DISTT, HEAPT>(answer_bin_file, topk, nq, answer_dists,
                             answer_ids);
  rc.RecordSection("write answers done");

  gather_vec_searched_per_query(index_path, pquery, nq, graph_search_topk, dq, block_size,
                                buf, bucket_labels);
  rc.RecordSection("gather statistics done");

  delete[] buf;
  delete[] bucket_labels;
  delete[] centroids_dists;
  delete[] pquery;
  delete[] answer_ids;
  delete[] answer_dists;

  rc.ElapseFromBegin("search bigann totally done");
}

#define BUILD_BBANN_DECL(DATAT, DISTT, HEAPT)                                  \
  template void build_bbann<DATAT, DISTT, HEAPT<DISTT, uint32_t>>(             \
      const std::string &raw_data_bin_file, const std::string &output_path,    \
      const int hnswM, const int hnswefC, MetricType metric_type,              \
      const int K1, const uint64_t block_size);

#define SEARCH_BBANN_DECL(DATAT, DISTT, HEAPT)                                 \
  template void search_bbann<DATAT, DISTT, HEAPT<DISTT, uint32_t>>(            \
      const std::string &index_path, const std::string &query_bin_file,        \
      const std::string &answer_bin_file, const int nprobe, const int hnsw_ef, \
      const int topk,                                                          \
      std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,             \
      const int K1, const uint64_t block_size,                                 \
      Computer<DATAT, DATAT, DISTT> &dis_computer);

BUILD_BBANN_DECL(uint8_t, uint32_t, CMin)
BUILD_BBANN_DECL(uint8_t, uint32_t, CMax)
BUILD_BBANN_DECL(int8_t, int32_t, CMin)
BUILD_BBANN_DECL(int8_t, int32_t, CMax)
BUILD_BBANN_DECL(float, float, CMin)
BUILD_BBANN_DECL(float, float, CMax)

SEARCH_BBANN_DECL(uint8_t, uint32_t, CMin)
SEARCH_BBANN_DECL(uint8_t, uint32_t, CMax)
SEARCH_BBANN_DECL(int8_t, int32_t, CMin)
SEARCH_BBANN_DECL(int8_t, int32_t, CMax)
SEARCH_BBANN_DECL(float, float, CMin)
SEARCH_BBANN_DECL(float, float, CMax)
