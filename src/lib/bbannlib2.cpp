#include "lib/bbannlib2.h"
#include "ann_interface.h"
#include "util/TimeRecorder.h"
#include "util/file_handler.h"
#include "util/heap.h"
#include "util/utils_inline.h"
#include <iostream>
#include <map>
#include <omp.h>
#include <stdint.h>
#include <string>
namespace bbann {

template <typename DATAT, typename DISTT>
void hierarchical_clusters(const BBAnnParameters para, const double avg_len) {
  TimeRecorder rc("hierarchical clusters");
  std::cout << "hierarchical clusters parameters:" << std::endl;
  std::cout << " output_path: " << para.indexPrefixPath
            << " vector avg length: " << avg_len
            << " block size: " << para.blockSize << std::endl;
  int K1 = para.K1;

  uint32_t cluster_size, cluster_dim, ids_size, ids_dim;
  uint32_t entry_num;

  std::string bucket_centroids_file =
      para.indexPrefixPath + "bucket-centroids.bin";
  std::string bucket_centroids_id_file =
      para.indexPrefixPath + "cluster-combine_ids.bin";
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

    int64_t global_start_position = centroids_id_writer.get_position();
    assert(global_start_position != -1);
    std::cout << "global_centroids_number: global_start_position: " << global_start_position << std::endl;

    for (uint32_t i = 0; i < K1; i++) {
      TimeRecorder rci("train-cluster-" + std::to_string(i));

      int64_t local_start_position = centroids_id_writer.get_position();
      assert(local_start_position != -1);
      std::cout << "global_centroids_number: local_start_position: " << local_start_position << std::endl;

      IOReader data_reader(getClusterRawDataFileName(para.indexPrefixPath, i));
      IOReader ids_reader(getClusterGlobalIdsFileName(para.indexPrefixPath, i));

      data_reader.read((char *)&cluster_size, sizeof(uint32_t));
      data_reader.read((char *)&cluster_dim, sizeof(uint32_t));
      ids_reader.read((char *)&ids_size, sizeof(uint32_t));
      ids_reader.read((char *)&ids_dim, sizeof(uint32_t));
      entry_num = (para.blockSize - sizeof(uint32_t)) /
                  (cluster_dim * sizeof(DATAT) + ids_dim * sizeof(uint32_t));
      centroids_dim = cluster_dim;
      assert(cluster_size == ids_size);
      assert(ids_dim == 1);
      assert(entry_num > 0);


      int64_t data_size = static_cast<int64_t>(cluster_size);
      // std::cout << "train cluster" << std::to_string(i) << "data files" << data_file << "size" << data_size << std::endl;
      DATAT *datai = new DATAT[data_size * cluster_dim * 1ULL];

      uint32_t *idi = new uint32_t[ids_size * ids_dim];
      uint32_t blk_num = 0;
      data_reader.read((char *)datai,
                       data_size * cluster_dim * sizeof(DATAT));
      ids_reader.read((char *)idi, ids_size * ids_dim * sizeof(uint32_t));

      IOWriter data_writer(getClusterRawDataFileName(para.indexPrefixPath, i),
                           ioreader::MEGABYTE * 100);

      // recursive style
      // recursive_kmeans<DATAT>(i, data_size, datai, idi, cluster_dim,
      //                         entry_num, blk_size, blk_num, data_writer,
      //                         centroids_writer, centroids_id_writer, 0, false,
      //                         avg_len);
      
      // non-recursive style
      std::list<struct ClusteringTask> todo;
      std::mutex todo_mutex;
      ClusteringTask init{0, data_size, 0};
      todo.push_back(init);
      std::mutex mutex;

      while (not todo.empty()) {
          std::vector<ClusteringTask> output_tasks;

          auto cur = todo.front();
          todo.pop_front();

          if (LevelType(cur.level) >= LevelType::BALANCE_LEVEL) {
              std::cout << "non_recursive_multilevel_kmeans: "
                        << "balance level, to parallel"
                        << std::endl;
              break;
          }

          std::cout << "non_recursive_multilevel_kmeans: "
                    << " cur.offset:" << cur.offset
                    << " cur.num_elems:" << cur.num_elems
                    << " cur.level:" << cur.level
                    << std::endl;
          non_recursive_multilevel_kmeans<DATAT>(i,
                                                 cur.num_elems,
                                                 datai,
                                                 idi,
                                                 cur.offset,
                                                 cluster_dim,
                                                 entry_num,
                                                 para.blockSize,
                                                 blk_num,
                                                 data_writer,
                                                 centroids_writer,
                                                 centroids_id_writer,
                                                 local_start_position,
                                                 cur.level,
                                                 mutex,
                                                 output_tasks,
                                                 false,
                                                 avg_len);

          for (auto & output_task : output_tasks) {
              todo.push_back(output_task);
          }
      }

      auto func = [&]() {
          while (true) {
              // gurantee the access of todo list
              std::unique_lock<std::mutex> lock(todo_mutex);
              if (todo.empty()) {
                  // finish
                  std::cout << "kmeans worker finish" << std::endl;
                  break;
              }
              auto cur = todo.front();
              todo.pop_front();
              lock.unlock();

              std::cout << "non_recursive_multilevel_kmeans: "
                        << " cur.offset:" << cur.offset
                        << " cur.num_elems:" << cur.num_elems
                        << " cur.level:" << cur.level
                        << std::endl;
              std::vector<ClusteringTask> output_tasks;
              non_recursive_multilevel_kmeans<DATAT>(i,
                                                     cur.num_elems,
                                                     datai,
                                                     idi,
                                                     cur.offset,
                                                     cluster_dim,
                                                     entry_num,
                                                     para.blockSize,
                                                     blk_num,
                                                     data_writer,
                                                     centroids_writer,
                                                     centroids_id_writer,
                                                     local_start_position,
                                                     cur.level,
                                                     mutex,
                                                     output_tasks,
                                                     false,
                                                     avg_len);
              assert(output_tasks.empty());
          }
      };

      size_t number_workers = 6;
      std::vector<std::thread> workers;
      for (size_t i = 0; i < number_workers; ++i) {
          workers.push_back(std::thread(func));
      }

      for (size_t i = 0; i < number_workers; ++i) {
          workers[i].join();
      }

      // global_centroids_number will update only once after all clustering
      // global_centroids_number += blk_num;

      delete[] datai;
      delete[] idi;
    }

    int64_t end_position = centroids_id_writer.get_position();
    assert(end_position != -1);
    std::cout << "global_centroids_number: end_position: " << end_position << std::endl;

    // centroid id is uint32_t type
    global_centroids_number += (end_position - global_start_position) / sizeof(uint32_t);
    std::cout << "calculate global_centroids_number by centroids id file position: "
              << "global_centroids_number " << global_centroids_number
              << std::endl;
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
  centroids_ids_meta_writer.write((char *)&global_centroids_number,sizeof(uint32_t));
  centroids_ids_meta_writer.write((char *)&centroids_id_dim, sizeof(uint32_t));
  centroids_meta_writer.close();
  centroids_ids_meta_writer.close();

  std::cout << "hierarchical_clusters generate " << global_centroids_number
            << " centroids" << std::endl;
  return;
}

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
  util::get_bin_metadata(raw_data_bin_file, nb, dim);
  int64_t sample_num = nb * consts::K1_SAMPLE_RATE;
  std::cout << "nb = " << nb << ", dim = " << dim
            << ", sample_num 4 K1: " << sample_num << std::endl;

  *centroids = new float[K1 * dim];
  sample_data = new DATAT[sample_num * dim];
  reservoir_sampling(raw_data_bin_file, sample_num, sample_data);
  rc.RecordSection("reservoir sample with sample rate: " +
                   std::to_string(consts::K1_SAMPLE_RATE) + " done");
  double mxl, mnl;
  int64_t stat_n = std::min(static_cast<int64_t>(1000000), sample_num);
  stat_length<DATAT>(sample_data, stat_n, dim, mxl, mnl, avg_len);
  rc.RecordSection("calculate " + std::to_string(stat_n) +
                   " vectors from sample_data done");
  std::cout << "max len: " << mxl << ", min len: " << mnl
            << ", average len: " << avg_len << std::endl;
  kmeans<DATAT>(sample_num, sample_data, dim, K1, *centroids, avg_len);
  rc.RecordSection("kmeans done");
  assert((*centroids) != nullptr);

  delete[] sample_data;
  rc.ElapseFromBegin("train cluster done.");
}

template <typename DATAT, typename DISTT>
void divide_raw_data(const BBAnnParameters para, const float *centroids) {
  TimeRecorder rc("divide raw data");
  std::cout << "divide_raw_data parameters:" << std::endl;
  std::cout << " raw_data_bin_file: " << para.dataFilePath
            << " output_path: " << para.indexPrefixPath
            << " centroids: " << centroids << " K1: " << para.K1 << std::endl;
  int K1 = para.K1;
  IOReader reader(para.dataFilePath);
  uint32_t nb, dim;
  reader.read((char *)&nb, sizeof(uint32_t));
  reader.read((char *)&dim, sizeof(uint32_t));
  uint32_t placeholder = 0, const_one = 1;
  std::vector<uint32_t> cluster_size(K1, 0);
  std::vector<std::ofstream> cluster_dat_writer(K1);
  std::vector<std::ofstream> cluster_ids_writer(K1);
  for (int i = 0; i < K1; i++) {
    cluster_dat_writer[i] = std::ofstream(
        getClusterRawDataFileName(para.indexPrefixPath, i), std::ios::binary);
    cluster_ids_writer[i] = std::ofstream(
        getClusterGlobalIdsFileName(para.indexPrefixPath, i), std::ios::binary);
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

template <typename dataT>
bool BBAnnIndex2<dataT>::LoadIndex(std::string &indexPathPrefix) {
  indexPrefix_ = indexPathPrefix;
  std::cout << "Loading: " << indexPrefix_;
  uint32_t bucket_num, dim;
  util::get_bin_metadata(getBucketCentroidsFileName(), bucket_num, dim);

  hnswlib::SpaceInterface<float> *space = nullptr;
  if (MetricType::L2 == metric_) {
    space = new hnswlib::L2Space(dim);
  } else if (MetricType::IP == metric_) {
    space = new hnswlib::InnerProductSpace(dim);
  } else {
    return false;
  }
  // load hnsw
  index_hnsw_ = std::make_shared<hnswlib::HierarchicalNSW<float>>(
      space, getHnswIndexFileName());
  indexPrefix_ = indexPathPrefix;
  return true;
}

template <typename DATAT, typename DISTT>
void search_bbann_queryonly(
    std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw,
    const BBAnnParameters para, const int topk, const DATAT *pquery,
    uint32_t *answer_ids, DISTT *answer_dists, uint32_t num_query,
    uint32_t dim) {
  TimeRecorder rc("search bigann");

  std::cout << "query numbers: " << num_query << " query dims: " << dim
            << std::endl;

  uint32_t *bucket_labels =
      new uint32_t[(int64_t)num_query * para.nProbe]; // 400K * nprobe

  // Search Graph====================================
  std::cout << "search graph parameters:" << std::endl;
  std::cout << " index_hnsw: " << index_hnsw << " num_query: " << num_query
            << " query dims: " << dim << " nprobe: " << para.nProbe
            << " refine_nprobe: " << para.efSearch
            << " pquery: " << static_cast<const void *>(pquery)
            << " bucket_labels: " << static_cast<void *>(bucket_labels)
            << std::endl;
  index_hnsw->setEf(para.efSearch);
#pragma omp parallel for
  for (int64_t i = 0; i < num_query; i++) {
    // auto queryi = pquery + i * dim;
    // todo: hnsw need to support query data is not float
    float *queryi = new float[dim];
    for (int j = 0; j < dim; j++)
      queryi[j] = (float)(*(pquery + i * dim + j));
    auto reti = index_hnsw->searchKnn(queryi, para.nProbe);
    auto p_labeli = bucket_labels + i * para.nProbe;
    while (!reti.empty()) {
      *p_labeli++ = reti.top().second;
      reti.pop();
    }
    delete[] queryi;
  }
  rc.ElapseFromBegin("search+graph+done.");
  rc.RecordSection("search buckets done.");

  uint32_t cid, bid;
  uint32_t gid;
  const uint32_t vec_size = sizeof(DATAT) * dim;
  const uint32_t entry_size = vec_size + sizeof(uint32_t);
  DATAT *vec;
  std::function<void(size_t, DISTT *, uint32_t *)> heap_heapify_func;
  std::function<bool(DISTT, DISTT)> cmp_func;
  std::function<void(size_t, DISTT *, uint32_t *, DISTT, uint32_t)>
      heap_swap_top_func;
  if (para.metric == MetricType::L2) {
    heap_heapify_func = heap_heapify<CMax<DISTT, uint32_t>>;
    cmp_func = CMax<DISTT, uint32_t>::cmp;
    heap_swap_top_func = heap_swap_top<CMax<DISTT, uint32_t>>;
  } else if (para.metric == MetricType::IP) {
    heap_heapify_func = heap_heapify<CMin<DISTT, uint32_t>>;
    cmp_func = CMin<DISTT, uint32_t>::cmp;
    heap_swap_top_func = heap_swap_top<CMin<DISTT, uint32_t>>;
  }

  // init answer heap
#pragma omp parallel for schedule(static, 128)
  for (int i = 0; i < num_query; i++) {
    auto ans_disi = answer_dists + topk * i;
    auto ans_idsi = answer_ids + topk * i;
    heap_heapify_func(topk, ans_disi, ans_idsi);
  }
  rc.RecordSection("heapify answers heaps");

  char *buf = new char[para.blockSize];
  auto dis_computer = select_computer<DATAT, DATAT, DISTT>(para.metric);

  /* flat */
  for (int64_t i = 0; i < num_query; ++i) {
    const auto ii = i * para.nProbe;
    const DATAT *q_idx = pquery + i * dim;

    for (int64_t j = 0; j < para.nProbe; ++j) {
      util::parse_global_block_id(bucket_labels[ii + j], cid, bid);
      auto fh =
          std::ifstream(getClusterRawDataFileName(para.indexPrefixPath, cid),
                        std::ios::binary);
      assert(!fh.fail());

      fh.seekg(bid * para.blockSize);
      fh.read(buf, para.blockSize);

      const uint32_t entry_num = *reinterpret_cast<uint32_t *>(buf);
      char *buf_begin = buf + sizeof(uint32_t);

      for (uint32_t k = 0; k < entry_num; ++k) {
        char *entry_begin = buf_begin + entry_size * k;
        vec = reinterpret_cast<DATAT *>(entry_begin);
        auto dis = dis_computer(vec, q_idx, dim);
        if (cmp_func(answer_dists[topk * i], dis)) {
          heap_swap_top_func(
              topk, answer_dists + topk * i, answer_ids + topk * i, dis,
              *reinterpret_cast<uint32_t *>(entry_begin + vec_size));
        }
      }
    }
  }

  delete[] buf;
  delete[] bucket_labels;
}

template <typename dataT>
void BBAnnIndex2<dataT>::BatchSearchCpp(const dataT *pquery, uint64_t dim,
                                        uint64_t numQuery, uint64_t knn,
                                        const BBAnnParameters para,
                                        uint32_t *answer_ids,
                                        distanceT *answer_dists) {
  std::cout << "Query: " << std::endl;

  search_bbann_queryonly<dataT, distanceT>(
      index_hnsw_, para, knn, pquery, answer_ids, answer_dists, numQuery, dim);
}

template <typename dataT>
void BBAnnIndex2<dataT>::BuildIndexImpl(const BBAnnParameters para) {
  auto index = std::make_unique<BBAnnIndex2<dataT>>(para.metric);
  index->BuildWithParameter(para);
}

template <typename dataT>
void BBAnnIndex2<dataT>::BuildWithParameter(const BBAnnParameters para) {
  std::cout << "Build start+ " << std::endl;
  TimeRecorder rc("build bigann");
  using distanceT = typename TypeWrapper<dataT>::distanceT;
  dataFilePath_ = para.dataFilePath;
  indexPrefix_ = para.indexPrefixPath;
  std::cout << "build bigann parameters:" << std::endl;
  std::cout << " raw_data_bin_file: " << dataFilePath_
            << " output_path: " << indexPrefix_ << " hnsw.M: " << para.hnswM
            << " hnsw.efConstruction: " << para.hnswefC << " K1: " << para.K1
            << " block size: " << para.blockSize << std::endl;

  float *centroids = nullptr;
  double avg_len;
  // sampling and do K1-means to get the first round centroids
  train_cluster<dataT>(dataFilePath_, indexPrefix_, para.K1, &centroids,
                       avg_len);
  assert(centroids != nullptr);
  rc.RecordSection("train cluster to get " + std::to_string(para.K1) +
                   " centroids done.");

  divide_raw_data<dataT, distanceT>(para, centroids);
  rc.RecordSection("divide raw data into " + std::to_string(para.K1) +
                   " clusters done");

  hierarchical_clusters<dataT, distanceT>(para, avg_len);
  rc.RecordSection("conquer each cluster into buckets done");

  build_graph(indexPrefix_, para.hnswM, para.hnswefC, para.metric);
  rc.RecordSection("build hnsw done.");

  // TODO: disable statistics
  // gather_buckets_stats(indexPrefix_, para.K1, para.blockSize);
  // rc.RecordSection("gather statistics done");

  delete[] centroids;
  rc.ElapseFromBegin("build bigann totally done.");
}

template <typename dataT>
std::tuple<std::vector<uint32_t>, std::vector<float>, std::vector<uint64_t>>
BBAnnIndex2<dataT>::RangeSearchCpp(const dataT *pquery, uint64_t dim,
                                   uint64_t numQuery, double radius,
                                   const BBAnnParameters para) {
  TimeRecorder rc("range search bbann");

  std::cout << "range search bigann parameters:" << std::endl;
  std::cout << " index_path: " << para.indexPrefixPath
            << " hnsw_ef: " << para.hnswefC << " radius: " << radius
            << " K1: " << para.K1 << std::endl;

  std::cout << "query numbers: " << numQuery << " query dims: " << dim
            << std::endl;
  /*
  for (int i = 0 ; i < numQuery; i++) {
    std::cout << "query " << i <<": ";
    for (int j = 0; j < dim; j++) {
      std::cout << pquery[i*dim + j] << " ";
    }
    std::cout << std::endl;
  }
   */

  std::vector<float> query_float;
  query_float.resize(numQuery * dim);
#pragma omp parallel for
  for (int64_t i = 0; i < numQuery * dim; i++) {
    query_float[i] = (float)pquery[i];
  }
  std::cout << " prepared query_float" << std::endl;
  // std::vector<uint32_t> *bucket_labels = new std::vector<uint32_t>[numQuery];
  std::vector<std::pair<uint32_t, uint32_t>> qid_bucketLabel;

  std::map<int, int> bucket_hit_cnt, hit_cnt_cnt, return_cnt;
  index_hnsw_->setEf(para.efSearch);
  // -- a function that conducts queries[a..b] and returns a list of <bucketid,
  // queryid> pairs; note: 1 bucketid may map to multiple queryid.
  auto run_hnsw_search = [&, this](int l,
                                   int r) -> std::vector<std::pair<int, int>> {
    std::vector<std::pair<int, int>> ret;
    for (int i = l; i < r; i++) {
      float *queryi = &query_float[i * dim];
      const auto reti = index_hnsw_->searchRange(queryi, para.nProbe, radius);
      for (auto const &[dist, bucket_label] : reti) {
        ret.emplace_back(std::make_pair(bucket_label, i));
      }
    }
    return ret;
  };
  int nparts_hnsw = 128;
  std::vector<std::pair<int, int>> bucketToQuery;
#pragma omp parallel for
  for (int partID = 0; partID < nparts_hnsw; partID++) {
    int low = partID * numQuery / nparts_hnsw;
    int high = (partID + 1) * numQuery / nparts_hnsw;
    auto part = run_hnsw_search(low, high);
    bucketToQuery.insert(bucketToQuery.end(), part.begin(), part.end());
  }
  rc.RecordSection(" query hnsw done");
  sort(bucketToQuery.begin(), bucketToQuery.end());
  rc.RecordSection("sort query results done");

  const uint32_t vec_size = sizeof(dataT) * dim;
  const uint32_t entry_size = vec_size + sizeof(uint32_t);

  // -- a function that reads the file for bucketid/queryid in
  // bucketToQuery[a..b]
  //
  auto run_bucket_scan =
      [&, this, para, pquery](int l, int r) -> std::list<qidIdDistTupleType> {
    /* return a list of tuple <queryid, id, dist>:*/
    std::list<qidIdDistTupleType> ret;
    std::vector<char> buf_v(para.blockSize);
    char *buf = &buf_v[0];
    auto dis_computer = select_computer<dataT, dataT, distanceT>(para.metric);
    auto reader = std::make_unique<CachedBucketReader>(para.indexPrefixPath);
    for (int i = l; i < r; i++) {
      const auto [bucketid, qid] = bucketToQuery[i];
      const dataT *q_idx = pquery + qid * dim;
      reader->readToBuf(bucketid, buf, para.blockSize);
      const uint32_t entry_num = *reinterpret_cast<uint32_t *>(buf);
      char *data_begin = buf + sizeof(uint32_t);

      for (uint32_t k = 0; k < entry_num; ++k) {
        char *entry_begin = data_begin + entry_size * k;
        auto dis =
            dis_computer(reinterpret_cast<dataT *>(entry_begin), q_idx, dim);
        if (dis < radius) {
          const uint32_t id =
              *reinterpret_cast<uint32_t *>(entry_begin + vec_size);
          ret.push_back(std::make_tuple(qid, id, dis));
        }
      }
    }
    std::cout << "Query number:" << r-l <<"  read count:" << reader->unique_reads_ << std::endl;
    return ret;
  };
  std::cout << "Need to access bucket data for " << bucketToQuery.size()
            << " times, " << std::endl;
  uint32_t totQuery = 0;
  uint32_t totReturn = 0;
  int nparts_block = 64;
  std::vector<qidIdDistTupleType> ans_list;
#pragma omp parallel for
  for (uint64_t partID = 0; partID < nparts_block; partID++) {
    uint32_t low = partID * bucketToQuery.size() / nparts_block;
    uint32_t high = (partID + 1) * bucketToQuery.size() / nparts_block;
    totQuery += (high - low);
    auto qid_id_dist = run_bucket_scan(low, high);
    totReturn += qid_id_dist.size();
    std::move(qid_id_dist.begin(), qid_id_dist.end(),
              std::back_inserter(ans_list));
    std::cout << "finished " << totQuery << "queries, returned " << totReturn
              << " answers" << std::endl;
  }
  rc.RecordSection("scan blocks done");
  sort(ans_list.begin(), ans_list.end());
  std::vector<uint32_t> ids;
  std::vector<float> dists;
  std::vector<uint64_t> lims(numQuery + 1);
  int lims_index = 0;
  for (auto const &[qid, ansid, dist] : ans_list) {
    //std::cout << qid << " " << ansid << " " << dist << std::endl;
    while (lims_index < qid) {
      lims[lims_index] = ids.size();
      // std::cout << "lims" << lims_index << "!" << lims[lims_index] << std::endl;
      lims_index++;
    }
    if (lims[qid] == 0 && qid == lims_index) {
      lims[qid] = ids.size();
      // std::cout << "lims" << qid << " " << lims[qid] << std::endl;
      lims_index++;
    }
    ids.push_back(ansid);
    dists.push_back(dist);
    // std::cout << "ansid " << ansid << " " << ids[lims[qid]] << std::endl;
  }
  while (lims_index <= numQuery) {
    lims[lims_index] = ids.size();
    lims_index++;
  }

  rc.RecordSection("format answer done");

  rc.ElapseFromBegin("range search bbann totally done");
  return std::make_tuple(ids, dists, lims);
}

#define BBANNLIB_DECL(dataT)                                                   \
  template bool BBAnnIndex2<dataT>::LoadIndex(std::string &indexPathPrefix);   \
  template void BBAnnIndex2<dataT>::BatchSearchCpp(                            \
      const dataT *pquery, uint64_t dim, uint64_t numQuery, uint64_t knn,      \
      const BBAnnParameters para, uint32_t *answer_ids,                        \
      distanceT *answer_dists);                                                \
  template void BBAnnIndex2<dataT>::BuildIndexImpl(                            \
      const BBAnnParameters para);                                             \
  template std::tuple<std::vector<uint32_t>, std::vector<float>,               \
                      std::vector<uint64_t>>                                   \
  BBAnnIndex2<dataT>::RangeSearchCpp(const dataT *pquery, uint64_t dim,        \
                                     uint64_t numQuery, double radius,         \
                                     const BBAnnParameters para);

BBANNLIB_DECL(float);
BBANNLIB_DECL(uint8_t);
BBANNLIB_DECL(int8_t);

#undef BBANNLIB_DECL

} // namespace bbann