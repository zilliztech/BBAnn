#include "lib/bbannlib2.h"
#include "ann_interface.h"
#include "util/TimeRecorder.h"
#include "util/file_handler.h"
#include "util/heap.h"
#include "util/utils_inline.h"
#include <iostream>
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

    for (uint32_t i = 0; i < K1; i++) {
      TimeRecorder rci("train-cluster-" + std::to_string(i));
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
      recursive_kmeans<DATAT>(i, data_size, datai, idi, cluster_dim, entry_num,
                              para.blockSize, blk_num, data_writer, centroids_writer,
                              centroids_id_writer, 0, false, avg_len);

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
  // rc.ElapseFromBegin("search+graph+done.");
  // for (int i = 0; i < 10; i++) {
  //   for (int j = 0; j < para.nProbe; j++)
  //     std::cout << bucket_labels[i * para.nProbe + j] << " ";
  //   std::cout << std::endl;
  // }
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

  std::function<void(const BBAnnParameters, const double)>
      hierarchical_clusters_func;
  std::function<void(const BBAnnParameters, const float *)>
      divide_raw_data_func;

  divide_raw_data_func = divide_raw_data<dataT, distanceT>;
  hierarchical_clusters_func = hierarchical_clusters<dataT, distanceT>;

  divide_raw_data_func(para, centroids);
  rc.RecordSection("divide raw data into " + std::to_string(para.K1) +
                   " clusters done");

  hierarchical_clusters_func(para, avg_len);
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
void BBAnnIndex2<dataT>::RangeSearchCpp(const dataT *pquery, uint64_t dim,
                                        uint64_t numQuery, double radius,
                                        const BBAnnParameters para,
                                        std::vector<std::vector<uint32_t>> &ids,
                                        std::vector<std::vector<float>> &dists,
                                        std::vector<uint64_t> &lims) {
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

  std::vector<uint32_t> *bucket_labels = new std::vector<uint32_t>[numQuery];

  index_hnsw_->setEf(para.hnswefC);
#pragma omp parallel for
  for (int64_t i = 0; i < numQuery; i++) {
    // todo: hnsw need to support query data is not float
    float *queryi = new float[dim];
    for (int j = 0; j < dim; j++)
      queryi[j] = (float)(*(pquery + i * dim + j));
    auto reti = index_hnsw_->searchRange(queryi, 20, radius);
    while (!reti.empty()) {
      bucket_labels[i].push_back(reti.top().second);
      reti.pop();
    }
    delete[] queryi;
  }
  rc.RecordSection("search buckets done.");

  uint32_t cid, bid;
  uint32_t gid;
  const uint32_t vec_size = sizeof(dataT) * dim;
  const uint32_t entry_size = vec_size + sizeof(uint32_t);
  dataT *vec;

  char *buf = new char[para.blockSize];
  auto dis_computer = select_computer<dataT, dataT, distanceT>(para.metric);

  uint64_t total_bucket_searched = 0;

  // for (int i = 0; i < numQuery; i++) {
  //   std::cout << bucket_labels[i].size() << " ";
  // }
  // std::cout << std::endl;
  /* flat */
  for (int64_t i = 0; i < numQuery; ++i) {
    if (i % 10000 == 0) {
      std::cout << i << "/" << numQuery << std::endl;
    }

    const dataT *q_idx = pquery + i * dim;

    for (int64_t j = 0; j < bucket_labels[i].size(); ++j) {
      util::parse_global_block_id(bucket_labels[i][j], cid, bid);
      auto fh = std::ifstream(getClusterRawDataFileName(para.indexPrefixPath, cid), std::ios::binary);
      assert(!fh.fail());

      fh.seekg(bid * para.blockSize);
      fh.read(buf, para.blockSize);

      const uint32_t entry_num = *reinterpret_cast<uint32_t *>(buf);
      char *buf_begin = buf + sizeof(uint32_t);

      for (uint32_t k = 0; k < entry_num; ++k) {
        char *entry_begin = buf_begin + entry_size * k;
        vec = reinterpret_cast<dataT *>(entry_begin);
        auto dis = dis_computer(vec, q_idx, dim);
        /*
          for (int qq = 0; qq < dim; qq++) {
            std::cout << q_idx[qq] << " ";
          }
          std::cout << std::endl;
          for (int qq = 0; qq < dim; qq++) {
            std::cout << vec[qq] << " ";
          }
          std::cout << "----" << dis << std::endl;

          std::cout << " " << dis << " " << radius << std::endl;
          */
        if (dis < radius) {
          dists[i].push_back(dis);
          ids[i].push_back(
              *reinterpret_cast<uint32_t *>(entry_begin + vec_size));
        }
      }
    }
    total_bucket_searched += bucket_labels[i].size();
  }
  rc.RecordSection("scan blocks done");

  int64_t idx = 0;
  for (int64_t i = 0; i < numQuery; ++i) {
    lims[i] = idx;
    idx += ids[i].size();
    // if (ids[i].size() > 0) {
    //   for (int j = 0; j < ids[i].size(); j++) {
    //     std::cout << dists[i][j] << " ";
    //   }
    //   std::cout << "---> " << i << std::endl;
    // }
  }
  lims[numQuery] = idx;

  rc.RecordSection("format answer done");

  std::cout << "Total bucket searched: " << total_bucket_searched << std::endl;

  delete[] bucket_labels;
  delete[] buf;
  rc.ElapseFromBegin("range search bbann totally done");
}

#define BBANNLIB_DECL(dataT)                                                   \
  template bool BBAnnIndex2<dataT>::LoadIndex(std::string &indexPathPrefix);   \
  template void BBAnnIndex2<dataT>::BatchSearchCpp(                            \
      const dataT *pquery, uint64_t dim, uint64_t numQuery, uint64_t knn,      \
      const BBAnnParameters para, uint32_t *answer_ids,                        \
      distanceT *answer_dists);                                                \
  template void BBAnnIndex2<dataT>::BuildIndexImpl(                            \
      const BBAnnParameters para);                                             \
  template void BBAnnIndex2<dataT>::RangeSearchCpp(                            \
      const dataT *pquery, uint64_t dim, uint64_t numQuery, double radius,     \
      const BBAnnParameters para, std::vector<std::vector<uint32_t>> &ids,     \
      std::vector<std::vector<float>> &dists, std::vector<uint64_t> &lims);

BBANNLIB_DECL(float);
BBANNLIB_DECL(uint8_t);
BBANNLIB_DECL(int8_t);

#undef BBANNLIB_DECL

} // namespace bbann