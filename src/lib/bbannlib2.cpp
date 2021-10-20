#include "lib/bbannlib2.h"
#include "ann_interface.h"
#include "util/TimeRecorder.h"
#include "util/file_handler.h"
#include "util/heap.h"
#include "util/utils_inline.h"
#include <iostream>
#include <memory>
#include <omp.h>
#include <stdint.h>
#include <string>
#include <unistd.h>

#include <fcntl.h> // open, pread
#include <libaio.h>
#include <stdlib.h> // posix_memalign
namespace bbann {

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
    uint32_t *answer_ids, DISTT *answer_dists, uint32_t nq, uint32_t dim) {
  TimeRecorder rc("search bigann");
  std::cout << " index_hnsw: " << index_hnsw << " num_query: " << nq
            << " query dims: " << dim << " nprobe: " << para.nProbe
            << " refine_nprobe: " << para.efSearch << std::endl;

  uint32_t *bucket_labels = new uint32_t[(int64_t)nq * para.nProbe];
  auto dis_computer = util::select_computer<DATAT, DATAT, DISTT>(para.metric);

  search_graph<DATAT>(index_hnsw, nq, dim, para.nProbe, para.efSearch, pquery,
                      bucket_labels, nullptr);
  rc.RecordSection("search buckets done.");

  uint32_t cid, bid;
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
  for (int i = 0; i < nq; i++) {
    auto ans_disi = answer_dists + topk * i;
    auto ans_idsi = answer_ids + topk * i;
    heap_heapify_func(topk, ans_disi, ans_idsi);
  }
  rc.RecordSection("heapify answers heaps");

  // step1
  std::unordered_map<uint32_t, std::pair<std::string, uint32_t>> blocks;
  std::unordered_map<uint32_t, std::vector<int64_t>>
      labels_2_qidxs; // label -> query idxs
  std::vector<uint32_t> locs;
  // std::unordered_map<uint32_t, uint32_t> inverted_locs;  // not required
  std::unordered_map<std::string, int> fds; // file -> file descriptor
  for (int64_t i = 0; i < nq; ++i) {
    const auto ii = i * para.nProbe;
    for (int64_t j = 0; j < para.nProbe; ++j) {
      auto label = bucket_labels[ii + j];
      if (blocks.find(label) == blocks.end()) {
        util::parse_global_block_id(label, cid, bid);
        std::string cluster_file_path =
            getClusterRawDataFileName(para.indexPrefixPath, cid);
        std::pair<std::string, uint32_t> blockPair(cluster_file_path, bid);
        blocks[label] = blockPair;

        if (fds.find(cluster_file_path) == fds.end()) {
          auto fd = open(cluster_file_path.c_str(), O_RDONLY | O_DIRECT);
          if (fd == 0) {
            std::cout << "open() failed, fd: " << fd
                      << ", file: " << cluster_file_path << ", errno: " << errno
                      << ", error: " << strerror(errno) << std::endl;
            exit(-1);
          }
          fds[cluster_file_path] = fd;
        }

        labels_2_qidxs[label] = std::vector<int64_t>{i};
        // inverted_locs[label] = locs.size();
        locs.push_back(label);
      }

      if (labels_2_qidxs.find(label) == labels_2_qidxs.end()) {
        labels_2_qidxs[label] = std::vector<int64_t>{i};
      } else {
        auto qidxs = labels_2_qidxs[label];
        if (std::find(qidxs.begin(), qidxs.end(), i) == qidxs.end()) {
          labels_2_qidxs[label].push_back(i);
        }
      }
    }
  }
  rc.RecordSection("calculate block position done");

  auto block_nums = blocks.size();
  std::cout << "block num: " << block_nums << "\tloc num: " << locs.size()
            << "\tnq: " << nq << "\tnprobe: " << para.nProbe
            << "\tblock_size: " << para.blockSize << std::endl;

  // step2 load unique blocks from IO
  std::vector<char *> block_bufs;
  block_bufs.resize(block_nums);
  std::cout << "block_bufs.size(): " << block_bufs.size() << std::endl;

  std::cout << "num of fds: " << fds.size() << std::endl;

  std::cout << "EAGAIN: " << EAGAIN << ", EFAULT: " << EFAULT
            << ", EINVAL: " << EINVAL << ", ENOMEM: " << ENOMEM
            << ", ENOSYS: " << ENOSYS << std::endl;

  auto max_events_num = util::get_max_events_num_of_aio();
  max_events_num = 1024;
  io_context_t ctx = 0;
  auto r = io_setup(max_events_num, &ctx);
  if (r) {
    std::cout << "io_setup() failed, returned: " << r
              << ", strerror(r): " << strerror(r) << ", errno: " << errno
              << ", error: " << strerror(errno) << std::endl;
    exit(-1);
  }

  auto n_batch = (block_nums + max_events_num - 1) / max_events_num;
  std::cout << "block_nums: " << block_nums << ", q_depth: " << max_events_num
            << ", n_batch: " << n_batch << std::endl;

  auto io_submit_threads_num = 8;
  auto io_wait_threads_num = 8;

  std::deque<std::mutex> heap_mtxs;
  heap_mtxs.resize(nq);

  for (auto n = 0; n < n_batch; n++) {
    auto begin = n * max_events_num;
    auto end = std::min(int((n + 1) * max_events_num), int(block_nums));
    auto num = end - begin;

    auto num_per_batch =
        (num + io_submit_threads_num - 1) / io_submit_threads_num;

    for (auto th = begin; th < end; th++) {
      auto r = posix_memalign((void **)(&block_bufs[th]), 512, para.blockSize);
      if (r != 0) {
        std::cout << "posix_memalign() failed, returned: " << r
                  << ", errno: " << errno << ", error: " << strerror(errno)
                  << std::endl;
        exit(-1);
      }
    }

#pragma omp parallel for
    for (auto th = 0; th < io_submit_threads_num; th++) {
      auto begin_th = begin + num_per_batch * th;
      auto end_th = std::min(int(begin_th + num_per_batch), end);
      auto num_th = end_th - begin_th;

      std::vector<struct iocb> ios(num_th);
      std::vector<struct iocb *> cbs(num_th, nullptr);
      for (auto i = begin_th; i < end_th; i++) {
        auto block = blocks[locs[i]];
        auto offset = block.second * para.blockSize;
        io_prep_pread(ios.data() + (i - begin_th), fds[block.first],
                      block_bufs[i], para.blockSize, offset);

        // Unfortunately, a lambda fundtion with capturing variables cannot
        // convert to a function pointer. But fortunately, in fact we only need
        // the location of label when the io is done.

        // auto callback = static_cast<void*>(locs.data() + i);
        // auto callback = (locs.data() + i);
        auto callback = new int[1];
        callback[0] = i;

        // io_set_callback(ios.data() + (i - begin_th), callback);
        ios[i - begin_th].data = callback;
      }

      for (auto i = 0; i < num_th; i++) {
        cbs[i] = ios.data() + i;
      }

      r = io_submit(ctx, num_th, cbs.data());
      if (r != num_th) {
        std::cout << "io_submit() failed, returned: " << r
                  << ", strerror(-r): " << strerror(-r) << ", errno: " << errno
                  << ", error: " << strerror(errno) << std::endl;
        exit(-1);
      }
    }

    auto wait_num_per_batch =
        (num + io_wait_threads_num - 1) / io_wait_threads_num;

#pragma omp parallel for
    for (auto th = 0; th < io_wait_threads_num; th++) {
      auto begin_th = begin + wait_num_per_batch * th;
      auto end_th = std::min(int(begin_th + wait_num_per_batch), end);
      auto num_th = end_th - begin_th;

      std::vector<struct io_event> events(num_th);

      r = io_getevents(ctx, num_th, num_th, events.data(), NULL);
      if (r != num_th) {
        std::cout << "io_getevents() failed, returned: " << r
                  << ", strerror(-r): " << strerror(-r) << ", errno: " << errno
                  << ", error: " << strerror(errno) << std::endl;
        exit(-1);
      }

      for (auto en = 0; en < num_th; en++) {
        auto loc = *reinterpret_cast<int *>(events[en].data);
        delete[] reinterpret_cast<int *>(events[en].data);
        auto label = locs[loc];
        // auto label = *reinterpret_cast<uint32_t *>(events[en].data);
        // char * buf = reinterpret_cast<char*>(events[en].obj->u.c.buf);
        char *buf = block_bufs[loc];
        const uint32_t entry_num = *reinterpret_cast<uint32_t *>(buf);
        char *buf_begin = buf + sizeof(uint32_t);

        auto nq_idxs = labels_2_qidxs[label];
        for (auto iter = 0; iter < nq_idxs.size(); iter++) {
          auto nq_idx = nq_idxs[iter];
          const DATAT *q_idx = pquery + nq_idx * dim;

          std::vector<DISTT> diss(entry_num);
          std::vector<uint32_t> ids(entry_num);
          for (uint32_t k = 0; k < entry_num; ++k) {
            char *entry_begin = buf_begin + entry_size * k;
            vec = reinterpret_cast<DATAT *>(entry_begin);
            auto dis = dis_computer(vec, q_idx, dim);
            auto id = *reinterpret_cast<uint32_t *>(entry_begin + vec_size);
            diss[k] = dis;
            ids[k] = id;
          }

          heap_mtxs[nq_idx].lock();
          for (auto k = 0; k < entry_num; k++) {
            auto dis = diss[k];
            auto id = ids[k];
            if (cmp_func(answer_dists[topk * nq_idx], dis)) {
              heap_swap_top_func(topk, answer_dists + topk * nq_idx,
                                 answer_ids + topk * nq_idx, dis, id);
            }
          }
          heap_mtxs[nq_idx].unlock();
        }
      }
    }

    for (auto th = begin; th < end; th++) {
      delete[] block_bufs[th];
    }
  }
  rc.RecordSection("async io done");

  /*gather_vec_searched_per_query(index_path, pquery, nq, nprobe, dq,
  block_size, buf, bucket_labels); rc.RecordSection("gather statistics done");*/

  for (auto iter = fds.begin(); iter != fds.end(); iter++) {
    close(iter->second);
  }
  rc.RecordSection("close fds done");

  delete[] bucket_labels;

  rc.ElapseFromBegin("search bigann totally done");
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
            << std::endl;

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

  // TODO()!!!!!!!!!!!!!!!)
  // gather_buckets_stats(indexPrefix_, para.K1, para.blockSize);
  rc.RecordSection("gather statistics done");

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
  auto dis_computer =
      util::select_computer<dataT, dataT, distanceT>(para.metric);

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
      auto fh = std::ifstream(getClusterRawDataFileName(cid), std::ios::binary);
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