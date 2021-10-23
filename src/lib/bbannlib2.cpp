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

template <typename dataT, typename distanceT>
bool BBAnnIndex2<dataT, distanceT>::LoadIndex(std::string &indexPathPrefix) {
  indexPrefix_ = indexPathPrefix;
  std::cout << "Loading: " << indexPrefix_;
  uint32_t bucket_num, dim;
  util::get_bin_metadata(getBucketCentroidsFileName(), bucket_num, dim);

  // hnswlib::SpaceInterface<distanceT> *space =
  auto *space = getDistanceSpace<dataT, distanceT>(metric_, dim);
  // load hnsw
  index_hnsw_ = std::make_shared<hnswlib::HierarchicalNSW<distanceT>>(
      space, getHnswIndexFileName());
  indexPrefix_ = indexPathPrefix;
  return true;
}

template <typename DATAT, typename DISTT>
void search_bbann_queryonly(
    std::shared_ptr<hnswlib::HierarchicalNSW<DISTT>> index_hnsw,
    const BBAnnParameters para, const int topk, const DATAT *pquery,
    uint32_t *answer_ids, DISTT *answer_dists, uint32_t nq, uint32_t dim) {
  TimeRecorder rc("search bigann");

  index_hnsw->setEf(para.efSearch);

  auto dis_computer = util::select_computer<DATAT, DATAT, DISTT>(para.metric);

  std::cout << "search bigann parameters:" << std::endl;
  std::cout << " index_path: " << para.indexPrefixPath
            << " nprobe: " << para.nProbe << " hnsw_ef: " << para.efSearch
            << " topk: " << topk << " K1: " << para.K1 << std::endl;

  auto nprobe = para.nProbe;

  auto bucket_labels = new uint32_t[(int64_t)nq * nprobe]; // 400K * nprobe

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

  uint32_t cid, bid;
  uint32_t gid;
  const uint32_t vec_size = sizeof(DATAT) * dim;
  const uint32_t code_size = sizeof(uint8_t) * dim;
  const uint32_t entry_size = para.vector_use_sq ? code_size + sizeof(uint32_t)
                                                 : vec_size + sizeof(uint32_t);

  // read min/min value in earch vector from file
  std::vector<DATAT> min_len(dim);
  std::vector<DATAT> max_len(dim);
  if (para.vector_use_sq) {
    std::string vector_sq_meta_file = getSQMetaFileName(para.indexPrefixPath);
    IOReader meta_reader(vector_sq_meta_file);
    meta_reader.read((char *)max_len.data(), sizeof(DATAT) * dim);
    meta_reader.read((char *)min_len.data(), sizeof(DATAT) * dim);
  }

  uint32_t max_cid = 0xff;
  std::unordered_map<uint32_t, int> fds; // cid -> file descriptor
  for (uint32_t i = 0; i < max_cid; i++) {
    std::string cluster_file_path =
        getClusterRawDataFileName(para.indexPrefixPath, i);
    auto fd = open(cluster_file_path.c_str(), O_RDONLY | O_DIRECT);
    if (fd == 0) {
      std::cout << "open() failed, fd: " << fd
                << ", file: " << cluster_file_path << ", errno: " << errno
                << ", error: " << strerror(errno) << std::endl;
      continue;
    }
    fds[i] = fd;
  }
  rc.RecordSection(std::string("open files done, number of clusters: ") +
                   std::to_string(fds.size()));

  auto max_blocks_num = 1024 * 1024;
  if (max_blocks_num > nq * nprobe) {
    max_blocks_num = nq * nprobe;
  }
  std::vector<char *> block_bufs(max_blocks_num);
  for (auto i = 0; i < max_blocks_num; i++) {
    auto r = posix_memalign((void **)(&block_bufs[i]), 4096, para.blockSize);
    if (r != 0) {
      std::cout << "posix_memalign() failed, returned: " << r
                << ", errno: " << errno << ", error: " << strerror(errno)
                << std::endl;
      exit(-1);
    }
  }
  rc.RecordSection("allocate cache memory");

  auto fio_way = [&](io_context_t aio_ctx, std::vector<char *> &bufs, int begin,
                     int end, int nr, int wait_nr) {
    auto page_cache_num = bufs.size();

    auto num = end - begin;

    if (num < nr) {
      nr = num;
    }

    if (num < wait_nr) {
      wait_nr = num;
    }

    if (nr < wait_nr) {
      wait_nr = nr;
    }

    std::vector<struct iocb> ios(num);
    std::vector<struct iocb *> cbs(num, nullptr);
    std::vector<struct io_event> events(num);
#pragma omp parallel for
    for (auto i = 0; i < num; i++) {
      auto loc = begin + i;
      auto label = bucket_labels[loc];
      uint32_t cid, bid;
      util::parse_global_block_id(label, cid, bid);
      io_prep_pread(ios.data() + i, fds[cid], bufs[loc % page_cache_num],
                    para.blockSize, bid * para.blockSize);

      cbs[i] = ios.data() + i;
    }

    auto done = 0;
    auto submitted = 0;
    auto to_submit_num = nr;

    while (done < num) {
      auto uppper = num - submitted;
      if (to_submit_num > uppper) {
        to_submit_num = uppper;
      }
      if (to_submit_num > nr) {
        to_submit_num = nr;
      }

      if (to_submit_num > 0) {
        auto r_submit =
            io_submit(aio_ctx, to_submit_num, cbs.data() + submitted);
        if (r_submit < 0) {
          std::cout << "io_submit() failed, returned: " << r_submit
                    << ", strerror(-r): " << strerror(-r_submit)
                    << ", begin: " << begin << ", end: " << end
                    << ", submitted: " << submitted << std::endl;
          exit(-1);
        }
        submitted += r_submit;
      }

      auto pending = submitted - done;
      if (wait_nr > pending) {
        wait_nr = pending;
      }
      auto r_done =
          io_getevents(aio_ctx, wait_nr, nr, events.data() + done, NULL);
      if (r_done < wait_nr) {
        std::cout << "io_getevents() failed, returned: " << r_done
                  << ", strerror(-): " << strerror(-r_done) << std::endl;
        exit(-1);
      }

      to_submit_num = r_done;
      done += r_done;
    }
  };

  // not thread safe.
  auto compare_by_label = [&](int q, int loc, const std::vector<char *> &bufs,
                              bool twice = false) {
    auto page_cache_num = bufs.size();

    char *buf = bufs[loc % page_cache_num];
    const DATAT *q_idx = pquery + q * dim;

    const uint32_t entry_num = *reinterpret_cast<uint32_t *>(buf);
    char *buf_begin = buf + sizeof(uint32_t);

    DATAT *vec;
    std::vector<DATAT> code_vec(dim);

    for (uint32_t k = 0; k < entry_num; ++k) {
      char *entry_begin = buf_begin + entry_size * k;

      if (para.vector_use_sq) {
        decode_uint8(max_len.data(), min_len.data(), code_vec.data(),
                     reinterpret_cast<uint8_t *>(entry_begin), 1, dim);
        vec = code_vec.data();
      } else {
        vec = reinterpret_cast<DATAT *>(entry_begin);
      }

      auto dis = dis_computer(vec, q_idx, dim);

      uint32_t id;
      if (para.vector_use_sq) {
        id = *reinterpret_cast<uint32_t *>(entry_begin + code_size);
      } else {
        id = *reinterpret_cast<uint32_t *>(entry_begin + vec_size);
      }

      if (cmp_func(answer_dists[topk * q], dis)) {
        heap_swap_top_func(topk, answer_dists + topk * q, answer_ids + topk * q,
                           dis, id);
      }
    }
  };

  auto compute = [&](const std::vector<char *> &bufs, int i,
                     bool twice = false) {
    const auto ii = i * nprobe;

    for (int64_t j = 0; j < nprobe; ++j) {
      compare_by_label(i, ii + j, bufs, twice);
    }
  };

  auto run_query = [&](int l, int r) {
    // step 1: search graph.
    search_graph<DATAT>(index_hnsw, r - l, dim, nprobe, para.efSearch,
                        pquery + l * dim, bucket_labels + l * nprobe, nullptr);

    auto num_jobs = 4;
    auto max_events_num = 1023;
    auto nr = 32;
    std::vector<io_context_t> ctxs(num_jobs, 0);
    for (auto i = 0; i < num_jobs; i++) {
      if (io_setup(max_events_num, &ctxs[i])) {
        std::cout << "io_setup() failed !" << std::endl;
        exit(-1);
      }
    }

#pragma omp parallel for
    for (auto i = 0; i < num_jobs; i++) {
      auto l_ = l + (r - l) * i / num_jobs;
      auto r_ = l + (r - l) * (i + 1) / num_jobs;
      if (r_ > r) {
        r_ = r;
      }

      auto begin = l_ * nprobe;
      auto end = r_ * nprobe;

      // step 2: io.
      fio_way(ctxs[i], block_bufs, begin, end, max_events_num, nr);

      // step 3: compute distance && heap sort.
#pragma omp parallel for
      for (auto j = l_; j < r_; j++) {
        compute(block_bufs, j);
      }
    }

    for (auto i = 0; i < num_jobs; i++) {
      io_destroy(ctxs[i]);
    }
  };

  auto n_batch = util::round_up_div(nq * nprobe, max_blocks_num);
  auto nq_per_batch = util::round_up_div(nq, n_batch);
  std::cout << "nq: " << nq << ", n_batch: " << n_batch
            << ", nq_per_batch: " << nq_per_batch << std::endl;

  auto run_batch_query = [&](int n) {
    auto q_begin = n * nq_per_batch;
    auto q_end = (n + 1) * nq_per_batch;
    if (q_end > nq) {
      q_end = nq;
    }

    auto threads_num = 16;
    auto num = q_end - q_begin;
    if (num < threads_num) {
      threads_num = num;
    }

#pragma omp parallel for
    for (auto i = 0; i < threads_num; i++) {
      auto l = q_begin + i * num / threads_num;
      auto r = q_begin + (i + 1) * num / threads_num;
      if (r > q_end) {
        r = q_end;
      }

      run_query(l, r);
    }
  };

  for (auto i = 0; i < n_batch; i++) {
    run_batch_query(i);
  }
  rc.RecordSection("query done");

  for (auto i = 0; i < max_blocks_num; i++) {
    delete[] block_bufs[i];
  }
  rc.RecordSection("release cache buf");

  for (auto iter = fds.begin(); iter != fds.end(); iter++) {
    close(iter->second);
  }
  rc.RecordSection("close fds done");

  delete[] bucket_labels;

  rc.ElapseFromBegin("search bigann totally done");
}

template <typename dataT, typename distanceT>
void BBAnnIndex2<dataT, distanceT>::BatchSearchCpp(
    const dataT *pquery, uint64_t dim, uint64_t numQuery, uint64_t knn,
    const BBAnnParameters para, uint32_t *answer_ids, distanceT *answer_dists) {
  std::cout << "Query: " << std::endl;

  search_bbann_queryonly<dataT, distanceT>(
      index_hnsw_, para, knn, pquery, answer_ids, answer_dists, numQuery, dim);
}

template <typename dataT, typename distanceT>
void BBAnnIndex2<dataT, distanceT>::BuildIndexImpl(const BBAnnParameters para) {
  auto index = std::make_unique<BBAnnIndex2<dataT, distanceT>>(para.metric);
  index->BuildWithParameter(para);
}

template <typename dataT, typename distanceT>
void BBAnnIndex2<dataT, distanceT>::BuildWithParameter(
    const BBAnnParameters para) {
  std::cout << "Build start+ " << std::endl;
  TimeRecorder rc("build bigann");
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
                       avg_len, para.vector_use_sq);
  assert(centroids != nullptr);
  rc.RecordSection("train cluster to get " + std::to_string(para.K1) +
                   " centroids done.");

  divide_raw_data<dataT, distanceT>(para, centroids);
  rc.RecordSection("divide raw data into " + std::to_string(para.K1) +
                   " clusters done");

  hierarchical_clusters<dataT, distanceT>(para, avg_len);
  rc.RecordSection("conquer each cluster into buckets done");

  build_graph<dataT, distanceT>(indexPrefix_, para.hnswM, para.hnswefC,
                                para.metric, para.blockSize, para.sample);
  rc.RecordSection("build hnsw done.");

  // TODO()!!!!!!!!!!!!!!!)
  // gather_buckets_stats(indexPrefix_, para.K1, para.blockSize);
  rc.RecordSection("gather statistics done");

  delete[] centroids;
  rc.ElapseFromBegin("build bigann totally done.");
}

template <typename dataT, typename distanceT>
std::tuple<std::vector<uint32_t>, std::vector<distanceT>, std::vector<uint64_t>>
BBAnnIndex2<dataT, distanceT>::RangeSearchCpp(const dataT *pquery, uint64_t dim,
                                              uint64_t numQuery, double radius,
                                              const BBAnnParameters para) {
  TimeRecorder rc("range search bbann");


  std::cout << "query numbers: " << numQuery << " query dims: " << dim
            << std::endl;

  std::vector<std::pair<uint32_t, uint32_t>> qid_bucketLabel;

  // std::map<int, int> bucket_hit_cnt, hit_cnt_cnt, return_cnt;
  index_hnsw_->setEf(para.efSearch);
  // -- a function that conducts queries[a..b] and returns a list of <bucketid,
  // queryid> pairs; note: 1 bucketid may map to multiple queryid.
  auto run_hnsw_search = [&, this](int l,
                                   int r) -> std::vector<std::pair<int, int>> {
    std::vector<std::pair<int, int>> ret;
    for (int i = l; i < r; i++) {
      const auto reti = index_hnsw_->searchRange(
          pquery + i * dim, para.rangeSearchProbeCount, radius*para.radiusFactor);
      for (auto const &[dist, bucket_label] : reti) {
        // convert the bucket label from 64bit to 32 bit
        uint32_t cid, bid, offset;
        bbann::util::parse_id(bucket_label, cid, bid, offset);
        auto bucket_label32 = bbann::util::gen_global_block_id(cid, bid);
        ret.emplace_back(std::make_pair(bucket_label32, i));
      }
    }
    return ret;
  };
  rc.RecordSection("prep query done");
  int nparts_hnsw = 128;
  std::vector<std::pair<int, int>> bucketToQuery;
#pragma omp parallel for
  for (int partID = 0; partID < nparts_hnsw; partID++) {
    int low = partID * numQuery / nparts_hnsw;
    int high = (partID + 1) * numQuery / nparts_hnsw;
    auto part = run_hnsw_search(low, high);
#pragma omp critical
    bucketToQuery.insert(bucketToQuery.end(), part.begin(), part.end());
  }
  rc.RecordSection(" query hnsw done");
  sort(bucketToQuery.begin(), bucketToQuery.end());
  rc.RecordSection("sort query results done");

  const uint32_t vec_size = sizeof(dataT) * dim;
  const uint32_t entry_size = vec_size + sizeof(uint32_t);
  AIOBucketReader reader(para.indexPrefixPath, para.aio_EventsPerBatch);
  // -- a function that reads the file for bucketid/queryid in
  // bucketToQuery[a..b]
  //
  auto run_bucket_scan =
      [&, this, para, pquery](int l, int r) -> std::list<qidIdDistTupleType> {
    /* return a list of tuple <queryid, id, dist>:*/
    std::list<qidIdDistTupleType> ret;

    auto dis_computer =
        util::select_computer<dataT, dataT, distanceT>(para.metric);
    std::vector<uint32_t> bucketIds;
    bucketIds.reserve(r - l);
    for (int i = l; i < r; i++) {
      bucketIds.emplace_back(bucketToQuery[i].first);
    }
    void *big_read_buf;
#pragma omp critical
    if (posix_memalign(&big_read_buf, 512, para.blockSize * (r - l)) != 0) {
      std::cerr << " err allocating  buf" << std::endl;
      exit(-1);
    }
    std::vector<uint32_t> resIds;
    resIds = reader.ReadToBuf(bucketIds, para.blockSize, big_read_buf);

    for (int i = l; i < r; i++) {
      const auto qid = bucketToQuery[i].second;
      const dataT *q_idx = pquery + qid * dim;
      char *buf = (char *)big_read_buf + resIds[i - l] * para.blockSize;
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
    free(big_read_buf);
    return ret;
  };
  // execute queries with "index mod (1<<shift) == thid".

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
#pragma omp critical
    std::move(qid_id_dist.begin(), qid_id_dist.end(),
              std::back_inserter(ans_list));
    rc.ElapseFromBegin("disk access part " + std::to_string(partID) +
                       "out of " + std::to_string(nparts_block) + " done");
  }
  rc.RecordSection("scan blocks done");
  sort(ans_list.begin(), ans_list.end());
  std::vector<uint32_t> ids;
  std::vector<distanceT> dists;
  std::vector<uint64_t> lims(numQuery + 1);
  int lims_index = 0;
  for (auto const &[qid, ansid, dist] : ans_list) {
    // std::cout << qid << " " << ansid << " " << dist << std::endl;
    while (lims_index < qid) {
      lims[lims_index] = ids.size();
      // std::cout << "lims" << lims_index << "!" << lims[lims_index] <<
      // std::endl;
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

#define BBANNLIB_DECL(dataT, distanceT)                                        \
  template bool BBAnnIndex2<dataT, distanceT>::LoadIndex(                      \
      std::string &indexPathPrefix);                                           \
  template void BBAnnIndex2<dataT, distanceT>::BatchSearchCpp(                 \
      const dataT *pquery, uint64_t dim, uint64_t numQuery, uint64_t knn,      \
      const BBAnnParameters para, uint32_t *answer_ids,                        \
      distanceT *answer_dists);                                                \
  template void BBAnnIndex2<dataT, distanceT>::BuildIndexImpl(                 \
      const BBAnnParameters para);                                             \
  template std::tuple<std::vector<uint32_t>, std::vector<distanceT>,           \
                      std::vector<uint64_t>>                                   \
  BBAnnIndex2<dataT, distanceT>::RangeSearchCpp(                               \
      const dataT *pquery, uint64_t dim, uint64_t numQuery, double radius,     \
      const BBAnnParameters para);

BBANNLIB_DECL(float, float);
BBANNLIB_DECL(uint8_t, uint32_t);
BBANNLIB_DECL(int8_t, int32_t);

#undef BBANNLIB_DECL

} // namespace bbann