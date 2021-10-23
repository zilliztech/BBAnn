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
  const uint32_t code_size = sizeof(uint8_t) * dim;
  const uint32_t entry_size = para.vector_use_sq ? code_size + sizeof(uint32_t) : vec_size + sizeof(uint32_t);


  // read min/min value in earch vector from file
  std::vector<DATAT> min_len(dim);
  std::vector<DATAT> max_len(dim);
  if (para.vector_use_sq) {
      std::string vector_sq_meta_file = getSQMetaFileName(para.indexPrefixPath);
      IOReader meta_reader(vector_sq_meta_file);
      meta_reader.read((char*)max_len.data(), sizeof(DATAT) * dim);
      meta_reader.read((char*)min_len.data(), sizeof(DATAT) * dim);
  }

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

  std::vector<int> fds; // file -> file descriptor
  fds.resize(para.K1);
  for (int i = 0; i < para.K1; i++) {
      std::string cluster_file_path = getClusterRawDataFileName(para.indexPrefixPath, i);
      auto fd = open(cluster_file_path.c_str(), O_RDONLY | O_DIRECT);
      if (fd == 0) {
          std::cout << "open() failed, fd: " << fd
                    << ", file: " << cluster_file_path << ", errno: " << errno
                    << ", error: " << strerror(errno) << std::endl;
          exit(-1);
      }
      fds[i] = fd;
  }

  std::unordered_map<uint32_t, std::unordered_set<int64_t>> labels_2_qidxs; // label -> query idxs
  for (int64_t i = 0; i < nq; ++i) {
    const auto ii = i * para.nProbe;
    for (int64_t j = 0; j < para.nProbe; ++j) {
      auto label = bucket_labels[ii + j];
      if (labels_2_qidxs.find(label) == labels_2_qidxs.end()) {
        labels_2_qidxs[label] = std::unordered_set<int64_t>{i};
      } else {
        labels_2_qidxs[label].insert(i);
      }
    }
  }

  int total = 0;
  // relations between postion to bucket label
  std::vector<uint32_t> locs;
  locs.resize(labels_2_qidxs.size());
  int i = 0;
  for (auto iter : labels_2_qidxs) {
    total += iter.second.size();
    locs[i] = iter.first;
    i++;
  }

  rc.RecordSection("calculate block position done");

  auto block_nums = labels_2_qidxs.size();
  std::cout << "block num: " << block_nums
            << "\tnq: " << nq << "\tnprobe: " << para.nProbe
            << "\ttotal: " << total
            << "\tblock_size: " << para.blockSize << std::endl;

  auto max_events_num = util::get_max_events_num_of_aio();
  max_events_num = 512;

auto fio_way = [&](io_context_t aio_ctx, std::vector<char *> &bufs, int begin, int end) {
    auto num = end - begin;
    std::vector<struct iocb> ios(num);
    std::vector<struct iocb *> cbs(num, nullptr);
    std::vector<struct io_event> events(num);
    for (auto i = 0; i < num; i++) {
        // find the correct position
        auto label = locs[begin + i];
        uint32_t cid, bid;
        util::parse_global_block_id(label, cid, bid);
        io_prep_pread(ios.data() + i, fds[cid], bufs[i], para.blockSize, bid * para.blockSize);
        cbs[i] = ios.data() + i;
    }

    auto submitted = 0;

    while (submitted != num) {
        auto r_submit = io_submit(aio_ctx, num - submitted, cbs.data() + submitted);
        if (r_submit < 0) {
            std::cout << "io_submit() failed, returned: " << r_submit
                      << ", strerror(-r): " << strerror(-r_submit)
                      << ", begin: " << begin << ", end: " << end
                      << ", submitted: " << submitted << std::endl;
            exit(-1);
        }
        submitted += r_submit;
    }

    auto done = 0;
    while (done != num) {
        auto r_done = io_getevents(aio_ctx, num, num, events.data(), NULL);
        if (r_done < 0) {
            std::cout << "io_getevents() failed, returned: " << r_done
                      << ", strerror(-): " << strerror(-r_done) << std::endl;
            exit(-1);
        }
        done += r_done;
    }
};

   int num_jobs = 1;
   std::vector<io_context_t> ctxs(num_jobs, 0);
   for (auto i = 0; i < num_jobs; i++) {
        if (io_setup(max_events_num, &ctxs[i])) {
            std::cout << "io_setup() failed !" << std::endl;
            exit(-1);
        }
   }

  std::vector<std::vector<char *>> taskQueues;
  taskQueues.resize(nq);
  std::mutex* locks = new std::mutex[nq];
  auto ioTask = [&](io_context_t aio_ctx, long threadStart, long threadEnd, int max_events_num) {
      std::cout<<"io task start, start " << threadStart << "end " << threadEnd << std::endl;
      int total = threadEnd - threadStart;
      int batch = (total + max_events_num - 1) / max_events_num ;
      int insert = 0;
      for (int i = 0; i < batch; i++) {
          long begin = threadStart + i * max_events_num;
          long end = std::min(begin + max_events_num, threadEnd);
          long batchNum = end - begin;
          std::vector<char *> block_bufs;
          block_bufs.resize(batchNum);
          for (int j = 0; j < batchNum; j++) {
              auto r = posix_memalign((void **) (&block_bufs[j]), 512, para.blockSize);
              if (r != 0) {
                  std::cout << "posix_memalign() failed, returned: " << r
                            << ", errno: " << errno << ", error: " << strerror(errno)
                            << std::endl;
                  exit(-1);
              }
          }
          fio_way(aio_ctx, block_bufs, begin, end);
          for (int j = 0; j < batchNum; j++) {
              auto nq_idxs = labels_2_qidxs[locs[j + begin]];
              for (const auto curNq: nq_idxs) {
                  locks[curNq].lock();
                  taskQueues[curNq].push_back(block_bufs[j]);
                  locks[curNq].unlock();
                  insert++;
              }
          }
      }
      std::cout<<"Stop with io inserted "<< insert << std::endl;
  };

  std::vector<std::thread> ioReaders;
  ioReaders.resize(num_jobs);
  long threadBatch = (block_nums + num_jobs -1) / num_jobs;
  for (int i =0; i < num_jobs; i++) {
      int threadStart = i * threadBatch;
      int threadEnd = (i + 1) * threadBatch;
      if (threadEnd > block_nums) {
          threadEnd = block_nums;
      }
      ioReaders[i] = std::thread(ioTask, ctxs[i], threadStart, threadEnd, max_events_num);
  }


    std::atomic<bool> stop (false);
    auto computer = [&](std::vector<std::vector<char *>> taskQueues, int nqStart, int nqEnd) {
        std::cout<<"computer start, start " << nqStart << "end " << nqEnd << std::endl;
        const uint32_t vec_size = sizeof(DATAT) * dim;
        const uint32_t entry_size = vec_size + sizeof(uint32_t);
        bool localStop = false;
        int processed = 0;
        while (true) {
            for (int nq_idx = nqStart; nq_idx < nqEnd; nq_idx++) {
                std::cout<<" handle " << nq_idx << std::endl;
                locks[nq_idx].lock();
                std::vector<char *> localTask;
                localTask.insert(localTask.begin(), taskQueues[nq_idx].begin(), taskQueues[nq_idx].end());
                taskQueues[nq_idx].clear();
                locks[nq_idx].unlock();
                std::cout<<" handle " << nq_idx << "with " << localTask.size()<< std::endl;

                if (localTask.empty()) {
                    continue;
                }
                // do the real caculation
                const DATAT *q_idx = pquery + nq_idx * dim;
                DATAT *vec;
                for (char* block : localTask) {
                    processed++;
                    const uint32_t entry_num = *reinterpret_cast<uint32_t *>(block);
                    std::cout<<"entry num " << entry_num << std::endl;
                    char *buf_begin = block + sizeof(uint32_t);
                    for (uint32_t k = 0; k < entry_num; ++k) {
                        char *entry_begin = buf_begin + entry_size * k;
                        uint32_t id;
                        if (para.vector_use_sq) {
                            std::vector<DATAT> code_vec(dim);
                            decode_uint8(max_len.data(), min_len.data(), code_vec.data(), reinterpret_cast<uint8_t *>(entry_begin), 1, dim);
                            vec = code_vec.data();
                            id = *reinterpret_cast<uint32_t *>(entry_begin + code_size);
                        } else {
                            vec = reinterpret_cast<DATAT *>(entry_begin);
                            id = *reinterpret_cast<uint32_t *>(entry_begin + vec_size);
                        }
                        auto dis = dis_computer(vec, q_idx, dim);
                        if (cmp_func(answer_dists[topk * nq_idx], dis)) {
                            heap_swap_top_func(topk, answer_dists + topk * nq_idx,
                                               answer_ids + topk * nq_idx, dis, id);
                        }
                    }
                    delete[] block;
                }
            }
            // last round
            if (localStop) {
                std::cout<<"Stop with the last round, exit with processed "<< processed << std::endl;
                break;
            }
            // make sure this happens after break, we need a final round to sweep out all the exist tasks
            localStop = stop;
        }
    };

    int32_t num_computer_jobs = 1;
    std::vector<std::thread> computers;
    computers.resize(num_computer_jobs);
    long nqPerThread =(nq + num_computer_jobs - 1)/ num_computer_jobs ;
    for (int i = 0; i < num_computer_jobs; i++) {
        int threadStart = i * nqPerThread;
        int threadEnd = (i + 1) * nqPerThread;
        if (threadEnd > nq) {
            threadEnd = nq;
        }
        computers[i] = std::thread(computer, taskQueues, threadStart, threadEnd);
    }

    for (auto& t: ioReaders) {
      t.join();
    }
    std::cout<< "IO Thread join!!" << std::endl;
    stop = true;
    for (auto& t: computers) {
        t.join();
    }
    std::cout<< "CPU Thread join!!" << std::endl;
  for (auto i = 0; i < num_jobs; i++) {
      io_destroy(ctxs[i]);
  }

  rc.RecordSection("async io/calculation done");

  /*gather_vec_searched_per_query(index_path, pquery, nq, nprobe, dq,
  block_size, buf, bucket_labels); rc.RecordSection("gather statistics done");*/

  for (int i = 0; i < para.K1; i++) {
        close(fds[i]);
  }
  rc.RecordSection("close fds done");

  delete[] bucket_labels;
  delete[] locks;

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
  // std::vector<uint32_t> *bucket_labels = new std::vector<uint32_t>[numQuery];
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
          pquery + i * dim, para.rangeSearchProbeCount, radius);
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

    auto dis_computer = util::select_computer<dataT, dataT, distanceT>(para.metric);
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
    std::cout << "Finished query number:" << r - l << std::endl;
    free(big_read_buf);
    std::cout << " relleased big_read_buf" << std::endl;
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
    std::cout << "finished " << totQuery << "queries, returned " << totReturn
              << " answers" << std::endl;
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