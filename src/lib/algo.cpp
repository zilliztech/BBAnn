#include "lib/algo.h"
#include "lib/ivf.h"
#include "util/constants.h"
#include "util/file_handler.h"
#include "util/statistics.h"
#include "util/utils_inline.h"
#include <iostream>
#include <memory>
#include <thread>

namespace bbann {
std::string Hello() { return "Hello!!!!"; }

template <typename DATAT, typename DISTT>
void search_graph(std::shared_ptr<hnswlib::HierarchicalNSW<DISTT>> index_hnsw,
                  const int nq, const int dq, const int nprobe,
                  const int refine_nprobe, const DATAT *pquery,
                  uint32_t *buckets_label, float *centroids_dist) {

  TimeRecorder rc("search graph");
  std::cout << "search graph parameters:" << std::endl;
  std::cout << " index_hnsw: " << index_hnsw << " nq: " << nq << " dq: " << dq
            << " nprobe: " << nprobe << " refine_nprobe: " << refine_nprobe
            << " pquery: " << static_cast<const void *>(pquery)
            << " buckets_label: " << static_cast<void *>(buckets_label)
            << std::endl;
  index_hnsw->setEf(refine_nprobe);
  bool set_distance = centroids_dist != nullptr;
#pragma omp parallel for
  for (int64_t i = 0; i < nq; i++) {
    // move the duplicate logic into inner loop
    // TODO optimize this
    float *queryi_dist = set_distance ? centroids_dist + i * nprobe : nullptr;
    auto reti = index_hnsw->searchKnn(pquery + i * dq, nprobe);
    auto p_labeli = buckets_label + i * nprobe;
    while (!reti.empty()) {
      uint32_t cid, bid, offset;
      bbann::util::parse_id(reti.top().second, cid, bid, offset);
      *p_labeli++ = bbann::util::gen_global_block_id(cid, bid);
      if (set_distance) {
        *queryi_dist++ = reti.top().first;
      }
      reti.pop();
    }
  }
  rc.ElapseFromBegin("search graph done.");
}

template <typename DATAT>
void train_cluster(const std::string &raw_data_bin_file,
                   const std::string &output_path, const int32_t K1,
                   float **centroids, double &avg_len, bool vector_use_sq) {
  TimeRecorder rc("train cluster");
  std::cout << "train_cluster parameters:" << std::endl;
  std::cout << " raw_data_bin_file: " << raw_data_bin_file
            << " output path: " << output_path << " K1: " << K1
            << " centroids: " << *centroids << std::endl;
  assert((*centroids) == nullptr);
  DATAT *sample_data = nullptr;
  uint32_t nb, dim;
  util::get_bin_metadata(raw_data_bin_file, nb, dim);
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

  if(vector_use_sq) {
      std::vector<DATAT> max_len;
      std::vector<DATAT> min_len;
      max_len.assign(dim, std::numeric_limits<DATAT>::min());
      min_len.assign(dim, std::numeric_limits<DATAT>::max());
      train_code<DATAT>(max_len.data(), min_len.data(), sample_data, sample_num, dim);
      std::string meta_file = getSQMetaFileName(output_path);
      IOWriter meta_writer(meta_file);
      meta_writer.write((char*)max_len.data(), sizeof(DATAT) * dim);
      meta_writer.write((char*)min_len.data(), sizeof(DATAT) * dim);
      for(int i =0; i< dim; i++) {
          std::cout<<"dim: "<<i<<"max: "<<max_len[i]<<", min: "<<min_len[i]<<std::endl;
      }
  }

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

    elkan_L2_assign<DATAT, float, DISTT>(block_buf, centroids, dim, ep - sp, K1,
                                         cluster_id.data(), dists.data());
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

template <typename DATAT, typename DISTT>
hnswlib::SpaceInterface<DISTT> *getDistanceSpace(MetricType metric_type,
                                                 uint32_t ndim) {
  hnswlib::SpaceInterface<DISTT> *space;
  if (MetricType::L2 == metric_type) {
    space = new hnswlib::L2Space<DATAT, DISTT>(ndim);
  } else if (MetricType::IP == metric_type) {
    space = new hnswlib::InnerProductSpace(ndim);
  } else {
    std::cout << "invalid metric_type = " << (int)metric_type << std::endl;
  }
  return space;
}

template <>
hnswlib::SpaceInterface<float> *
getDistanceSpace<float, float>(MetricType metric_type, uint32_t ndim) {
  hnswlib::SpaceInterface<float> *space;
  if (MetricType::L2 == metric_type) {
    space = new hnswlib::L2Space<float, float>(ndim);
  } else if (MetricType::IP == metric_type) {
    space = new hnswlib::InnerProductSpace(ndim);
  } else {
    std::cout << "invalid metric_type = " << (int)metric_type << std::endl;
  }
  return space;
}

template <>
hnswlib::SpaceInterface<int> *
getDistanceSpace<int8_t, int>(MetricType metric_type, uint32_t ndim) {
  hnswlib::SpaceInterface<int> *space;
  if (MetricType::L2 == metric_type) {
    space = new hnswlib::L2Space<int8_t, int32_t>(ndim);
  } else {
    std::cout << "invalid metric_type = " << (int)metric_type << std::endl;
  }
  return space;
}

template <>
hnswlib::SpaceInterface<uint32_t> *
getDistanceSpace<uint8_t, uint32_t>(MetricType metric_type, uint32_t ndim) {
  hnswlib::SpaceInterface<uint32_t> *space;
  if (MetricType::L2 == metric_type) {
    space = new hnswlib::L2Space<uint8_t, uint32_t>(ndim);
  } else {
    std::cout << "invalid metric_type = " << (int)metric_type << std::endl;
  }
  return space;
}

template <typename DATAT, typename DISTT>
void build_graph(const std::string &index_path, const int hnswM,
                 const int hnswefC, MetricType metric_type,
                 const uint64_t block_size, const int32_t sample) {
  TimeRecorder rc("create_graph_index");
  std::cout << "build hnsw parameters:" << std::endl;
  std::cout << " index_path: " << index_path << " hnsw.M: " << hnswM
            << " hnsw.efConstruction: " << hnswefC
            << " metric_type: " << (int)metric_type << std::endl;

  DATAT *pdata = nullptr;
  uint32_t *pids = nullptr;
  uint32_t nblocks, ndim, nids, nidsdim;

  util::read_bin_file<DATAT>(index_path + "bucket-centroids.bin", pdata,
                             nblocks, ndim);
  rc.RecordSection("load centroids of buckets done");
  std::cout << "there are " << nblocks << " of dimension " << ndim
            << " points of hnsw" << std::endl;
  assert(pdata != nullptr);
  hnswlib::SpaceInterface<DISTT> *space =
      getDistanceSpace<DATAT, DISTT>(metric_type, ndim);
  util::read_bin_file<uint32_t>(index_path + "cluster-combine_ids.bin", pids,
                                nids, nidsdim);
  rc.RecordSection("load combine ids of buckets done");
  std::cout << "there are " << nids << " of dimension " << nidsdim
            << " combine ids of hnsw" << std::endl;
  assert(pids != nullptr);
  assert(nblocks == nids);
  assert(nidsdim == 1);

  const uint32_t vec_size = sizeof(DATAT) * ndim;
  const uint32_t entry_size = vec_size + sizeof(uint32_t);

  auto index_hnsw = std::make_shared<hnswlib::HierarchicalNSW<DISTT>>(
      space, sample * nblocks, hnswM, hnswefC);
  uint32_t cid0, bid0;
  bbann::util::parse_global_block_id(pids[0], cid0, bid0);
  // add centroids
  index_hnsw->addPoint(pdata, bbann::util::gen_id(cid0, bid0, 0));
  if (sample != 1) {
    // add sample data
    std::string cluster_file_path =
        index_path + CLUSTER + std::to_string(cid0) + "-" + RAWDATA + BIN;
    auto fh = std::ifstream(cluster_file_path, std::ios::binary);
    assert(!fh.fail());
    char *buf = new char[block_size];
    fh.seekg(bid0 * block_size);
    fh.read(buf, block_size);
    const uint32_t entry_num = *reinterpret_cast<uint32_t *>(buf);
    int bucketSample = sample - 1;
    if (bucketSample > entry_num) {
      bucketSample = entry_num;
    }
    char *buf_begin = buf + sizeof(uint32_t);
    for (int64_t j = 0; j < bucketSample; j++) {
      // TODO, pick vectors smarter?
      char *entry_begin = buf_begin + entry_size * j;
      index_hnsw->addPoint(reinterpret_cast<DATAT *>(entry_begin),
                           bbann::util::gen_id(cid0, bid0, j + 1));
    }
    delete[] buf;
    fh.close();
  }
#pragma omp parallel for
  for (int64_t i = 1; i < nblocks; i++) {
    uint32_t cid, bid;
    bbann::util::parse_global_block_id(pids[i], cid, bid);
    index_hnsw->addPoint(pdata + i * ndim, bbann::util::gen_id(cid, bid, 0));
    if (sample != 1) {
      // add sample data
      std::string cluster_file_path =
          index_path + CLUSTER + std::to_string(cid) + "-" + RAWDATA + BIN;
      auto fh = std::ifstream(cluster_file_path, std::ios::binary);
      assert(!fh.fail());
      char *buf = new char[block_size];
      fh.seekg(bid * block_size);
      fh.read(buf, block_size);
      const uint32_t entry_num = *reinterpret_cast<uint32_t *>(buf);
      int bucketSample = sample - 1;
      if (bucketSample > entry_num) {
        bucketSample = entry_num;
      }
      char *buf_begin = buf + sizeof(uint32_t);
      for (int64_t j = 0; j < bucketSample; j++) {
        // TODO, pick vectors smarter?
        char *entry_begin = buf_begin + entry_size * j;
        index_hnsw->addPoint(reinterpret_cast<DATAT *>(entry_begin),
                             bbann::util::gen_id(cid, bid, j + 1));
      }
      delete[] buf;
      fh.close();
    }
  }
  std::cout << "hnsw totally add " << nblocks << " points" << std::endl;
  rc.RecordSection("create index hnsw done");
  index_hnsw->saveIndex(index_path + HNSW + INDEX + BIN);
  rc.RecordSection("hnsw save index done");
  delete[] pdata;
  pdata = nullptr;
  delete[] pids;
  pids = nullptr;
  rc.ElapseFromBegin("create index hnsw totally done");
}

template <typename T>
void reservoir_sampling(const std::string &data_file, const size_t sample_num,
                        T *sample_data) {
  assert(sample_data != nullptr);
  std::random_device rd;
  auto x = rd();
  std::mt19937 generator((unsigned)x);
  uint32_t nb, dim;
  size_t ntotal, ndims;
  IOReader reader(data_file);
  reader.read((char *)&nb, sizeof(uint32_t));
  reader.read((char *)&dim, sizeof(uint32_t));
  ntotal = nb;
  ndims = dim;
  std::unique_ptr<T[]> tmp_buf = std::make_unique<T[]>(ndims);
  for (size_t i = 0; i < sample_num; i++) {
    auto pi = sample_data + ndims * i;
    reader.read((char *)pi, ndims * sizeof(T));
  }
  for (size_t i = sample_num; i < ntotal; i++) {
    reader.read((char *)tmp_buf.get(), ndims * sizeof(T));
    std::uniform_int_distribution<size_t> distribution(0, i);
    size_t rand = (size_t)distribution(generator);
    if (rand < sample_num) {
      memcpy((char *)(sample_data + ndims * rand), tmp_buf.get(),
             ndims * sizeof(T));
    }
  }
}

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
  util::get_bin_metadata(getClusterRawDataFileName(para.indexPrefixPath, 0), cluster_size, cluster_dim);

  std::vector<DATAT> max_len(cluster_dim);
  std::vector<DATAT> min_len(cluster_dim);

  if (para.vector_use_sq) {
    std::cout << "vector sq: read max/min from meta file" << std::endl;
    std::string meta_file = getSQMetaFileName(para.indexPrefixPath);
    IOReader meta_reader(meta_file);
    meta_reader.read((char*)max_len.data(), sizeof(DATAT) * cluster_dim);
    meta_reader.read((char*)min_len.data(), sizeof(DATAT) * cluster_dim);
  }

  {
    IOWriter centroids_writer(bucket_centroids_file);
    IOWriter centroids_id_writer(bucket_centroids_id_file);
    centroids_writer.write((char *)&placeholder, sizeof(uint32_t));
    centroids_writer.write((char *)&placeholder, sizeof(uint32_t));
    centroids_id_writer.write((char *)&placeholder, sizeof(uint32_t));
    centroids_id_writer.write((char *)&placeholder, sizeof(uint32_t));

    int64_t global_start_position = centroids_id_writer.get_position();
    assert(global_start_position != -1);
    // std::cout << "global_centroids_number: global_start_position: "
    //           << global_start_position << std::endl;

    for (uint32_t i = 0; i < K1; i++) {
      TimeRecorder rci("train-cluster-" + std::to_string(i));

      int64_t local_start_position = centroids_id_writer.get_position();
      assert(local_start_position != -1);
      // std::cout << "global_centroids_number: local_start_position: "
      //           << local_start_position << std::endl;

      IOReader data_reader(getClusterRawDataFileName(para.indexPrefixPath, i));
      IOReader ids_reader(getClusterGlobalIdsFileName(para.indexPrefixPath, i));

      data_reader.read((char *)&cluster_size, sizeof(uint32_t));
      data_reader.read((char *)&cluster_dim, sizeof(uint32_t));
      ids_reader.read((char *)&ids_size, sizeof(uint32_t));
      ids_reader.read((char *)&ids_dim, sizeof(uint32_t));

      if (para.vector_use_sq) {
        entry_num = (para.blockSize - sizeof(uint32_t)) /
                    (cluster_dim * sizeof(uint8_t) + ids_dim * sizeof(uint32_t));
      } else {
        entry_num = (para.blockSize - sizeof(uint32_t)) /
                    (cluster_dim * sizeof(DATAT) + ids_dim * sizeof(uint32_t));
      }

      centroids_dim = cluster_dim;
      assert(cluster_size == ids_size);
      assert(ids_dim == 1);
      assert(entry_num > 0);

      int64_t data_size = static_cast<int64_t>(cluster_size);
      // std::cout << "train cluster" << std::to_string(i) << "data files" <<
      // data_file << "size" << data_size << std::endl;
      DATAT *datai = new DATAT[data_size * cluster_dim * 1ULL];

      uint32_t *idi = new uint32_t[ids_size * ids_dim];
      uint32_t blk_num = 0;
      data_reader.read((char *)datai, data_size * cluster_dim * sizeof(DATAT));
      ids_reader.read((char *)idi, ids_size * ids_dim * sizeof(uint32_t));

      IOWriter data_writer(getClusterRawDataFileName(para.indexPrefixPath, i),
                           ioreader::MEGABYTE * 100);

      // recursive style
      // recursive_kmeans<DATAT>(i, data_size, datai, idi, cluster_dim,
      //                         entry_num, blk_size, blk_num, data_writer,
      //                         centroids_writer, centroids_id_writer, 0,
      //                         false, avg_len);

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
          // std::cout << "non_recursive_multilevel_kmeans: "
          //           << "balance level, to parallel" << std::endl;
          break;
        }

        // std::cout << "non_recursive_multilevel_kmeans: "
        //           << " cur.offset:" << cur.offset
        //           << " cur.num_elems:" << cur.num_elems
        //           << " cur.level:" << cur.level << std::endl;
        non_recursive_multilevel_kmeans<DATAT>(
            i, // the index of k1 round k-means
            cur.num_elems, // number vectors in this cluster
            datai, // buffer to place all vectors
            idi,   // buffer to place all ids
            cur.offset,     // the offset of to clustering data in this round
            cluster_dim,    // the dimension of vector
            entry_num,      // threshold, determines when to stop recursive clustering
            para.blockSize, // general 4096, determines how many vectors can be placed in a block
            blk_num,        // output variable, number block output in this round clustering
            data_writer,          // file writer 1: to output base vectors
            centroids_writer,     // file writer 2: to output centroid vectors
            centroids_id_writer,  // file writer 3: to output centroid ids
            local_start_position, // the start position of all centroids id
            cur.level,    // n-th round recursive clustering, start with 0
            mutex,        // mutext to protect write out centroids
            output_tasks, // output clustering tasks
            para.vector_use_sq, // whether use scalar quantization on base vector
            max_len, // the max values in each dimension
            min_len, // the min values in each dimension
            false,   // k-means parameter
            avg_len  // k-means parameter
            );

        for (auto &output_task : output_tasks) {
          todo.push_back(output_task);
        }
      }

      auto func = [&]() {
        while (true) {
          // gurantee the access of todo list
          std::unique_lock<std::mutex> lock(todo_mutex);
          if (todo.empty()) {
            // finish
            // std::cout << "kmeans worker finish" << std::endl;
            break;
          }
          auto cur = todo.front();
          todo.pop_front();
          lock.unlock();

          // std::cout << "non_recursive_multilevel_kmeans: "
          //           << " cur.offset:" << cur.offset
          //           << " cur.num_elems:" << cur.num_elems
          //           << " cur.level:" << cur.level << std::endl;
          std::vector<ClusteringTask> output_tasks;
          non_recursive_multilevel_kmeans<DATAT>(
                  i, // the index of k1 round k-means
                  cur.num_elems, // number vectors in this cluster
                  datai, // buffer to place all vectors
                  idi,   // buffer to place all ids
                  cur.offset,     // the offset of to clustering data in this round
                  cluster_dim,    // the dimension of vector
                  entry_num,      // threshold, determines when to stop recursive clustering
                  para.blockSize, // general 4096, determines how many vectors can be placed in a block
                  blk_num,        // output variable, number block output in this round clustering
                  data_writer,          // file writer 1: to output base vectors
                  centroids_writer,     // file writer 2: to output centroid vectors
                  centroids_id_writer,  // file writer 3: to output centroid ids
                  local_start_position, // the start position of all centroids id
                  cur.level,    // n-th round recursive clustering, start with 0
                  mutex,        // mutext to protect write out centroids
                  output_tasks, // output clustering tasks
                  para.vector_use_sq, // whether use scalar quantization on base vector
                  max_len, // the max values in each dimension
                  min_len, // the min values in each dimension
                  false,   // k-means parameter
                  avg_len  // k-means parameter
          );
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
    // std::cout << "global_centroids_number: end_position: " << end_position
    //           << std::endl;

    // centroid id is uint32_t type
    global_centroids_number +=
        (end_position - global_start_position) / sizeof(uint32_t);
    // std::cout
    //     << "calculate global_centroids_number by centroids id file position: "
    //     << "global_centroids_number " << global_centroids_number << std::endl;
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

  // std::cout << "hierarchical_clusters generate " << global_centroids_number
  //           << " centroids" << std::endl;
  return;
}

#define ALGO_LIB_DECL(DATAT)                                                   \
  template void train_cluster<DATAT>(                                          \
      const std::string &raw_data_bin_file, const std::string &output_path,    \
      const int32_t K1, float **centroids, double &avg_len,                    \
      bool vector_use_sq=false);                                               \
  template void reservoir_sampling<DATAT>(const std::string &data_file,        \
                                          const size_t sample_num,             \
                                          DATAT *sample_data);

#define ALGO_LIB_DECL_2(DATAT, DISTT)                                          \
  template void divide_raw_data<DATAT, DISTT>(const BBAnnParameters para,      \
                                              const float *centroids);         \
  template void hierarchical_clusters<DATAT, DISTT>(                           \
      const BBAnnParameters para, const double avg_len);                       \
  template void search_graph<DATAT, DISTT>(                                    \
      std::shared_ptr<hnswlib::HierarchicalNSW<DISTT>> index_hnsw,             \
      const int nq, const int dq, const int nprobe, const int refine_nprobe,   \
      const DATAT *pquery, uint32_t *buckets_label, float *centroids_dist);    \
  template void build_graph<DATAT, DISTT>(                                     \
      const std::string &index_path, const int hnswM, const int hnswefC,       \
      MetricType metric_type, const uint64_t block_size,                       \
      const int32_t sample);

ALGO_LIB_DECL(float);
ALGO_LIB_DECL(uint8_t);
ALGO_LIB_DECL(int8_t);

ALGO_LIB_DECL_2(float, float);
ALGO_LIB_DECL_2(uint8_t, uint32_t);
ALGO_LIB_DECL_2(int8_t, int32_t);

#undef ALGO_LIB_DECL_2
#undef ALGO_LIB_DECL

} // namespace bbann