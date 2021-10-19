#include "bbann.h"

// search disk-based index
// strategy is hnsw + ivf + pq + refine

/*
 * args:
 * 1. data type(string): float or uint8 or int8
 * 2. index path(string): a string end with '/' denotes the directory that where
 * the index related file locates
 * 3. query data file(string): binary file of query data
 * 4. answer file(string): file name to store answer
 * 5. ground_truth file(string): file name where ground_truth stores
 * 6. nprobe(int): number of buckets to query
 * 7. hnsw ef(int): number of buckets to be candidate
 * 8. topk(int): number of answers 4 each query
 * 9. K1(int): number of centroids of the first round kmeans
 * 10. metric type(string): metric type
 * 11. page per block: the number of pages in a block
 */

int main(int argc, char **argv) {
  TimeRecorder rc("main");
  if (argc != 12) {
    std::cout << "Usage: << " << argv[0] << " data_type(float or uint8 or int8)"
              << " index path"
              << " query data file"
              << " answer file"
              << " ground truth file"
              << " nprobe"
              << " hnsw ef"
              << " topk"
              << " K1"
              << " metric type"
              << " page per block" << std::endl;
    return 1;
  }
  // parse parameters
  std::string index_path(argv[2]);
  std::string query_file(argv[3]);
  std::string answer_file(argv[4]);
  std::string ground_truth_file(argv[5]);
  int nprobe = std::stoi(argv[6]);
  int hnsw_ef = std::stoi(argv[7]);
  int topk = std::stoi(argv[8]);
  int K1 = std::stoi(argv[9]);
  auto metric_type = get_metric_type_by_name(std::string(argv[10]));
  const uint64_t block_size = std::stoul(argv[11]) * PAGESIZE;

  assert(metric_type != MetricType::None);

  if ('/' != *index_path.rbegin())
    index_path += '/';

  std::string hnsw_index_file = index_path + HNSW + INDEX + BIN;
  std::string bucket_centroids_file = index_path + BUCKET + CENTROIDS + BIN;

  uint32_t bucket_num, dim;
  get_bin_metadata(bucket_centroids_file, bucket_num, dim);

  if (argv[1] == std::string("float")) {
    Computer<float, float, float> dis_computer; // refine computer
    hnswlib::SpaceInterface<float> *space = nullptr;

    if (MetricType::L2 == metric_type) {
        space = new hnswlib::L2Space<float, float>(dim);
    } else if (MetricType::IP == metric_type) {
        space = new hnswlib::InnerProductSpace(dim);
    }
    // load hnsw
    auto index_hnsw =std::make_shared<hnswlib::HierarchicalNSW<float>>(space, hnsw_index_file);
    if (MetricType::L2 == metric_type) {
      dis_computer = L2sqr<const float, const float, float>;
      search_bbann<float, float, CMax<float, uint32_t>>(
          index_path, query_file, answer_file, nprobe, hnsw_ef, topk,
          index_hnsw, K1, block_size, dis_computer);
    } else if (MetricType::IP == metric_type) {
      dis_computer = IP<const float, const float, float>;
      search_bbann<float, float, CMin<float, uint32_t>>(
          index_path, query_file, answer_file, nprobe, hnsw_ef, topk,
          index_hnsw, K1, block_size, dis_computer);
    }
    // calculate_recall<float>(ground_truth_file, answer_file, topk);
    recall<float, uint32_t>(ground_truth_file, answer_file, metric_type, true,
                            false);
  } else if (argv[1] == std::string("uint8")) {
    Computer<uint8_t, uint8_t, uint32_t> dis_computer; // refine computer
      hnswlib::SpaceInterface<uint32_t> *space = nullptr;
      if (MetricType::L2 == metric_type) {
          space = new hnswlib::L2Space<uint8_t, uint32_t>(dim);
      } else if (MetricType::IP == metric_type) {
          std::cout << "Not support metric IP with int8" << std::endl;
          return - 1;
      }
      // load
      std::shared_ptr<hnswlib::HierarchicalNSW<uint32_t>> index_hnsw = std::make_shared<hnswlib::HierarchicalNSW<uint32_t>>(space, hnsw_index_file);
      rc.RecordSection("load hnsw done.");

      dis_computer = L2sqr<const uint8_t, const uint8_t, uint32_t>;
      search_bbann<uint8_t, uint32_t, CMax<uint32_t, uint32_t>>(
          index_path, query_file, answer_file, nprobe, hnsw_ef, topk,
          index_hnsw, K1, block_size, dis_computer);

    // calculate_recall<uint32_t>(ground_truth_file, answer_file, topk);
    recall<uint32_t, uint32_t>(ground_truth_file, answer_file, metric_type,
                               true, false);
  } else if (argv[1] == std::string("int8")) {
      hnswlib::SpaceInterface<int> *space = nullptr;
      if (MetricType::L2 == metric_type) {
          space = new hnswlib::L2Space<int8_t, int32_t>(dim);
      } else if (MetricType::IP == metric_type) {
          std::cout << "Not support metric IP with int8" << std::endl;
          return - 1;
      }
      // load
      std::shared_ptr<hnswlib::HierarchicalNSW<int32_t>> index_hnsw = std::make_shared<hnswlib::HierarchicalNSW<int32_t>>(space, hnsw_index_file);
      rc.RecordSection("load hnsw done.");
      Computer<int8_t, int8_t, int32_t> dis_computer; // refine computer

      dis_computer = L2sqr<const int8_t, const int8_t, int32_t>;
      search_bbann<int8_t, int32_t, CMax<int32_t, uint32_t>>(
          index_path, query_file, answer_file, nprobe, hnsw_ef, topk,
          index_hnsw, K1, block_size, dis_computer);

    // calculate_recall<uint32_t>(ground_truth_file, answer_file, topk);
    recall<int32_t, uint32_t>(ground_truth_file, answer_file, metric_type, true,
                              false);
  }

  rc.ElapseFromBegin(" totally done.");
  return 0;
}
