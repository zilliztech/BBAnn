// ----------------------------------------------------------------------------------------------------
#include <sys/mman.h>
#include <cassert>
#include <cstdint>
#include <algorithm>
// ----------------------------------------------------------------------------------------------------
#include <iostream>
#include <fstream>
#include <vector>
#include "bbann.h"
//---------------------------------------------------------------------------
namespace {
//---------------------------------------------------------------------------
//    Idea from
//    template <typename DATAT>
//    void search_flat(const std::string &index_path, const DATAT *pquery,
//                     const uint32_t nq, const uint32_t dq, const int nprobe,
//                     uint32_t *labels)
void build_histogram_centroid(const std::string& index_input_path, const std::string& output_path, const int num_bins, float min, float max, int n_th /*top-k*/) {
    const std::string bucket_centroids_file = index_input_path + BUCKET + CENTROIDS + BIN;
    const std::string bucket_centroids_id_file = index_input_path + CLUSTER + COMBINE_IDS + BIN;
    const int nprobe = n_th;

    /// bucket-centroids.bin
    float *cen_data = nullptr;
    uint32_t num_centriods, dim;
    read_bin_file<float>(bucket_centroids_file, cen_data, num_centriods, dim);
    std::cout << "number of centroids in " << bucket_centroids_file << " :" << num_centriods << std::endl;
    std::cout << "DIM: " << dim << std::endl;

    /// cluster-combine_ids.bin
    uint32_t *cen_ids = nullptr;
    uint32_t temp_num_centriods, temp_dim;
    read_bin_file<uint32_t>(bucket_centroids_id_file, cen_ids, temp_num_centriods, temp_dim);
    std::cout << "number of centroids: " << bucket_centroids_id_file << " :" << temp_num_centriods << std::endl;
    std::cout << "DIM: " << temp_dim << std::endl;
    assert(num_centriods == temp_num_centriods);
    assert(dim == temp_dim);

    /// Dummy Query File as Centroid itself.
//    uint32_t num_queries = num_centriods;
    uint32_t num_queries = 300'000;  /// aka number of samples
    float* pquery = new float[num_queries * dim];  /// aka sample data
    reservoir_sampling(bucket_centroids_file, num_queries, pquery);
    uint32_t query_dim = dim;
    std::cout << "number of queries: " << num_queries << std::endl;
    std::cout << "query_dim: " << query_dim << std::endl;
    assert(dim == query_dim);

    uint32_t *bucket_labels = new uint32_t[(int64_t)num_queries * nprobe];  /// vector id
    float *values = new float[(int64_t)num_queries * nprobe];               /// distance

    knn_2<CMax<float, uint32_t>, float, float>(
            pquery, cen_data, num_queries, num_centriods, dim, nprobe, values, bucket_labels,
            L2sqr<const float, const float, float>);

//    // Not necessary
//    for (uint64_t i = 0; i < (uint64_t)num_queries * nprobe; ++i) {
//        bucket_labels[i] = cen_ids[bucket_labels[i]];
//    }

//    for (int i = 0; i < num_queries; ++i) {
//        std::cout << "query " << i << " := ";
//        for (int j = 0; j < nprobe; ++j) {
//            std::cout << nprobe * i + j << ":" << values[nprobe * i + j] << " | ";
//        }
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;

    // Dummy Copy into std::vector && Build a Histogram
    // distance_vecs[0] for all 0-NN a.k.a. top-1
    // distance_vecs[1] for all 1-NN a.k.a. top-2
    // ...
    // distance_vecs[nprobe-1] for all (nprobe-1)-NN a.k.a. top-nprobe
    std::vector<std::vector<float>> distance_vecs(nprobe, std::vector<float>(num_queries, 0.0));

    for (int i = 0; i < num_queries; ++i) {
        for (int j = 0; j < nprobe; ++j) {
            distance_vecs[j][i] = values[nprobe * i + j];
        }
    }

//    std::cout << "========================================" << std::endl;
//    for (int i = 0; i < nprobe; ++i) {
//        std::cout << "top " << i << " := ";
//        for (int j = 0; j < num_queries; ++j) {
//            std::cout << distance_vecs[i][j] << " | ";
//        }
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;

    delete[] values;

    // CSV output
    std::ofstream output(output_path, std::ios::binary);
    assert(output.is_open());
    output << "value range,counter,percentage,nth,HistogramType" << std::endl;  // CSV's header
    int nprobe_it = 0;
    for (auto& distance_vec : distance_vecs) {
        std::cout << nprobe_it << " ROUND STARTING" << std::endl;
        std::sort(distance_vec.begin(), distance_vec.end());
        if (min < 0) min = distance_vec.front();
        if (max < 0) max = distance_vec.back();
        assert(min < max);

        const int num_histogram_sperator = num_bins - 1;
        std::vector<uint64_t> range_counter(num_bins, 0);
        const float range_width = (max - min) / num_bins;
        // [range_start， range_end)， but for the last [range_start, max]
        for (const auto& distance : distance_vec) {
            const int index = (distance == max) ? num_histogram_sperator : (distance - min) / range_width;
            assert(index >= 0 && index <= num_histogram_sperator);
            ++range_counter[index];
        }

        std::vector<double> range_percentage(num_bins, 0.0);
        for (int i = 0; i < range_percentage.size(); ++i) range_percentage[i] = 1.0 * range_counter[i] / num_queries;

        {
            // COUT output
            uint64_t sum_counter = 0;
            double sum_percentage = 0.0;
            for (int i = 0; i < num_bins; ++i) {
                sum_counter += range_counter[i];
                sum_percentage += range_percentage[i];
                std::cout << "[" << i << "]'s range ["
                          << min + range_width * i
                          << ", "
                          << ((i == num_bins - 1) ? (max) : (min + range_width * (i + 1)))
                          << ((i == num_bins - 1) ? "]" : ")")
                          << "  :=  "
                          << range_counter[i] << "  " << range_percentage[i] * 100.0 << "%" << std::endl;
            }
            assert(sum_percentage == 1.0 || std::fabs(sum_percentage - 1.0) <= 0.00001);
            assert(sum_counter == num_queries);
            std::cout << "The min of N-th Distance: " << distance_vec.front() << std::endl;
            std::cout << "The max of N-th Distance: " << distance_vec.back() << std::endl;
            std::cout << "The median of N-th Distance: " << distance_vec[num_queries / 2] << std::endl;
        }

        {
            // CSV output
            assert(output.is_open());
            for (int i = 0; i < num_bins; ++i) {
                output << "["
                       << min + range_width * i
                       << " ~ "
                       << ((i == num_bins - 1) ? (max) : (min + range_width * (i + 1)))
                       << ((i == num_bins - 1) ? "]" : ")")
                       << "," << range_counter[i]
                       << "," << range_percentage[i]
                       << "," << nprobe_it
                       << "," << "CENTROID"
                       << std::endl;
            }
        }
        std::cout << nprobe_it << " ROUND ENDING" << std::endl;
        ++nprobe_it;
    }
    output.close();
}
//---------------------------------------------------------------------------
} // namespace
//---------------------------------------------------------------------------
int main() {
    std::cout << "Please type in the index index_input_path into the std::cin:" << std::endl;
    std::cout << "Example: \"/data/index/BBANN-BIGANN-32-500-128-1/\"" << std::endl;
    std::string index_input_path;
    std::cin >> index_input_path;
    std::cout << "The input index_input_path is: " << index_input_path << std::endl;

    std::cout << "Please type in the output file input_path into the std::cin:" << std::endl;
    std::cout << "Example: \"/home/jigao/Desktop/histogram.csv\"" << std::endl;
    std::string output_path;
    std::cin >> output_path;
    std::cout << "The output_path is: " << output_path << std::endl;

//    std::cout << "Please type in the data type: uint8, int8 or float32: " << std::endl;
//    std::cout << "Example: \"uint8\"" << std::endl;
//    std::string data_type;
//    std::cin >> data_type;
//    if (data_type != "uint8" && data_type != "int8" && data_type != "float32") {
//        std::cerr << "Wrong data type: " << data_type << std::endl;
//        return 1;
//    }
//    std::cout << "The data type is: " << data_type << std::endl;

    std::cout << "Please type in the index of NN of centroid into the std::cin:" << std::endl;
    std::cout << "Example: 10 for getting distance SQR between query and its 9-th NN neighbor" << std::endl;
    std::cout << "1-th NN as this input is actually the 0-th NN in the ground-truth file." << std::endl;
    std::cout << "This input must >= 1." << std::endl;
    int n_th;
    std::cin >> n_th;
    std::cout << "The input index is: " << n_th << std::endl;
    if (n_th <= 0) return 1;

    std::cout << "Please type in the min of the histogram's bin:" << std::endl;
    std::cout << "Example: 0" << std::endl;
    float min;
    std::cin >> min;

    std::cout << "Please type in the max of the histogram's bin:" << std::endl;
    std::cout << "Example: 100000" << std::endl;
    float max;
    std::cin >> max;

    std::cout << "Please type in the number of bins of the histogram:" << std::endl;
    std::cout << "Example: 20" << std::endl;
    int num_bins;
    std::cin >> num_bins;
    std::cout << "The input number of bins is: " << num_bins << std::endl;

    build_histogram_centroid(index_input_path, output_path, num_bins, min, max, n_th);

//    if (data_type == "uint8") {
//        build_histogram_centroid(index_input_path, output_path, num_bins, min, max, n_th);
//    } else if (data_type == "int8") {
//        build_histogram_centroid(index_input_path, output_path, num_bins, min, max, n_th);
//    } else if (data_type == "float32") {
//        build_histogram_centroid(index_input_path, output_path, num_bins, min, max, n_th);
//    }
    return 0;
}
//---------------------------------------------------------------------------
