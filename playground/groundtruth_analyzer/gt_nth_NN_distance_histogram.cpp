// ----------------------------------------------------------------------------------------------------
#include <sys/mman.h>
#include <cassert>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
//---------------------------------------------------------------------------
namespace {
//---------------------------------------------------------------------------
void build_histogram_ground_truth(const std::string& input_path, const std::string& output_path, int n_th, int num_bins, float min, float max) {
    // The ground truth binary files for k-NN search consist of the following information:
    //   num_queries(uint32_t) K-NN(uint32) followed by
    //   num_queries X K x sizeof(uint32_t) bytes of data representing the IDs of the K-nearest neighbors of the queries,
    //   followed by num_queries X K x sizeof(float) bytes of data representing the distances to the corresponding points.
    std::ifstream input(input_path, std::ios::binary);
    uint32_t num_queries;
    uint32_t knn;
    assert(input.is_open());
    input.read(reinterpret_cast<char*>(&num_queries), sizeof(num_queries));
    input.read(reinterpret_cast<char*>(&knn), sizeof(knn));
    std::cout << "number of queries: " << num_queries << std::endl;
    std::cout << "k-NN: " << knn << std::endl;
    assert(knn >= n_th);

    // Read all NN's vector id
    for (size_t i = 0; i < num_queries; ++i) {
        uint32_t nn_id;
        for (size_t j = 0; j < knn; ++j) {
            input.read(reinterpret_cast<char*>(&nn_id), sizeof(nn_id));
        }
    }

    std::vector<float> distance_vec(num_queries, 0.0);
    for (size_t i = 0; i < num_queries; ++i) {
        float distance;
        for (size_t j = 0; j < knn; ++j) {
            input.read(reinterpret_cast<char*>(&distance), sizeof(distance));
            if (j == n_th) {
                distance_vec[i] = distance;  /// This distance is L2 SQR
            }
        }
    }
    input.close();

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
        for (int i = 0; i < range_percentage.size(); ++i) {
            sum_counter += range_counter[i];
            sum_percentage += range_percentage[i];
            std::cout << "[" << i << "]'s range ["
                      << min + range_width * i
                      << ", "
                      << ((i == range_percentage.size() - 1) ? (max) : (min + range_width * (i + 1)))
                      << ((i == range_percentage.size() - 1) ? "]" : ")")
                      << "  :=  "
                      << range_counter[i] << "  " << range_percentage[i] * 100.0 << "%" << std::endl;
        }
        assert(sum_percentage == 1.0 || std::fabs(sum_percentage - 1.0) <= 0.00001);
        assert(sum_counter == num_queries);
        std::cout << "The min of N-th Distance SQR: " << distance_vec.front() << std::endl;
        std::cout << "The max of N-th Distance SQR: " << distance_vec.back() << std::endl;
        std::cout << "The median of N-th Distance SQR: " << distance_vec[num_queries / 2] << std::endl;
    }

    {
        // CSV output
        std::ofstream output(output_path, std::ios::binary);
        assert(output.is_open());
        output << "value range,counter,percentage,nth,HistogramType" << std::endl;  // CSV's header
        for (int i = 0; i < range_percentage.size(); ++i) {
            output << "["
                   << min + range_width * i
                   << " ~ "
                   << ((i == range_percentage.size() - 1) ? (max) : (min + range_width * (i + 1)))
                   << ((i == range_percentage.size() - 1) ? "]" : ")")
                   << "," << range_counter[i]
                   << "," << range_percentage[i]
                   << "," << n_th
                   << "," << "GT"
                   << std::endl;
        }
        output.close();
    }
}
//---------------------------------------------------------------------------
} // namespace
//---------------------------------------------------------------------------
int main() {
    // TODO(jigao): No Range Search Dataset as Input
    std::cout << "Please type in the input ground-truth file input_path into the std::cin:" << std::endl;
    std::cout << "Example: \"/home/jigao/Desktop/GT_10M_v2/GT_10M/bigann-10M\"" << std::endl;
    std::string input_path;
    std::cin >> input_path;
    std::cout << "The input input_path is: " << input_path << std::endl;

    std::cout << "Please type in the output file input_path into the std::cin:" << std::endl;
    std::cout << "Example: \"/home/jigao/Desktop/histogram.csv\"" << std::endl;
    std::string output_path;
    std::cin >> output_path;
    std::cout << "The output_path is: " << output_path << std::endl;

    std::cout << "Please type in the index of NN into the std::cin:" << std::endl;
    std::cout << "Example: 10 for getting distance SQR between query and its 10-th NN neighbor" << std::endl;
    std::cout << "0-th NN as this input is actually the 0-th NN in the ground-truth file." << std::endl;
    std::cout << "This input must >= 0." << std::endl;
    int n_th;
    std::cin >> n_th;
    std::cout << "The input index is: " << n_th << std::endl;

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

    build_histogram_ground_truth(input_path, output_path, n_th, num_bins, min, max);
    return 0;
}
//---------------------------------------------------------------------------
