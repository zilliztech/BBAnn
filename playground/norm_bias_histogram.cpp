// ----------------------------------------------------------------------------------------------------
#include <cassert>
#include <cstdint>
#include <algorithm>
#include <random>
// ----------------------------------------------------------------------------------------------------
#include <iostream>
#include <fstream>
//---------------------------------------------------------------------------
namespace {
//---------------------------------------------------------------------------
constexpr int TOP_K = 10;  // I only care about top10 NN.
static_assert(TOP_K >= 1 && TOP_K <= 100);
//---------------------------------------------------------------------------
// TODO: only for text-to-image with float. :(
void generate_norm_bias_histogram(const std::string& base_input_path, const std::string& query_input_path, const std::string& output_path, const int num_bins) {
    // All datasets are in the common binary format that starts with
    // 8 bytes of data consisting of num_points(uint32_t) num_dimensions(uint32)
    // followed by num_pts X num_dimensions x sizeof(type) bytes of data stored one vector after another.
    std::ifstream base_input(base_input_path, std::ios::binary);
    uint32_t num_points;
    uint32_t num_dimensions;
    assert(base_input.is_open());
    base_input.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));
    base_input.read(reinterpret_cast<char*>(&num_dimensions), sizeof(num_dimensions));
    assert(num_dimensions == 200 && "ONLY FOR TEST TO IMAGE.");
    std::cout << "number of points: " << num_points << std::endl;
    std::cout << "number of dimensions: " << num_dimensions << std::endl;
    std::vector<float> norm_vec(num_points, 0.0);  // to be sorted

    std::cout << "Start to process the vector file." << std::endl;
    float ele;
    for (int i = 0; i < num_points; ++i) {
        float norm_squared = 0.0;
        for (int j = 0; j < num_dimensions; ++j) {
            base_input.read(reinterpret_cast<char*>(&ele), sizeof(ele));
            norm_squared += ele * ele;
        }
        norm_vec[i] = std::sqrt(norm_squared);
    }
    base_input.close();
    const std::vector<float> unsorted_norm_vec = norm_vec;
    std::cout << "End of processing the vector file. Start to do in-memory sort." << std::endl;

    // ONLY IN-MEMORY SORTING!
    std::sort(norm_vec.begin(), norm_vec.end());
    const float min = norm_vec.front();
    const float max = norm_vec.back();
    assert(max <= 1.0);
    std::cout << "End of sorting. Start to build histogram's bins." << std::endl;

    const int num_histogram_sperator = num_bins - 1;
    // [a, b), [b, c), [c, d] => 3 bins
    // histogram_sperators := {b, c} => 2 histogram sperators
    std::vector<float> histogram_sperators;
    for (int i = num_points / num_bins; i < num_points; i += (num_points / num_bins)) histogram_sperators.emplace_back(norm_vec[i]);
    assert(histogram_sperators.size() == num_histogram_sperator);
    std::cout << "End of building histogram's bins. Start to process the queries' Ground Truth file." << std::endl;

    std::ifstream query_input(query_input_path, std::ios::binary);
    uint32_t num_queries;
    uint32_t knn;
    assert(query_input.is_open());
    query_input.read(reinterpret_cast<char*>(&num_queries), sizeof(num_queries));
    query_input.read(reinterpret_cast<char*>(&knn), sizeof(knn));
    std::cout << "number of queries: " << num_queries << std::endl;
    std::cout << "k-NN: " << knn << std::endl;

    // Collect all NN id, top-k.
    std::vector<uint32_t> all_nn_id;
    // TODO: to set
    all_nn_id.reserve(num_queries * TOP_K);
    for (size_t i = 0; i < num_queries; ++i) {
        uint32_t nn_id;  // NN vector ID
        for (size_t j = 0; j < knn; ++j) {
            query_input.read(reinterpret_cast<char*>(&nn_id), sizeof(nn_id));
            if (j < TOP_K) all_nn_id.emplace_back(nn_id);  // skip the rest
        }
    }
    assert(all_nn_id.size() == num_queries * TOP_K);

    // Counter NN's norm in histogram.
    std::vector<uint64_t> range_counter(num_bins, 0);
    // [range_start， range_end)， but for the last [range_start, max]
    for (const auto& nn_id : all_nn_id) {
        const int index = [&]() {
            for (int i = 0; i < histogram_sperators.size(); ++i) {
                if (unsorted_norm_vec[nn_id] < histogram_sperators[i]) return i;
            }
            return num_histogram_sperator; // the last range => larger than the histogram_sperators[-1]
        } ();
        assert(index >= 0 && index <= num_histogram_sperator);
        ++range_counter[index];
    }

    std::vector<double> range_percentage(num_bins, 0.0);
    for (int i = 0; i < range_percentage.size(); ++i) {
        range_percentage[i] = 1.0 * range_counter[i] / all_nn_id.size();
    }
    std::cout << "End of building histogram." << std::endl;

    {
        // COUT output
        uint64_t sum_counter = 0;
        double sum_percentage = 0.0;
        for (int i = 0; i < range_percentage.size(); ++i) {
            sum_counter += range_counter[i];
            sum_percentage += range_percentage[i];
            std::cout << "[" << i << "]'s range ["
                      << ((i == 0) ? (min) : (histogram_sperators[i - 1]))
                      << ", "
                      << ((i == range_percentage.size() - 1) ? (max) : (histogram_sperators[i]))
                      << ((i == range_percentage.size() - 1) ? "]" : ")")
                      << "  :=  "
                      << range_counter[i] << "  " << range_percentage[i] * 100.0 << "%" << std::endl;
        }
        assert(sum_percentage == 1.0 || std::fabs(sum_percentage - 1.0) <= 0.00001);
        assert(sum_counter == num_points);
        std::cout << "The min of Norm: " << norm_vec.front() << std::endl;
        std::cout << "The max of Norm: " << norm_vec.back() << std::endl;
        std::cout << "The median of Norm: " << norm_vec[num_points / 2] << std::endl;
    }

    {
        // CSV output
        std::ofstream output(output_path, std::ios::binary);
        assert(output.is_open());
        output << "norm value range,counter,percentage" << std::endl;  // CSV's header
        for (int i = 0; i < range_percentage.size(); ++i) {
            output << "["
                      << ((i == 0) ? (min) : (histogram_sperators[i - 1]))
                      << " ~ "
                      << ((i == range_percentage.size() - 1) ? (max) : (histogram_sperators[i]))
                      << ((i == range_percentage.size() - 1) ? "]" : ")")
                      << "," << range_counter[i]
                      << "," << range_percentage[i] << std::endl;
        }
        output.close();
    }
}
//---------------------------------------------------------------------------
} // namespace
//---------------------------------------------------------------------------
int main() {
    std::cout << "Please type in the input base vector file base_input_path into the std::cin:" << std::endl;
    std::cout << "Example: \"/home/jigao/Desktop/Yandex.TexttoImage.base.10M.fdata\"" << std::endl;
    std::string base_input_path;
    std::cin >> base_input_path;
    std::cout << "The input base_input_path is: " << base_input_path << std::endl;

    std::cout << "Please type in the input query & ground truth file query_input_path into the std::cin:" << std::endl;
    std::cout << "Example: \"/home/jigao/Desktop/text2image-10M-gt\"" << std::endl;
    std::string query_input_path;
    std::cin >> query_input_path;
    std::cout << "The input query_input_path is: " << query_input_path << std::endl;

    std::cout << "Please type in the output file input_path into the std::cin:" << std::endl;
    std::cout << "Example: \"/home/jigao/Desktop/histogram.csv\"" << std::endl;
    std::string output_path;
    std::cin >> output_path;
    std::cout << "The output_path is: " << output_path << std::endl;

    std::cout << "Please type in the number of bins of the histogram:" << std::endl;
    std::cout << "Example: 20" << std::endl;
    int num_bins;
    std::cin >> num_bins;
    std::cout << "The input number of bins is: " << num_bins << std::endl;

    generate_norm_bias_histogram(base_input_path, query_input_path, output_path, num_bins);

    return 0;
}
//---------------------------------------------------------------------------
