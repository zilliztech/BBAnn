// ----------------------------------------------------------------------------------------------------
#include <sys/mman.h>
#include <cassert>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
//---------------------------------------------------------------------------
namespace {
//---------------------------------------------------------------------------
void read_ground_truth(const std::string& input_path, const int slice_size, const std::string& output_path) {
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

    std::ofstream output(output_path, std::ios::binary);
    assert(output.is_open());
    std::cout << "number of slices: " << slice_size << std::endl;
    output.write(reinterpret_cast<const char*>(&slice_size), sizeof(slice_size));
    output.write(reinterpret_cast<char*>(&knn), sizeof(knn));


    for (size_t i = 0; i < num_queries; ++i) {
        uint32_t nn_id;
        if (i < slice_size) {
            for (size_t j = 0; j < knn; ++j) {
                input.read(reinterpret_cast<char*>(&nn_id), sizeof(nn_id));
                output.write(reinterpret_cast<char *>(&nn_id), sizeof(nn_id));
            }
        } else {
            for (size_t j = 0; j < knn; ++j) {
                input.read(reinterpret_cast<char*>(&nn_id), sizeof(nn_id));
            }
        }
    }

    for (size_t i = 0; i < num_queries; ++i) {
        float distance;
        if (i < slice_size) {
            for (size_t j = 0; j < knn; ++j) {
                input.read(reinterpret_cast<char*>(&distance), sizeof(distance));
                output.write(reinterpret_cast<char *>(&distance), sizeof(distance));
            }
        } else {
            for (size_t j = 0; j < knn; ++j) {
                input.read(reinterpret_cast<char*>(&distance), sizeof(distance));
            }
        }
    }

    input.close();
    output.close();
}
//---------------------------------------------------------------------------
} // namespace
//---------------------------------------------------------------------------
int main() {
    std::cout << "NO Range Search Dataset as Input" << std::endl;
    std::cout << "Please type in the input file input_path into the std::cin:" << std::endl;
    std::cout << "Example: \"/home/jigao/Desktop/GT_10M_v2/GT_10M/bigann-10M\"" << std::endl;
    std::string input_path;
    std::cin >> input_path;
    std::cout << "The input input_path is: " << input_path << std::endl;

    std::cout << "Please type in the size of slice into the std::cin:" << std::endl;
    std::cout << "Example: \"1000\"" << std::endl;
    size_t slice_size;
    std::cin >> slice_size;
    std::cout << "The size of slice is: " << slice_size << std::endl;

    std::cout << "Please type in the output file input_path into the std::cin:" << std::endl;
    std::cout << "Example: \"/home/jigao/Desktop/slice.u8bin\"" << std::endl;
    std::string output_path;
    std::cin >> output_path;
    std::cout << "The output_path is: " << output_path << std::endl;

    read_ground_truth(input_path, slice_size, output_path);
    return 0;
}
//---------------------------------------------------------------------------
