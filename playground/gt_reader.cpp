// ----------------------------------------------------------------------------------------------------
#include <sys/mman.h>
#include <cassert>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <iostream>
#include <fstream>
//---------------------------------------------------------------------------
namespace {
//---------------------------------------------------------------------------
template <typename T>
void read_ground_truth(const std::string& input_path) {
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

    uint32_t min_nn_id = std::numeric_limits<uint32_t>::max();
    uint32_t max_nn_id = std::numeric_limits<uint32_t>::min();
    for (size_t i = 0; i < num_queries; ++i) {
        std::cout << i << "-th query: ";
        uint32_t nn_id;
        for (size_t j = 0; j < knn; ++j) {
            input.read(reinterpret_cast<char*>(&nn_id), sizeof(nn_id));
            min_nn_id = std::min(min_nn_id, nn_id);
            max_nn_id = std::max(max_nn_id, nn_id);
            std::cout << nn_id << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;

    T min_distance = std::numeric_limits<T>::max();
    T max_distance = std::numeric_limits<T>::min();
    for (size_t i = 0; i < num_queries; ++i) {
        std::cout << i << "-th query: ";
        T distance;
        for (size_t j = 0; j < knn; ++j) {
            input.read(reinterpret_cast<char*>(&distance), sizeof(distance));
            min_distance = std::min<T>(min_distance, distance);
            max_distance = std::max<T>(max_distance, distance);
            if constexpr (std::is_same_v<T, float>) {
                std::cout << distance << " ";
            } else {
                std::cout << static_cast<int32_t>(distance) << " ";
            }
        }
        std::cout << std::endl;
    }

    std::cout << "min nn id: " << min_nn_id << std::endl;
    std::cout << "max nn id: " << max_nn_id << std::endl;
    if constexpr (std::is_same_v<T, float>) {
        std::cout << "min distance: " << min_distance << std::endl;
        std::cout << "min distance: " << max_distance << std::endl;
    } else {
        std::cout << "min distance: " << static_cast<int32_t>(min_distance) << std::endl;
        std::cout << "min distance: " << static_cast<int32_t>(max_distance) << std::endl;
    }

    input.close();
}
//---------------------------------------------------------------------------
} // namespace
//---------------------------------------------------------------------------
int main() {
    std::cout << "Please type in the input file input_path into the std::cin:" << std::endl;
    std::cout << "Example: \"/home/jigao/Desktop/GT_10M_v2/GT_10M/bigann-10M\"" << std::endl;
    std::string input_path;
    std::cin >> input_path;
    std::cout << "The input input_path is: " << input_path << std::endl;

    std::cout << "Please type in the data type: uint8, int8 or float32: " << std::endl;
    std::cout << "Example: \"uint8\"" << std::endl;
    std::string data_type;
    std::cin >> data_type;
    if (data_type != "uint8" && data_type != "int8" && data_type != "float32") {
        std::cerr << "Wrong data type: " << data_type << std::endl;
        return 1;
    }
    std::cout << "The data type is: " << data_type << std::endl;

    if (data_type == "uint8") {
        read_ground_truth<uint8_t>(input_path);
    } else if (data_type == "int8") {
        read_ground_truth<int8_t>(input_path);
    } else if (data_type == "float32") {
        read_ground_truth<float>(input_path);
    }
    return 0;
}
//---------------------------------------------------------------------------
