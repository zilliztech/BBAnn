// ----------------------------------------------------------------------------------------------------
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/mman.h>
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

    for (size_t i = 0; i < num_queries; ++i) {
        std::cout << i << "-th query: ";
        uint32_t ele;
        for (size_t j = 0; j < knn; ++j) {
            input.read(reinterpret_cast<char*>(&ele), sizeof(ele));
            std::cout << ele << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;

    for (size_t i = 0; i < num_queries; ++i) {
        std::cout << i << "-th query: ";
        float ele;
        for (size_t j = 0; j < knn; ++j) {
            input.read(reinterpret_cast<char*>(&ele), sizeof(ele));
            std::cout << ele << " ";
        }
        std::cout << std::endl;
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

    read_ground_truth(input_path);

    return 0;
}
//---------------------------------------------------------------------------
