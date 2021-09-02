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
// TODO: only for UINT8. :(
void generate_sample(const std::string& input_path, const int sample_size, const std::string& output_path) {
    // All datasets are in the common binary format that starts with
    // 8 bytes of data consisting of num_points(uint32_t) num_dimensions(uint32)
    // followed by num_pts X num_dimensions x sizeof(type) bytes of data stored one vector after another.
    std::ifstream input(input_path, std::ios::binary);
    uint32_t num_points;
    uint32_t num_dimensions;
    assert(input.is_open());
    input.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));
    input.read(reinterpret_cast<char*>(&num_dimensions), sizeof(num_dimensions));
    assert(num_points >= sample_size);
    std::cout << "number of points: " << num_points << std::endl;
    std::cout << "number of dimensions: " << num_dimensions << std::endl;

    std::ofstream output(output_path, std::ios::binary);
    assert(output.is_open());
    uint8_t ele;

    std::vector<size_t> random_indexes(sample_size, 0);
    std::mt19937 generator(1);
    for (int i = 0; i < sample_size; ++i) {
        random_indexes[i] = generator() % num_points;
    }
    std::sort(random_indexes.begin(), random_indexes.end());

    for (int i = 0; i < sample_size; ++i) {
        input.seekg(random_indexes[i] * sizeof(ele) * num_dimensions);
        for (int j = 0; j < num_dimensions; ++j) {
            input.read(reinterpret_cast<char*>(&ele), sizeof(ele));
            output.write(reinterpret_cast<char *>(&ele), sizeof(ele));
        }
        std::cout << std::endl;
    }
    output.close();


    {
        // Validation: reading the ouput file
//        std::ifstream output(output_path, std::ios::binary);
//        assert(output.is_open());
//        uint8_t ele;
//        for (int i = 0; i < sample_size; ++i) {
//            for (int j = 0; j < num_dimensions; ++j) {
//                output.read(reinterpret_cast<char*>(&ele), sizeof(ele));
//                int temp = ele;
//                std::cout << temp << " ";
//            }
//            std::cout << std::endl;
//        }
    }
}
//---------------------------------------------------------------------------
} // namespace
//---------------------------------------------------------------------------
int main() {
    std::cout << "Please type in the input file input_path into the std::cin:" << std::endl;
    std::cout << "Example: \"/home/jigao/Desktop/learn.100M.u8bin\"" << std::endl;
    std::string input_path;
    std::cin >> input_path;
    std::cout << "The input input_path is: " << input_path << std::endl;

    std::cout << "Please type in the size of samples into the std::cin:" << std::endl;
    std::cout << "Example: \"1000\"" << std::endl;
    size_t sample_size = 1000;
    std::cin >> sample_size;
    std::cout << "The size of samples is: " << sample_size << std::endl;


    std::cout << "Please type in the output file input_path into the std::cin:" << std::endl;
    std::cout << "Example: \"/home/jigao/Desktop/sample.u8bin\"" << std::endl;
    std::string output_path;
    std::cin >> output_path;
    std::cout << "The output_path is: " << output_path << std::endl;

    generate_sample(input_path, sample_size, output_path);

    return 0;
}
//---------------------------------------------------------------------------
