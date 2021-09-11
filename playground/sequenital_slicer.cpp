// ----------------------------------------------------------------------------------------------------
#include <sys/mman.h>
#include <cassert>
#include <cstdint>
#include <algorithm>
// ----------------------------------------------------------------------------------------------------
#include <iostream>
#include <fstream>
//---------------------------------------------------------------------------
namespace {
//---------------------------------------------------------------------------
template <typename T>
void generate_slice(const std::string& input_path, const int slice_size, const std::string& output_path) {
    // All datasets are in the common binary format that starts with
    // 8 bytes of data consisting of num_points(uint32_t) num_dimensions(uint32)
    // followed by num_pts X num_dimensions x sizeof(type) bytes of data stored one vector after another.
    std::ifstream input(input_path, std::ios::binary);
    uint32_t num_points;
    uint32_t num_dimensions;
    assert(input.is_open());
    input.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));
    input.read(reinterpret_cast<char*>(&num_dimensions), sizeof(num_dimensions));
    assert(num_points >= slice_size);
    std::cout << "number of points: " << num_points << std::endl;
    std::cout << "number of dimensions: " << num_dimensions << std::endl;

    std::ofstream output(output_path, std::ios::binary);
    assert(output.is_open());
    output.write(reinterpret_cast<const char*>(&slice_size), sizeof(slice_size));
    output.write(reinterpret_cast<char*>(&num_dimensions), sizeof(num_dimensions));

    T ele;
    for (int i = 0; i < slice_size; ++i) {
        for (int j = 0; j < num_dimensions; ++j) {
            input.read(reinterpret_cast<char*>(&ele), sizeof(ele));
            output.write(reinterpret_cast<char *>(&ele), sizeof(ele));
        }
        std::cout << std::endl;
    }
    input.close();
    output.close();

    {
        // Validation: reading the output file
        std::ifstream output(output_path, std::ios::binary);
        assert(output.is_open());
        output.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));
        assert(num_points == slice_size);
        output.read(reinterpret_cast<char*>(&num_dimensions), sizeof(num_dimensions));
        T ele;
        for (int i = 0; i < slice_size; ++i) {
            for (int j = 0; j < num_dimensions; ++j) {
                output.read(reinterpret_cast<char*>(&ele), sizeof(ele));
                int temp = ele;
                std::cout << temp << " ";
            }
            std::cout << std::endl;
        }
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

    std::cout << "Please type in the size of slice into the std::cin:" << std::endl;
    std::cout << "Example: \"1000\"" << std::endl;
    size_t slice_size;
    std::cin >> slice_size;
    std::cout << "The size of slice is: " << slice_size << std::endl;

    std::cout << "Please type in the data type: uint8, int8 or float32: " << std::endl;
    std::cout << "Example: \"uint8\"" << std::endl;
    std::string data_type;
    std::cin >> data_type;
    if (data_type != "uint8" && data_type != "int8" && data_type != "float32") {
        std::cerr << "Wrong data type: " << data_type << std::endl;
        return 1;
    }
    std::cout << "The data type is: " << data_type << std::endl;

    std::cout << "Please type in the output file input_path into the std::cin:" << std::endl;
    std::cout << "Example: \"/home/jigao/Desktop/slice.u8bin\"" << std::endl;
    std::string output_path;
    std::cin >> output_path;
    std::cout << "The output_path is: " << output_path << std::endl;

    if (data_type == "uint8" || data_type == "int8") {
        generate_slice<uint8_t>(input_path, slice_size, output_path);
    } else if (data_type == "float32") {
        generate_slice<float>(input_path, slice_size, output_path);
    }
    return 0;
}
//---------------------------------------------------------------------------
