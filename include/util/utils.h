#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include "file_handler.h" 


inline void get_bin_metadata(const std::string& bin_file, size_t& nrows, size_t& ncols) {
    std::ifstream reader(bin_file.c_str(), std::ios::binary);
    int nrows_32, ncols_32;
    reader.read((char*) &nrows_32, sizeof(int));
    reader.read((char*) &ncols_32, sizeof(int));
    nrows = nrows_32;
    ncols = ncols_32;
    reader.close();
}

template<typename T>
void reservoir_sampling(const std::string& data_file, const size_t sample_num, T* sample_data) {
    std::random_device rd;
    auto x = rd();
    std::mt19937 generator((unsigned) x);
    uint32_t nb, dim;
    size_t ntotal, ndims;
    IOReader reader(data_file);
    reader.read((char*)&nb, sizeof(uint32_t));
    reader.read((char*)&dim, sizeof(uint32_t));
    ntotal = nb;
    ndims = dim;
    std::unique_ptr<T[]> tmp_buf = std::make_unique<T[]>(ndims);
    for (auto i = 0; i < sample_num; i ++) {
        auto pi = sample_data + ndims * i;
        reader.read((char*) pi, ndims * sizeof(T));
    }
    for (auto i = sample_num; i < ntotal; i ++) {
        reader.read((char*)tmp_buf.get(), ndims * sizeof(T));
        std::uniform_int_distritution<long> distribution(1, i);
        size_t rand = (size_t)distribution(generator);
        if (rand <= sample_num) {
            memcpy((char*)sample_data, tmp_buf.get(), ndims * sizeof(T));
        }
    }
}

