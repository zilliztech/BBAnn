#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include "file_handler.h" 


inline void get_bin_metadata(const std::string& bin_file, size_t& nrows, size_t& ncols) {
    std::ifstream reader(bin_file.c_str(), std::ios::binary);
    uint32_t nrows_32, ncols_32;
    reader.read((char*) &nrows_32, sizeof(uint32_t));
    reader.read((char*) &ncols_32, sizeof(uint32_t));
    nrows = nrows_32;
    ncols = ncols_32;
    reader.close();
}

inline void set_bin_metadata(const std::string& bin_file, const uint32_t& nrows, const uint32_t& ncols) {
    std::ofstream writer(bin_file.c_str(), std::ios::binary);
    writer.seekp(0);
    writer.write((char*) &nrows, sizeof(uint32_t));
    writer.write((char*) &ncols, sizeof(uint32_t));
    writer.close();
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

template<typename T>
void write_bin_file(const std::string& file_name, T* data, uint32_t n,
                    uint32_t dim) {
    std::ofstream writer(file_name, std::ios::binary);

    writer.write((char*)&n, sizeof(uint32_t));
    writer.write((char*)&dim, sizeof(uint32_t));
    writer.write((char*)data, sizeof(T) * n * dim);

    writer.close();
}

uint64_t gen_id(const uint32_t cid, const uint32_t bid, const uint32_t off) {
    uint64_t ret = 0;
    ret |= (cid & 0xff);
    ret <<= 24;
    ret |= (bid & 0xffffff);
    ret <<= 32;
    ret |= (off & 0xffffffff);
}

void parse_id(uint64_t id, uint32_t& cid, uint32_t& bid, uint32_t& off) {
    off = (id & 0xffffffff);
    id >>= 32;
    bid = (id & 0xffffff);
    id >>= 24;
    cid = (id & 0xff);
}

