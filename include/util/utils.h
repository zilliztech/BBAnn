#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <functional>
#include <memory>
#include "file_handler.h" 
#include "distance.h"
#include "defines.h"


template <typename T1, typename T2, typename R>
using Computer = std::function<R(const T1*, const T2*, int n)>;


template<typename T1, typename T2, typename R>
Computer<T1, T2, R> select_computer(MetricType metric_type) {
    switch (metric_type) {
        case MetricType::L2:
            return L2sqr<const T1, const T2, R>;
            break;
        case MetricType::IP:
            return IP<const T1, const T2, R>;
            break;
    }
}

inline void get_bin_metadata(const std::string& bin_file, uint32_t& nrows, uint32_t& ncols) {
    std::ifstream reader(bin_file.c_str(), std::ios::binary);
    reader.read((char*) &nrows, sizeof(uint32_t));
    reader.read((char*) &ncols, sizeof(uint32_t));
    reader.close();
    std::cout << "get meta from " << bin_file << ", nrows = " << nrows << ", ncols = " << ncols << std::endl;
}

inline void set_bin_metadata(const std::string& bin_file, const uint32_t& nrows, const uint32_t& ncols) {
    std::ofstream writer(bin_file.c_str(), std::ios::binary);
    writer.seekp(0);
    writer.write((char*) &nrows, sizeof(uint32_t));
    writer.write((char*) &ncols, sizeof(uint32_t));
    writer.close();
    std::cout << "set meta to " << bin_file << ", nrows = " << nrows << ", ncols = " << ncols << std::endl;
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
        std::uniform_int_distribution<size_t> distribution(1, i);
        size_t rand = (size_t)distribution(generator);
        if (rand <= sample_num) {
            memcpy((char*)sample_data, tmp_buf.get(), ndims * sizeof(T));
        }
    }
}

template<typename T>
inline void write_bin_file(const std::string& file_name, T* data, uint32_t n,
                    uint32_t dim) {
    assert(data != nullptr);
    std::ofstream writer(file_name, std::ios::binary);

    writer.write((char*)&n, sizeof(uint32_t));
    writer.write((char*)&dim, sizeof(uint32_t));
    writer.write((char*)data, sizeof(T) * n * dim);

    writer.close();
    std::cout << "read binary file from " << file_name << " done in ... seconds, n = "
              << n << ", dim = " << dim << std::endl;
}

template<typename T>
inline void read_bin_file(const std::string& file_name, T* data, uint32_t& n,
                    uint32_t& dim) {
    std::ifstream reader(file_name, std::ios::binary);

    reader.read((char*)&n, sizeof(uint32_t));
    reader.read((char*)&dim, sizeof(uint32_t));
    if (data == nullptr) {
        data = new T[n * dim];
    }
    reader.read((char*)data, sizeof(T) * n * dim);

    reader.close();
    std::cout << "write binary file to " << file_name << " done in ... seconds, n = "
              << n << ", dim = " << dim << std::endl;
}

inline uint64_t gen_id(const uint32_t cid, const uint32_t bid, const uint32_t off) {
    uint64_t ret = 0;
    ret |= (cid & 0xff);
    ret <<= 24;
    ret |= (bid & 0xffffff);
    ret <<= 32;
    ret |= (off & 0xffffffff);
    return ret;
}

inline void parse_id(uint64_t id, uint32_t& cid, uint32_t& bid, uint32_t& off) {
    off = (id & 0xffffffff);
    id >>= 32;
    bid = (id & 0xffffff);
    id >>= 24;
    cid = (id & 0xff);
}

inline uint64_t gen_refine_id(const uint32_t cid, const uint32_t offset, const uint32_t queryid) {
    uint64_t ret = 0;
    ret |= (cid & 0xff);
    ret <<= 32;
    ret |= (offset & 0xffffffff);
    ret <<= 24;
    ret |= (queryid & 0xffffff);
    return ret;
}

inline void parse_refine_id(uint64_t id, uint32_t& cid, uint32_t& offset, uint32_t& queryid) {
    queryid = (id & 0xffffff);
    queryid >>= 24;
    offset = (id & 0xffffffff);
    id >>= 32;
    cid = (id & 0xff);
}

inline MetricType get_metric_type_by_name(const std::string& mt_name) {
    if (mt_name == std::string("L2"))
        return MetricType::L2;
    if (mt_name == std::string("IP"))
        return MetricType::IP;
    return MetricType::None;
}

