#pragma once
#include <iostream>
#include <fstream>
#include <sstream>



class IOReader {
 public:
     IOReader() {}
 private:
  // underlying ifstream
    std::ifstream reader;
  // # bytes to cache in one shot read
    uint64_t cache_size = 0;
  // underlying buf for cache
    char* cache_buf = nullptr;
  // offset into cache_buf for cur_pos
    uint64_t cur_off = 0;
  // file size
    uint64_t fsize = 0;

};


class IOWriter {
};
