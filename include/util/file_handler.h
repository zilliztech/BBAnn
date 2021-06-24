#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cstring>
#include "util/constants.h"


class IOReader {
 public:
     IOReader(const std::string& file_name, const uint64_t cache_size = GIGABYTE)
         : cache_size_(cache_size), cur_off_(0) {
        reader_.open(file_name, std::ios::binary | std::ios::ate);
        assert(reader_.is_open() == true);
        reader_.seekg(0, reader_.end);
        fsize_ = reader_.tellg();
        reader_.seekg(0, reader_.beg);
        assert(cache_size_ > 0);
        cache_size_ = (std::min)(cache_size_, fsize_);
        cache_buf_ = (char*) malloc(cache_size_);
        assert(cache_buf_ != nullptr);
        reader_.read(cache_buf_, cache_size_);
     }

     ~IOReader() {
         free(cache_buf_);
         reader_.close();
     }

     uint64_t get_file_size() { return fsize_; }

     void read(char* read_buf, const size_t n_bytes) {
         assert(read_buf != nullptr);
         assert(cache_buf_ != nullptr);
         if (cur_off_ + n_bytes <= cache_size_) {
             memcpy(read_buf, cache_buf_ + cur_off_, n_bytes);
             cur_off_ += n_bytes;
         } else {
             auto in_cache = cache_size_ - cur_off_;
             assert(n_bytes - in_cache <= fsize_ - reader_.tellg());
             memcpy(read_buf, cache_buf_ + cur_off_, in_cache);
             cur_off_ = cache_size_;
             reader_.read(read_buf + in_cache, n_bytes - in_cache);
             if (cache_size_ <= fsize_ - reader_.tellg()) {
                 reader_.read(cache_buf_, cache_size_);
                 cur_off_ = 0;
             }
         }
     }

 private:
  // underlying ifstream
    std::ifstream reader_;
  // # bytes to cache in one shot read
    uint64_t cache_size_ = 0;
  // underlying buf for cache
    char* cache_buf_ = nullptr;
  // offset into cache_buf for cur_pos
    uint64_t cur_off_ = 0;
  // file size
    uint64_t fsize_ = 0;
};

class IOWriter {
 public:
    IOWriter(const std::string& file_name, const uint64_t cache_size = GIGABYTE)
    : cache_size_(cache_size), cur_off_(0) {
        writer_.open(file_name, std::ios::binary);
        assert(writer_.is_open() == true);
        assert(cache_size_ > 0);
        cache_buf_ = (char*) malloc(cache_size_);
        assert(cache_buf_ != nullptr);
    }
    ~IOWriter() {
        if (cur_off_ > 0) {
            flush();
        }
        free(cache_buf_);
        writer_.close();
    }

    uint64_t get_file_size() { return fsize_; }

    void write(char* buff, const size_t n_bytes) {
        assert(buff != nullptr);
        if (n_bytes + cur_off_ <= cache_size_) {
            memcpy(cache_buf_ + cur_off_, buff, n_bytes);
            cur_off_ += n_bytes;
        } else {
            writer_.write(cache_buf_, cur_off_);
            fsize_ += cur_off_;
            writer_.write(buff, n_bytes);
            fsize_ += n_bytes;
            memset(cache_buf_, 0, cache_size_);
            cur_off_ = 0;
        }
    }

    void flush() {
        if (cur_off_ > 0) {
            writer_.write(cache_buf_, cur_off_);
            fsize_ += cur_off_;
        }
    }

 private:
  // underlying ofstream
  std::ofstream writer_;
  // # bytes to cache for one shot write
  uint64_t cache_size_ = 0;
  // underlying buf for cache
  char* cache_buf_ = nullptr;
  // offset into cache_buf for cur_pos
  uint64_t cur_off_ = 0;

  // file size
  uint64_t fsize_ = 0;
};
