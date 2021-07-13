#pragma once
#define _LARGEFILE64_SOURCE
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include "util/constants.h"


class IOReader {
 public:
     IOReader(const std::string& file_name, const int64_t cache_size = GIGABYTE)
         : cache_size_(cache_size), cur_off_(0) {
        fd_ = open(file_name.c_str(), O_RDONLY);
        assert(fd_ != -1);
        fsize_ = lseek64(fd_, 0, SEEK_END);
        lseek64(fd_, 0, SEEK_SET);
        assert(cache_size_ > 0);
        cache_size_ = (std::min)(cache_size_, fsize_);
        cache_buf_ = (char*) malloc(cache_size_);
        assert(cache_buf_ != nullptr);
        int64_t read_bytes = read(fd_, cache_buf_, cache_size_);
        assert(read_bytes == cache_size_);
     }

     ~IOReader() {
         free(cache_buf_);
         int err = close(fd_);
         assert(err == 0);
     }

     int64_t get_file_size() { return fsize_; }

     void Read(char* read_buf, const int64_t n_bytes) {
         assert(read_buf != nullptr);
         assert(cache_buf_ != nullptr);
         if (cur_off_ + n_bytes <= cache_size_) {
             memcpy(read_buf, cache_buf_ + cur_off_, n_bytes);
             cur_off_ += n_bytes;
         } else {
             auto in_cache = cache_size_ - cur_off_;
             assert(n_bytes - in_cache <= fsize_ - lseek64(fd_, 0, SEEK_CUR));
             memcpy(read_buf, cache_buf_ + cur_off_, in_cache);
             cur_off_ = cache_size_;
             read(fd_, read_buf + in_cache, n_bytes - in_cache);
             if (cache_size_ <= fsize_ - lseek64(fd_, 0, SEEK_CUR)) {
                 read(fd_, cache_buf_, cache_size_);
                 cur_off_ = 0;
             }
         }
     }

 private:
  // file descriptor
    int fd_;
  // # bytes to cache in one shot read
    int64_t cache_size_ = 0;
  // underlying buf for cache
    char* cache_buf_ = nullptr;
  // offset into cache_buf for cur_pos
    int64_t cur_off_ = 0;
  // file size
    int64_t fsize_ = 0;
};

class IOWriter {
 public:
    IOWriter(const std::string& file_name, const int64_t cache_size = GIGABYTE)
    : cache_size_(cache_size), cur_off_(0) {
        fd_ = open(file_name.c_str(), O_WRONLY);
        assert(fd_ != -1);
        assert(cache_size_ > 0);
        cache_buf_ = (char*) malloc(cache_size_);
        assert(cache_buf_ != nullptr);
    }
    ~IOWriter() {
        if (cur_off_ > 0) {
            flush();
        }
        free(cache_buf_);
        int err = close(fd_);
        assert(err == 0);
    }

    int64_t get_file_size() { return fsize_; }

    void Write(char* buff, const int64_t n_bytes) {
        assert(buff != nullptr);
        if (n_bytes + cur_off_ <= cache_size_) {
            memcpy(cache_buf_ + cur_off_, buff, n_bytes);
            cur_off_ += n_bytes;
        } else {
            write(fd_, cache_buf_, cur_off_);
            fsize_ += cur_off_;
            write(fd_, buff, n_bytes);
            fsize_ += n_bytes;
            memset(cache_buf_, 0, cache_size_);
            cur_off_ = 0;
        }
    }

    void flush() {
        if (cur_off_ > 0) {
            write(fd_, cache_buf_, cur_off_);
            fsize_ += cur_off_;
        }
    }

 private:
  // file descriptor
  int fd_;
  // # bytes to cache for one shot write
  int64_t cache_size_ = 0;
  // underlying buf for cache
  char* cache_buf_ = nullptr;
  // offset into cache_buf for cur_pos
  int64_t cur_off_ = 0;

  // file size
  int64_t fsize_ = 0;
};
