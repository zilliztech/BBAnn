#pragma once
#include "util/utils_inline.h"
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

namespace ioreader {
constexpr static uint64_t KILOBYTE = 1024;
constexpr static uint64_t MEGABYTE = 1024 * 1024;
constexpr static uint64_t GIGABYTE = 1024 * 1024 * 1024;
} // namespace ioreader

class IOReader {

public:
  IOReader(const std::string &file_name,
           const uint64_t cache_size = ioreader::GIGABYTE)
      : cache_size_(cache_size), cur_off_(0) {
    reader_.open(file_name, std::ios::binary | std::ios::ate);
    assert(reader_.is_open() == true);
    reader_.seekg(0, reader_.end);
    fsize_ = reader_.tellg();
    reader_.seekg(0, reader_.beg);
    assert(cache_size_ > 0);
    cache_size_ = (std::min)(cache_size_, fsize_);
    cache_buf_ = (char *)malloc(cache_size_);
    assert(cache_buf_ != nullptr);
    reader_.read(cache_buf_, cache_size_);
  }

  ~IOReader() {
    free(cache_buf_);
    reader_.close();
  }

  uint64_t get_file_size() { return fsize_; }

  void read(char *read_buf, const uint64_t n_bytes) {
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
  char *cache_buf_ = nullptr;
  // offset into cache_buf for cur_pos
  uint64_t cur_off_ = 0;
  // file size
  uint64_t fsize_ = 0;
};

class IOWriter {
public:
  IOWriter(const std::string &file_name,
           const uint64_t cache_size = ioreader::GIGABYTE)
      : cache_size_(cache_size), cur_off_(0) {
    writer_.open(file_name, std::ios::binary);
    std::cout << "writing file" << file_name << std::endl;
    assert(writer_.is_open() == true);
    assert(cache_size_ > 0);
    cache_buf_ = (char *)malloc(cache_size_);
    assert(cache_buf_ != nullptr);
  }
  ~IOWriter() {
    if (cur_off_ > 0) {
      flush();
    }
    free(cache_buf_);
    writer_.close();
  }

  // returns current position in the output stream
  int64_t get_position() { return cur_pos_; }

  uint64_t get_file_size() { return fsize_; }

  void write(char *buff, const uint64_t n_bytes) {
    assert(buff != nullptr);
    cur_pos_ += n_bytes;

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
  char *cache_buf_ = nullptr;
  // offset into cache_buf for cur_pos
  uint64_t cur_off_ = 0;
  uint64_t cur_pos_ = 0;

    // file size
  uint64_t fsize_ = 0;
};

namespace bbann {
inline std::string getClusterRawDataFileName(std::string prefix,
                                             int cluster_id) {
  return prefix + "cluster-" + std::to_string(cluster_id) + "-raw_data.bin";
}
inline std::string getClusterGlobalIdsFileName(std::string prefix,
                                               int cluster_id) {
  return prefix + "cluster-" + std::to_string(cluster_id) + "-global_ids.bin";
}

class CachedBucketReader {
public:
  CachedBucketReader(std::string prefix)
      : prefix_(prefix), last_cid_(-1), last_bid_(-1), unique_reads_(0) {}
  void readToBuf(int bucketid, char *buf, int blockSize) {
    uint32_t cid, bid;
    util::parse_global_block_id(bucketid, cid, bid);
    if (last_cid_ != cid) {
      fh_ = std::ifstream(getClusterRawDataFileName(prefix_, cid),
                          std::ios::binary);
      fh_.seekg(bid * blockSize);
      fh_.read(buf, blockSize);
      last_cid_ = cid;
      last_bid_ = bid;
      unique_reads_++;
      return;
    }
    if (last_bid_ != bid) {
      last_bid_ = bid;
      fh_.seekg(bid * blockSize);
      fh_.read(buf, blockSize);
      unique_reads_++;
    }
  }

  int last_cid_, last_bid_;
  int unique_reads_;
  std::ifstream fh_;
  std::string prefix_;
};

} // namespace bbann