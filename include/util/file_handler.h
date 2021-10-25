#pragma once
#include "aio_reader.h"
#include "util/utils_inline.h"

#include <cassert>
#include <cstring>
#include <fcntl.h> // open
#include <fstream>
#include <iostream>
#include <libaio.h>
#include <mutex>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <unistd.h> // pread

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

constexpr int MAX_EVENTS_NUM = 1023;
/*
class CtxManager {
public:
  static CtxManager &GetInstance() {
    std::call_once(flag_, &CtxManager::init);
    return *ins_;
  }

  io_context_t get_ctx() { return ctx_; }

private:
  CtxManager() = default;
  ~CtxManager() { io_destroy(ctx_); }
  CtxManager(const CtxManager &) = delete;
  CtxManager &operator=(const CtxManager &) = delete;

  static void init() {
    ins_ = new CtxManager();
    ins_->ctx_ = 0;
    auto r = io_setup(MAX_EVENTS_NUM, &ins_->ctx_);
    if (r) {
      std::cout << "io_setup() failed!"
                << " r: " << r << std::endl;
      exit(-1);
    }
  }

  io_context_t ctx_ = 0;
  static std::once_flag flag_;
  static CtxManager *ins_;
};
*/
// std::once_flag CtxManager::flag_;
// CtxManager *CtxManager::ins_ = nullptr;

template <class Tp> struct NAlloc {
  typedef Tp value_type;
  NAlloc() = default;
  template <class T> NAlloc(const NAlloc<T> &) {}

  Tp *allocate(std::size_t n) {
    n *= sizeof(Tp);
    Tp *p;
    if (posix_memalign(reinterpret_cast<void **>(&p), 512, sizeof(Tp) * n) !=
        0) {
      throw std::bad_alloc();
    }

    std::cout << "allocating " << n << " bytes @ " << p << '\n';
    return p;
  }

  void deallocate(Tp *p, std::size_t n) {
    std::cout << "deallocating " << n * sizeof *p << " bytes @ " << p << "\n\n";
    ::operator delete(p);
  }
};

class AIOBucketReader {
public:
  // returns a vector of bucketSize * q bytes, and a vector of res_id
  // where q is the actual unique blocks fetched from file.
  // the block at bucketSize*resid[i] is the result of the bucketIds[i];
  AIOBucketReader(std::string prefix, int eventsPerBatch)
      : eventsPerBatch_(eventsPerBatch), prefix_(prefix) {}

  std::vector<uint32_t> ReadToBuf(const std::vector<uint32_t> &bucketIds,
                                  int blockSize, void *ans) {

    int n = bucketIds.size();
    std::vector<AIORequest> req;
    std::vector<uint32_t> resId(n);
    // FIXED BUG: we can not clear cid_to_fd here!!!!!!!!!! cid_to_fd can only
    // be accessed from the critical section.
    for (int i = 0; i < n; i++) {
      uint32_t cid, bid;
      util::parse_global_block_id(bucketIds[i], cid, bid);
      if (i)
        if (bucketIds[i] == bucketIds[i - 1]) {
          resId[i] = resId[i - 1];
          continue;
        }
      resId[i] = req.size();
      AIORequest r;
      r.fd = cid;
      r.buf = reinterpret_cast<char *>(ans) + req.size() * blockSize;
      r.offset = bid * blockSize;
      r.size = blockSize;
      req.emplace_back(r);
    }

    {
      // Critical section: only 1 thread is supposed to enter this.
      const std::lock_guard<std::mutex> lock(mutex_);
      for (auto &r : req) {
        r.fd = getFd(r.fd);
      }
      io_context_t ctx = 0;
      auto max_events_num = 512; //!!!!!!!!!TODO(AIO_PARAMETERS!)
      io_setup(max_events_num, &ctx);
      AIORead(ctx, req, {}, 32, max_events_num);
      io_destroy(ctx);

      for (const auto &[cid, fd] : cid_to_fd) {
        close(fd);
      }
      cid_to_fd.clear();
    }
    return resId;
  }

  int getFd(uint32_t cid) {
    if (!cid_to_fd.count(cid)) {
      cid_to_fd[cid] = open(getClusterRawDataFileName(prefix_, cid).c_str(),
                            O_DIRECT | O_RDONLY);
    }
    return cid_to_fd[cid];
  }

  std::unordered_map<int, int> cid_to_fd;
  std::mutex mutex_;
  std::string prefix_;
  int eventsPerBatch_;
};
class CachedBucketReader {
public:
  CachedBucketReader(std::string prefix)
      : prefix_(prefix), last_cid_(-1), last_bid_(-1), unique_reads_(0) {}
  void readToBuf(int bucketid, char *buf, int blockSize) {
    uint32_t cid, bid;
    bbann::util::parse_global_block_id(bucketid, cid, bid);
    if (last_cid_ != cid) {
      fh_ = std::ifstream(bbann::getClusterRawDataFileName(prefix_, cid),
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
