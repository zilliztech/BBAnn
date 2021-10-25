
#include "aio_reader.h"
#include "util/TimeRecorder.h"
#include <fcntl.h> // open
#include <fstream> // ofstream
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h> // pread
#include <vector>

const char *TEST_FILE = "/data/tmp/test_aio_reader.log";

inline static void
SyncRead(std::vector<AIORequest> &read_reqs,
         const std::vector<std::function<void(AIORequest)>> &callbacks) {
  TimeRecorder rc("sync read");

  auto total = read_reqs.size();
  auto num_callback = callbacks.size();

  for (auto i = 0; i < total; i++) {
    if (!pread(read_reqs[i].fd, read_reqs[i].buf, read_reqs[i].size,
               read_reqs[i].offset)) {
      exit(-1);
    }
    if (i < num_callback && callbacks[i] != nullptr) {
      callbacks[i](read_reqs[i]);
    }
  }

  rc.RecordSection("sync read done");

  rc.ElapseFromBegin("sync read done");
}

template <typename T>
static inline void write_test_file(const std::string &file = TEST_FILE,
                                   size_t num = 1024, size_t repeat_num = 1) {

  std::ofstream stream(file, std::ofstream::out | std::ofstream::trunc |
                                 std::ofstream::binary);

  if (!stream.good()) {
    std::cout << "cannot open " << file << std::endl;
    exit(-1);
  }

  for (auto i = 0; i < num; i++) {
    for (auto j = 0; j < repeat_num; j++) {
      stream << static_cast<T>(i);
    }
  }
}

int main(int argc, char *argv[]) {
  TimeRecorder rc("test aio reader");

  using T = char;

  size_t num = 128;
  size_t dup = 512;
  auto size = dup;
  size_t test_num = 256 * 1024;

  write_test_file<T>(TEST_FILE, num, dup);

  auto fd = open(TEST_FILE, O_DIRECT | O_RDONLY);

  std::vector<AIORequest> sync_reqs, async_reqs;
  std::vector<std::function<void(AIORequest)>> callbacks;

  auto cb = [&](AIORequest req, int i) {
    auto buf = reinterpret_cast<T *>(req.buf);
    for (auto j = 0; j < dup; j++) {
      auto expected = static_cast<T>(i);
      auto got = *(buf + i);
      if (got != expected) {
        std::cout << "read wrong, i: " << i << ", expected: " << (int)expected
                  << ", got: " << (int)got << std::endl;
        exit(-1);
      }
    }
  };

  for (auto i = 0; i < test_num; i++) {
    callbacks.emplace_back(std::bind(cb, std::placeholders::_1, i % num));
  }
  rc.RecordSection("prepare callback");

  for (auto i = 0; i < test_num; i++) {
    AIORequest req;
    if (posix_memalign(reinterpret_cast<void **>(&req.buf), 512, size) != 0) {
      exit(-1);
    }
    req.fd = fd;
    req.size = size;
    req.offset = sizeof(T) * dup * (i % num);
    async_reqs.emplace_back(req);
  }
  rc.RecordSection("prepare async requests");

  io_context_t ctx = 0;
  auto max_events_num = 1023;
  io_setup(max_events_num, &ctx);
  // AsyncRead(ctx, async_reqs, callbacks, max_events_num);
  AIORead(ctx, async_reqs, callbacks, 64);
  io_destroy(ctx);
  rc.RecordSection("async read done");

  for (auto i = 0; i < test_num; i++) {
    AIORequest req;
    if (posix_memalign(reinterpret_cast<void **>(&req.buf), 512, size) != 0) {
      exit(-1);
    }
    req.fd = fd;
    req.size = size;
    req.offset = sizeof(T) * dup * (i % num);
    sync_reqs.emplace_back(req);
  }
  rc.RecordSection("prepare sync requests");

  SyncRead(sync_reqs, callbacks);
  rc.RecordSection("sync read done");

  for (auto i = 0; i < test_num; i++) {
    delete[] sync_reqs[i].buf;
    delete[] async_reqs[i].buf;
  }

  std::remove(TEST_FILE);

  rc.ElapseFromBegin("done");

  return 0;
}
