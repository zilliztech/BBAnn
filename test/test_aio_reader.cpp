

龙际全
你们已经成为联系人，可以开始聊天了
#pragma once

#include <functional>
#include <iostream>
#include <libaio.h>
#include <thread>
#include <vector>
struct AIORequest {
  char *buf;
  int fd;
  size_t offset;
  size_t size;
};

inline static void
AsyncRead(io_context_t ctx, std::vector<AIORequest> &read_reqs,
          std::vector<std::function<void(AIORequest)>> callbacks,
          int max_events_num = 512, int io_submit_threads_num = 8,
          int io_wait_threads_num = 8) {

  auto total = read_reqs.size();
  auto num_callback = callbacks.size();

  auto eventsPerBatch =
      (total + max_events_num - 1) / max_events_num; // events per batch.

  auto func = [&](int begin_th, int end_th) {
    auto num_th = end_th - begin_th;

    std::vector<struct iocb> ios(num_th);
    std::vector<struct iocb *> cbs(num_th, nullptr);
    for (auto i = begin_th; i < end_th; i++) {
      io_prep_pread(ios.data() + (i - begin_th), read_reqs[i].fd,
                    read_reqs[i].buf, read_reqs[i].size, read_reqs[i].offset);

      auto callback = new int[1];
      callback[0] = i;

      ios[i - begin_th].data = callback;
    }
    std::cout << "I am here" << std::endl;

    for (auto i = 0; i < num_th; i++) {
      cbs[i] = ios.data() + i;
    }

    auto r = io_submit(ctx, num_th, cbs.data());
    if (r != num_th) {
      std::cout << "io_submit() failed, returned: " << r
                << ", strerror(-r): " << strerror(-r) << std::endl;
      exit(-1);
    }
  };

  for (auto n = 0; n < eventsPerBatch; n++) {
    auto begin = n * max_events_num;
    auto end = std::min(int((n + 1) * max_events_num), int(total));
    auto numEventsThisBatch = end - begin;

    auto num_per_batch = (numEventsThisBatch + io_submit_threads_num - 1) /
                         io_submit_threads_num;
    std::cout << num_per_batch << std::endl;

    std::vector<std::thread> pool, pool2;

    for (auto th = 0; th < io_submit_threads_num; th++) {
      auto begin_th = begin + num_per_batch * th;
      auto end_th = std::min(int(begin_th + num_per_batch), end);
      pool.emplace_back(std::thread(func, begin_th, end_th));
    };
    for (auto &t : pool)
      t.join();
    std::cout << "I am here after pool 1" << std::endl;
    auto wait_num_per_batch =
        (numEventsThisBatch + io_wait_threads_num - 1) / io_wait_threads_num;

    auto func2 = [&](int begin_th, int end_th) {
      auto num_th = end_th - begin_th;

      std::vector<struct io_event> events(num_th);

      auto r = io_getevents(ctx, num_th, num_th, events.data(), NULL);
      if (r != num_th) {
        std::cout << "io_getevents() failed, returned: " << r
                  << ", strerror(-r): " << strerror(-r) << std::endl;
        exit(-1);
      }

      for (auto en = 0; en < num_th; en++) {
        auto idx = *reinterpret_cast<int *>(events[en].data);
        delete[] reinterpret_cast<int *>(events[en].data);
        if (idx < num_callback && callbacks[idx] != nullptr) {
          callbacks[idx](read_reqs[idx]);
        }
      }
    };

    for (auto th = 0; th < io_wait_threads_num; th++) {
      auto begin_th = begin + wait_num_per_batch * th;
      auto end_th = std::min(int(begin_th + wait_num_per_batch), end);
      pool2.emplace_back(std::thread(func, begin_th, end_th));
    }

    for (auto &t : pool2)
      t.join();
  }
}
展开
#pragma once

#include <functional>
#include <iostream>
#include <libaio.h>
#include <thread>
#include <vector>

struct AIORequest
{
    char *buf;
    int fd;
    size_t offset;
    size_t size;
};

inline static void
AsyncRead(io_context_t ctx, std::vector<AIORequest> &read_reqs,
          const std::vector<std::function<void(AIORequest)>>& callbacks,
          int eventsNumPerBatch = 512,
          int io_submit_threads_num = 8,
          int io_wait_threads_num = 1)
{

    auto total = read_reqs.size();
    auto num_callback = callbacks.size();

    eventsNumPerBatch = std::min(eventsNumPerBatch, 1023);

    auto eventsBatchNum =
        (total + eventsNumPerBatch - 1) / eventsNumPerBatch;

    auto submit_func = [&](int begin_th, int end_th)
    {
        auto num_th = end_th - begin_th;

        std::vector<struct iocb> ios(num_th);
        std::vector<struct iocb *> cbs(num_th, nullptr);
        for (auto i = begin_th; i < end_th; i++)
        {
            io_prep_pread(ios.data() + (i - begin_th), read_reqs[i].fd,
                          read_reqs[i].buf, read_reqs[i].size, read_reqs[i].offset);

            auto callback = new int[1];
            callback[0] = i;

            ios[i - begin_th].data = callback;
        }

        for (auto i = 0; i < num_th; i++)
        {
            cbs[i] = ios.data() + i;
        }

        auto r = io_submit(ctx, num_th, cbs.data());
        if (r != num_th)
        {
            std::cout << "io_submit() failed, returned: " << r
                      << ", strerror(-r): " << strerror(-r) << std::endl;
            exit(-1);
        }
    };

    auto wait_func = [&](int begin_th, int end_th)
    {
        auto num_th = end_th - begin_th;

        std::vector<struct io_event> events(num_th);

        auto r = io_getevents(ctx, num_th, num_th, events.data(), NULL);
        if (r != num_th)
        {
            std::cout << "io_getevents() failed, returned: " << r
                      << ", strerror(-r): " << strerror(-r) << std::endl;
            exit(-1);
        }

        for (auto en = 0; en < num_th; en++)
        {
            auto idx = *reinterpret_cast<int *>(events[en].data);
            delete[] reinterpret_cast<int *>(events[en].data);
            if (idx < num_callback && callbacks[idx] != nullptr)
            {
                callbacks[idx](read_reqs[idx]);
            }
        }
    };

    for (auto n = 0; n < eventsBatchNum; n++)
    {
        auto begin = n * eventsNumPerBatch;
        auto end = std::min(int((n + 1) * eventsNumPerBatch), int(total));
        auto numEventsThisBatch = end - begin;

        auto num_per_batch = (numEventsThisBatch + io_submit_threads_num - 1) /
                             io_submit_threads_num;

        // std::vector<std::thread> submit_pool, wait_pool;

#pragma omp parallel for
        for (auto th = 0; th < io_submit_threads_num; th++)
        {
            auto begin_th = begin + num_per_batch * th;
            auto end_th = std::min(int(begin_th + num_per_batch), end);
            submit_func(begin_th, end_th);
            // submit_pool.emplace_back(std::thread(submit_func, begin_th, end_th));
        };

        // for (auto &t : submit_pool)
        //     t.join();

        auto wait_num_per_batch =
            (numEventsThisBatch + io_wait_threads_num - 1) / io_wait_threads_num;

// #pragma omp parallel for
        for (auto th = 0; th < io_wait_threads_num; th++)
        {
            auto begin_th = begin + wait_num_per_batch * th;
            auto end_th = std::min(int(begin_th + wait_num_per_batch), end);
            wait_func(begin_th, end_th);
            // wait_pool.emplace_back(std::thread(wait_func, begin_th, end_th));
        }

        // for (auto &t : wait_pool)
        //     t.join();
    }
}
#include <unistd.h> // pread
#include <fcntl.h>  // open
#include <string>
#include <vector>
#include <fstream>  // ofstream
#include <stdlib.h>
#include <stdio.h>
#include "aiowrapper/aio_reader.h"
#include "util/TimeRecorder.h"

const char* TEST_FILE = "/data/tmp/test_aio_reader.log";

inline static void
SyncRead(std::vector<AIORequest>& read_reqs,
        const std::vector<std::function<void(AIORequest)>>& callbacks
) {
    TimeRecorder rc("sync read");

    auto total = read_reqs.size();
    auto num_callback = callbacks.size();

    for (auto i = 0; i < total; i++) {
        if (!pread(read_reqs[i].fd, read_reqs[i].buf, read_reqs[i].size, read_reqs[i].offset)) {
            exit(-1);
        }
        if (i < num_callback && callbacks[i] != nullptr) {
            callbacks[i](read_reqs[i]);
        }
    }

    rc.RecordSection("sync read done");

    rc.ElapseFromBegin("sync read done");
}

template<typename T>
static inline void
write_test_file(const std::string& file=TEST_FILE,
                size_t num=1024,
                size_t repeat_num=1) {

    std::ofstream stream(file, std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);

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

int
main(int argc, char* argv[]) {
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
        auto buf = reinterpret_cast<T*>(req.buf);
        for (auto j = 0; j < dup; j++) {
            auto expected = static_cast<T>(i);
            auto got = *(buf + i);
            if (got != expected) {
                std::cout << "read wrong, i: " << i
                        << ", expected: " << expected
                        << ", got: " << got
                        << std::endl;
                exit(-1);
            }
        }
    };

    for (auto i = 0; i < test_num; i++) {
        callbacks.emplace_back(std::bind(cb, std::placeholders::_1, i  % num));
    }
    rc.RecordSection("prepare callback");

    for (auto i = 0; i < test_num; i++) {
        AIORequest req;
        if (posix_memalign(reinterpret_cast<void**>(&req.buf), 512, size) != 0) {
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
    AsyncRead(ctx, async_reqs, callbacks, max_events_num);
    io_destroy(ctx);
    rc.RecordSection("async read done");

    for (auto i = 0; i < test_num; i++) {
        AIORequest req;
        if (posix_memalign(reinterpret_cast<void**>(&req.buf), 512, size) != 0) {
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


