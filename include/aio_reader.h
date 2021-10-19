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
          const std::vector<std::function<void(AIORequest)>> &callbacks,
          int eventsNumPerBatch = 512, int io_submit_threads_num = 8,
          int io_wait_threads_num = 1) {

  auto total = read_reqs.size();
  auto num_callback = callbacks.size();

  eventsNumPerBatch = std::min(eventsNumPerBatch, 1023);

  auto eventsBatchNum = (total + eventsNumPerBatch - 1) / eventsNumPerBatch;

  auto submit_func = [&](int begin_th, int end_th) {
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

  auto wait_func = [&](int begin_th, int end_th) {
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

  for (auto n = 0; n < eventsBatchNum; n++) {
    auto begin = n * eventsNumPerBatch;
    auto end = std::min(int((n + 1) * eventsNumPerBatch), int(total));
    auto numEventsThisBatch = end - begin;

    auto num_per_batch = (numEventsThisBatch + io_submit_threads_num - 1) /
                         io_submit_threads_num;

#pragma omp parallel for
    for (auto th = 0; th < io_submit_threads_num; th++) {
      auto begin_th = begin + num_per_batch * th;
      auto end_th = std::min(int(begin_th + num_per_batch), end);
      submit_func(begin_th, end_th);
    };

    auto wait_num_per_batch =
        (numEventsThisBatch + io_wait_threads_num - 1) / io_wait_threads_num;

    for (auto th = 0; th < io_wait_threads_num; th++) {
      auto begin_th = begin + wait_num_per_batch * th;
      auto end_th = std::min(int(begin_th + wait_num_per_batch), end);
      wait_func(begin_th, end_th);
    }
  }
}
