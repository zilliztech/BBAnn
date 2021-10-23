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

inline static void AIORead(
    io_context_t ctx, std::vector<AIORequest> &read_reqs,
    const std::vector<std::function<void(AIORequest)>> &callbacks,
    int eventsPerBatch = 32, int maxEventsInQueue = 1023) {
  std::vector<struct iocb> ios(read_reqs.size());
  std::vector<struct iocb *> cbs(read_reqs.size(), nullptr);

  if (maxEventsInQueue > read_reqs.size()) {
    maxEventsInQueue = read_reqs.size();
  }
  if (maxEventsInQueue < eventsPerBatch) {
    eventsPerBatch = maxEventsInQueue;
  }
  int num_callback = callbacks.size();

  int finishedEvents = 0;
  int eventToAdd = eventsPerBatch;

  std::vector<struct io_event> resultEvents(maxEventsInQueue);
  int numEventsToAdd = maxEventsInQueue;
  int submittedEvents = 0;
  while (finishedEvents < read_reqs.size()) {
    int upper = read_reqs.size() - submittedEvents;
    if (upper < numEventsToAdd) {
      numEventsToAdd = upper;
    }

    // Need to add numEventsToAdd reqests.
    if (numEventsToAdd > 0) {
      for (int i = submittedEvents; i < submittedEvents + numEventsToAdd; i++) {
        io_prep_pread(ios.data() + i, read_reqs[i].fd, read_reqs[i].buf,
                      read_reqs[i].size, read_reqs[i].offset);
#ifdef CALLBACK_ENABLED
        auto callback = new int[1];
        callback[0] = i;
        ios[i].data = callback;
#endif
        cbs[i] = ios.data() + i;
      }
      auto submitted = io_submit(ctx, numEventsToAdd, cbs.data() + submittedEvents);
      submittedEvents += numEventsToAdd;
      if (submitted < 0) {
        std::cout << "io_submit() failed, returned: " << submitted
                  << ", strerror(-): " << strerror(-submitted) << std::endl;
        return;
        // exit(-1);
      }
    }

    int r = io_getevents(ctx, 1, maxEventsInQueue, resultEvents.data(), NULL);

    if (r < 0) {
      std::cout << "io_getevents() failed, returned: " << r
                << ", strerror(-): " << strerror(-r) << std::endl;
      return;                
      // exit(-1);
    }
#ifdef CALLBACK_ENABLED
    for (auto en = 0; en < r; en++) {
      auto idx = *reinterpret_cast<int *>(resultEvents[en].data);
      if (idx < num_callback && callbacks[idx] != nullptr) {
        callbacks[idx](read_reqs[idx]);
      }
    }
#endif
    finishedEvents += r;
    numEventsToAdd = r;
  }
}
