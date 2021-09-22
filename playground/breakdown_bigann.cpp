// ---------------------------------------------------------------------------
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <fcntl.h>
#include <liburing.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include "omp.h"
// ---------------------------------------------------------------------------
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <algorithm>
#include <vector>
#include <string>
#include <chrono>

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <fcntl.h>
#include <liburing.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
// ---------------------------------------------------------------------------
#include <chrono>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <fstream>
// ---------------------------------------------------------------------------
#include "util/distance.h"
#include "util.h"
#include <benchmark/benchmark.h>
#include "omp.h"
// ---------------------------------------------------------------------------
using namespace std;
using namespace std::chrono;
// ---------------------------------------------------------------------------
namespace {
// ---------------------------------------------------------------------------
auto base_file_path = "/data/base.1B.u8bin"; // TODO: The path of the file
// ---------------------------------------------------------------------------
//  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
// ---------------------------------------------------------------------------
static void InMemory(benchmark::State& st) {
    std::ifstream input(base_file_path, std::ios::binary);
    uint32_t num_points;
    uint32_t num_dimensions;
    assert(input.is_open());
    input.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));
    input.read(reinterpret_cast<char*>(&num_dimensions), sizeof(num_dimensions));
    uint32_t slice_size = st.range(0);
    assert(num_points >= slice_size);
    const auto size = num_dimensions * slice_size * sizeof(uint8_t);
    std::vector<uint8_t> buffer(size);
    uint64_t sum = 0;
    if (input.read(reinterpret_cast<char*>(buffer.data()), size)) {
        for (auto _ : st) {
            for (int i = 0; i < slice_size; ++i){
                const int len = L2sqr<uint8_t, uint8_t, int>(buffer.data(), buffer.data() + num_dimensions * i, num_dimensions);
                sum += len;
            }
        }
    }
//    st.counters["num_points"] = num_points;
//    st.counters["num_dimensions"] = num_dimensions;
    st.counters["slice_size"] = slice_size;
    st.counters["SUM of L2 SQR"] = sum;
}
// ---------------------------------------------------------------------------
static void InMemory_MT(benchmark::State& st) {
    std::ifstream input(base_file_path, std::ios::binary);
    uint32_t num_points;
    uint32_t num_dimensions;
    assert(input.is_open());
    input.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));
    input.read(reinterpret_cast<char*>(&num_dimensions), sizeof(num_dimensions));
    uint32_t slice_size = st.range(0);
    assert(num_points >= slice_size);
    const auto size = num_dimensions * slice_size * sizeof(uint8_t);
    std::vector<uint8_t> buffer(size);
    uint64_t sum = 0;
    if (input.read(reinterpret_cast<char*>(buffer.data()), size)) {
        for (auto _ : st) {
#pragma omp parallel for
            for (int i = 0; i < slice_size; ++i){
                const int len = L2sqr<uint8_t, uint8_t, int>(buffer.data(), buffer.data() + num_dimensions * i, num_dimensions);
                sum += len;
            }
        }
    }
//    st.counters["num_points"] = num_points;
//    st.counters["num_dimensions"] = num_dimensions;
    st.counters["slice_size"] = slice_size;
    st.counters["SUM of L2 SQR"] = sum;
}
// ---------------------------------------------------------------------------
static void SequentialRead(benchmark::State& st) {
    uint32_t slice_size = st.range(0);
    clean_page_cache();
    constexpr size_t BLOCK_SZ = 128 * sizeof(uint8_t);
    for (auto _ : st) {
        int fd = open(base_file_path, O_RDONLY);
        assert(fd != -1);
        void *buf;
        if (posix_memalign(&buf, 4096, BLOCK_SZ)) return;
        for (size_t i = 0; i < slice_size; i++) {
            // Sequential read a file
            int size = pread(fd, buf, BLOCK_SZ, i * 128 * sizeof(float) + 8 /*header*/);
            if (size != BLOCK_SZ) return;
        }
        free(buf);
        close(fd);
    }
    st.counters["slice_size"] = slice_size;
}
// ---------------------------------------------------------------------------
static void SequentialRead_HT(benchmark::State& st) {
    uint32_t slice_size = st.range(0);
    clean_page_cache();
    constexpr size_t BLOCK_SZ = 128 * sizeof(uint8_t);
    for (auto _ : st) {
        int fd = open(base_file_path, O_RDONLY);
        assert(fd != -1);
        void *buf;
        if (posix_memalign(&buf, 4096, BLOCK_SZ)) return;
#pragma omp parallel for
        for (size_t i = 0; i < slice_size; i++) {
            // Sequential read a file
            int size = pread(fd, buf, BLOCK_SZ, i * 128 * sizeof(float) + 8 /*header*/);
//            if (size != BLOCK_SZ) return;
        }
        free(buf);
        close(fd);
    }
    st.counters["slice_size"] = slice_size;
}
// ---------------------------------------------------------------------------
static void iouringRead_DirectIO(benchmark::State& st) {
    // https://git.kernel.dk/cgit/liburing/tree/examples/io_uring-test.c
    float max_D = 0.0f;
    struct io_uring ring;
    int i, fd, ret, pending, done;
    struct io_uring_sqe *sqe;
    struct io_uring_cqe *cqe;
    struct iovec *iovecs;
    struct stat sb;
    ssize_t fsize;
    void *buf;

    clean_page_cache();
    constexpr size_t BLOCK_SZ = 128 * sizeof(uint8_t);
    uint32_t slice_size = st.range(0);
    auto QUEUE_DEPTH = 4096;
    assert((sb.st_size - 8) / 128 /*DIM*/ / sizeof(uint8_t) >= slice_size);
    for (auto _ : st) {
        ret = io_uring_queue_init(QUEUE_DEPTH, &ring, 0);
        if (ret < 0) {
            fprintf(stderr, "queue_init: %s\n", strerror(-ret));
            return;
        }

        fd = open(base_file_path, O_RDONLY | O_DIRECT);
        if (fd < 0) {
            perror("open");
            return;
        }
        if (fstat(fd, &sb) < 0) {
            perror("fstat");
            return;
        }

        fsize = 0;
        iovecs = static_cast<iovec *>(calloc(QUEUE_DEPTH, sizeof(struct iovec)));
        for (i = 0; i < QUEUE_DEPTH; i++) {
            if (posix_memalign(&buf, 4096, BLOCK_SZ)) return;
            iovecs[i].iov_base = buf;
            iovecs[i].iov_len = BLOCK_SZ;
            fsize += BLOCK_SZ;
        }

        for (i = 0; i < slice_size; i++) {
            sqe = io_uring_get_sqe(&ring);
            if (!sqe) break;
            io_uring_prep_readv(sqe, fd, &iovecs[i], 1, i * BLOCK_SZ + 8 /*header*/);
            if (i * BLOCK_SZ + 8 > sb.st_size) break;
        }

        ret = io_uring_submit(&ring);
        if (ret < 0) {
            fprintf(stderr, "io_uring_submit: %s\n", strerror(-ret));
            return;
        } else if (ret != i) {
            fprintf(stderr, "io_uring_submit submitted less %d\n", ret);
            return;
        }

        done = 0;
        pending = ret;
        fsize = 0;
        for (i = 0; i < pending; i++) {
            ret = io_uring_wait_cqe(&ring, &cqe);
            if (ret < 0) {
                fprintf(stderr, "io_uring_wait_cqe: %s\n", strerror(-ret));
                return;
            }

            done++;
            ret = 0;
            if (cqe->res != BLOCK_SZ) {
                fprintf(stderr, "#done=%d, ret=%d, wanted BLOCK_SZ\n", done, cqe->res);
                ret = 1;
            }
            fsize += cqe->res;
            io_uring_cqe_seen(&ring, cqe);
            if (ret) break;
        }

        close(fd);
        io_uring_queue_exit(&ring);
    }
}
// ---------------------------------------------------------------------------
BENCHMARK(InMemory)->Iterations(10)->Unit(benchmark::kMillisecond)->Arg(1'000'000)->Arg(10'000'000);
BENCHMARK(InMemory_MT)->Iterations(10)->Unit(benchmark::kMillisecond)->Arg(1'000'000)->Arg(10'000'000);
BENCHMARK(SequentialRead)->Iterations(10)->Unit(benchmark::kMillisecond)->Arg(1'000'000)->Arg(10'000'000);
BENCHMARK(SequentialRead_HT)->Iterations(10)->Unit(benchmark::kMillisecond)->Arg(1'000'000)->Arg(10'000'000);
//BENCHMARK(iouringRead_DirectIO)->Iterations(10)->Unit(benchmark::kMillisecond)->Arg(1'000'000)->Arg(10'000'000);
// ---------------------------------------------------------------------------
}  // namespace
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
}
// ---------------------------------------------------------------------------
