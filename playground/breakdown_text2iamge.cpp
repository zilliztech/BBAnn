// ---------------------------------------------------------------------------
#include <vector>
#include <chrono>
#include <iostream>
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
auto base_file_path = "/data/diskann-T2-baseline-indices/text2image-1B/base.10M.fbin"; // TODO: The path of the file
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
    const auto size = num_dimensions * slice_size * sizeof(float);
    std::vector<uint8_t> buffer(size);
    double sum = 0;
    if (input.read(reinterpret_cast<char*>(buffer.data()), size)) {
        for (auto _ : st) {
            for (int i = 0; i < slice_size; ++i){
                const double len = IP_PLAIN(reinterpret_cast<float*>(buffer.data()), reinterpret_cast<float*>(buffer.data() + num_dimensions * sizeof(float) * i), num_dimensions);
                sum += len;
            }
        }
    }
//    st.counters["num_points"] = num_points;
//    st.counters["num_dimensions"] = num_dimensions;
    st.counters["slice_size"] = slice_size;
    st.counters["SUM of IP"] = sum;
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
    const auto size = num_dimensions * slice_size * sizeof(float);
    std::vector<uint8_t> buffer(size);
    double sum = 0;
    if (input.read(reinterpret_cast<char*>(buffer.data()), size)) {
        for (auto _ : st) {
#pragma omp parallel for reduction(+: sum)
            for (int i = 0; i < slice_size; ++i){
                const double len = IP_PLAIN(reinterpret_cast<float*>(buffer.data()), reinterpret_cast<float*>(buffer.data() + num_dimensions * sizeof(float) * i), num_dimensions);
                sum += len;
            }
        }
    }
//    st.counters["num_points"] = num_points;
//    st.counters["num_dimensions"] = num_dimensions;
    st.counters["slice_size"] = slice_size;
    st.counters["SUM of IP"] = sum;
}
// ---------------------------------------------------------------------------
static void InMemory_SIMD(benchmark::State& st) {
    std::ifstream input(base_file_path, std::ios::binary);
    uint32_t num_points;
    uint32_t num_dimensions;
    assert(input.is_open());
    input.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));
    input.read(reinterpret_cast<char*>(&num_dimensions), sizeof(num_dimensions));
    uint32_t slice_size = st.range(0);
    assert(num_points >= slice_size);
    const auto size = num_dimensions * slice_size * sizeof(float);
    std::vector<uint8_t> buffer(size);
    double sum = 0;
    if (input.read(reinterpret_cast<char*>(buffer.data()), size)) {
        for (auto _ : st) {
            for (int i = 0; i < slice_size; ++i){
                const double len = IP<float, float, float>(reinterpret_cast<float*>(buffer.data()), reinterpret_cast<float*>(buffer.data() + num_dimensions * sizeof(float) * i), num_dimensions);
                sum += len;
            }
        }
    }
//    st.counters["num_points"] = num_points;
//    st.counters["num_dimensions"] = num_dimensions;
    st.counters["slice_size"] = slice_size;
    st.counters["SUM of IP"] = sum;
}
// ---------------------------------------------------------------------------
static void InMemory_SIMD_MT(benchmark::State& st) {
    std::ifstream input(base_file_path, std::ios::binary);
    uint32_t num_points;
    uint32_t num_dimensions;
    assert(input.is_open());
    input.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));
    input.read(reinterpret_cast<char*>(&num_dimensions), sizeof(num_dimensions));
    uint32_t slice_size = st.range(0);
    assert(num_points >= slice_size);
    const auto size = num_dimensions * slice_size * sizeof(float);
    std::vector<uint8_t> buffer(size);
    double sum = 0;
    if (input.read(reinterpret_cast<char*>(buffer.data()), size)) {
        for (auto _ : st) {
#pragma omp parallel for reduction(+: sum)
            for (int i = 0; i < slice_size; ++i){
                const double len = IP<float, float, float>(reinterpret_cast<float*>(buffer.data()), reinterpret_cast<float*>(buffer.data() + num_dimensions * sizeof(float) * i), num_dimensions);
                sum += len;
            }
        }
    }
//    st.counters["num_points"] = num_points;
//    st.counters["num_dimensions"] = num_dimensions;
    st.counters["slice_size"] = slice_size;
    st.counters["SUM of IP"] = sum;
}
// ---------------------------------------------------------------------------
static void CPP_Fread(benchmark::State& st) {
    std::ifstream input(base_file_path, std::ios::binary);
    uint32_t num_points;
    uint32_t num_dimensions;
    assert(input.is_open());
    input.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));
    input.read(reinterpret_cast<char*>(&num_dimensions), sizeof(num_dimensions));
    uint32_t slice_size = st.range(0);
    assert(num_points >= slice_size);
    const auto size = num_dimensions * slice_size * sizeof(float);
    for (auto _ : st) {
        st.PauseTiming(); clean_page_cache(); st.ResumeTiming();
        std::vector<uint8_t> buffer(size);
        input.read(reinterpret_cast<char*>(buffer.data()), size);
    }
//    st.counters["num_points"] = num_points;
//    st.counters["num_dimensions"] = num_dimensions;
    st.counters["slice_size"] = slice_size;
}
// ---------------------------------------------------------------------------
BENCHMARK(InMemory)->Iterations(10)->Unit(benchmark::kMillisecond)->Arg(1'000'000)->Arg(10'000'000);
BENCHMARK(InMemory_MT)->Iterations(10)->Unit(benchmark::kMillisecond)->Arg(1'000'000)->Arg(10'000'000);
BENCHMARK(InMemory_SIMD)->Iterations(10)->Unit(benchmark::kMillisecond)->Arg(1'000'000)->Arg(10'000'000);
BENCHMARK(InMemory_SIMD_MT)->Iterations(10)->Unit(benchmark::kMillisecond)->Arg(1'000'000)->Arg(10'000'000);
BENCHMARK(CPP_Fread)->Iterations(10)->Unit(benchmark::kMillisecond)->Arg(1'000'000)->Arg(10'000'000);
// ---------------------------------------------------------------------------
}  // namespace
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
}
// ---------------------------------------------------------------------------
