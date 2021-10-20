#include <iostream>
#include <vector>

#include "hnswlib/hnswalg.h"
#include "hnswlib/hnswlib.h"
#include "util/utils.h"

using CODE_T = float;
using DIS_T = float;
const char *Learn_Path = "../../data/Yandex-Text-to-Image/base.10M.fdata";
const char *Query_Path = "../../data/Yandex-Text-to-Image/query.public.100K.fbin";
#define Dis_Compare CMin<float, int>
#define Dis_Computer IP<const CODE_T, const CODE_T, DIS_T>
#define OUT_PUT "yandex_text_to_image"
#define METRIC MetricType::IP

uint32_t nb, nq, dim;
float *xb, *xq;

const int Base_Batch = 50000;
const int Query_Batch = 1;

const float radius = 2.4;

const int hnswM = 32;
const int hnswefC = 200;
const int hnswef = Base_Batch;

int main() {
    read_bin_file(Query_Path, xq, nq, dim);
    nq = Query_Batch;

    read_bin_file(Learn_Path, xb, nb, dim);
    nb = Base_Batch;

    uint32_t *ids = new uint32_t[nb];
    for (uint32_t i = 0; i < nb; ++i) {
        ids[i] = i;
    }

    hnswlib::SpaceInterface<float> *space = new hnswlib::L2Space<float, float>(dim);

    auto index_hnsw = std::make_shared<hnswlib::HierarchicalNSW<float>>(
        space, nb, hnswM, hnswefC);
    index_hnsw->addPoint(xb, ids[0]);
#pragma omp parallel for
    for (int64_t i = 1; i < nb; i++) {
        index_hnsw->addPoint(xb + i * dim, ids[i]);
    }

    std::cout << "Build done" << std::endl;

    index_hnsw->setEf(hnswef);

    // for (float radius = 2.4; radius > 0; radius -= 0.1) {
    auto reti = index_hnsw->searchRange(xq, 20, radius);
    std::cout << radius << " size: " << reti.size() << std::endl;

    while (!reti.empty()) {
        auto x = reti.top();
        // std::cout << x.second << " " << x.first << std::endl;
        reti.pop();
    }
    // }

    std::cout << "Search done" << std::endl;

    delete space;
    delete[] xb;
    delete[] xq;

    return 0;
}