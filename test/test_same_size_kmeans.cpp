#include "flat/flat.h"
#include "ivf/ivf_flat.h"
#include "ivf/same_size_kmeans.h"
#include "util/merge.h"
#include "util/read_file.h"
#include "util/utils.h"

#include "hnswlib/hnswalg.h"
#include "hnswlib/hnswlib.h"
#include "hnswlib/space_ui8_l2.h"

#include <fstream>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

// test ivfflat

#define BigAnn

#ifdef BigAnn
using CODE_T = uint8_t;
using DIS_T = uint32_t;
const char *Learn_Path = "../../data/BIGANN/base.1M.u8bin";
const char *Query_Path = "../../data/BIGANN/query.public.10K.128.u8bin";
#define Dis_Compare CMax<DIS_T, uint32_t>
#define Dis_Computer L2sqr<const CODE_T, const CODE_T, DIS_T>
#define OUT_PUT "bigann"
#define METRIC MetricType::L2
#endif

#ifdef Yandex_Text_to_Image
using CODE_T = float;
using DIS_T = float;
const char *Learn_Path = "data/Yandex_Text-to-Image/query.learn.50M.fbin";
const char *Query_Path = "data/Yandex_Text-to-Image/query.public.100K.fbin";
#define Dis_Compare CMin<float, int>
#define Dis_Computer IP<const CODE_T, const CODE_T, DIS_T>
#define OUT_PUT "yandex_text_to_image"
#define METRIC MetricType::IP
#endif

timeval t1, t2;
long int getTime(timeval end, timeval start) {
  return 1000 * (end.tv_sec - start.tv_sec) +
         (end.tv_usec - start.tv_usec) / 1000;
}

uint32_t nb, nq, dim;
CODE_T *xb, *xq;
uint32_t topk = 10;

const int Base_Batch = 10000;
const int Query_Batch = 10000;

DIS_T *global_dis = new DIS_T[Query_Batch * topk];
uint32_t *global_lab = new uint32_t[Query_Batch * topk];

DIS_T *tmp_dis = new DIS_T[Query_Batch * topk];
uint32_t *tmp_lab = new uint32_t[Query_Batch * topk];

void Flat(int batch_from, int batch_num) {
  gettimeofday(&t1, 0);

  // knn_1<Dis_Compare, CODE_T, CODE_T> (xq, xb, nq, batch_num, dim, topk,
  // tmp_dis, tmp_lab, Dis_Computer);
  knn_1<Dis_Compare, CODE_T, CODE_T>(xq, xb, nq, batch_num, dim, topk,
                                     global_dis, global_lab, Dis_Computer);

  gettimeofday(&t2, 0);
  printf("flat seg %d cost %ldms\n", batch_from / Base_Batch, getTime(t2, t1));

  // merge<Dis_Compare>(global_dis, global_lab, tmp_dis, tmp_lab, nq, topk,
  // batch_from);
}

uint32_t nlist = 100;
uint32_t nprobe = 20;
float *centroids = nullptr;
std::vector<std::vector<CODE_T>> codes;
std::vector<std::vector<uint32_t>> ids;

void IVF_Train(int nx) {
  centroids = new float[nlist * dim];
  int64_t *assign = new int64_t[nx];

  gettimeofday(&t1, 0);

  // kmeans(batch_num, xb, dim, nlist, centroids);
  same_size_kmeans(nx, xb, dim, nlist, centroids, assign);

  gettimeofday(&t2, 0);
  printf("kmeans cost %ld ms\n", getTime(t2, t1));

    codes.resize(nlist);
    ids.resize(nlist);

#pragma omp parallel
    {
        int64_t nt = omp_get_num_threads();
        int64_t rank = omp_get_thread_num();

        // this thread is taking care of centroids c0:c1
        int64_t c0 = (nlist * rank) / nt;
        int64_t c1 = (nlist * (rank + 1)) / nt;

        for (int64_t i = 0; i < nx; i++) {
            int64_t ci = assign[i];
            if (ci >= c0 && ci < c1) {
                codes[ci].resize(codes[ci].size() + dim);
                memcpy(codes[ci].data() + ids[ci].size() * dim,
                       xb + i * dim, dim * sizeof(CODE_T));
                ids[ci].push_back(i);
            }
        }
    }

  delete[] assign;
}

void IVF_Search(int batch_num, int iter = 1) {

  long min_time = std::numeric_limits<long>::max();
  while (iter--) {
    gettimeofday(&t1, 0);
    // ivf_flat_search<Dis_Compare, CODE_T, DIS_T>(
    //     nq, xq, dim, nlist, centroids, codes, ids, nprobe, topk, tmp_dis,
    //     tmp_lab, Dis_Computer);
    ivf_flat_search<Dis_Compare, CODE_T, DIS_T>(
        nq, xq, dim, nlist, centroids, codes, ids, nprobe, topk, global_dis,
        global_lab, Dis_Computer);

    gettimeofday(&t2, 0);
    min_time = std::min(min_time, getTime(t2, t1));
  }

  printf("min query cost %ld ms\n", min_time);
}

void save_answer(const std::string answer_bin_file, DIS_T *answer_dists,
                 uint32_t *answer_ids) {
  std::ofstream answer_writer(answer_bin_file, std::ios::binary);
  answer_writer.write((char *)&nq, sizeof(uint32_t));
  answer_writer.write((char *)&topk, sizeof(uint32_t));

  // for (int i = 0; i < nq; i ++) {
  //     auto ans_disi = answer_dists + topk * i;
  //     auto ans_idsi = answer_ids + topk * i;
  //     heap_reorder<Dis_Compare>(topk, ans_disi, ans_idsi);
  // }

  uint32_t tot = nq * topk;
  answer_writer.write((char *)answer_ids, tot * sizeof(uint32_t));
  answer_writer.write((char *)answer_dists, tot * sizeof(DIS_T));

  answer_writer.close();
}

int main() {
  // init heap
  heap_heapify<Dis_Compare>(Query_Batch * topk, global_dis, global_lab);

  get_bin_metadata(Query_Path, nq, dim);
  read_bin_file(Query_Path, xq, nq, dim);
  nq = Query_Batch;

  // for each learn
  read_bin_file(Learn_Path, xb, nb, dim);
  nb = Base_Batch;

#if 0
    Flat(0, nb);

    std::ofstream answer_writer("flat_gt.bin", std::ios::binary);
    answer_writer.write((char*)&nq, sizeof(uint32_t));
    answer_writer.write((char*)&topk, sizeof(uint32_t));

    uint32_t tot = nq * topk;
    answer_writer.write((char*)global_lab, tot * sizeof(uint32_t));
    float tmp_gt_dis[tot];
    for (int i = 0; i < tot; ++i) {
        tmp_gt_dis[i] = static_cast<float>(global_dis[i]);
    }
    // gt file needs to be float
    answer_writer.write((char*)tmp_gt_dis, tot * sizeof(float));
    answer_writer.close();
#else
  codes.clear();
  ids.clear();
  IVF_Train(nb);
  IVF_Search(nb);

  save_answer("ivf_answer.bin", global_dis, global_lab);
#endif

  // for (auto i = 0; i < nq; ++i) {
  //     for (auto j = 0; j < topk; ++j) {
  //         std::cout << global_dis[i*topk+j] << " " << global_lab[i*topk+j] <<
  //         std::endl;
  //     }
  // }

  recall<DIS_T, uint32_t>("flat_gt.bin", "ivf_answer.bin", METRIC);

#if 0
    // hnsw
    int M = 24;
    int EF_B = 128;
    int EF_Q = 96;
    hnswlib::SpaceInterface<int> *space = new hnswlib::L2Space<int8_t, int>(dim);
    hnswlib::HierarchicalNSW<int> *index = new hnswlib::HierarchicalNSW<int>(space, nb, M, EF_B);

    std::vector<std::vector<std::pair<int, size_t>>> rst;
    rst.resize(nq);

    timeval b1;
    gettimeofday(&b1, 0);

    index->addPoint(xb, 0);
#pragma omp parallel for
    for (int i = 1; i < nb; ++i) {
        index->addPoint(xb + i * dim, i);
    }

    timeval b2;
    gettimeofday(&b2, 0);

    printf("ntotal = %d build cost %zu\n", nb, getTime(b2, b1));

    index->setEf(EF_Q);

    timeval t0;
    gettimeofday(&t0, 0);

#pragma omp parallel for
    for (int i = 0; i < nq; i++) {
        auto ret = index->searchKnnCloserFirst(xq + i * dim, topk);
        rst[i].swap(ret);
    }

    timeval t1;
    gettimeofday(&t1, 0);

    printf("nq = %d query cost %zu\n", nq, getTime(t1, t0));

    freopen("bigann_hnsw.txt","w",stdout);
    for(int i=0;i<nq;i++){
        for(int j=0;j<topk;j++){
            printf("%d %zu\n", rst[i][j].first, rst[i][j].second);
        }
        printf("\n");
    }
#endif

  delete[] global_dis;
  delete[] global_lab;
  delete[] tmp_dis;
  delete[] tmp_lab;

  delete[] xb;
  delete[] xq;

  return 0;
}
