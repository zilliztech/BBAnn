#define TEST_RPQ

#include "pq/pq_residual.h"
#include "util/heap.h"
#include "flat/flat.h"
#include "util/read_file.h"
#include "util/merge.h"
#include "ivf/kmeans.h"
#include "ivf/ivf_flat.h"

#include <sys/time.h>

#define BigAnn
// #define Yandex_Text_to_Image

#ifdef BigAnn
    using CODE_T = uint8_t;
    using DIS_T = int32_t;
    const char* Learn_Path = "../../data/BIGANN/learn.100M.u8bin";
    const char* Query_Path = "../../data/BIGANN/query.public.10K.u8bin";
    #define Dis_Compare     CMax<int32_t,uint32_t>
    #define Dis_Computer    L2sqr<const CODE_T, const CODE_T, DIS_T>
    #define PQ_DIS_Computer L2sqr<const CODE_T, const float, float>
    #define OUT_PUT         "bigann"
#endif

#ifdef Yandex_Text_to_Image
    using CODE_T = float;
    using DIS_T = float;
    const char* Learn_Path = "../../data/Yandex-Text-to-Image/query.learn.50M.fbin";
    const char* Query_Path = "../../data/Yandex-Text-to-Image/query.public.100K.fbin";
    #define Dis_Compare     CMin<float,int>
    #define Dis_Computer    IP<const CODE_T, const CODE_T, DIS_T>
    #define PQ_DIS_Computer IP<const CODE_T, const float, float>
    #define OUT_PUT         "yandex_text_to_image"
#endif

const int Base_Batch = 5000000;
const int Query_Batch = 1000;
int topk = 1000;
int nb, nq, dim;
CODE_T *xb, *xq;

uint8_t m = 32;
uint8_t nbits = 8;

timeval t1, t2;
long int getTime(timeval end, timeval start) {
    return 1000*(end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1000;
}

DIS_T *tmp_dis = new DIS_T[Query_Batch * topk];
uint32_t *tmp_lab = new uint32_t[Query_Batch * topk];

void Output(const char* file_name, DIS_T *dis, uint32_t *lab) {
    FILE *fi = fopen(file_name, "w");
    for(int i=0;i<nq;i++){
        for(int j=0;j<topk;j++){
            fprintf(fi, "%f %d\n", (float)dis[i*topk+j], lab[i*topk+j]);
        }
        fprintf(fi, "\n");
    }
    fclose(fi);
}

void Flat(int batch_from, int batch_num) {
    gettimeofday(&t1, 0);

    knn_2<Dis_Compare, CODE_T, CODE_T> (
        xq, xb, nq, batch_num, dim, topk, tmp_dis, tmp_lab, Dis_Computer);

    gettimeofday(&t2, 0);
    printf("flat seg %d cost %ldms\n", batch_from/Base_Batch, getTime(t2,t1));

    char file_name[64];
    sprintf(file_name, "%s_flat_segment_%d.txt",OUT_PUT, batch_from/Base_Batch);
    Output(file_name, tmp_dis, tmp_lab);
}

int main() {
    FILE *fi;
    // just query 1 batch
    fi = read_file_head(Query_Path, &nq, &dim);
    assert(fi);
    xq = new CODE_T[Query_Batch * dim];

    read_file_data(fi, Query_Batch, dim, xq);
    fclose(fi);
    nq = Query_Batch;

    // for each learn
    fi = read_file_head(Learn_Path, &nb, &dim);
    assert(fi);
    xb = new CODE_T[Base_Batch * dim];

    int batch_num = read_file_data(fi, Base_Batch, dim, xb);
    fclose(fi);

    // Flat(0, batch_num);
    // return 0;

    // pq
    PQResidualQuantizer<Dis_Compare, CODE_T, uint8_t> pq(dim, m, nbits, MetricType::L2);

    int32_t train_size = std::min(65536, Base_Batch);
    gettimeofday(&t1, 0);

    int nlist = 4096;
    float *centroids = new float [nlist*dim];
    kmeans(batch_num, xb, dim, nlist, centroids, false, 508);

    gettimeofday(&t2, 0);
    printf("kmeans cost %ldms\n", getTime(t2,t1));


    std::vector<std::vector<CODE_T>> codes;
    std::vector<std::vector<uint32_t>> ids;
    ivf_flat_insert(batch_num, xb, dim,
                    nlist, centroids,
                    codes, ids);

    gettimeofday(&t1, 0);
    printf("ivf_flat_insert cost %ldms\n", getTime(t1,t2));

    std::vector<uint32_t> buckets(nlist, 0);
    CODE_T *xb2 = new CODE_T[dim * batch_num];
    float *cent2 = new float[dim * batch_num];
    uint32_t *id2 = new uint32_t[batch_num];

    int64_t train_count = 0;
    {
        auto xb2_tmp = xb2;
        auto cent2_tmp = cent2;
        auto id2_tmp = id2;
        for(size_t i=0;i<codes.size();i++){
            buckets[i] = ids[i].size();
            printf("%d %d\n", i, buckets[i]);

            memcpy(xb2_tmp, codes[i].data(), codes[i].size()*sizeof(CODE_T));
            xb2_tmp += codes[i].size();

            memcpy(id2_tmp, ids[i].data(), ids[i].size()*sizeof(uint32_t));
            id2_tmp += ids[i].size();

            if ((cent2_tmp - cent2) / dim < 100000){
                for(size_t j=0;j<ids[i].size();j++){
                    memcpy(cent2_tmp, centroids + i * dim, dim*sizeof(float));
                    cent2_tmp += dim;
                }
            }
        }

        train_count = (cent2_tmp - cent2) / dim;
    }

    gettimeofday(&t2, 0);
    printf("prepare cost %ldms\n", getTime(t2,t1));

    pq.train(train_count, xb2, cent2);
    delete[] cent2;

    gettimeofday(&t1, 0);
    printf("pq.train cost %ldms\n", getTime(t1,t2));

    float *precomputer_table = nullptr;
    pq.encode_vectors_and_save(precomputer_table, batch_num, xb2, centroids, buckets, "");

    delete[] xb2;
    delete[] precomputer_table;


    gettimeofday(&t2, 0);
    printf("encode_vectors_and_save cost %ldms\n", getTime(t2,t1));


/*
    int nprobe = 4;
    float *val = new float[nq * nprobe];
    uint32_t *label = new uint32_t[nq * nprobe];
    knn_2<CMax<float,int>,CODE_T,float>(xq, centroids, nq, nlist, dim, nprobe,
                                        val, label, L2sqr<const CODE_T, const float, float>);
*/

#pragma omp parallel for
    for (int32_t q=0; q<nq; q++){
        auto cent_tmp = centroids;
        auto id_tmp = id2;
        auto pcodes = pq.get_codes();

        float* precomputer_table = nullptr;
        pq.calc_precompute_table(precomputer_table, xq+q*dim);
        for (int32_t i=0; i<nlist; i++){
            pq.search(precomputer_table, xq+q*dim, cent_tmp, pcodes, buckets[i],
                      topk, tmp_dis+q*topk, tmp_lab+q*topk,
                      (i==nlist-1), (i==0), (uint64_t)id_tmp, 0 , 0);
            cent_tmp += dim;
            id_tmp += buckets[i];
            pcodes += buckets[i] * (m + sizeof(float));
        }
        delete[] precomputer_table;
    }

    Output(OUT_PUT "_rpq.txt", tmp_dis, tmp_lab);

    delete []tmp_dis;
    delete []tmp_lab;

    delete []xb;
    delete []xq;

    return 0;
}
