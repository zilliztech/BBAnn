#include "ivf/same_size_kmeans.h"
#include "util/heap.h"
#include "flat/flat.h"
#include "util/read_file.h"
#include "util/merge.h"

#include <sys/time.h>

#define BigAnn
// #define Yandex_Text_to_Image

#ifdef BigAnn
    using CODE_T = uint8_t;
    using DIS_T = int32_t;
    const char* Learn_Path = "../../data/BIGANN/base.10M.u8bin";
    const char* Query_Path = "../../data/BIGANN/query.public.10K.u8bin";
    #define Dis_Compare     CMax<int,int>
    #define Dis_Computer    L2sqr<const CODE_T, const CODE_T, DIS_T>
    #define METRIC_TYPE     MetricType::L2
    #define OUT_PUT         "bigann"
#endif

#ifdef Yandex_Text_to_Image
    using CODE_T = float;
    using DIS_T = float;
    const char* Learn_Path = "../../data/Yandex-Text-to-Image/base.10M.fdata";
    const char* Query_Path = "../../data/Yandex-Text-to-Image/query.public.100K.fbin";
    #define Dis_Compare     CMin<float,int>
    #define Dis_Computer    IP<const CODE_T, const CODE_T, DIS_T>
    #define METRIC_TYPE     MetricType::IP
    #define OUT_PUT         "yandex_text_to_image"
#endif

const int Base_Batch = 10000;
// const int Query_Batch = 1000;
int topk = 1000;
int nb, nq, dim;
CODE_T *xb, *xq;

timeval t1, t2;
long int getTime(timeval end, timeval start) {
    return 1000*(end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1000;
}

int main() {

    FILE *fi;
    // just query 1 batch
    // fi = read_file_head(Query_Path, &nq, &dim);
    // assert(fi);
    // xq = new CODE_T[Query_Batch * dim];

    // read_file_data(fi, Query_Batch, dim, xq);
    // fclose(fi);
    // nq = Query_Batch;

    fi = read_file_head(Learn_Path, &nb, &dim);
    assert(fi);
    xb = new CODE_T[Base_Batch * dim];

    int batch_num = read_file_data(fi, Base_Batch, dim, xb);
    int k = 1000;

    float* centroids = new float[k];
    same_size_kmeans(batch_num, xb, dim, 100, centroids);

    delete[] centroids;
    delete[] xb;
    // delete[] xq;

    return 0;
}
