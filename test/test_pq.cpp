#include "pq/pq.h"
#include "utils/heap.h"
#include "flat/flat.h"
#include "utils/read_file.h"
#include "utils/merge.h"

#include <sys/time.h>

#define BigAnn
// #define Yandex_Text_to_Image

#ifdef BigAnn
    using CODE_T = uint8_t;
    using DIS_T = int32_t;
    const char* Learn_Path = "../../data/BIGANN/learn.100M.u8bin";
    const char* Query_Path = "../../data/BIGANN/query.public.10K.u8bin";
    #define Dis_Compare     CMax<int,int>
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

uint8_t m = 8;
uint8_t nbits = 8;

timeval t1, t2;
long int getTime(timeval end, timeval start) {
    return 1000*(end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1000;
}

DIS_T *global_dis = new DIS_T[Query_Batch * topk];
int *global_lab = new int[Query_Batch * topk];

DIS_T *tmp_dis = new DIS_T[Query_Batch * topk];
int *tmp_lab = new int[Query_Batch * topk];

void Output(const char* file_name, DIS_T *dis, int *lab) {
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

    merge<Dis_Compare>(global_dis, global_lab, tmp_dis, tmp_lab, nq, topk, batch_from);
}

int main() {

    heap_heapify<Dis_Compare>(Query_Batch * topk, global_dis, global_lab);

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

    // Flat(0, batch_num);

    // pq
    ProductQuantizer<Dis_Compare, CODE_T, uint8_t> pq(batch_num, dim, m, nbits);


    int32_t train_size = 65536;
    gettimeofday(&t1, 0);
    pq.train(train_size, xb);
    gettimeofday(&t2, 0);
    printf("Train %d cost %ldms\n", train_size, getTime(t2,t1));

    gettimeofday(&t1, 0);
    pq.encode_vectors(batch_num, xb);
    gettimeofday(&t2, 0);
    printf("Encode %d cost %ldms\n", batch_num, getTime(t2,t1));

    gettimeofday(&t1, 0);
    pq.search(nq, xq, topk, global_dis, global_lab, PQ_DIS_Computer);
    gettimeofday(&t2, 0);
    printf("Search nq %d, topk %d,cost %ldms\n", nq, topk, getTime(t2,t1));

    fclose(fi);

    // Output(OUT_PUT "_flat.txt", global_dis, global_lab);
    // Output(OUT_PUT "_ivf.txt", global_dis, global_lab);
    Output(OUT_PUT "_pq.txt", global_dis, global_lab);

    delete []global_dis;
    delete []global_lab;
    delete []tmp_dis;
    delete []tmp_lab;

    delete []xb;
    delete []xq;

    return 0;
}
