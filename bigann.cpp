#include "read_file.h"
#include "distance.h"
#include "merge.h"

#include "flat.h"
#include "ivf_flat.h"

#include "space_ui8_l2.h"
#include "hnswlib.h"
#include "hnswalg.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#define BigAnn

#ifdef BigAnn
    using CODE_T = uint8_t;
    using DIS_T = int32_t;
    const char* Learn_Path = "data/BIGANN/learn.100M.u8bin";
    const char* Query_Path = "data/BIGANN/query.public.10K.u8bin";
    #define Dis_Compare     CMax<int,int>
    #define Dis_Computer    L2sqr<const CODE_T, const CODE_T, DIS_T>
    #define OUT_PUT         "bigann"
#endif

timeval t1, t2;
long int getTime(timeval end, timeval start) {
    return 1000*(end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1000;
}

int nb, nq, dim;
CODE_T *xb, *xq;
int topk = 1000;

const int Base_Batch = 5000000;
const int Query_Batch = 1000;

DIS_T *global_dis = new int[Query_Batch * topk];
int *global_lab = new int[Query_Batch * topk];

DIS_T *tmp_dis = new int[Query_Batch * topk];
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
    printf("flat seg %d cost %dms\n", batch_from/Base_Batch, getTime(t2,t1));

    char file_name[64];
    sprintf(file_name, "%s_flat_segment_%d.txt",OUT_PUT, batch_from/Base_Batch);
    Output(file_name, tmp_dis, tmp_lab);

    merge<Dis_Compare>(global_dis, global_lab, tmp_dis, tmp_lab, nq, topk, batch_from);
}

int nlist = 2048;
int nprobe = 24;
float* centroids = nullptr;
std::vector<std::vector<CODE_T>> codes;
std::vector<std::vector<int32_t>> ids;

void IVF_Train(int batch_from, int batch_num) {
    centroids = new float[nlist * dim];

    gettimeofday(&t1, 0);

    kmeans(batch_num, xb, dim, nlist, centroids);

    gettimeofday(&t2, 0);
    printf("kmeans cost %dms\n", getTime(t2,t1));

    FILE *fi=fopen(OUT_PUT "_centroids_2048.bin", "w");
    fwrite(centroids, sizeof(float), nlist * dim, fi);
    fclose(fi);
}

void IVF_Insert(int batch_from, int batch_num) {
    gettimeofday(&t1, 0);

    ivf_flat_insert(batch_num, xb, dim, nlist, centroids, codes, ids);

    gettimeofday(&t2, 0);
    printf("insert seg %d cost %dms\n", batch_from/Base_Batch, getTime(t2,t1));

    char file_name[64];
    sprintf(file_name, "%s_ivf_segment_%d.bin",OUT_PUT, batch_from/Base_Batch);
    FILE *fi=fopen(file_name, "w");
    fwrite(&batch_from, sizeof(int), 1, fi);
    fwrite(&nlist, sizeof(int), 1, fi);
    for (int i=0;i<nlist;i++){
        int s=ids[i].size();
        fwrite(&s, sizeof(int), 1, fi);
        fwrite(ids[i].data(), sizeof(int), s, fi);
        fwrite(codes[i].data(), dim * sizeof(CODE_T), s, fi);
    }
    fclose(fi);

    for(int i=0;i<nlist;i++){
        printf("%f\t%d\n",
            sqrt(IP<float,float,float>(centroids+i*dim, centroids+i*dim, dim)),
            ids[i].size());
    }
}

void IVF_Search(int batch_from, int batch_num) {
    gettimeofday(&t1, 0);

    ivf_flat_search<Dis_Compare, CODE_T, DIS_T>(
        nq, xq, dim, nlist, centroids, codes, ids, nprobe, topk, tmp_dis, tmp_lab, Dis_Computer);

    gettimeofday(&t2, 0);
    printf("query seg %d cost %dms\n", batch_from/Base_Batch, getTime(t2,t1));

    char file_name[64];
    sprintf(file_name, "%s_ivf_%d_segment_%d.txt",OUT_PUT, nprobe, batch_from/Base_Batch);
    Output(file_name, tmp_dis, tmp_lab);

    merge<Dis_Compare>(global_dis, global_lab, tmp_dis, tmp_lab, nq, topk, batch_from);
}

int main() {
    // init heap
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

    for (int batch_from = 0; batch_from < nb; batch_from += Base_Batch) {
        int batch_num = read_file_data(fi, Base_Batch, dim, xb);

//        Flat(batch_from, batch_num);

        if (batch_from == 0) {
            IVF_Train(batch_from, batch_num);
        }
        codes.clear();
        ids.clear();
        IVF_Insert(batch_from, batch_num);
        IVF_Search(batch_from, batch_num);

        nprobe = 48;
        IVF_Search(batch_from, batch_num);

        break;
    }
    fclose(fi);

    // Output(OUT_PUT "_flat.txt", global_dis, global_lab);
    // Output(OUT_PUT "_ivf.txt", global_dis, global_lab);

#if 0
    // hnsw
    int M = 24;
    int EF_B = 128;
    int EF_Q = 96;
    hnswlib::SpaceInterface<int> *space = new hnswlib::Ui8L2Space(dim);
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

    delete []global_dis;
    delete []global_lab;
    delete []tmp_dis;
    delete []tmp_lab;

    delete []xb;
    delete []xq;

    return 0;
}