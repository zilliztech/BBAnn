#include "read_file.h"
#include "distance.h"
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
    const char* OUT_PUT1 = "bigann_flat.txt";
    const char* OUT_PUT2 = "bigann_ivf_16.txt";
    const char* OUT_PUT3 = "bigann_ivf_32.txt";
    const char* OUT_PUT4 = "bigann_ivf_64.txt";
#endif


timeval t1, t2;
long int getTime(timeval end, timeval start) {
    return 1000*(end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1000;
}

int nb, nq, dim;
CODE_T *xb, *xq;
int topk = 1000;



int main() {
    const int Base_Batch = 5000000;
    const int Query_Batch = 1000;

    FILE *fi;
    fi = read_file_head(Learn_Path, &nb, &dim);
    xb = new CODE_T[Base_Batch * dim];
    if (fi) {
        read_file_data(fi, Base_Batch, dim, xb);
        fclose(fi);
    }

    fi = read_file_head(Query_Path, &nq, &dim);
    xq = new CODE_T[Query_Batch * dim];
    if (fi) {
        read_file_data(fi, Query_Batch, dim, xq);
        fclose(fi);
    }

    nb = Base_Batch;
    nq = Query_Batch;
    printf("%d %d %d %d\n", nb, nq, dim, topk);

    DIS_T *dis = new int[nq*topk];
    int *lab = new int[nq*topk];

    auto Output = [&](const char* file_name) {
        fi = fopen(file_name, "w");
        for(int i=0;i<nq;i++){
            for(int j=0;j<topk;j++){
                fprintf(fi, "%f %d\n", (float)dis[i*topk+j], lab[i*topk+j]);
            }
            fprintf(fi, "\n");
        }
        fclose(fi);
    };

#if 1
    // Flat
    gettimeofday(&t1, 0);

    knn_1<Dis_Compare, CODE_T, CODE_T> (
        xq, xb, nq, nb, dim, topk, dis, lab, Dis_Computer);

    gettimeofday(&t2, 0);
    printf("cost %dms\n",getTime(t2,t1));

    Output(OUT_PUT1);

#endif

#if 1
    // IVF
    std::vector<std::vector<CODE_T>> codes;
    std::vector<std::vector<int32_t>> ids;
    int nlist = 2048;
    int nprobe = 16;
    float* centroids = new float[nlist * dim];

    gettimeofday(&t1, 0);

    kmeans(500000, xb, dim, nlist, centroids);

    gettimeofday(&t2, 0);
    printf("kmeans cost %dms\n",getTime(t2,t1));

    ivf_flat_insert(nb, xb, dim, nlist, centroids, codes, ids);

    gettimeofday(&t1, 0);
    printf("insert cost %dms\n",getTime(t1,t2));

    ivf_flat_search<Dis_Compare, CODE_T, DIS_T>(
        nq, xq, dim, nlist, centroids, codes, ids, nprobe, topk, dis, lab, Dis_Computer);

    gettimeofday(&t2, 0);
    printf("query cost %dms\n",getTime(t2,t1));

    Output(OUT_PUT2);

    gettimeofday(&t1, 0);

    ivf_flat_search<Dis_Compare, CODE_T, DIS_T>(
        nq, xq, dim, nlist, centroids, codes, ids, 32, topk, dis, lab, Dis_Computer);

    gettimeofday(&t2, 0);
    printf("query cost %dms\n",getTime(t2,t1));

    Output(OUT_PUT3);

    gettimeofday(&t1, 0);

    ivf_flat_search<Dis_Compare, CODE_T, DIS_T>(
        nq, xq, dim, nlist, centroids, codes, ids, 48, topk, dis, lab, Dis_Computer);

    gettimeofday(&t2, 0);
    printf("query cost %dms\n",getTime(t2,t1));

    Output(OUT_PUT4);
#endif

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

    delete []dis;
    delete []lab;

    delete []xb;
    delete []xq;

    return 0;
}