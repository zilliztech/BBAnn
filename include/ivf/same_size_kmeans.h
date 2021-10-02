#pragma once

#include <omp.h>

#include "util/distance.h"
#include "util/random.h"
#include "util/utils.h"
#include <algorithm>
#include <assert.h>
#include <deque>
#include <memory>
#include <string.h>
#include <unistd.h>
#include "hnswlib/hnswlib.h"
#include "interface/kaHIP_interface.h"

// #define SSK_LOG

template <typename T>
void ssk_compute_dist_tab(int64_t nx, const T *x_in, int64_t dim, int64_t k,
                          const float *centroids, float *dis_tab) {
#pragma omp parallel for
  for (int64_t i = 0; i < nx; ++i) {
    const T *x = x_in + i * dim;
    const int64_t ii = i * k;
    for (int64_t j = 0; j < k; ++j) {
      dis_tab[ii + j] =
          L2sqr<const T, const float, float>(x, centroids + j * dim, dim);
    }
  }
}

void ssk_init_assign(int64_t nx, int64_t k, uint64_t max_target_size,
                     const float *dis_tab, int64_t *hassign, int64_t *assign) {
  uint64_t remain = nx;
  std::vector<float> min_max_dist_diff(nx);
  std::vector<int64_t> min_cluster_ids(nx);
  bool idx = 0;
  // rotate vector each iteration
  std::vector<std::vector<int64_t>> points(2, std::vector<int64_t>(nx));

  for (int64_t i = 0; i < nx; ++i) {
    points[idx][i] = i;
  }

  while (remain) {
    for (int64_t i = 0; i < remain; ++i) {
      float min_dis = std::numeric_limits<float>::max();
      float max_dis = 0;
      int64_t min_cluster_id = -1;
      const auto x = points[idx][i];

      for (int64_t j = 0; j < k; ++j) {
        if (hassign[j] < max_target_size) {
          auto dis = dis_tab[x * k + j];
          if (dis < min_dis) {
            min_dis = dis;
            min_cluster_id = j;
          }
          max_dis = std::max(max_dis, dis);
        }
      }
      // this should not happen as the max_target_size is a ceiling
      // so there is at least one of the clusters could fit the vector
      assert(min_cluster_id != -1);
      min_cluster_ids[x] = min_cluster_id;
      min_max_dist_diff[x] = min_dis - max_dis;
    }

    std::sort(points[idx].begin(), points[idx].begin() + remain,
              [&](const auto &x, const auto &y) {
                return min_max_dist_diff[x] < min_max_dist_diff[y];
              });

    int64_t j = 0;
    for (int64_t i = 0; i < remain; ++i) {
      const auto x = points[idx][i];
      const auto c = min_cluster_ids[x];

      if (hassign[c] < max_target_size) {
        assign[x] = c;
        ++hassign[c];
      } else {
        points[!idx][j++] = x;
      }
    }

    remain = j;
    idx = !idx;
  }
}

void ssk_print_cluster_size_stats(int64_t k, const int64_t *hassign) {
  float mini = std::numeric_limits<float>::max(), maxi = 0, avg = 0;
  for (int64_t i = 0; i < k; ++i) {
    avg += hassign[i];
    mini = std::min(mini, 1.0f * hassign[i]);
    maxi = std::max(maxi, 1.0f * hassign[i]);
  }
  std::cout << "avg: " << avg / k << " min: " << mini << " max: " << maxi
            << std::endl;
}

template <typename T>
void same_size_kmeans(int64_t nx, const T *x_in, int64_t dim, int64_t k,
                      float *centroids, int64_t *assign, bool kmpp = false,
                      float avg_len = 0.0, int64_t niter = 10,
                      int64_t seed = 1234) {
      assert(x_in != nullptr);
    assert(centroids != nullptr);
    assert(assign != nullptr);

    assert(nx > 0);
    assert(k > 0);
    std::cout<<"input cluster size="<<nx<<std::endl;
    uint64_t max_target_size = (nx + k - 1) / k;
    uint64_t min_target_size = nx / k;

    // the nubmer of vectors in the cluster
    int64_t *hassign = new int64_t[k];

    memset(hassign, 0, sizeof(int64_t) * k);
    int* xadj=new int[nx+1];
    xadj[0]=0;
    std::vector<int> neighborhood;
    int npts = (int) nx;
    int kint = (int) k;
//    printf("reaching same_size_kmenas\n");
    {
        std::vector<std::vector<int>> neighbors(nx, std::vector<int>());
        {
            hnswlib::SpaceInterface<float> *space;
            space = new hnswlib::L2Space(dim);
            auto index_hnsw = std::make_shared<hnswlib::HierarchicalNSW<float>>(
                    space, nx, 16, 100);
//	printf("start to build hnsw\n");
#pragma omp parallel for
            for (int64_t i = 0; i < nx; i++) {
                index_hnsw->addPoint(x_in + i * dim, i);
            }
//	printf("finish building hnsw for gp\n");
            for (int i = 0; i < nx; i++) {
                std::priority_queue<std::pair<float, long unsigned int>> pq=index_hnsw->searchKnn(x_in+i*dim,10);
                while (!pq.empty()){
                    int neighbor_id=(int)pq.top().second;
                    if (neighbor_id!=i) {
                        neighbors[i].push_back(neighbor_id);
                    }
                    pq.pop();
                }
            }
            //std::cout<<"finish initializing neighbors"<<std::endl;
        }
        for (int i=0;i<nx;i++){
            for (int j:neighbors[i]){
                auto iter=std::find(neighbors[j].begin(),neighbors[j].end(),i);
                if (iter==neighbors[j].end()){
                    neighbors[j].push_back(i);
                }
            }
        }
        //std::cout<<"finish transform to undirect"<<std::endl;
        for (int i=0;i<nx;i++){
            xadj[i+1]=xadj[i]+neighbors[i].size();
            for (int j:neighbors[i]){
                neighborhood.push_back(j);
            }
        }
        //std::cout<<"finish initialize graph"<<std::endl;
    }
//    for (int j=0;j<nx;j++){
//	    int n_n=xadj[j+1]-xadj[j];
//    printf("neighbor size of node %d is %d, the ids of neighbors are ",j,n_n);
//    for (int i=xadj[j];i<xadj[j+1];i++){
//	    printf("%d ",neighborhood[i]);
//    }
//    printf("\n");
//    }
    double imbalance = 0.025;
    int edge_cut     = 0;
    int* vwgt        = NULL;
    int* adjcwgt     = NULL;
    int* int_assign=new int[nx];
    kaffpa(&npts, vwgt, xadj, adjcwgt, neighborhood.data(), &kint, &imbalance, false, 0, ECO, & edge_cut,
           int_assign);
//    printf("finish partitioning graph\n");
    int max_id=0;
    for (int i=0;i<nx;i++){
	    assign[i]=(int64_t)(int_assign[i]);
	    if (max_id<assign[i]) max_id=assign[i];
    }
    int* counting=new int[max_id+1];
    for (int i=0;i<=max_id;i++){
	    counting[i]=0;
    }
    for (int i=0;i<nx;i++){
	    counting[assign[i]]++;
    }
    for (int i=0;i<=max_id;i++){
	    printf("%d ",counting[i]);
    }
    printf("\n");
//    printf("start to compute centroids\n");
    compute_centroids(dim,k,nx,x_in,assign,hassign,centroids,avg_len);
//    for (int i=0;i<dim;i++){
//        printf("%f ",centroids[i]);
//    }
//    printf("\n");
    delete[] hassign;
    delete[] int_assign;
}
