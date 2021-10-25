#pragma once

#include "heap.h"

// Id Type: C::TI
// Distance type: C::T

template <class C>
void merge(typename C::T *dis1, typename C::TI *id1, typename C::T *dis2,
           typename C::TI *id2, int64_t nq, int64_t topk, int64_t data2_base) {

  using DIS_T = typename C::T;
  using ID_T = typename C::TI;

#pragma omp parallel
  {
    DIS_T *work_dis = new DIS_T[topk];
    ID_T *work_id = new ID_T[topk];
#pragma omp for
    for (int64_t q_i = 0; q_i < nq; q_i++) {
      auto d1 = dis1 + q_i * topk;
      auto i1 = id1 + q_i * topk;
      auto d2 = dis2 + q_i * topk;
      auto i2 = id2 + q_i * topk;

      int64_t i = 0, j = 0, k = 0;
      while (i < topk) {
        if (C::cmp(d1[j], d2[k])) {
          work_dis[i] = d2[k];
          work_id[i] = i2[k] + data2_base;
          k++;
        } else {
          work_dis[i] = d1[j];
          work_id[i] = i1[j];
          j++;
        }
        i++;
      }

      memcpy(d1, work_dis, topk * sizeof(DIS_T));
      memcpy(i1, work_id, topk * sizeof(ID_T));
    }

    delete[] work_dis;
    delete[] work_id;
  }
}