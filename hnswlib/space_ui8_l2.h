#pragma once
#include "hnswlib.h"
#include "../util/distance.h"

#include <stdlib.h>

namespace hnswlib {

int uit_l2(const void *a, const void *b, const void *d) {
    return L2sqr<const u_int8_t, const u_int8_t, int>((const u_int8_t*)a, (const u_int8_t*)b, *(size_t*)d);
}

class Ui8L2Space : public SpaceInterface<int> {
  private:
    DISTFUNC<int> fstdistfunc_;
    size_t data_size_;
    size_t dim_;
  public:
    Ui8L2Space(size_t dim) {
        fstdistfunc_ = uit_l2;
        dim_ = dim;
        data_size_ = dim;
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<int> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &data_size_;
    }

    ~Ui8L2Space() {}
};

}
