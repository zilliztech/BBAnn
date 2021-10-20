#pragma once
#include "hnswlib/hnswlib.h"
#include "util/distance.h"

#include <stdlib.h>

namespace hnswlib {
    int32_t int_l2(const void *a, const void *b, const void *d) {
        return ::L2sqr<int8_t, int8_t, int32_t>((int8_t*)a, (int8_t*)b, *(size_t*)d);
    }

    template<>
    class L2Space<int8_t, int32_t> : public SpaceInterface<int32_t> {
    private:
        DISTFUNC<int32_t> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        L2Space(size_t dim) {
            fstdistfunc_ = int_l2;
            dim_ = dim;
            data_size_ = dim;
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<int32_t> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &data_size_;
        }

        ~L2Space() {}
    };

    uint32_t uint_l2(const void *a, const void *b, const void *d) {
        return ::L2sqr<uint8_t, uint8_t, uint32_t>((uint8_t*)a, (uint8_t*)b, *(size_t*)d);
    }

    template<>
    class L2Space<uint8_t, uint32_t> : public SpaceInterface<uint32_t> {
    private:
        DISTFUNC<uint32_t> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        L2Space(size_t dim) {
            fstdistfunc_ = uint_l2;
            dim_ = dim;
            data_size_ = dim;
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<uint32_t> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &data_size_;
        }

        ~L2Space() {}
    };

}
