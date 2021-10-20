#pragma once
#include "hnswlib.h"

namespace sq_hnswlib {

    static float
    L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr,const float* codes,bool both_codes) {
        if (both_codes) {
            auto *pVect1 = (uint8_t *) pVect1v;
            auto *pVect2 = (uint8_t *) pVect2v;
            size_t qty = *((size_t *) qty_ptr);

            float res = 0;
            for (size_t i = 0; i < qty; i++) {
                float t = codes[(*pVect1) * qty + i] - codes[(*pVect2) * qty + i];
                pVect1++;
                pVect2++;
                res += t * t;
            }
            return (res);
        }
        else {
            auto *pVect1 = (float *) pVect1v;
            auto *pVect2 = (uint8_t *) pVect2v;
            size_t qty = *((size_t *) qty_ptr);

            float res = 0;
            for (size_t i = 0; i < qty; i++) {
                float t = (*pVect1) - codes[(*pVect2) * qty + i];
                pVect1++;
                pVect2++;
                res += t * t;
            }
            return (res);
        }
    }

#if defined(USE_AVX)

    // Favor using AVX if available.
    static float
    L2SqrSIMD16Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr, const float* codes,bool both_codes) {
        size_t qty = *((size_t *) qty_ptr);
        float *pVect1 = new float [qty];
        float *pVect2 = new float [qty];
        if(both_codes) {
            for (unsigned i = 0; i < qty; i++) {
                pVect1[i] = codes[(*(uint8_t*)(pVect1v))*qty+i];
                pVect2[i] = codes[(*(uint8_t*)(pVect2v))*qty+i];
            }
        } else {
            for (unsigned i = 0; i < qty; i++) {
                pVect1[i] = (*(uint8_t*)(pVect1v))*qty+i;
                pVect2[i] = codes[(*(uint8_t*)(pVect2v))*qty+i];
            }
        }

        float PORTABLE_ALIGN32 TmpRes[8];
        size_t qty16 = qty >> 4;

        const float *pEnd1 = pVect1 + (qty16 << 4);

        __m256 diff, v1, v2;
        __m256 sum = _mm256_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        }

        _mm256_store_ps(TmpRes, sum);
        delete [] pVect1;
        delete [] pVect2;
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
    }

    static float
    L2SqrSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr, const float* codes,bool both_codes) {
        size_t qty = *((size_t *) qty_ptr);
        float *pVect1 = new float [qty];
        float *pVect2 = new float [qty];
        if(both_codes) {
            for (unsigned i = 0; i < qty; i++) {
                pVect1[i] = codes[(*(uint8_t*)(pVect1v))*qty+i];
                pVect2[i] = codes[(*(uint8_t*)(pVect2v))*qty+i];
            }
        } else {
            for (unsigned i = 0; i < qty; i++) {
                pVect1[i] = (*(uint8_t*)(pVect1v))*qty+i;
                pVect2[i] = codes[(*(uint8_t*)(pVect2v))*qty+i];
            }
        }

        float PORTABLE_ALIGN32 TmpRes[8];

        size_t qty4 = qty >> 2;

        const float *pEnd1 = pVect1 + (qty4 << 2);

        __m128 diff, v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }
        _mm_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    }


    static float
    L2SqrSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr, const float* codes,bool both_codes) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty16 = qty >> 4 << 4;
        float res = L2SqrSIMD16Ext(pVect1v, pVect2v, &qty16, codes, both_codes);
        void *pVect1;
        void *pVect2;
        if(both_codes) {
            pVect1 = (uint8_t *)pVect1v + qty16;
        } else {
            pVect1 = (float *)pVect1v + qty16;
        }
        pVect2 = (uint8_t *)pVect2v + qty16;
        size_t qty_left = qty - qty16;
        float res_tail = L2Sqr(pVect1, pVect2, &qty_left, codes, both_codes);
        return (res + res_tail);
    }

    static float
    L2SqrSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr, const float* codes,bool both_codes) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty4 = qty >> 2 << 2;

        float res = L2SqrSIMD4Ext(pVect1v, pVect2v, &qty4, codes, both_codes);
        size_t qty_left = qty - qty4;
        void *pVect1;
        void *pVect2;
        if(both_codes) {
            pVect1 = (uint8_t *)pVect1v + qty4;
        } else {
            pVect1 = (float *)pVect1v + qty4;
        }

        float res_tail = L2Sqr(pVect1, pVect2, &qty_left, codes, both_codes);

        return (res + res_tail);
    }
#endif

    class L2Space : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        L2Space(size_t dim) {
            fstdistfunc_ = L2Sqr;
        #if defined(USE_SSE)
            if (dim % 16 == 0)
                fstdistfunc_ = L2SqrSIMD16Ext;
            else if (dim % 4 == 0)
                fstdistfunc_ = L2SqrSIMD4Ext;
            else if (dim > 16)
                fstdistfunc_ = L2SqrSIMD16ExtResiduals;
            else if (dim > 4)
                fstdistfunc_ = L2SqrSIMD4ExtResiduals;
        #endif
            dim_ = dim;
            data_size_ = dim * sizeof(uint8_t);
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &dim_;
        }

        ~L2Space() {}
    };

//    static int
//    L2SqrI4x(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr) {
//
//        size_t qty = *((size_t *) qty_ptr);
//        int res = 0;
//        unsigned char *a = (unsigned char *) pVect1;
//        unsigned char *b = (unsigned char *) pVect2;
//
//        qty = qty >> 2;
//        for (size_t i = 0; i < qty; i++) {
//
//            res += ((*a) - (*b)) * ((*a) - (*b));
//            a++;
//            b++;
//            res += ((*a) - (*b)) * ((*a) - (*b));
//            a++;
//            b++;
//            res += ((*a) - (*b)) * ((*a) - (*b));
//            a++;
//            b++;
//            res += ((*a) - (*b)) * ((*a) - (*b));
//            a++;
//            b++;
//        }
//        return (res);
//    }
//
//    static int L2SqrI(const void* __restrict pVect1, const void* __restrict pVect2, const void* __restrict qty_ptr) {
//        size_t qty = *((size_t*)qty_ptr);
//        int res = 0;
//        unsigned char* a = (unsigned char*)pVect1;
//        unsigned char* b = (unsigned char*)pVect2;
//
//        for(size_t i = 0; i < qty; i++)
//        {
//            res += ((*a) - (*b)) * ((*a) - (*b));
//            a++;
//            b++;
//        }
//        return (res);
//    }
//
//    class L2SpaceI : public SpaceInterface<int> {
//
//        DISTFUNC<int> fstdistfunc_;
//        size_t data_size_;
//        size_t dim_;
//    public:
//        L2SpaceI(size_t dim) {
//            if(dim % 4 == 0) {
//                fstdistfunc_ = L2SqrI4x;
//            }
//            else {
//                fstdistfunc_ = L2SqrI;
//            }
//            dim_ = dim;
//            data_size_ = dim * sizeof(unsigned char);
//        }
//
//        size_t get_data_size() {
//            return data_size_;
//        }
//
//        DISTFUNC<int> get_dist_func() {
//            return fstdistfunc_;
//        }
//
//        void *get_dist_func_param() {
//            return &dim_;
//        }
//
//        ~L2SpaceI() {}
//    };


}