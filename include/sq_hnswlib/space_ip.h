#pragma once
#include "hnswlib.h"

namespace sq_hnswlib {

    static float
    InnerProduct(const void *pVect1, const void *pVect2, const void *qty_ptr,const float* codes,bool both_codes) {
        size_t qty = *((size_t *) qty_ptr);
        float res = 0;
        if (both_codes) {
            for (unsigned i = 0; i < qty; i++) {
                res += codes[(*(uint8_t*)(pVect1))*qty+i] * codes[(*(uint8_t*)(pVect2))*qty+i];
            }
        }
        else{
            for (unsigned i = 0; i < qty; i++) {
                res += ((float*)(pVect1))[i] * codes[(*(uint8_t*)(pVect2))*qty+i];
            }
        }
        return (1.0f - res);

    }


#if defined(USE_AVX)
//// Favor using AVX if available.
    static float
    InnerProductSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr, const float* codes, bool both_codes) {

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


        size_t qty16 = qty / 16;
        size_t qty4 = qty / 4;

        const float *pEnd1 = pVect1 + 16 * qty16;
        const float *pEnd2 = pVect1 + 4 * qty4;

        __m256 sum256 = _mm256_set1_ps(0);

        while (pVect1 < pEnd1) {
            //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

            __m256 v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            __m256 v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
        }

        __m128 v1, v2;
        __m128 sum_prod = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));

        while (pVect1 < pEnd2) {
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
        }

        _mm_store_ps(TmpRes, sum_prod);
        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
        delete [] pVect1;
        delete [] pVect2;
        return 1.0f - sum;
    }


    static float
    InnerProductSIMD16Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr, const float* codes, bool both_codes) {
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

        size_t qty16 = qty / 16;


        const float *pEnd1 = pVect1 + 16 * qty16;

        __m256 sum256 = _mm256_set1_ps(0);

        while (pVect1 < pEnd1) {
            //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

            __m256 v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            __m256 v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
        }

        _mm256_store_ps(TmpRes, sum256);
        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
        delete [] pVect1;
        delete [] pVect2;
        return 1.0f - sum;
    }


    static float
    InnerProductSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr, const float* codes, bool both_codes) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty16 = qty >> 4 << 4;
        float res = InnerProductSIMD16Ext(pVect1v, pVect2v, &qty16, codes, both_codes);
        void *pVect1;
        void *pVect2;
        if(both_codes) {
            pVect1 = (uint8_t*)pVect1v + qty16;
        } else {
            pVect1 = (float*)pVect1v + qty16;
        }
        pVect2 = (uint8_t*)pVect2v + qty16;

        size_t qty_left = qty - qty16;
        float res_tail = InnerProduct(pVect1, pVect2, &qty_left, codes, both_codes);
        return res + res_tail - 1.0f;
    }

    static float
    InnerProductSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr, const float* codes, bool both_codes) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty4 = qty >> 2 << 2;
        void *pVect1;
        void *pVect2;
        float res = InnerProductSIMD4Ext(pVect1v, pVect2v, &qty4, codes, both_codes);
        size_t qty_left = qty - qty4;

        if(both_codes) {
            pVect1 = (uint8_t*)pVect1v + qty4;
        } else {
            pVect1 = (float*)pVect1v + qty4;
        }
        pVect2 = (uint8_t*)pVect2v + qty4;

        float res_tail = InnerProduct(pVect1, pVect2, &qty_left, codes, both_codes);

        return res + res_tail - 1.0f;
    }
#endif

    class InnerProductSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        InnerProductSpace(size_t dim) {
            fstdistfunc_ = InnerProduct;
    #if defined(USE_AVX)
            if (dim % 16 == 0)
                fstdistfunc_ = InnerProductSIMD16Ext;
            else if (dim % 4 == 0)
                fstdistfunc_ = InnerProductSIMD4Ext;
            else if (dim > 16)
                fstdistfunc_ = InnerProductSIMD16ExtResiduals;
            else if (dim > 4)
                fstdistfunc_ = InnerProductSIMD4ExtResiduals;
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

    ~InnerProductSpace() {}
    };


}
