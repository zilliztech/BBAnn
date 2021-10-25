#pragma once
#include "hnswlib.h"
#

namespace sq_hnswlib {

static float L2Sqr_float(const void *pVect1v, const void *pVect2v,
                         const void *qty_ptr) {
  float *pVect1 = (float *)pVect1v;
  float *pVect2 = (float *)pVect2v;
  size_t qty = *((size_t *)qty_ptr);

  float res = 0;
  for (size_t i = 0; i < qty; i++) {
    float t = *pVect1 - *pVect2;
    pVect1++;
    pVect2++;
    res += t * t;
  }
  return (res);
}

#if defined(USE_AVX)

// Favor using AVX if available.
static float L2SqrSIMD16Ext(const void *pVect1v, const void *pVect2v,
                            const void *qty_ptr) {
  float *pVect1 = (float *)pVect1v;
  float *pVect2 = (float *)pVect2v;
  size_t qty = *((size_t *)qty_ptr);
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
  return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] +
         TmpRes[6] + TmpRes[7];
}

#elif defined(USE_SSE)

static float L2SqrSIMD16Ext(const void *pVect1v, const void *pVect2v,
                            const void *qty_ptr) {
  float *pVect1 = (float *)pVect1v;
  float *pVect2 = (float *)pVect2v;
  size_t qty = *((size_t *)qty_ptr);
  float PORTABLE_ALIGN32 TmpRes[8];
  size_t qty16 = qty >> 4;

  const float *pEnd1 = pVect1 + (qty16 << 4);

  __m128 diff, v1, v2;
  __m128 sum = _mm_set1_ps(0);

  while (pVect1 < pEnd1) {
    //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    diff = _mm_sub_ps(v1, v2);
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    diff = _mm_sub_ps(v1, v2);
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    diff = _mm_sub_ps(v1, v2);
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

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
#endif

#if defined(USE_SSE) || defined(USE_AVX)
static float L2SqrSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v,
                                     const void *qty_ptr) {
  size_t qty = *((size_t *)qty_ptr);
  size_t qty16 = qty >> 4 << 4;
  float res = L2SqrSIMD16Ext(pVect1v, pVect2v, &qty16);
  float *pVect1 = (float *)pVect1v + qty16;
  float *pVect2 = (float *)pVect2v + qty16;

  size_t qty_left = qty - qty16;
  float res_tail = L2Sqr_float(pVect1, pVect2, &qty_left);
  return (res + res_tail);
}
#endif

#ifdef USE_SSE
static float L2SqrSIMD4Ext(const void *pVect1v, const void *pVect2v,
                           const void *qty_ptr) {
  float PORTABLE_ALIGN32 TmpRes[8];
  float *pVect1 = (float *)pVect1v;
  float *pVect2 = (float *)pVect2v;
  size_t qty = *((size_t *)qty_ptr);

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

static float L2SqrSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v,
                                    const void *qty_ptr) {
  size_t qty = *((size_t *)qty_ptr);
  size_t qty4 = qty >> 2 << 2;

  float res = L2SqrSIMD4Ext(pVect1v, pVect2v, &qty4);
  size_t qty_left = qty - qty4;

  float *pVect1 = (float *)pVect1v + qty4;
  float *pVect2 = (float *)pVect2v + qty4;
  float res_tail = L2Sqr_float(pVect1, pVect2, &qty_left);

  return (res + res_tail);
}
#endif

static float L2Sqr(const void *pVect1, const void *pVect2, const void *qty_ptr,
                   const float *codes, bool both_codes) {
  size_t dim = *((size_t *)qty_ptr);
  std::vector<float> vec1(dim);
  std::vector<float> vec2(dim);
  if (both_codes) {
    for (auto i = 0; i < dim; ++i) {
      // vec1 decode:
      uint8_t vec_code1 = *((uint8_t *)(pVect1) + i);
      uint32_t code_offest1 = i * 256 + vec_code1;
      vec1[i] = codes[code_offest1];
      // vec2 decode:
      uint8_t vec_code2 = *((uint8_t *)(pVect2) + i);
      uint32_t code_offest2 = i * 256 + vec_code2;
      vec2[i] = codes[code_offest2];
    }
  } else {
    vec1.assign((float *)pVect1, (float *)pVect1 + dim);
    for (auto i = 0; i < dim; ++i) {
      // vec2 decode:
      uint8_t vec_code2 = *((uint8_t *)(pVect2) + i);
      uint32_t code_offest2 = i * 256 + vec_code2;
      vec2[i] = codes[code_offest2];
    }
  }

#if defined(USE_SSE) || defined(USE_AVX)
  if (dim % 16 == 0)
    L2SqrSIMD16Ext((void *)vec1.data(), (void *)vec2.data(), qty_ptr);
  else if (dim % 4 == 0)
    L2SqrSIMD4Ext((void *)vec1.data(), (void *)vec2.data(), qty_ptr);
  else if (dim > 16)
    L2SqrSIMD16ExtResiduals((void *)vec1.data(), (void *)vec2.data(), qty_ptr);
  else if (dim > 4)
    L2SqrSIMD4ExtResiduals((void *)vec1.data(), (void *)vec2.data(), qty_ptr);
#else
  L2Sqr_float((void *)vec1.data(), (void *)vec2.data(), qty_ptr);
#endif
}

class L2Space : public SpaceInterface<float> {

  DISTFUNC<float> fstdistfunc_;
  size_t data_size_;
  size_t dim_;

public:
  L2Space(size_t dim) {
    fstdistfunc_ = L2Sqr;
    dim_ = dim;
    data_size_ = dim * sizeof(uint8_t);
  }

  size_t get_data_size() { return data_size_; }

  DISTFUNC<float> get_dist_func() { return fstdistfunc_; }

  void *get_dist_func_param() { return &dim_; }

  ~L2Space() {}
};

} // namespace sq_hnswlib