#pragma once
#include <stddef.h>
#include <stdio.h>
#include <immintrin.h>
#include <stdint.h>
#include <assert.h>


// Data type: T1, T2
// Distance type: R

template<typename T1, typename T2, typename R>
R L2sqr(T1 *a, T2 *b, size_t n) {
    size_t i = 0;
    R dis = 0, dif;
    switch(n & 7) {
        default:
            while (n > 7) {
                n -= 8; dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
                case 7: dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
                case 6: dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
                case 5: dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
                case 4: dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
                case 3: dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
                case 2: dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
                case 1: dif=(R)a[i]-(R)b[i]; dis+=dif*dif; i++;
            }
    }
    return dis;
}


template<>
float L2sqr<float, float, float>(float *a, float *b, size_t n) {
    __m256 msum1 = _mm256_setzero_ps();

    while (n >= 8) {
        __m256 mx = _mm256_loadu_ps (a); a += 8;
        __m256 my = _mm256_loadu_ps (b); b += 8;
        const __m256 a_m_b1 = mx - my;
        msum1 += a_m_b1 * a_m_b1;
        n -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 +=       _mm256_extractf128_ps(msum1, 0);

    if (n >= 4) {
        __m128 mx = _mm_loadu_ps (a); a += 4;
        __m128 my = _mm_loadu_ps (b); b += 4;
        const __m128 a_m_b1 = mx - my;
        msum2 += a_m_b1 * a_m_b1;
        n -= 4;
    }

    msum2 = _mm_hadd_ps (msum2, msum2);
    msum2 = _mm_hadd_ps (msum2, msum2);
    float dis = _mm_cvtss_f32 (msum2);

    if (n > 0) {
        float dif;
        switch (n) {
            case 3:
                dif = *(a+2) - *(b+2); dis += dif *dif;
            case 2:
                dif = *(a+1) - *(b+1); dis += dif * dif;
            case 1:
                dif = *a - *b; dis += dif * dif;
        }

    }
    return  dis;
}


template<>
int L2sqr<uint8_t, uint8_t, int>(uint8_t *a, uint8_t *b, size_t n) {
    __m256i msum1 = _mm256_setzero_si256();

    while (n >= 16) {
        __m128i charx = _mm_loadu_si128((__m128i*) a); a += 16;
        __m128i chary = _mm_loadu_si128((__m128i*) b); b += 16;

        __m256i shortx = _mm256_cvtepu8_epi16 (charx);
        __m256i shorty = _mm256_cvtepu8_epi16 (chary);

        __m256i temp = _mm256_madd_epi16(shortx, shortx) + _mm256_madd_epi16(shorty, shorty) -2* _mm256_madd_epi16(shortx, shorty);
        msum1 = msum1 + temp;
        //shortx = _mm256_subs_epi16(shortx, shorty);
        //msum1 += _mm256_madd_epi16(shortx, shortx);
        n -= 16;
    }

    int16_t xshortbuffer[16];
    int16_t yshortbuffer[16];
    if(n > 0) {
        for(int i=0; i< 16; i++) {
            if(i<n) {
                xshortbuffer[i] = (int16_t)a[i];
                yshortbuffer[i] = (int16_t)b[i];
            } else {
                xshortbuffer[i] = 0;
                yshortbuffer[i] = 0;
            }
        }

        __m256i shortx = _mm256_loadu_si256((__m256i*) xshortbuffer); a += 16;
        __m256i shorty = _mm256_loadu_si256((__m256i*) yshortbuffer); b += 16;

        __m256i temp = _mm256_madd_epi16(shortx, shortx) + _mm256_madd_epi16(shorty, shorty) -2* _mm256_madd_epi16(shortx, shorty);
        msum1 = msum1 + temp;

    }

    __m128i msum2 = _mm256_extractf128_si256(msum1, 1);
    msum2 +=       _mm256_extractf128_si256(msum1, 0);

    msum2 = _mm_hadd_epi32 (msum2, msum2);
    msum2 = _mm_hadd_epi32 (msum2, msum2);
    return _mm_cvtsi128_si32(msum2);
}

template<>
int L2sqr<int8_t, int8_t, int>(int8_t *a, int8_t *b, size_t n) {
    __m256i msum1 = _mm256_setzero_si256();

    while (n >= 16) {

        __m128i charx = _mm_loadu_si128((__m128i*) a); a += 16;
        __m128i chary = _mm_loadu_si128((__m128i*) b); b += 16;

        __m256i shortx = _mm256_cvtepi8_epi16 (charx);
        __m256i shorty = _mm256_cvtepi8_epi16 (chary);

        //	__m256i temp = _mm256_madd_epi16(shortx, shortx) + _mm256_madd_epi16(shorty, shorty) -2* _mm256_madd_epi16(shortx, shorty);
        //	msum1 = msum1 + temp;
        shortx =_mm256_subs_epi16(shortx, shorty);
        msum1 += _mm256_madd_epi16(shortx, shortx);
        n -= 16;
    }

    if(n > 0) {
        int16_t xshortbuffer[16];
        int16_t yshortbuffer[16];
        for(int i=0; i< 16; i++) {
            if(i < n) {
                xshortbuffer[i] =(int16_t) a[i];
                yshortbuffer[i] =(int16_t) b[i];
            } else
            {
                xshortbuffer[i] = 0;
                yshortbuffer[i] = 0;
            }
        }

        __m256i shortx = _mm256_loadu_si256((__m256i*) xshortbuffer); a += 16;
        __m256i shorty = _mm256_loadu_si256((__m256i*) yshortbuffer); b += 16;

        __m256i temp = _mm256_madd_epi16(shortx, shortx) + _mm256_madd_epi16(shorty, shorty) -2* _mm256_madd_epi16(shortx, shorty);
        msum1 = msum1 + temp;

    }

    __m128i msum2 = _mm256_extractf128_si256(msum1, 1);
    msum2 +=       _mm256_extractf128_si256(msum1, 0);

    msum2 = _mm_hadd_epi32 (msum2, msum2);
    msum2 = _mm_hadd_epi32 (msum2, msum2);
    return _mm_cvtsi128_si32(msum2);
}

template<>
float L2sqr<int8_t, float, float>(int8_t *a, float *b, size_t n) {
    float afloatbuffer[256];
    for (int i=0; i< n; i++) {
        afloatbuffer[i] = (float) a[i];
    }
    return L2sqr<float, float, float>(afloatbuffer, b, n);
}

template<>
float L2sqr<uint8_t, float, float>(uint8_t *a, float *b, size_t n) {
    float afloatbuffer[256];
    for (int i=0; i< n; i++) {
        afloatbuffer[i] = (float) a[i];
    }
    return L2sqr<float, float, float>(afloatbuffer, b, n);
}

template<typename T1, typename T2, typename R>
R IP(T1 *a, T2 *b, size_t n) {
    size_t i = 0;
    R dis = 0;
    switch(n & 7) {
        default:
            while (n > 7) {
                n -= 8; dis+=(R)a[i]*(R)b[i]; i++;
                case 7: dis+=(R)a[i]*(R)b[i]; i++;
                case 6: dis+=(R)a[i]*(R)b[i]; i++;
                case 5: dis+=(R)a[i]*(R)b[i]; i++;
                case 4: dis+=(R)a[i]*(R)b[i]; i++;
                case 3: dis+=(R)a[i]*(R)b[i]; i++;
                case 2: dis+=(R)a[i]*(R)b[i]; i++;
                case 1: dis+=(R)a[i]*(R)b[i]; i++;
            }
    }
    return dis;
}

template<typename T1, typename T2, typename R>
void compute_residual(T1* x, T2* c, R* res, size_t d) {
    size_t i = 0;
    switch(d & 7) {
        default:
            while (d > 7) {
                d -= 8; res[i]=(R)x[i]-(R)c[i]; i++;
                case 7: res[i]=(R)x[i]-(R)c[i]; i++;
                case 6: res[i]=(R)x[i]-(R)c[i]; i++;
                case 5: res[i]=(R)x[i]-(R)c[i]; i++;
                case 4: res[i]=(R)x[i]-(R)c[i]; i++;
                case 3: res[i]=(R)x[i]-(R)c[i]; i++;
                case 2: res[i]=(R)x[i]-(R)c[i]; i++;
                case 1: res[i]=(R)x[i]-(R)c[i]; i++;
            }
    }
}

template<>
float IP<float, float, float>(float* a, float* b, size_t n) {
    __m256 msum1 = _mm256_setzero_ps();

    while (n >= 8) {
        __m256 mx = _mm256_loadu_ps (a); a += 8;
        __m256 my = _mm256_loadu_ps (b); b += 8;
        msum1 = _mm256_fmadd_ps(mx, my, msum1);
        n -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 +=       _mm256_extractf128_ps(msum1, 0);

    if (n >= 4) {
        __m128 mx = _mm_loadu_ps (a); a += 4;
        __m128 my = _mm_loadu_ps (b); b += 4;
        msum2 = _mm_fmadd_ps(mx, my, msum2);
        n -= 4;
    }

    msum2 = _mm_hadd_ps (msum2, msum2);
    msum2 = _mm_hadd_ps (msum2, msum2);
    float dis =_mm_cvtss_f32 (msum2);

    if (n > 0) {
        switch (n) {
            case 3:
                dis += (*(a+2)) * (*(b+2));
            case 2:
                dis += (*(a+1)) * (*(b+1));
            case 1:
                dis += (*a) * (*b);
        }
    }

    return  dis;
}

// A vector multiply a matrix
// args:　
// a: subquery vector;
// b: m centroids vectors, b is m*n matrix;
// c: result;
// n: sub_dim;
// m: the number of centroids, m is divisible by 32.
template<typename T1>
void compute_lookuptable_IP(T1* a, float* b, float* c, size_t n, size_t m) {
    float* a_buffer = new float[n];
    for (int i=0; i<n; i++) {
        a_buffer[i] = (float) a[i];
    }

    compute_lookuptable_IP<float>(a_buffer, b, c, n, m);

    delete[] a_buffer;
}

template<>
void compute_lookuptable_IP<float>(float* a, float* b, float* c, size_t n, size_t m) {

    size_t offest = 0;
    __m256 msum1, msum2, msum3, msum4;

    while ( offest < m ) {
        size_t dim = 0;
        float* y = b + offest;

        msum1 = _mm256_setzero_ps();
        msum2 = _mm256_setzero_ps();
        msum3 = _mm256_setzero_ps();
        msum4 = _mm256_setzero_ps();

        while ( dim < n ) {

            __m256 mx =  _mm256_set1_ps(*(a+dim));

            __m256 my1 =  _mm256_loadu_ps (y);     //b0-7;
            __m256 my2 =  _mm256_loadu_ps (y + 8); //b8-15;
            __m256 my3 =  _mm256_loadu_ps (y + 16);//b16-23;
            __m256 my4 =  _mm256_loadu_ps (y + 24);//b24-31;

            msum1 = _mm256_fmadd_ps(mx, my1, msum1);
            msum2 = _mm256_fmadd_ps(mx, my2, msum2);
            msum3 = _mm256_fmadd_ps(mx, my3, msum3);
            msum4 = _mm256_fmadd_ps(mx, my4, msum4);

            y = y + m;
            dim ++;
        }

        _mm256_storeu_ps(c, msum1);
        _mm256_storeu_ps(c + 8,  msum2);
        _mm256_storeu_ps(c + 16, msum3);
        _mm256_storeu_ps(c + 24, msum4);
        offest = offest + 32;
        c += 32;
    }
    return ;
}

template<typename T1>
void compute_lookuptable_L2(T1* a, float* b, float* c, size_t n, size_t m) {
    float* a_buffer = new float[n];
    for (int i=0; i<n; i++) {
        a_buffer[i] = (float) a[i];
    }

    compute_lookuptable_L2<float>(a_buffer, b, c, n, m);

    delete[] a_buffer;
}


template<>
void compute_lookuptable_L2<float>(float* a, float* b, float* c, size_t n, size_t m) {

    size_t offest = 0;
    __m256 msum1, msum2, msum3, msum4;

    while ( offest < m ) {
        size_t dim = 0;
        float* y = b + offest;

        msum1 = _mm256_setzero_ps();
        msum2 = _mm256_setzero_ps();
        msum3 = _mm256_setzero_ps();
        msum4 = _mm256_setzero_ps();

        while ( dim < n ) {

            __m256 mx =  _mm256_set1_ps(*(a+dim));

            __m256 my1 =  _mm256_loadu_ps (y);     //b0-7;
            __m256 my2 =  _mm256_loadu_ps (y + 8); //b8-15;
            __m256 my3 =  _mm256_loadu_ps (y + 16);//b16-23;
            __m256 my4 =  _mm256_loadu_ps (y + 24);//b24-31;

            my1 = _mm256_sub_ps(mx, my1);
            msum1 = _mm256_fmadd_ps(my1, my1, msum1);
            my2 = _mm256_sub_ps(mx, my2);
            msum2 = _mm256_fmadd_ps(my2, my2, msum2);
            my3 = _mm256_sub_ps(mx, my3);
            msum3 = _mm256_fmadd_ps(my3, my3, msum3);
            my4 = _mm256_sub_ps(mx, my4);
            msum4 = _mm256_fmadd_ps(my4, my4, msum4);


            y = y + m;
            dim ++;
        }

        _mm256_storeu_ps(c, msum1);
        _mm256_storeu_ps(c + 8,  msum2);
        _mm256_storeu_ps(c + 16, msum3);
        _mm256_storeu_ps(c + 24, msum4);
        offest = offest + 32;
        c += 32;
    }
    return ;
}

void matrix_transpose(const float* src, float* des, int64_t row, int64_t col) {
    assert(src != nullptr);
    assert(des != nullptr);

    for (int64_t i = 0; i < row; ++i) {
        for (int64_t j = 0; j < col; ++j) {
            *(des + j * row + i) = *(src + i * col + j);
        }
    }
}