#pragma once

#include "clusting.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <string.h>

class PQ {
private:
    int32_t d, dsub, K;
    uint8_t m, nbits;

    float* centroids = nullptr;
    uint8_t* codes = nullptr;
public:
    PQ(int32_t _d, uint8_t _m, uint8_t _nbits)
    : d(_d), m(_m), nbits(_nbits) {
        if (d % m != 0) {
            std::cerr << "Dimension d = " << d
                      << "is not divisible by m = " << m
                      << std::endl;
        }

        dsub = d / m;
        K = 1 << nbits;

        centroids = new float[m * K * dsub];
    };

    ~PQ() {
        if (centroids != nullptr) {
            delete[] centroids;
        }
    }

    template<typename T>
    void train(int32_t n, const T* x);

    template<typename T>
    void PQ::encode_vectors(int32_t n, const T* x);

    // T: type of the database vector
    // R: type of the distance
    template<class C, typename T, typename R>
    void search(int32_t n, const T* x, int32_t topk,
                   typename C::T* dist, typename C::TI* labels);
};

template<typename T>
void PQ::train(int32_t n, const T* x) {
    T *xs = new T[n * dsub];

    for (uint8_t i = 0; i < m; ++i) {

        // get slice of x in subspace m_i
        for (int j = 0; j < n; ++j) {
            memcpy(xs + j * dsub, x + j * d + i * dsub, sizeof(T));
        }

        // compute centroids
        kmeans(n, xs, dsub, K, centroids + i * K * dsub);
    }

    delete[] xs;
};

template<typename T>
void PQ::encode_vectors(int32_t n, const T* x) {
    
}

template<class C, typename T, typename R>
void search(int32_t n, const T* x, int32_t topk,
            typename C::T* dist, typename C::TI* labels) {
    
}