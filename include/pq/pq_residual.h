#pragma once

#include "pq/pq.h"

/*
    Term 1: ||q-c||^2 (compute during search)
    Term 2: ||r||^2 + 2cr (Precompute and save alongside each vector)
    Term 3: -2qr (pq table)
*/

// T: type of the database vector
// U: type of codes ID (PQ representation)
template <class C, typename T, typename U>
class PQResidualQuantizer {
private:
    ProductQuantizer* pq;

    // for easy access
    int32_t d;
    uint8_t m;

    void compute_residual(const T*, const float* x, float* residual);
public:
    PQResidualQuantizer(int32_t _d, uint8_t _m, uint8_t _nbits);

    ~PQResidualQuantizer();

    void train(int32_t n, const T* x);
    
    void save_centroids(const std::string& save_file);

    void encode_vectors_and_save(
            float*& precomputer_table,
            int32_t n,
            const T* x,
            const float* ivf_centroids,
            const std::string& file_path);
};

template <class C, typename T, typename U>
PQResidualQuantizer<C, T, U>::PQResidualQuantizer(
        int32_t _d,
        uint8_t _m,
        uint8_t _nbits)
            : d(_d), m(_m) {
    pq = new ProductQuantizer(_d, _m, _nbits);
}

template <class C, typename T, typename U>
PQResidualQuantizer<C, T, U>::~PQResidualQuantizer() {
    delete pq;
}

template <class C, typename T, typename U>
void PQResidualQuantizer<C, T, U>::train(int32_t n, const T* x) {
    pq->train(n, x);
}

template <class C, typename T, typename U>
void PQResidualQuantizer<C, T, U>::save_centroids(const std::string& save_file) {
    pq->save_centroids((save_file);
}

template<class C, typename T, typename U>
void PQResidualQuantizer<C, T, U>::encode_vectors_and_save(
        float*& precomputer_table,
        int32_t n,
        const T* x,
        const float* ivf_centroids,
        const std::string& file_path) {
    pq->encode_vectors(precomputer_table, n, x, false);
    std::vector<float> term2s(n, 0);
    const U* c = pq->get_codes();
    const float* ivf_c = ivf_centroids;

    // precompute term2
    for (int i = 0; i < n; ++i, c += m) {
        // TODO: reconstruct
        term2 += norm_L2sqr<U, float>(c, d);
        term2 += 2.0f * IP(ivf_c, c, d);
    }

    uint32_t wm = m + sizeof(float);
    std::ofstream code_writer(file_path, std::ios::binary | std::ios::out);
    code_writer.write((char*)&n, sizeof(uint32_t));
    code_writer.write((char*)&wm, sizeof(uint32_t));

    c = pq->get_codes();
    const float* t2 = term2s.data();
    const size_t c_size = m * sizeof(U);
    for (int i = 0; i < n; ++i, c += m, ++t2) {
        code_writer.write((char*)c, c_size);
        code_writer.write((char*)t2, sizeof(float));
    }
    code_writer.close();
    std::cout << "PQResidualQuantizer encode " << n << " vectors with m = " << m << "and term2 into file "
              << file_path << std::endl;
}

template <class C, typename T, typename U>
void search() {

}