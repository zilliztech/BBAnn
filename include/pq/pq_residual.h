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
    ProductQuantizer<C, T, U>* pq;

    // for easy access
    int64_t d, K;
    uint32_t m;

    void compute_residual(const T*, const float* x, float* residual);
public:
    PQResidualQuantizer(int64_t _d, uint32_t _m, uint32_t _nbits);

    ~PQResidualQuantizer();

    void train(int64_t n, const T* x);
    
    void save_centroids(const std::string& save_file);

    void load_centroids(const std::string& load_file);

    void encode_vectors_and_save(
            float*& precomputer_table,
            int64_t n,
            const T* x,
            const float* ivf_centroids,
            const std::string& file_path);

    void search(
        float* precompute_table,
        const T* q,
        const float* centroid,
        const U* pcodes,
        int64_t n,
        int64_t topk,
        typename C::T* values,
        typename C::TI* labels,
        PQ_Computer<T> computer,
        bool reorder, 
        bool heapify,
        const uint32_t& cid, 
        const uint32_t& off,
        const uint32_t& qid);
};

template <class C, typename T, typename U>
PQResidualQuantizer<C, T, U>::PQResidualQuantizer(
        int64_t _d,
        uint32_t _m,
        uint32_t _nbits)
            : d(_d), m(_m) {
    pq = new ProductQuantizer<C, T, U>(_d, _m, _nbits);
    K = 1 << _nbits;
}

template <class C, typename T, typename U>
PQResidualQuantizer<C, T, U>::~PQResidualQuantizer() {
    delete pq;
}

template <class C, typename T, typename U>
void PQResidualQuantizer<C, T, U>::train(int64_t n, const T* x) {
    pq->train(n, x);
}

template <class C, typename T, typename U>
void PQResidualQuantizer<C, T, U>::save_centroids(const std::string& save_file) {
    pq->save_centroids(save_file);
}    

template <class C, typename T, typename U>
void PQResidualQuantizer<C, T, U>::load_centroids(const std::string& load_file) {
    pq->load_centroids(load_file);
}

template<class C, typename T, typename U>
void PQResidualQuantizer<C, T, U>::encode_vectors_and_save(
        float*& precomputer_table,
        int64_t n,
        const T* x,
        const float* ivf_centroids,
        const std::string& file_path) {
    pq->encode_vectors(precomputer_table, n, x, false);
    std::vector<float> term2s(n, 0);
    const U* c = pq->get_codes();
    const float* ivf_c = ivf_centroids;
    float* r = new float[d];

    // precompute term2
    for (int i = 0; i < n; ++i, c += m, ivf_c += d) {
        pq->reconstruct(r, c);
        term2s[i] += norm_L2sqr<U, float>(r, d);
        term2s[i] += 2.0f * IP(ivf_c, r, d);
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
    delete[] r;
    std::cout << "PQResidualQuantizer encode " << n << " vectors with m = " << m << "and term2 into file "
              << file_path << std::endl;
}

template <class C, typename T, typename U>
void PQResidualQuantizer<C, T, U>::search(
        float* precompute_table,
        const T* q,
        const float* centroid,
        const U* pcodes,
        int64_t n,
        int64_t topk,
        typename C::T* values,
        typename C::TI* labels,
        PQ_Computer<T> computer,
        bool reorder, 
        bool heapify,
        const uint32_t& cid, 
        const uint32_t& off,
        const uint32_t& qid) {

    assert(precompute_table != nullptr);
    const float* dis_tab = precompute_table;

    auto* __restrict val_ = values;
    auto* __restrict ids_ = labels;

    if (heapify)
        heap_heapify<C>(topk, val_, ids_);
    const U* c = pcodes;

    for (int j = 0; j < n; ++j) {
        // term 1
        float dis = L2sqr<T, float, float>(q, centroid, d);

        // term 2
        dis += *reinterpret_cast<float *>(pcodes + m);

        // term 3
        const float* __restrict dt = dis_tab;
        float term3 = 0;
        for (int mm = 0; mm < m; ++mm, dt += K) {
            term3 += dt[*c++];
        }
        dis -= 2.0f * term3;

        if (C::cmp(val_[0], dis)) {
            heap_swap_top<C>(topk, val_, ids_, dis, gen_refine_id(cid, off + j, qid));
        }
    }
    if (reorder)
        heap_reorder<C>(topk, val_, ids_);
}