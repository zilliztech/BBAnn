#pragma once

#include "pq/pq.h"

/*
  For L2
    Term 1: ||q-c||^2 (compute during search)
    Term 2: ||r||^2 + 2cr (Precompute and save alongside each vector)
    Term 3: -2qr (pq table)

  For IP
    qc (compute once for each query)
    qr (LUT)
*/

// TODO: refactor this to inherit from pq and overload some functions

// T: type of the database vector
// U: type of codes ID (PQ representation)
template <class C, typename T, typename U>
class PQResidualQuantizer {
private:
    // for easy access
    int64_t d, K;
    uint32_t m;
public:
    PQResidualQuantizer(int64_t _d, uint32_t _m, uint32_t _nbits);

    ~PQResidualQuantizer();

    ProductQuantizer<C, T, U>* pq;

    uint32_t getM() { return m; }

    void encode_vectors_and_save(
            float*& precomputer_table,
            int64_t n,
            const T* x,
            const float* ivf_cen,
            const std::vector<uint32_t>& buckets,
            uint64_t& bucket_cnt,
            const std::string& file_path,
            MetricType metric_type);

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
        const uint32_t& qid,
        MetricType metric_type);
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

template<class C, typename T, typename U>
void PQResidualQuantizer<C, T, U>::encode_vectors_and_save(
        float*& precomputer_table,
        int64_t n,
        const T* x,
        const float* ivf_cen,
        const std::vector<uint32_t>& buckets,
        uint64_t& bucket_cnt,
        const std::string& file_path,
        MetricType metric_type) {
    assert(ivf_cen != nullptr);

    pq->encode_vectors(precomputer_table, n, x, false, true, ivf_cen);

    if (MetricType::L2 == metric_type) {
        std::vector<float> term2s(n, 0);

        // these codes are local, for each cluster
        const U* c = pq->get_codes();
        float* r = new float[d];
        const float* ivf_c = ivf_cen + bucket_cnt * d;

        // precompute term2
        int64_t cnt = 0;
        for (int i = 0; i < buckets.size(); ++i, ivf_c += d) {
            for (int j = 0; j < buckets[i]; ++j, c += m) {
                pq->reconstruct(r, c);
                term2s[cnt] += norm_L2sqr<const float, float>(r, d);
                term2s[cnt] += 2.0f * IP<const float, const float, float>(ivf_c, r, d);


                ++cnt;
            }
            ++bucket_cnt;
        }
        assert(cnt == n);

        uint32_t wm = m + sizeof(float);
        std::ofstream code_writer(file_path, std::ios::binary | std::ios::out);
        code_writer.write((char*)&n, sizeof(uint32_t));
        code_writer.write((char*)&wm, sizeof(uint32_t));

        c = pq->get_codes();
        const size_t c_size = m * sizeof(U);
        for (int i = 0; i < n; ++i, c += m) {
            code_writer.write((char*)c, c_size);
            code_writer.write((char*)&term2s[i], sizeof(float));
        }

        code_writer.close();
        delete[] r;
        std::cout << "PQResidualQuantizer encode " << n << " vectors with m = " << m << " and term2 into file "
                  << file_path << std::endl;
    } else if (MetricType::IP == metric_type) {
        std::ofstream code_writer(file_path, std::ios::binary | std::ios::out);
        const U* c = pq->get_codes();

        code_writer.write((char*)&n, sizeof(uint32_t));
        code_writer.write((char*)&m, sizeof(uint32_t));
        code_writer.write((char*)c, n * m * sizeof(U));

        code_writer.close();

        std::cout << "ProductQuantizer encode " << n << " vectors with m = " << m << " into file "
                  << file_path << std::endl;
    } else {
        std::cerr << "Unrecognized metric type: " << static_cast<int>(metric_type) << std::endl;
    }
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
        const uint32_t& qid,
        MetricType metric_type) {

    assert(precompute_table != nullptr);
    const float* dis_tab = precompute_table;

    auto* __restrict val_ = values;
    auto* __restrict ids_ = labels;

    if (heapify)
        heap_heapify<C>(topk, val_, ids_);
    const U* c = pcodes;


    // term 1
    float term1 = L2sqr<const T, const float, float>(q, centroid, d);

    if (MetricType::L2 == metric_type) {
        for (int j = 0; j < n; ++j) {
            float dis = term1;

            // term 3
            const float* __restrict dt = dis_tab;
            float term3 = 0;
            for (int mm = 0; mm < m; ++mm, dt += K) {
                term3 += dt[*c++];
            }
            dis -= 2.0f * term3;

            // term 2
            dis += *reinterpret_cast<const float *>(c);


            c += sizeof(float);


            if (C::cmp(val_[0], dis)) {
                heap_swap_top<C>(topk, val_, ids_, dis, gen_refine_id(cid, off + j, qid));
            }
        }
    } else {
        // compute qc
        float qc = IP<const T, const float, float>(q, centroid, d);

        for (int j = 0; j < n; ++j) {
            float dis = qc;
            // compute qr
            const float* __restrict dt = dis_tab;
            for (int mm = 0; mm < m; ++mm, dt += K) {
                dis += dt[*c++];
            }

            if (C::cmp(val_[0], dis)) {
                heap_swap_top<C>(topk, val_, ids_, dis, gen_refine_id(cid, off + j, qid));
            }
        }
    }

    if (reorder)
        heap_reorder<C>(topk, val_, ids_);
}