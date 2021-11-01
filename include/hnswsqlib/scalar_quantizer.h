#pragma once
#include "hnswlib.h"
#include "util/utils_inline.h"
#include <vector>

namespace hnswsqlib {
    class ScalarQuantizer{

        std::vector<float> codebook_;
        size_t dim_code_num_;
        size_t total_codes_num_;
        size_t codes_size_;
        size_t dim_;

    public:
        ScalarQuantizer(size_t dim) {
            dim_ = dim;
            dim_code_num_ = 2;
            total_codes_num_ = dim_ * dim_code_num_;
            codes_size_ = total_codes_num_ * sizeof(float);
            codebook_.assign(total_codes_num_, 0.0);
        }

        ScalarQuantizer(size_t dim, float* codebook) {
            dim_ = dim;
            dim_code_num_ = 2;
            total_codes_num_ = dim_ * dim_code_num_;
            codes_size_ = total_codes_num_ * sizeof(float);
            codebook_.assign(codebook, codebook + total_codes_num_);
        }

        inline void train_codebook(float* data, int64_t n) {
            bbann::train_code<float>(codebook_.data(), (float*)(codebook_.data() + dim_), data, n, (uint32_t)dim_);
        }

        void encode_code(float* data, uint8_t* codes, int64_t n) {
            bbann::encode_uint8<float>(codebook_.data(), codebook_.data() + dim_, data, codes, n, (uint32_t)dim_);
        }

        void decode_code(float* data, uint8_t* codes, int64_t n) {
            bbann::decode_uint8<float>(codebook_.data(), codebook_.data() + dim_, data, codes, n, (uint32_t)dim_);
        }

        size_t get_codebook_size() {
            return codes_size_;
        }

        size_t get_dim() {
            return dim_;
        }

        float* get_codebook() {
            return codebook_.data();
        }

        size_t get_dim_code_num() {
            return dim_code_num_;
        }

        void set_codes(float* max_array, float* min_array) {
            memcpy(codebook_.data(), max_array, sizeof(float) * dim_);
            memcpy(codebook_.data() + dim_, min_array, sizeof(float) * dim_);
        }

        ~ScalarQuantizer() {}
    };


}