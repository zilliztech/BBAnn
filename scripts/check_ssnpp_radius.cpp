#include <iostream>
#include <vector>
#include <fstream>
#include <stdint.h>
#include <map>
#include <string>

#include "util/utils.h"
#include "util/distance.h"

std::string gt_path = "/home/pat/datasets/Facebook-SimSearchNet++/ssnpp-10M";
std::string q_path = "/home/pat/datasets/Facebook-SimSearchNet++/FB_ssnpp_public_queries.u8bin";
std::string base_path = "/home/pat/datasets/Facebook-SimSearchNet++/FB_ssnpp_database.10M.u8bin";

int main () {
	std::vector<std::vector<int32_t>> v;
	int32_t nq, total_res;
	read_comp_range_search<int32_t, int32_t>(v, gt_path, nq, total_res);

    uint8_t *xq = nullptr;
    uint32_t nxq, dim;
    read_bin_file(q_path, xq, nxq, dim);

    auto fh = std::ifstream(base_path, std::ios::binary);
    const uint64_t beg = 2 * sizeof(uint32_t);
    const uint64_t vec_size = sizeof(uint8_t) * dim;

    uint64_t cnt = 0, upper_cnt = 0;

    char *vec = new char[vec_size];

    for (auto i = 0; i < nq; ++i) {
        uint8_t *q = xq + i * dim;
        for (const auto& id : v[i]) {
            fh.seekg(beg + id * vec_size);
            fh.read(vec, vec_size);

            auto dis = L2sqr<const uint8_t, const uint8_t, uint32_t>(q, reinterpret_cast<uint8_t*>(vec), dim);

            if (dis > 60000) {
                // std::cout << "radius greater than 60000 " << dis << std::endl;
                // goto outter;
                ++cnt;

                if (dis > 96237) {
                    ++upper_cnt;
                }
            }
        }
    }

    std::cout << "> 60000: " << cnt << " > 96237: " << upper_cnt << std::endl;

outter:
    delete[] vec;

	return 0;
}
