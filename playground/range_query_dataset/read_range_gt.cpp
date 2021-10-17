#include <iostream>
#include <vector>
#include <fstream>
#include <stdint.h>
#include <map>
#include <string>
#include <cassert>

// https://zilliverse.feishu.cn/sheets/shtcnZciTchM20wJtRF5W4t1BGd
// This value is a median in num_results_per_query.
constexpr int pivot = 6505;

template<typename FILE_IDT, typename IDT>
void read_comp_range_search(std::vector<std::vector<IDT>>& query_results_vec, const std::string& input_path, int32_t& num_queries, int32_t& total_res) {
    // The ground truth binary files for range search consist of the following information:
    // num_queries(int32_t) followed by the total number of results total_res(int32_t)
    // followed by num_queries X size(int32_t) bytes corresponding to num_results_per_query for each query,
    // followed by total_res X sizeof(int32_t) bytes corresponding to the IDs of the neighbors of each query one after the other.
    std::ifstream input(input_path, std::ios::binary);
    assert(input.is_open());
    input.read((char*)&num_queries, sizeof(int32_t));
    input.read((char*)&total_res, sizeof(int32_t));
    std::cout << "number of queries: " << num_queries << std::endl;
    std::cout << "total number of results total_res: " << total_res << std::endl;

    query_results_vec.resize(num_queries);
    int32_t total_num_results = 0;
    std::vector<int> offset_50_percentile;
    for (int i = 0; i < num_queries; ++i) {
        int32_t num_results_per_query;
        input.read((char*)&num_results_per_query, sizeof(num_results_per_query));
        query_results_vec[i].resize(num_results_per_query);
        total_num_results += num_results_per_query;
        std::cout << i << "-th query has # results: " << num_results_per_query << std::endl;
        if (num_results_per_query >= pivot) offset_50_percentile.push_back(i);
    }
    std::cout << "CHECKSUM total_num_results: " << total_num_results << std::endl;
    if (total_num_results != total_res) {
        std::cout << "total_num_results != total_res" << std::endl;
        return;
    }

    FILE_IDT t_id;
    for (uint32_t i = 0; i < num_queries; ++i) {
        for (uint32_t j = 0; j < query_results_vec[i].size(); ++j) {
            input.read((char*)&t_id, sizeof(FILE_IDT));
            query_results_vec[i][j] = static_cast<IDT>(t_id);
        }
    }

    input.close();

    std::cout << "====================" << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << "offset_50_percentile: #=" << offset_50_percentile.size() << std::endl;
    std::cout << "{";
    for (const auto& offset : offset_50_percentile) std::cout << offset << ",";
    std::cout << std::endl;
}

int main () {
    std::cout << "Please type in the input file input_path into the std::cin:" << std::endl;
    std::cout << "Example: \"/home/pat/datasets/Facebook-SimSearchNet++/ssnpp-1B-gt\"" << std::endl;
    std::string input_path;
    std::cin >> input_path;
    std::cout << "The input input_path is: " << input_path << std::endl;

    int32_t num_queries, total_res;
    std::vector<std::vector<int32_t>> query_results_vec;
	read_comp_range_search<int32_t, int32_t>(query_results_vec, input_path, num_queries, total_res);

	std::map<int, int> m;
	for (const auto& query_results : query_results_vec) ++m[query_results.size()];

	std::cout << "# Result of Query,# Query" << std::endl;
	for (const auto& pair : m) {
		std::cout << pair.first << "," << pair.second << std::endl;
	}
	return 0;
}
