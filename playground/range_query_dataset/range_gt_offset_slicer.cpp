// ----------------------------------------------------------------------------------------------------
#include <sys/mman.h>
#include <cassert>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <set>
//---------------------------------------------------------------------------
namespace {
//---------------------------------------------------------------------------
template<typename FILE_IDT, typename IDT>
void read_ground_truth(const std::string& input_path, const std::string& output_path) {
    // Dataset SSN (range query): query with number of results >= 50 percentile num_results_per_query
    std::vector<int> offset{623,648,878,1336,1650,1962,2284,2391,3275,3532,3542,4135,4389,5118,5720,5751,6204,7027,7306,9459,9477,10381,10626,11261,12508,13210,13273,13502,13667,14615,15172,17451,17584,18202,18413,18594,19377,19973,20102,20467,20560,20566,21212,21488,21556,22075,22505,22794,22944,22994,23845,23977,24256,24286,25282,25728,26012,26310,26658,27034,28810,28888,29032,29088,29329,30385,30900,31576,32371,32633,33327,34140,34174,34189,34190,34264,34522,34986,35568,35707,35938,36260,36320,36402,36520,38185,38797,39095,40110,40762,40963,41306,41329,41385,41417,41440,43460,43682,44874,45252,45408,45423,45641,45782,45847,46353,46792,46960,47594,47893,47898,48006,48269,48717,48793,48903,49784,50064,50066,50157,50229,50635,51291,51319,51753,52329,52579,52611,53349,53550,54081,55093,56664,56799,57515,57841,58244,58510,59733,60363,60479,60663,60745,61491,61962,62234,62435,62785,63278,63282,63632,63754,64336,64915,65416,65437,65588,65679,67560,68238,68365,68826,68837,69013,69695,69907,69986,70685,70790,70800,71193,71325,71346,71385,71689,72490,72618,72692,72757,72881,73185,73330,73334,73684,73765,75841,76342,76417,76470,76474,77018,78065,78194,78353,78370,78596,79123,79287,80340,80609,81312,81321,82184,82488,82512,83471,83626,83747,84046,84200,84340,84431,84484,84566,85089,85269,85280,85760,86135,86751,87093,87826,88130,90006,90053,90144,90541,90543,91952,92256,92980,93548,93679,93743,94168,94835,95274,95673,95731,96532,96898,97087,98047,98539,99099,99588};

    std::set<int> ht(offset.begin(), offset.end());
    if (ht.size() != offset.size()) {
        std::cout << "SIZE MISMATCH!" << std::endl;
        return;
    }
    int32_t slice_size = ht.size();

    // The ground truth binary files for range search consist of the following information:
    // num_queries(int32_t) followed by the total number of results total_res(int32_t)
    // followed by num_queries X size(int32_t) bytes corresponding to num_results_per_query for each query,
    // followed by total_res X sizeof(int32_t) bytes corresponding to the IDs of the neighbors of each query one after the other.
    std::ifstream input(input_path, std::ios::binary);
    assert(input.is_open());
    int32_t num_queries, total_res;
    input.read((char*)&num_queries, sizeof(int32_t));
    input.read((char*)&total_res, sizeof(int32_t));
    std::cout << "number of queries: " << num_queries << std::endl;
    std::cout << "total number of results total_res: " << total_res << std::endl;

    std::ofstream output(output_path, std::ios::binary);
    assert(output.is_open());
    std::cout << "number of slices: " << slice_size << std::endl;
    output.write(reinterpret_cast<const char*>(&slice_size), sizeof(slice_size));

    std::vector<std::vector<int32_t>> query_results_vec;
    query_results_vec.resize(num_queries);
    int32_t total_num_results = 0;
    for (int i = 0; i < num_queries; ++i) {
        int32_t num_results_per_query;
        input.read((char*)&num_results_per_query, sizeof(int32_t));
        query_results_vec[i].resize(num_results_per_query);
        total_num_results += num_results_per_query;
    }
    std::cout << "CHECKSUM total_num_results: " << total_num_results << std::endl;

    int32_t slice_total_num_results = 0;
    for (const auto& index : ht) {
        slice_total_num_results += query_results_vec[index].size();
    }
    std::cout << "slice_total_num_results: " << slice_total_num_results << std::endl;

    output.write(reinterpret_cast<const char*>(&slice_total_num_results), sizeof(slice_total_num_results));
    for (const auto& index : ht) {
        int32_t num_results_per_query = query_results_vec[index].size();
        output.write(reinterpret_cast<const char*>(&num_results_per_query), sizeof(num_results_per_query));
    }

    int32_t t_id;
    for (size_t i = 0; i < num_queries; ++i) {
        if (ht.find(i) != ht.end()) {
            std::cout << "i:" << i << " | size:"  << query_results_vec[i].size() << std::endl;
            for (uint32_t j = 0; j < query_results_vec[i].size(); ++j) {
                input.read((char*)&t_id, sizeof(t_id));
                output.write(reinterpret_cast<const char*>(&t_id), sizeof(t_id));
            }
        } else {
            for (uint32_t j = 0; j < query_results_vec[i].size(); ++j) {
                input.read((char*)&t_id, sizeof(t_id));
            }
        }
    }

    input.close();
    output.close();
}
//---------------------------------------------------------------------------
} // namespace
//---------------------------------------------------------------------------
int main() {
    std::cout << "ONLY Range Search Dataset as Input" << std::endl;
    std::cout << "Please type in the input file input_path into the std::cin:" << std::endl;
    std::cout << "Example: \"/home/pat/datasets/Facebook-SimSearchNet++/ssnpp-1B-gt\"" << std::endl;
    std::string input_path;
    std::cin >> input_path;
    std::cout << "The input input_path is: " << input_path << std::endl;

    std::cout << "Please type in the output file input_path into the std::cin:" << std::endl;
    std::cout << "Example: \"/home/jigao/Desktop/slice.u8bin\"" << std::endl;
    std::string output_path;
    std::cin >> output_path;
    std::cout << "The output_path is: " << output_path << std::endl;

    read_ground_truth<int32_t, int32_t>(input_path, output_path);
    return 0;
}
//---------------------------------------------------------------------------
