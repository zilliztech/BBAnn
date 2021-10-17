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
//---------------------------------------------------------------------------
namespace {
//---------------------------------------------------------------------------
void read_ground_truth(const std::string& input_path, const std::string& output_path) {
        // OFFSET with distance <= 5000
//        std::vector<int> offset{
//                3
//                ,50
//                ,101
//                ,111
//                ,140
//                ,163
//                ,249
//                ,293
//                ,425
//                ,434
//                ,436
//                ,484
//                ,493
//                ,595
//                ,601
//                ,646
//                ,665
//                ,689
//                ,703
//                ,713
//                ,752
//                ,846
//                ,849
//                ,850
//                ,984
//                ,1166
//                ,1186
//                ,1193
//                ,1258
//                ,1310
//                ,1443
//                ,1521
//                ,1584
//                ,1652
//                ,1767
//                ,1774
//                ,1780
//                ,1786
//                ,1809
//                ,1826
//                ,1831
//                ,1845
//                ,1893
//                ,1986
//                ,2055
//                ,2078
//                ,2092
//                ,2123
//                ,2129
//                ,2162
//                ,2287
//                ,2322
//                ,2388
//                ,2393
//                ,2438
//                ,2480
//                ,2500
//                ,2521
//                ,2574
//                ,2575
//                ,2713
//                ,2746
//                ,2793
//                ,2875
//                ,2881
//                ,2954
//                ,3019
//                ,3053
//                ,3142
//                ,3167
//                ,3177
//                ,3210
//                ,3224
//                ,3274
//                ,3278
//                ,3321
//                ,3431
//                ,3655
//                ,3713
//                ,3744
//                ,3774
//                ,3813
//                ,3861
//                ,3872
//                ,3875
//                ,3879
//                ,3899
//                ,3919
//                ,3945
//                ,4059
//                ,4070
//                ,4095
//                ,4114
//                ,4148
//                ,4150
//                ,4198
//                ,4203
//                ,4222
//                ,4226
//                ,4227
//                ,4240
//                ,4348
//                ,4419
//                ,4440
//                ,4474
//                ,4522
//                ,4538
//                ,4552
//                ,4657
//                ,4685
//                ,4707
//                ,4729
//                ,4784
//                ,4862
//                ,4864
//                ,4881
//                ,4901
//                ,4904
//                ,5009
//                ,5016
//                ,5019
//                ,5126
//                ,5138
//                ,5187
//                ,5195
//                ,5282
//                ,5302
//                ,5307
//                ,5320
//                ,5323
//                ,5326
//                ,5338
//                ,5430
//                ,5468
//                ,5469
//                ,5472
//                ,5493
//                ,5528
//                ,5550
//                ,5608
//                ,5645
//                ,5676
//                ,5753
//                ,5804
//                ,5844
//                ,5880
//                ,5918
//                ,5933
//                ,6053
//                ,6068
//                ,6069
//                ,6070
//                ,6122
//                ,6171
//                ,6202
//                ,6242
//                ,6275
//                ,6284
//                ,6458
//                ,6532
//                ,6573
//                ,6640
//                ,6654
//                ,6670
//                ,6722
//                ,6785
//                ,6840
//                ,6860
//                ,6914
//                ,6950
//                ,6973
//                ,6993
//                ,7010
//                ,7053
//                ,7142
//                ,7146
//                ,7197
//                ,7252
//                ,7276
//                ,7290
//                ,7295
//                ,7298
//                ,7371
//                ,7404
//                ,7459
//                ,7512
//                ,7530
//                ,7534
//                ,7539
//                ,7555
//                ,7638
//                ,7774
//                ,7781
//                ,7791
//                ,7826
//                ,7871
//                ,7938
//                ,7939
//                ,7963
//                ,7978
//                ,7980
//                ,8047
//                ,8072
//                ,8104
//                ,8145
//                ,8167
//                ,8184
//                ,8226
//                ,8254
//                ,8296
//                ,8309
//                ,8360
//                ,8403
//                ,8454
//                ,8492
//                ,8521
//                ,8531
//                ,8612
//                ,8629
//                ,8639
//                ,8658
//                ,8712
//                ,8729
//                ,8743
//                ,8779
//                ,8804
//                ,8835
//                ,8840
//                ,8864
//                ,8906
//                ,8927
//                ,8976
//                ,8981
//                ,8988
//                ,9033
//                ,9057
//                ,9085
//                ,9086
//                ,9093
//                ,9094
//                ,9183
//                ,9228
//                ,9255
//                ,9293
//                ,9359
//                ,9509
//                ,9541
//                ,9547
//                ,9560
//                ,9572
//                ,9575
//                ,9584
//                ,9615
//                ,9639
//                ,9657
//                ,9766
//                ,9848
//                ,9871
//                ,9922
//                ,9941
//                ,9962
//                ,9966
//                ,9982
//                ,9994
//        };

//    // OFFSET with distance <= 1000
//    std::vector<int> offset{
//        111
//        ,434
//        ,713
//        ,846
//        ,850
//        ,3655
//        ,4904
//        ,5019
//        ,5645
//        ,7530
//        ,7980
//        ,8835};

    /// Distance(query, 20th Centriod) < Distance(query, 9NN GT)
//    std::vector<int> offset{4};
//    std::vector<int> offset{0, 1, 2, 3, 4};

    std::vector<int> offset; for (int i = 0; i < 500; ++i) offset.emplace_back(i);

    std::unordered_set<int> ht(offset.begin(), offset.end());
    if (ht.size() != offset.size()) {
        std::cout << "SIZE MISMATCH!" << std::endl;
        return;
    }
    int slice_size = ht.size();

    // The ground truth binary files for k-NN search consist of the following information:
    //   num_queries(uint32_t) K-NN(uint32) followed by
    //   num_queries X K x sizeof(uint32_t) bytes of data representing the IDs of the K-nearest neighbors of the queries,
    //   followed by num_queries X K x sizeof(float) bytes of data representing the distances to the corresponding points.
    std::ifstream input(input_path, std::ios::binary);
    uint32_t num_queries;
    uint32_t knn;
    assert(input.is_open());
    input.read(reinterpret_cast<char*>(&num_queries), sizeof(num_queries));
    input.read(reinterpret_cast<char*>(&knn), sizeof(knn));
    std::cout << "number of queries: " << num_queries << std::endl;
    std::cout << "k-NN: " << knn << std::endl;

    std::ofstream output(output_path, std::ios::binary);
    assert(output.is_open());
    std::cout << "number of slices: " << slice_size << std::endl;
    output.write(reinterpret_cast<const char*>(&slice_size), sizeof(slice_size));
    output.write(reinterpret_cast<char*>(&knn), sizeof(knn));


    for (size_t i = 0; i < num_queries; ++i) {
        uint32_t nn_id;
        if (ht.find(i) != ht.end()) {
            for (size_t j = 0; j < knn; ++j) {
                input.read(reinterpret_cast<char*>(&nn_id), sizeof(nn_id));
                output.write(reinterpret_cast<char *>(&nn_id), sizeof(nn_id));
            }
        } else {
            for (size_t j = 0; j < knn; ++j) {
                input.read(reinterpret_cast<char*>(&nn_id), sizeof(nn_id));
            }
        }
    }

    for (size_t i = 0; i < num_queries; ++i) {
        float distance;
        if (ht.find(i) != ht.end()) {
            for (size_t j = 0; j < knn; ++j) {
                input.read(reinterpret_cast<char*>(&distance), sizeof(distance));
                output.write(reinterpret_cast<char *>(&distance), sizeof(distance));
            }
        } else {
            for (size_t j = 0; j < knn; ++j) {
                input.read(reinterpret_cast<char*>(&distance), sizeof(distance));
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
    std::cout << "NO Range Search Dataset as Input" << std::endl;
    std::cout << "Please type in the input file input_path into the std::cin:" << std::endl;
    std::cout << "Example: \"/home/jigao/Desktop/GT_10M_v2/GT_10M/bigann-10M\"" << std::endl;
    std::string input_path;
    std::cin >> input_path;
    std::cout << "The input input_path is: " << input_path << std::endl;

    std::cout << "Please type in the output file input_path into the std::cin:" << std::endl;
    std::cout << "Example: \"/home/jigao/Desktop/slice.u8bin\"" << std::endl;
    std::string output_path;
    std::cin >> output_path;
    std::cout << "The output_path is: " << output_path << std::endl;

    read_ground_truth(input_path, output_path);
    return 0;
}
//---------------------------------------------------------------------------
