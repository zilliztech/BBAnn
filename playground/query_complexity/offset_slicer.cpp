// ----------------------------------------------------------------------------------------------------
#include <sys/mman.h>
#include <cassert>
#include <cstdint>
#include <algorithm>
#include <unordered_set>
// ----------------------------------------------------------------------------------------------------
#include <iostream>
#include <fstream>
//---------------------------------------------------------------------------
namespace {
//---------------------------------------------------------------------------
template <typename T>
void generate_slice(const std::string& input_path, int32_t slice_size, const std::string& output_path) {
// OFFSET with distance <= 5000
//std::vector<int> offset{
//        3
//        ,50
//        ,101
//        ,111
//        ,140
//        ,163
//        ,249
//        ,293
//        ,425
//        ,434
//        ,436
//        ,484
//        ,493
//        ,595
//        ,601
//        ,646
//        ,665
//        ,689
//        ,703
//        ,713
//        ,752
//        ,846
//        ,849
//        ,850
//        ,984
//        ,1166
//        ,1186
//        ,1193
//        ,1258
//        ,1310
//        ,1443
//        ,1521
//        ,1584
//        ,1652
//        ,1767
//        ,1774
//        ,1780
//        ,1786
//        ,1809
//        ,1826
//        ,1831
//        ,1845
//        ,1893
//        ,1986
//        ,2055
//        ,2078
//        ,2092
//        ,2123
//        ,2129
//        ,2162
//        ,2287
//        ,2322
//        ,2388
//        ,2393
//        ,2438
//        ,2480
//        ,2500
//        ,2521
//        ,2574
//        ,2575
//        ,2713
//        ,2746
//        ,2793
//        ,2875
//        ,2881
//        ,2954
//        ,3019
//        ,3053
//        ,3142
//        ,3167
//        ,3177
//        ,3210
//        ,3224
//        ,3274
//        ,3278
//        ,3321
//        ,3431
//        ,3655
//        ,3713
//        ,3744
//        ,3774
//        ,3813
//        ,3861
//        ,3872
//        ,3875
//        ,3879
//        ,3899
//        ,3919
//        ,3945
//        ,4059
//        ,4070
//        ,4095
//        ,4114
//        ,4148
//        ,4150
//        ,4198
//        ,4203
//        ,4222
//        ,4226
//        ,4227
//        ,4240
//        ,4348
//        ,4419
//        ,4440
//        ,4474
//        ,4522
//        ,4538
//        ,4552
//        ,4657
//        ,4685
//        ,4707
//        ,4729
//        ,4784
//        ,4862
//        ,4864
//        ,4881
//        ,4901
//        ,4904
//        ,5009
//        ,5016
//        ,5019
//        ,5126
//        ,5138
//        ,5187
//        ,5195
//        ,5282
//        ,5302
//        ,5307
//        ,5320
//        ,5323
//        ,5326
//        ,5338
//        ,5430
//        ,5468
//        ,5469
//        ,5472
//        ,5493
//        ,5528
//        ,5550
//        ,5608
//        ,5645
//        ,5676
//        ,5753
//        ,5804
//        ,5844
//        ,5880
//        ,5918
//        ,5933
//        ,6053
//        ,6068
//        ,6069
//        ,6070
//        ,6122
//        ,6171
//        ,6202
//        ,6242
//        ,6275
//        ,6284
//        ,6458
//        ,6532
//        ,6573
//        ,6640
//        ,6654
//        ,6670
//        ,6722
//        ,6785
//        ,6840
//        ,6860
//        ,6914
//        ,6950
//        ,6973
//        ,6993
//        ,7010
//        ,7053
//        ,7142
//        ,7146
//        ,7197
//        ,7252
//        ,7276
//        ,7290
//        ,7295
//        ,7298
//        ,7371
//        ,7404
//        ,7459
//        ,7512
//        ,7530
//        ,7534
//        ,7539
//        ,7555
//        ,7638
//        ,7774
//        ,7781
//        ,7791
//        ,7826
//        ,7871
//        ,7938
//        ,7939
//        ,7963
//        ,7978
//        ,7980
//        ,8047
//        ,8072
//        ,8104
//        ,8145
//        ,8167
//        ,8184
//        ,8226
//        ,8254
//        ,8296
//        ,8309
//        ,8360
//        ,8403
//        ,8454
//        ,8492
//        ,8521
//        ,8531
//        ,8612
//        ,8629
//        ,8639
//        ,8658
//        ,8712
//        ,8729
//        ,8743
//        ,8779
//        ,8804
//        ,8835
//        ,8840
//        ,8864
//        ,8906
//        ,8927
//        ,8976
//        ,8981
//        ,8988
//        ,9033
//        ,9057
//        ,9085
//        ,9086
//        ,9093
//        ,9094
//        ,9183
//        ,9228
//        ,9255
//        ,9293
//        ,9359
//        ,9509
//        ,9541
//        ,9547
//        ,9560
//        ,9572
//        ,9575
//        ,9584
//        ,9615
//        ,9639
//        ,9657
//        ,9766
//        ,9848
//        ,9871
//        ,9922
//        ,9941
//        ,9962
//        ,9966
//        ,9982
//        ,9994
//    };

//    // OFFSET with distance <= 1000
//    std::vector<int> offset{
//    111
//    ,434
//    ,713
//    ,846
//    ,850
//    ,3655
//    ,4904
//    ,5019
//    ,5645
//    ,7530
//    ,7980
//    ,8835};

//    /// Distance(query, 20th Centriod) < Distance(query, 9NN GT)
//    std::vector<int> offset{4};

//    std::vector<int> offset{0, 1, 2, 3, 4};

    // Dataset SSN (range query): query with number of results >= 50 percentile num_results_per_query
    std::vector<int> offset{623,648,878,1336,1650,1962,2284,2391,3275,3532,3542,4135,4389,5118,5720,5751,6204,7027,7306,9459,9477,10381,10626,11261,12508,13210,13273,13502,13667,14615,15172,17451,17584,18202,18413,18594,19377,19973,20102,20467,20560,20566,21212,21488,21556,22075,22505,22794,22944,22994,23845,23977,24256,24286,25282,25728,26012,26310,26658,27034,28810,28888,29032,29088,29329,30385,30900,31576,32371,32633,33327,34140,34174,34189,34190,34264,34522,34986,35568,35707,35938,36260,36320,36402,36520,38185,38797,39095,40110,40762,40963,41306,41329,41385,41417,41440,43460,43682,44874,45252,45408,45423,45641,45782,45847,46353,46792,46960,47594,47893,47898,48006,48269,48717,48793,48903,49784,50064,50066,50157,50229,50635,51291,51319,51753,52329,52579,52611,53349,53550,54081,55093,56664,56799,57515,57841,58244,58510,59733,60363,60479,60663,60745,61491,61962,62234,62435,62785,63278,63282,63632,63754,64336,64915,65416,65437,65588,65679,67560,68238,68365,68826,68837,69013,69695,69907,69986,70685,70790,70800,71193,71325,71346,71385,71689,72490,72618,72692,72757,72881,73185,73330,73334,73684,73765,75841,76342,76417,76470,76474,77018,78065,78194,78353,78370,78596,79123,79287,80340,80609,81312,81321,82184,82488,82512,83471,83626,83747,84046,84200,84340,84431,84484,84566,85089,85269,85280,85760,86135,86751,87093,87826,88130,90006,90053,90144,90541,90543,91952,92256,92980,93548,93679,93743,94168,94835,95274,95673,95731,96532,96898,97087,98047,98539,99099,99588};


    std::unordered_set<int> ht(offset.begin(), offset.end());
    if (ht.size() != offset.size()) {
        std::cout << "SIZE MISMATCH!" << std::endl;
        return;
    }
    slice_size = ht.size();

    // All datasets are in the common binary format that starts with
    // 8 bytes of data consisting of num_points(uint32_t) num_dimensions(uint32)
    // followed by num_pts X num_dimensions x sizeof(type) bytes of data stored one vector after another.
    std::ifstream input(input_path, std::ios::binary);
    uint32_t num_points;
    uint32_t num_dimensions;
    assert(input.is_open());
    input.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));
    input.read(reinterpret_cast<char*>(&num_dimensions), sizeof(num_dimensions));
    assert(num_points >= slice_size);
    std::cout << "number of points: " << num_points << std::endl;
    std::cout << "number of dimensions: " << num_dimensions << std::endl;

    std::ofstream output(output_path, std::ios::binary);
    assert(output.is_open());
    std::cout << "number of slices: " << slice_size << std::endl;
    output.write(reinterpret_cast<const char*>(&slice_size), sizeof(slice_size));
    output.write(reinterpret_cast<char*>(&num_dimensions), sizeof(num_dimensions));

    T ele;
    for (int i = 0; i < num_points; ++i) {
        if (ht.find(i) != ht.end()) {
            for (int j = 0; j < num_dimensions; ++j) {
                input.read(reinterpret_cast<char*>(&ele), sizeof(ele));
                output.write(reinterpret_cast<char *>(&ele), sizeof(ele));
            }
        } else {
            for (int j = 0; j < num_dimensions; ++j) {
                input.read(reinterpret_cast<char*>(&ele), sizeof(ele));
            }
        }
    }
    input.close();
    output.close();

    {
        // Validation: reading the output file
        std::cout << "Validation: reading the output file" << std::endl;
        std::ifstream output(output_path, std::ios::binary);
        assert(output.is_open());
        output.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));
        assert(num_points == slice_size);
        output.read(reinterpret_cast<char*>(&num_dimensions), sizeof(num_dimensions));
        std::cout << "number of points: " << num_points << std::endl;
        std::cout << "number of dimensions: " << num_dimensions << std::endl;
        T ele;
        for (int i = 0; i < slice_size; ++i) {
            for (int j = 0; j < num_dimensions; ++j) {
                output.read(reinterpret_cast<char*>(&ele), sizeof(ele));
                int temp = ele;
                std::cout << temp << ",";
            }
            std::cout << std::endl;
        }
    }
}
//---------------------------------------------------------------------------
} // namespace
//---------------------------------------------------------------------------
int main() {
    std::cout << "Please type in the input file input_path into the std::cin:" << std::endl;
    std::cout << "Example: \"/home/jigao/Desktop/learn.100M.u8bin\"" << std::endl;
    std::string input_path;
    std::cin >> input_path;
    std::cout << "The input input_path is: " << input_path << std::endl;
//
//    std::cout << "Please type in the size of slice into the std::cin:" << std::endl;
//    std::cout << "Example: \"1000\"" << std::endl;
//    size_t slice_size;
//    std::cin >> slice_size;
//    std::cout << "The size of slice is: " << slice_size << std::endl;

    std::cout << "Please type in the data type: uint8, int8 or float32: " << std::endl;
    std::cout << "Example: \"uint8\"" << std::endl;
    std::string data_type;
    std::cin >> data_type;
    if (data_type != "uint8" && data_type != "int8" && data_type != "float32") {
        std::cerr << "Wrong data type: " << data_type << std::endl;
        return 1;
    }
    std::cout << "The data type is: " << data_type << std::endl;

    std::cout << "Please type in the output file input_path into the std::cin:" << std::endl;
    std::cout << "Example: \"/home/jigao/Desktop/slice.u8bin\"" << std::endl;
    std::string output_path;
    std::cin >> output_path;
    std::cout << "The output_path is: " << output_path << std::endl;

    if (data_type == "uint8" || data_type == "int8") {
        generate_slice<uint8_t>(input_path, 0, output_path);
    } else if (data_type == "float32") {
        generate_slice<float>(input_path, 0, output_path);
    }
    return 0;
}
//---------------------------------------------------------------------------
