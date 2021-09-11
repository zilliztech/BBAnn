//
// Created by cqy on 21-9-10.
//

#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <unistd.h>
#include <stdio.h>
#include "ivf/clustering.h"
#include "util/file_handler.h"
using namespace std;
int64_t blk_size = 12*1024;
int32_t test_recursive_kmeans() {
    uint32_t nb;
    uint32_t dim;

    int32_t blk_num = 0;
    const char* file_name = "/home/cqy/dataset/learn.100M.u8bin";


    IOReader data_reader(file_name);
    data_reader.read((char*)&nb, sizeof(uint32_t));
    data_reader.read((char*)&dim, sizeof(uint32_t));
    nb = 1000000;

    uint8_t* data = new uint8_t[dim*nb];
    int32_t* ids = new int32_t[nb];
    float aver_norm = 0.0;
    float norm = 0.0;
    cout<<"dataset size "<<nb<<" dim"<< dim << endl;
    data_reader.read((char*)data, sizeof(uint8_t)* dim*(uint64_t)nb);
    cout<<"calculate "<<endl;
    cout<<"calculate1 "<<endl;
    for (int i=0; i<nb; i++) {
        ids[i] =i;
        norm = 0.0;
        for (int j=0; j<dim; j++) {
            norm += data[i*dim+j]* data[i*dim+j];
        }
        aver_norm += norm;

    }
    aver_norm = aver_norm/nb;
    cout<<"aver norm"<<aver_norm<<endl;
    IOWriter centroids_id_writer("centroids_id");
    IOWriter centroids_writer("centroids");
    IOWriter data_writer("data");
    cout<<"begin recursive_kmeans"<<endl;
    int threshold = (blk_size / (dim*sizeof(int8_t) + sizeof(uint32_t)));
    cout<<"threshold"<<threshold<<endl;
    //(int, uint32_t&, uint8_t*&, uint32_t*&, uint32_t&, int&, int64_t&, int32_t&, IOWriter&, IOWriter&, IOWriter&)
    recursive_kmeans<uint8_t>(0, (int64_t)nb, data, ids, (int64_t)dim, threshold, blk_size, blk_num, data_writer, centroids_writer, centroids_id_writer);
    return blk_num;

}
void test_hierarchical_clusters(int blk_num ) {
    IOReader result_reader("data");
    int min = 100000;
    int max = 0;
    int64_t total = 0;
    uint8_t temp_size = 0;
    char* buf = new char [blk_size];

    for (int i=0; i<blk_num; i++) {
        result_reader.read(buf, blk_size);
        temp_size = buf[0];
        total += temp_size;
        if (temp_size < min) min = temp_size;
        if (temp_size > max) max = temp_size;
    }
    cout << "blk number :" << blk_num << "aver.num" << total/blk_num <<"min num "<<(int)min<<" max num "<<(int)max<<endl;
    return ;
}

int main() {
    int32_t blk_num = test_recursive_kmeans();
    test_hierarchical_clusters(blk_num);
    return 0;
}