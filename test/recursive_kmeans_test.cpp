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

void  hierarchical_clusters() {
  //  TimeRecorder rc("hierarchical clusters");
    std::cout << "hierarchical clusters parameters:" << std::endl;


    uint32_t cluster_size, cluster_dim, ids_size, ids_dim;
    int32_t threshold;
    uint32_t entry_num;

    std::string bucket_centroids_file = "centroids";
    std::string bucket_centroids_id_file = "centroids_id";
    IOWriter centroids_writer(bucket_centroids_file);
    IOWriter centroids_id_writer(bucket_centroids_id_file);
    uint32_t placeholder = 1;
    char* data_blk_buf = new char[blk_size];
    centroids_writer.write((char*)&placeholder, sizeof(uint32_t));
    centroids_writer.write((char*)&placeholder, sizeof(uint32_t));
    centroids_id_writer.write((char*)&placeholder, sizeof(uint32_t));
    centroids_id_writer.write((char*)&placeholder, sizeof(uint32_t));
    uint32_t global_centroids_number = 0;
    uint32_t centroids_dim = 0;

    for (uint32_t i=0; i < 1; i++) {

        std::string data_file = "/home/cqy/dataset/learn.100M.u8bin";
        IOReader data_reader("/home/cqy/dataset/learn.100M.u8bin");
        //IOReader ids_reader(ids_file);

        data_reader.read((char*)&cluster_size, sizeof(uint32_t));
        data_reader.read((char*)&cluster_dim, sizeof(uint32_t));
        ids_size = cluster_size;
        ids_dim = 1;
        //ids_reader.read((char*)&ids_size, sizeof(uint32_t));
        //ids_reader.read((char*)&ids_dim, sizeof(uint32_t));

        centroids_dim = cluster_dim;
        assert(cluster_size == ids_size);
        //assert(ids_dim == 1);
       // assert(threshold > 0);
        uint8_t* datai = new uint8_t[cluster_size * cluster_dim];
        int32_t* idi = new int32_t[ids_size * ids_dim];
        int32_t blk_num = 0;
        data_reader.read((char*)datai, cluster_size * cluster_dim * sizeof(uint8_t));
        for (int i=0;i<ids_size;i++)
        {
            idi[i]=i;
        }
       // ids_reader.read((char*)idi, ids_size * ids_dim * sizeof(uint32_t));
       // cluster_size = 1000000;
        IOWriter data_writer("temp", MEGABYTE * 100);
        memset(data_blk_buf, 0, blk_size);
        *(uint32_t*)(data_blk_buf + 0 * sizeof(uint32_t)) = cluster_size;
        *(uint32_t*)(data_blk_buf + 1 * sizeof(uint32_t)) = cluster_dim;
        *(uint32_t*)(data_blk_buf + 2 * sizeof(uint32_t)) = blk_size;
        *(uint32_t*)(data_blk_buf + 3 * sizeof(uint32_t)) = entry_num;
        *(uint32_t*)(data_blk_buf + 4 * sizeof(uint32_t)) = placeholder;
        *(uint32_t*)(data_blk_buf + 5 * sizeof(uint32_t)) = i;
        data_writer.write((char*)data_blk_buf, blk_size);
        cout<<cluster_size<<" "<<cluster_dim<<endl;
        for(int i=0; i < cluster_dim; i++) {
            cout<<int(datai[(cluster_size-1)*centroids_dim+i])<<" ";
        }
        //cout<<int(datai[(cluster_size-1)*centroids_dim])<<endl;
        int entry_size = (cluster_dim * sizeof(uint8_t) + ids_dim * sizeof(uint32_t));
        threshold = blk_size / entry_size;
        cout<<"threshold"<<threshold<<endl;
        recursive_kmeans<uint8_t>(i, 10000000, datai, idi, cluster_dim, threshold, blk_size, blk_num,
                         data_writer, centroids_writer, centroids_id_writer, false);

        global_centroids_number += blk_num;

        //write back the data's placeholder:
        ofstream data_meta_writer("temp", std::ios::binary | std::ios::in);
        data_meta_writer.seekp(4 * sizeof(uint32_t));
        data_meta_writer.write((char*)&blk_num, sizeof(uint32_t));
        data_meta_writer.close();
        delete [] datai;
        delete [] idi;
    }
    delete [] data_blk_buf;

    // write back the centroids' placeholder:
    uint32_t id_dim = 1;
    ofstream centroids_meta_writer(bucket_centroids_file, std::ios::binary | std::ios::in);
    ofstream centroids_ids_meta_writer(bucket_centroids_id_file, std::ios::binary | std::ios::in);
    centroids_meta_writer.seekp(0);
    centroids_meta_writer.write((char*)&global_centroids_number, sizeof(uint32_t));
    centroids_meta_writer.write((char*)&centroids_dim, sizeof(uint32_t));
    centroids_ids_meta_writer.seekp(0);
    centroids_ids_meta_writer.write((char*)&global_centroids_number, sizeof(uint32_t));
    centroids_ids_meta_writer.write((char*)&id_dim, sizeof(uint32_t));
    centroids_meta_writer.close();
    centroids_ids_meta_writer.close();

    std::cout<<"hierarchical_clusters generate "<<global_centroids_number<<" centroids"<<std::endl;
    return ;
}

int32_t test_recursive_kmeans() {
    uint32_t nb;
    uint32_t dim;

    int32_t blk_num = 0;
    const char* file_name = "/home/cqy/dataset/learn.100M.u8bin";


    IOReader data_reader(file_name);
    data_reader.read((char*)&nb, sizeof(uint32_t));
    data_reader.read((char*)&dim, sizeof(uint32_t));
    nb = 10000000;

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
    int32_t threshold = (blk_size / (dim*sizeof(int8_t) + sizeof(uint32_t)));
    cout<<"threshold"<<threshold<<endl;
    //(int, uint32_t&, uint8_t*&, uint32_t*&, uint32_t&, int&, int64_t&, int32_t&, IOWriter&, IOWriter&, IOWriter&)
    recursive_kmeans<uint8_t>(0, (int64_t)nb, data, ids, (int64_t)dim, threshold, blk_size, blk_num, data_writer, centroids_writer, centroids_id_writer);
    cout<<blk_num<<endl;
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
  //  int32_t blk_num = test_recursive_kmeans();
   // test_hierarchical_clusters(blk_num);
    hierarchical_clusters();
    return 0;
}