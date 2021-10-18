//
// Created by cqy on 21-9-10.
//

#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <unistd.h>
#include <stdio.h>
#include "ivf/hierarchical_kmeans.h"
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
        uint32_t* idi = new uint32_t[ids_size * ids_dim];
        uint32_t blk_num = 0;
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

        int entry_size = (cluster_dim * sizeof(uint8_t) + ids_dim * sizeof(uint32_t));
        threshold = blk_size / entry_size;
        cout<<"threshold"<<threshold<<endl;

        recursive_kmeans<uint8_t>(i, 1000000, datai, idi, cluster_dim, threshold, blk_size, blk_num, data_writer, centroids_writer, centroids_id_writer, 0, false);

        global_centroids_number += blk_num;

        delete [] datai;
        delete [] idi;
    }
    delete [] data_blk_buf;


    std::cout<<"hierarchical_clusters generate "<<global_centroids_number<<" centroids"<<std::endl;
    return ;
}

void test_split_clusters_half() {
    int64_t dim = 128;
    int64_t k = 2;
    int64_t n = 200;
    int64_t* hassign = nullptr;
    uint8_t* x_in = new uint8_t [n*dim];
    cout <<"init x_in"<<endl;
    for (int i=0; i<n ;i++) {
        for (int j=0; j<dim; j++) {
            x_in[i*dim + j] = (i*11+j)%255;
            //x_in[i*dim + j] = 1;
        }
    }
    int64_t * assign = new int64_t[n];
    memset(assign, 0, n);
    cout<<"init centroids"<<endl;
    float* centroids = new float [k*dim];
    for (int i=0; i<k; i++) {
        for(int j=0;j<dim;j++){
            centroids[i*dim + j] = i*1.1+j;
        }
    }
    cout<<"begin split"<<endl;
    split_clusters_half<uint8_t> (dim, k, n, x_in, hassign, assign, centroids);
    if(hassign) {
        cout<<"delete hassign"<<endl;
        delete [] hassign;
    }
    //cout<<"hassign"<<endl;
  //  for (int i =0; i<k;i++) cout<<hassign[i]<<endl;
    cout<<"assign"<<endl;
    for (int i=0;i<n;i++) cout<<assign[i]<<endl;
    cout<<"centroids"<<endl;
    for (int i=0;i<k;i++) {
        cout<<endl;
        for (int j=0;j<dim;j++)
        cout<<centroids[i*dim +j]<<" ";
    }

}


void test_split_10m() {
    uint32_t cluster_size, cluster_dim, ids_size, ids_dim;
    int32_t threshold;
    uint32_t entry_num;

    std::string data_file = "/home/cqy/dataset/learn.100M.u8bin";
    IOReader data_reader("/home/cqy/dataset/learn.100M.u8bin");
    //IOReader ids_reader(ids_file);

    data_reader.read((char*)&cluster_size, sizeof(uint32_t));
    data_reader.read((char*)&cluster_dim, sizeof(uint32_t));
    uint8_t* buf = new uint8_t [cluster_size * cluster_dim];
    data_reader.read((char*)buf, sizeof(uint8_t) * cluster_dim * cluster_size);
}

int main() {
  //  int32_t blk_num = test_recursive_kmeans();
   // test_hierarchical_clusters(blk_num);
    hierarchical_clusters();
   //test_split_clusters_half();
    return 0;
}