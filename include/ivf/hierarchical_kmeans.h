#pragma once

#include "ivf/kmeans.h"
#include "ivf/balanced_kmeans.h"
#include "ivf/same_size_kmeans.h"

template <typename T>
void recursive_kmeans(uint32_t k1_id, int64_t cluster_size, T* data, uint32_t* ids, int64_t dim, uint32_t threshold, const uint64_t blk_size,
                      uint32_t& blk_num, IOWriter& data_writer, IOWriter& centroids_writer, IOWriter& centroids_id_writer,
                      bool kmpp = false, float avg_len = 0.0, int64_t niter = 10, int64_t seed = 1234) {

    float weight = 0;
    int vector_size = sizeof(T) * dim;
    int id_size = sizeof(uint32_t);
    int64_t k2;
    if ( weight!=0 && cluster_size > SAME_SIZE_THRESHOLD) {
        k2 = int64_t(sqrt(cluster_size/threshold)) + 1;
    } else {
        k2 = int64_t(cluster_size/threshold) + 1;
    }

    k2 = k2 < MAX_CLUSTER_K2 ? k2 : MAX_CLUSTER_K2;
    float* k2_centroids = new float[k2 * dim];

    std::vector<int64_t> cluster_id(cluster_size, -1);
    std::vector<float> dists(cluster_size, -1);
    std::vector<float> bucket_pre_size(k2 + 1, 0);

    if(cluster_size <= SAME_SIZE_THRESHOLD) {
        //use same size kmeans or graph partition 
        same_size_kmeans<T>(cluster_size, data, dim, k2, k2_centroids, cluster_id.data(), kmpp, niter, seed);
    } else {

        kmeans<T>(cluster_size, data, dim, k2, k2_centroids, kmpp, avg_len, niter, seed);
        // Dynamic balance constraint K-means:
        //balanced_kmeans<T>(cluster_size, data, dim, k2, k2_centroids, weight, kmpp, avg_len, niter, seed);

        if( weight!=0 && cluster_size <= KMEANS_THRESHOLD ) {
            dynamic_assign<T, float, float>(data, k2_centroids, dim, cluster_size, k2, weight, cluster_id.data(), dists.data());
        } else {
            elkan_L2_assign<T, float, float>(data, k2_centroids, dim, cluster_size, k2, cluster_id.data(), dists.data());
        }

        split_clusters_half(dim, k2, cluster_size, data, nullptr, cluster_id.data(), k2_centroids, avg_len);
    }


    //dists is useless, so delete first
    std::vector<float>().swap(dists);

    for (int i=0; i<cluster_size; i++) {
        bucket_pre_size[cluster_id[i]+1]++;
    }
    for (int i=1; i <= k2; i++) {
        bucket_pre_size[i] += bucket_pre_size[i-1];
    }

    //reorder thr data and ids by their cluster id
    T* x_temp = new T[cluster_size * dim];
    uint32_t* ids_temp = new uint32_t[cluster_size];
    int64_t offest;
    memcpy(x_temp, data, cluster_size * vector_size);
    memcpy(ids_temp, ids, cluster_size * id_size);
    for(int i=0; i < cluster_size; i++) {
        offest = (bucket_pre_size[cluster_id[i]]++);
        ids[offest] = ids_temp[i];
        memcpy(data + offest * dim, x_temp + i * dim, vector_size);
    }
    delete [] x_temp;
    delete [] ids_temp;

    int64_t bucket_size;
    int64_t bucket_offest;
    int entry_size = vector_size + id_size;
    uint32_t global_id;

    char* data_blk_buf = new char[blk_size];
    for(int i=0; i < k2; i++) {
        if (i == 0) {
            bucket_size = bucket_pre_size[i];
            bucket_offest = 0;
        } else {
            bucket_size = bucket_pre_size[i] - bucket_pre_size[i - 1];
            bucket_offest = bucket_pre_size[i - 1];
        }
        // std::cout<<"after kmeans : centroids i"<<i<<" has vectors "<<(int)bucket_size<<std::endl;
        if (bucket_size <= threshold) {
            //write a blk to file
            memset(data_blk_buf, 0, blk_size);
            *reinterpret_cast<uint32_t*>(data_blk_buf) = bucket_size;
            char* beg_address = data_blk_buf + sizeof(uint32_t);

            for (int j = 0; j < bucket_size; j++) {
                memcpy(beg_address + j * entry_size, data + dim * (bucket_offest + j), vector_size);
                memcpy(beg_address + j * entry_size + vector_size, ids + bucket_offest + j, id_size);
            }
            global_id = gen_global_block_id(k1_id, blk_num);

            data_writer.write((char *) data_blk_buf, blk_size);
            centroids_writer.write((char *) (k2_centroids + i * dim), sizeof(float) * dim);
            centroids_id_writer.write((char *) (&global_id), sizeof(uint32_t));
            blk_num++;

        } else {
            recursive_kmeans(k1_id, (uint32_t)bucket_size, data + bucket_offest * dim, ids + bucket_offest, dim, threshold, blk_size,
                             blk_num, data_writer, centroids_writer, centroids_id_writer, kmpp, avg_len, niter, seed);
        }
    }
    delete [] data_blk_buf;
    delete [] k2_centroids;

}