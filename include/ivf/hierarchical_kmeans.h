#pragma once

#include "ivf/kmeans.h"
#include "ivf/balanced_kmeans.h"
#include "ivf/same_size_kmeans.h"

template<typename T>
void find_nearest_large_bucket (
        const T *  x,
        const float * centroids,
        int64_t nx, int64_t k, int64_t dim, int64_t * hassign, int64_t * transform_table,
        int64_t large_bucket_min_limit, int64_t small_bucket_max_limit, std::vector<int64_t>& assign)
{
    std::vector<int64_t> new_assign(nx, -1);
#pragma omp parallel for
    for(int i = 0; i < nx; i++) {
        auto *x_i =  x + i * dim;
        int min_id = 0;
        float min_dist ;
        float dist ;
        if (hassign[assign[i]] > small_bucket_max_limit) {
            new_assign[i] = transform_table[assign[i]];
        } else {
            min_dist = L2sqr<const T, const float ,float >(x_i, centroids + transform_table[min_id] * dim, dim);

            for (int j = min_id; j < k; j++) {

                dist = L2sqr<const T, const float, float>(x_i, centroids + transform_table[j] *dim , dim);
                if(dist < min_dist) {
                    min_dist = dist;
                    min_id = j;
                }
            }
            new_assign[i] = min_id;
        }
    }
    assign.assign(new_assign.begin(), new_assign.end());
}

template <typename T>
void merge_clusters(LevelType level, int64_t dim, int64_t nx, int64_t& k, const T* x,
                    std::vector<int64_t>& assign, std::vector<float>& centroids, float avg_len = 0.0) {

    int64_t* hassign = new int64_t [k];
    memset(hassign, 0, sizeof(int64_t) * k);
    for (int i = 0; i < nx; i++) {
        hassign[assign[i]]++;
    }

    int64_t large_bucket_min_limit;
    int64_t small_bucket_max_limit;

    if(level == LevelType::FIRST_LEVEL || level == LevelType::SECOND_LEVEL) {

        large_bucket_min_limit = MAX_SAME_SIZE_THRESHOLD;
        small_bucket_max_limit = MAX_SAME_SIZE_THRESHOLD;

    } else if (level == LevelType::THIRTH_LEVEL || level == LevelType::BALANCE_LEVEL) {

        large_bucket_min_limit = MAX_SAME_SIZE_THRESHOLD;
        small_bucket_max_limit = MIN_SAME_SIZE_THRESHOLD;

    } else {
        //error
        assert(level < LevelType::FINAL_LEVEL);

    }

    //find the new k2 and centroids:
    int64_t new_k = 0;
    int64_t large_bucket_num = 0;
    int64_t * transform_table = new int64_t [k];
    for (int i=0; i < k; i++ ) {
        if(hassign[i] >= large_bucket_min_limit) {
            transform_table[i] = large_bucket_num;
            large_bucket_num++;
        } else {
            transform_table[i] = -1;
        }
    }
    new_k += large_bucket_num;
    for (int i = 0; i < k; i++) {
        if (hassign[i] >= small_bucket_max_limit && transform_table[i] == -1) {
            transform_table[i] = new_k;
            new_k++;
        }
    }
    if(new_k == k) {
      //  delete [] transform_table;
        return ;
    }
    new_k = large_bucket_num ? new_k : new_k + 1; // add a bucket for all small bucket

    int64_t * new_hassign = new int64_t [new_k];
    float * new_centroids = new float[dim * new_k];
    for (int i = 0; i < k; i++) {
        if(transform_table[i] != -1) {
            memcpy(new_centroids + transform_table[i] * dim, centroids.data() + i * dim, dim);
        }
    }
    if (large_bucket_num) {

        find_nearest_large_bucket<T>(x, new_centroids, nx, large_bucket_num, dim, hassign, transform_table,
                large_bucket_min_limit, small_bucket_max_limit, assign);

        compute_centroids<T>(dim, new_k, nx, x, assign.data(), new_hassign, new_centroids, avg_len);

    } else {

        float * __restrict merge_centroid = new_centroids + (new_k - 1) * dim;
        int64_t merge_centroid_id = new_k -1;
        memset(merge_centroid, 0, sizeof(float) * dim);
#pragma omp parallel for
        for (int i = 0; i < nx; i++) {
            auto * __restrict x_in = x + i *dim;
            if(transform_table[assign[i]] == -1) {
                for (int d = 0; d < dim; d++) {
                    merge_centroid[d] += x_in[d];
                }
                assign[i] = merge_centroid_id;
            } else {
                assign[i] = transform_table[assign[i]];
            }
        }

        if (avg_len != 0.0) {
            float len = avg_len / sqrt(IP<float, float, double>(merge_centroid, merge_centroid, dim));
            for (int64_t j = 0; j < dim; j++){
                merge_centroid[j] *= len;
            }
        } else {
            float norm = 1.0 / hassign[merge_centroid_id];
            for (int64_t j = 0; j < dim; j++) {
                merge_centroid[j] *= norm;
            }
        }
    }

    //update meta :
    k = new_k;
    centroids.assign(new_centroids, new_centroids + k * dim);

    delete [] new_centroids;
    delete [] new_hassign;
    delete [] transform_table;
    delete [] hassign;


    return ;
}


template <typename T>
void recursive_kmeans(uint32_t k1_id, int64_t cluster_size, T* data, uint32_t* ids, int64_t dim, uint32_t threshold, const uint64_t blk_size,
                      uint32_t& blk_num, IOWriter& data_writer, IOWriter& centroids_writer, IOWriter& centroids_id_writer, int level,
                      bool kmpp = false, float avg_len = 0.0, int64_t niter = 10, int64_t seed = 1234) {
    std::cout<< "level" <<level<<" cluster_size"<<cluster_size<<std::endl;

    float weight = 0;
    int vector_size = sizeof(T) * dim;
    int id_size = sizeof(uint32_t);
    int64_t k2;
    bool do_same_size_kmeans = (LevelType (level) == LevelType ::FINAL_LEVEL) ||
            (LevelType (level) == LevelType ::SECOND_LEVEL && cluster_size >= MIN_SAME_SIZE_THRESHOLD && cluster_size <= MAX_SAME_SIZE_THRESHOLD);
    if(do_same_size_kmeans) {
        k2 = int64_t(cluster_size/threshold) + 1;
    } else {
        k2 = int64_t(sqrt(cluster_size/threshold)) + 1;
    }
    /*
    if ( weight!=0 && cluster_size > SAME_SIZE_THRESHOLD) {
        k2 = int64_t(sqrt(cluster_size/threshold)) + 1;
    } else {
        k2 = int64_t(cluster_size/threshold) + 1;
    }*/

    k2 = k2 < MAX_CLUSTER_K2 ? k2 : MAX_CLUSTER_K2;
    //float* k2_centroids = new float[k2 * dim];
    std::vector<float> k2_centroids(k2 * dim, 0.0);
    std::vector<int64_t> cluster_id(cluster_size, -1);



    if(do_same_size_kmeans) {
        //use same size kmeans or graph partition

        same_size_kmeans<T>(cluster_size, data, dim, k2, k2_centroids.data(), cluster_id.data(), kmpp, avg_len, niter, seed);

    } else {
        kmeans<T>(cluster_size, data, dim, k2, k2_centroids.data(), kmpp, avg_len, niter, seed);
        // Dynamic balance constraint K-means:
        // balanced_kmeans<T>(cluster_size, data, dim, k2, k2_centroids, weight, kmpp, avg_len, niter, seed);
        std::vector<float> dists(cluster_size, -1);
        if( weight!=0 && cluster_size <= KMEANS_THRESHOLD ) {
            dynamic_assign<T, float, float>(data, k2_centroids.data(), dim, cluster_size, k2, weight, cluster_id.data(), dists.data());
        } else {
            elkan_L2_assign<T, float, float>(data, k2_centroids.data(), dim, cluster_size, k2, cluster_id.data(), dists.data());
        }

        //dists is useless, so delete first
        std::vector<float>().swap(dists);

        merge_clusters<T>((LevelType)level, dim, cluster_size, k2, data, cluster_id, k2_centroids, avg_len);

        //split_clusters_half(dim, k2, cluster_size, data, nullptr, cluster_id.data(), k2_centroids, avg_len);
    }

    std::vector<float> bucket_pre_size(k2 + 1, 0);

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
            centroids_writer.write((char *) (k2_centroids.data() + i * dim), sizeof(float) * dim);
            centroids_id_writer.write((char *) (&global_id), sizeof(uint32_t));
            blk_num++;

        } else {
            recursive_kmeans(k1_id, (uint32_t)bucket_size, data + bucket_offest * dim, ids + bucket_offest, dim, threshold, blk_size,
                             blk_num, data_writer, centroids_writer, centroids_id_writer, level + 1, kmpp, avg_len, niter, seed);
        }
    }
    delete [] data_blk_buf;
}